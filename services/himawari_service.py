import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from adapters.himawari_adapter import (
    BAND_DESCRIPTION_EN,
    BAND_CATALOG,
    CHINA_EXTENT,
    DEFAULT_LATEST_DELAY_MINUTES,
    DEFAULT_WINDOW_HOURS,
    LATLON_RESOLUTION,
    build_latlon_grid,
    latest_himawari_slot,
    normalize_himawari_meta,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Himawari"
STATIC_PREFIX = "/data/Himawari"
BEIJING_OFFSET = timedelta(hours=8)
DISPLAY_IMAGE_PATTERNS = ("*.webp", "*.png")


def _as_posix(path: Path | None) -> str | None:
    return str(path).replace("\\", "/") if path else None


def _png_url(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    try:
        relative = path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        return None
    return f"{STATIC_PREFIX}/{relative.as_posix()}"


def _display_image_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in DISPLAY_IMAGE_PATTERNS:
        files.extend(folder.glob(pattern))
    return files


def _window_hours_env() -> int:
    env = os.environ
    key = "HIMAWARI_WINDOW_HOURS" if "HIMAWARI_WINDOW_HOURS" in env else "HIMAWARI_BACKFILL_HOURS"
    try:
        return max(1, int(env.get(key, DEFAULT_WINDOW_HOURS)))
    except ValueError:
        return DEFAULT_WINDOW_HOURS


def _latest_delay_minutes_env() -> int:
    try:
        return max(0, int(os.environ.get("HIMAWARI_LATEST_DELAY_MINUTES", DEFAULT_LATEST_DELAY_MINUTES)))
    except ValueError:
        return DEFAULT_LATEST_DELAY_MINUTES


def _product_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("key") or "").strip()


def _existing_path(value: str | None, meta_path: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return path
    if meta_path:
        for folder in ("latlon", "composites"):
            if folder not in path.parts:
                continue
            relative_parts = path.parts[path.parts.index(folder) :]
            candidate = meta_path.parent.parent.joinpath(*relative_parts)
            if candidate.exists():
                return candidate
    return None


def _select_default_png(meta_json: dict[str, Any] | None, fallback: list[Path], meta_path: Path | None = None) -> Path | None:
    if not meta_json:
        return fallback[0] if fallback else None

    for key in ("B13", "B14", "B08"):
        for item in meta_json.get("variables", []):
            if _product_name(item) == key and (path := _existing_path(item.get("png"), meta_path)):
                return path

    for key in ("true_color", "natural_color", "water_vapor_enhanced"):
        for item in meta_json.get("composites", []):
            if _product_name(item) == key and (path := _existing_path(item.get("png"), meta_path)):
                return path

    return fallback[0] if fallback else None


def _with_png_urls(items: list[dict[str, Any]], meta_path: Path | None = None) -> list[dict[str, Any]]:
    enriched = []
    for item in items:
        path = _existing_path(item.get("png"), meta_path)
        item_copy = {**item}
        item_copy.pop("png_data_url", None)
        enriched.append({**item_copy, "png_url": _png_url(path)})
    return enriched


def _parse_observation_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _scene_time_from_path(meta_path: Path) -> datetime | None:
    try:
        date = meta_path.parents[2].name
        scene_time = meta_path.parents[1].name
        return datetime.strptime(f"{date}{scene_time}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _scene_time_from_scene_dir(scene_dir: Path) -> datetime | None:
    try:
        return datetime.strptime(f"{scene_dir.parent.name}{scene_dir.name}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _legend_ticks(vmin: float, vmax: float, count: int = 4) -> list[str]:
    if count <= 1 or vmax <= vmin:
        return [f"{vmin:g}", f"{vmax:g}"]
    step = (vmax - vmin) / (count - 1)
    return [f"{vmin + step * index:g}" for index in range(count)]


def _partial_variable_from_png(scene_dir: Path, png_path: Path, grid: dict[str, Any]) -> dict[str, Any] | None:
    band = png_path.stem.upper()
    catalog = BAND_CATALOG.get(band)
    if not catalog:
        return None
    vmin = float(catalog["vmin"])
    vmax = float(catalog["vmax"])
    return {
        "name": catalog["key"],
        "long_name": catalog["plain_name"],
        "short_name": catalog["key"],
        "raw_name": catalog["key"],
        "name_cn": catalog["name_zh"],
        "unit": catalog["unit"],
        "display_unit": catalog["display_unit"],
        "shape": [grid["ny"], grid["nx"]],
        "dims": ["lat", "lon"],
        "level": None,
        "missing": None,
        "stats": {"min": None, "max": None, "mean": None, "std": None},
        "category": catalog["category"],
        "description": catalog["description"],
        "description_zh": catalog["description"],
        "description_en": BAND_DESCRIPTION_EN.get(band),
        "wavelength": catalog["wavelength"],
        "vmin": vmin,
        "vmax": vmax,
        "legend_ticks": _legend_ticks(vmin, vmax),
        "png": png_path.as_posix(),
        "float32": None,
        "netcdf": None,
    }


def _read_partial_png_entry(scene_dir: Path) -> dict[str, Any] | None:
    if (scene_dir / "meta" / "scene.meta.json").exists():
        return None
    observed = _scene_time_from_scene_dir(scene_dir)
    if not observed:
        return None
    png_files = sorted(
        [item for item in _display_image_files(scene_dir / "latlon") if item.stem.upper().startswith("B")],
        key=lambda item: item.stem,
    )
    if not png_files:
        return None
    grid = build_latlon_grid(CHINA_EXTENT, LATLON_RESOLUTION)
    variables = [
        variable
        for png_path in png_files
        if (variable := _partial_variable_from_png(scene_dir, png_path, grid))
    ]
    if not variables:
        return None
    meta_path = scene_dir / "meta" / "scene.meta.json"
    meta_json = normalize_himawari_meta(
        {
            "scene_id": f"{scene_dir.parent.name}_{scene_dir.name}",
            "satellite": "Himawari-9",
            "observation_time": observed.isoformat().replace("+00:00", "Z"),
            "projection": grid["projection"],
            "grid_type": grid["grid_type"],
            "extent": grid["extent"],
            "resolution": grid["resolution"],
            "grid": {"nx": grid["nx"], "ny": grid["ny"]},
            "variables": variables,
            "composites": [],
            "loaded_bands": [item["name"] for item in variables],
            "source_raw_dir": (scene_dir / "raw").as_posix(),
            "raw_file_count": len(list((scene_dir / "raw").glob("HS_H*.DAT*"))),
            "retention_managed": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "extra": {"status": "partial"},
        },
        meta_path,
    )
    return {
        "path": meta_path,
        "meta": meta_json,
        "observed": observed,
        "scene_id": meta_json.get("scene_id") or scene_dir.name,
    }


def _read_meta_entry(meta_path: Path) -> dict[str, Any] | None:
    try:
        with meta_path.open("r", encoding="utf-8") as file:
            meta_json = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    meta_json = normalize_himawari_meta(meta_json, meta_path)
    observed = _parse_observation_time(meta_json.get("observation_time")) or _scene_time_from_path(meta_path)
    return {
        "path": meta_path,
        "meta": meta_json,
        "observed": observed,
        "scene_id": meta_json.get("scene_id") or meta_path.parents[1].name,
    }


def _display_window_bounds(
    now: datetime | None = None,
    retention_hours: int = DEFAULT_WINDOW_HOURS,
    delay_minutes: int = DEFAULT_LATEST_DELAY_MINUTES,
) -> tuple[datetime, datetime]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    latest_date, latest_time = latest_himawari_slot(now=current, delay_minutes=delay_minutes)
    latest = datetime.strptime(f"{latest_date}{latest_time}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return latest - timedelta(hours=retention_hours), latest


def _meta_entries(
    retention_hours: int = DEFAULT_WINDOW_HOURS,
    now: datetime | None = None,
    delay_minutes: int = DEFAULT_LATEST_DELAY_MINUTES,
) -> list[dict[str, Any]]:
    entries = []
    for meta_path in list(DATA_DIR.glob("*/*/meta/scene.meta.json")) + list(DATA_DIR.glob("*.meta.json")):
        if entry := _read_meta_entry(meta_path):
            entries.append(entry)
    for scene_dir in DATA_DIR.glob("*/*"):
        if scene_dir.is_dir() and (entry := _read_partial_png_entry(scene_dir)):
            entries.append(entry)
    if not entries:
        return []
    cutoff, latest = _display_window_bounds(now=now, retention_hours=retention_hours, delay_minutes=delay_minutes)
    sorted_entries = sorted(
        entries,
        key=lambda item: (item["observed"] or datetime.min.replace(tzinfo=timezone.utc), item["path"].as_posix()),
    )
    window_entries = [
        entry
        for entry in sorted_entries
        if not entry["observed"] or cutoff <= entry["observed"] <= latest
    ]
    return window_entries or sorted_entries


def _timeline_item(entry: dict[str, Any]) -> dict[str, Any]:
    observed = entry["observed"]
    label_time = observed + BEIJING_OFFSET if observed else None
    return {
        "scene_id": entry["scene_id"],
        "time": observed.isoformat().replace("+00:00", "Z") if observed else entry["meta"].get("observation_time"),
        "label": label_time.strftime("%m-%d %H:%M") if label_time else entry["scene_id"],
        "label_timezone": "Asia/Shanghai",
    }


def _select_entry(entries: list[dict[str, Any]], scene_id: str | None = None) -> dict[str, Any] | None:
    if not entries:
        return None
    if scene_id:
        for entry in entries:
            if entry["scene_id"] == scene_id:
                return entry
    return entries[-1]


def get_display_data(
    scene_id: str | None = None,
    retention_hours: int | None = None,
    now: datetime | None = None,
    delay_minutes: int | None = None,
) -> dict[str, Any]:
    retention_hours = _window_hours_env() if retention_hours is None else retention_hours
    delay_minutes = _latest_delay_minutes_env() if delay_minutes is None else delay_minutes
    entries = _meta_entries(retention_hours=retention_hours, now=now, delay_minutes=delay_minutes)
    selected = _select_entry(entries, scene_id)
    meta_files = [entry["path"] for entry in entries]
    png_files = sorted(
        [
            *[path for pattern in DISPLAY_IMAGE_PATTERNS for path in DATA_DIR.glob(f"*/*/latlon/{pattern}")],
            *[path for pattern in DISPLAY_IMAGE_PATTERNS for path in DATA_DIR.glob(f"*/*/composites/{pattern}")],
            *[path for pattern in DISPLAY_IMAGE_PATTERNS for path in DATA_DIR.glob(pattern)],
        ],
        key=lambda item: (item.stat().st_mtime, item.as_posix()),
        reverse=True,
    )

    meta_json = None
    if selected:
        meta_json = selected["meta"]

    meta_path = selected["path"] if selected else None
    png_path = _select_default_png(meta_json, png_files, meta_path)
    variables = _with_png_urls(meta_json.get("variables", []), meta_path) if meta_json else []
    composites = _with_png_urls(meta_json.get("composites", []), meta_path) if meta_json else []

    return {
        "business_type": "Himawari",
        "meta_file": _as_posix(meta_path),
        "meta_json": meta_json,
        "weather_info": meta_json.get("weather_info") if meta_json else None,
        "extent": (meta_json.get("extent") or meta_json.get("bbox")) if meta_json else None,
        "grid": meta_json.get("grid") if meta_json else None,
        "timeline": [_timeline_item(entry) for entry in entries],
        "variables": variables,
        "composites": composites,
        "products": composites + variables,
        "png": _as_posix(png_path),
        "png_url": _png_url(png_path),
        "png_files": [_as_posix(path) for path in png_files if "latlon" in path.parts or "composites" in path.parts],
    }
