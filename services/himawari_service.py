import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from adapters.himawari_adapter import normalize_himawari_meta


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Himawari"
STATIC_PREFIX = "/data/Himawari"


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


def _meta_entries(retention_hours: int = 24) -> list[dict[str, Any]]:
    entries = []
    for meta_path in list(DATA_DIR.glob("*/*/meta/scene.meta.json")) + list(DATA_DIR.glob("*.meta.json")):
        if entry := _read_meta_entry(meta_path):
            entries.append(entry)
    if not entries:
        return []
    latest = max((entry["observed"] for entry in entries if entry["observed"]), default=None)
    if latest:
        cutoff = latest - timedelta(hours=retention_hours)
        entries = [entry for entry in entries if not entry["observed"] or entry["observed"] >= cutoff]
    return sorted(entries, key=lambda item: (item["observed"] or datetime.min.replace(tzinfo=timezone.utc), item["path"].as_posix()))


def _timeline_item(entry: dict[str, Any]) -> dict[str, Any]:
    observed = entry["observed"]
    return {
        "scene_id": entry["scene_id"],
        "time": observed.isoformat().replace("+00:00", "Z") if observed else entry["meta"].get("observation_time"),
        "label": observed.strftime("%m-%d %H:%M") if observed else entry["scene_id"],
    }


def _select_entry(entries: list[dict[str, Any]], scene_id: str | None = None) -> dict[str, Any] | None:
    if not entries:
        return None
    if scene_id:
        for entry in entries:
            if entry["scene_id"] == scene_id:
                return entry
    return entries[-1]


def get_display_data(scene_id: str | None = None, retention_hours: int = 24) -> dict[str, Any]:
    entries = _meta_entries(retention_hours=retention_hours)
    selected = _select_entry(entries, scene_id)
    meta_files = [entry["path"] for entry in entries]
    png_files = sorted(
        list(DATA_DIR.glob("*/*/latlon/*.png")) + list(DATA_DIR.glob("*/*/composites/*.png")) + list(DATA_DIR.glob("*.png")),
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
        "png": _as_posix(png_path),
        "png_url": _png_url(png_path),
        "png_files": [_as_posix(path) for path in png_files if "latlon" in path.parts],
    }
