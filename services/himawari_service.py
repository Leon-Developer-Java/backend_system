import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from adapters.himawari_adapter import (
    DEFAULT_LATEST_DELAY_MINUTES,
    DEFAULT_WINDOW_HOURS,
    latest_himawari_slot,
    normalize_himawari_meta,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Himawari"
STATIC_PREFIX = "/data/Himawari"
BEIJING_OFFSET = timedelta(hours=8)


def _as_posix(path: Path | None) -> str | None:
    return str(path).replace("\\", "/") if path else None


def _webp_url(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    try:
        relative = path.resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        return None
    return f"{STATIC_PREFIX}/{relative.as_posix()}"


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
    if Path(value).suffix.lower() != ".webp":
        return None
    path = Path(value)
    if path.exists():
        return path
    if meta_path:
        scene_dir = meta_path.parent.parent
        # 处理 diff/*/latlon/ 路径
        if "diff" in path.parts:
            diff_idx = path.parts.index("diff")
            candidate = scene_dir.joinpath(*path.parts[diff_idx:])
            if candidate.exists():
                return candidate
            return None
        # 处理直接的 latlon/ 或 composites/ 路径
        for folder in ("latlon", "composites"):
            if folder not in path.parts:
                continue
            candidate = scene_dir.joinpath(*path.parts[path.parts.index(folder):])
            if candidate.exists():
                return candidate
    return None


def _select_default_webp(meta_json: dict[str, Any] | None, fallback: list[Path], meta_path: Path | None = None) -> Path | None:
    if not meta_json:
        return None

    for key in ("B13", "B14", "B08"):
        for item in meta_json.get("variables", []):
            if _product_name(item) == key and (path := _existing_path(item.get("webp") or item.get("png"), meta_path)):
                return path

    for key in ("true_color", "natural_color", "water_vapor_enhanced"):
        for item in meta_json.get("composites", []):
            if _product_name(item) == key and (path := _existing_path(item.get("webp") or item.get("png"), meta_path)):
                return path

    return fallback[0] if fallback else None


def _with_webp_urls(items: list[dict[str, Any]], meta_path: Path | None = None) -> list[dict[str, Any]]:
    enriched = []
    for item in items:
        path = _existing_path(item.get("webp") or item.get("png"), meta_path)
        if not path:
            continue
        item_copy = {**item}
        item_copy.pop("png_data_url", None)
        item_copy.pop("png", None)
        item_copy["webp"] = _as_posix(path)
        item_copy["image"] = _as_posix(path)
        item_copy["webp_url"] = _webp_url(path)
        item_copy["image_url"] = _webp_url(path)
        assets = item.get("resolution_assets")
        if isinstance(assets, dict):
            cleaned_assets = {}
            for key, asset in assets.items():
                if not isinstance(asset, dict):
                    continue
                asset_path = _existing_path(asset.get("webp") or asset.get("png"), meta_path)
                if not asset_path:
                    continue
                asset_copy = {**asset}
                asset_copy.pop("png", None)
                asset_copy["webp"] = _as_posix(asset_path)
                asset_copy["image"] = _as_posix(asset_path)
                asset_copy["webp_url"] = _webp_url(asset_path)
                asset_copy["image_url"] = _webp_url(asset_path)
                cleaned_assets[key] = asset_copy
            item_copy["resolution_assets"] = cleaned_assets
        enriched.append(item_copy)
    return enriched


def _scene_webp_files(meta_path: Path | None) -> list[Path]:
    if not meta_path:
        return []
    scene_dir = meta_path.parent.parent
    files = [
        *scene_dir.glob("latlon/*.webp"),
        *scene_dir.glob("composites/*.webp"),
    ]
    return sorted(files, key=lambda item: (item.stat().st_mtime, item.as_posix()), reverse=True)


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
    meta_json = None
    if selected:
        meta_json = selected["meta"]

    meta_path = selected["path"] if selected else None
    webp_files = _scene_webp_files(meta_path)
    webp_path = _select_default_webp(meta_json, webp_files, meta_path)
    variables = _with_webp_urls(meta_json.get("variables", []), meta_path) if meta_json else []
    composites = _with_webp_urls(meta_json.get("composites", []), meta_path) if meta_json else []

    return {
        "business_type": "Himawari",
        "meta_file": _as_posix(meta_path),
        "meta_json": meta_json,
        "weather_info": meta_json.get("weather_info") if meta_json else None,
        "resolution_options": (meta_json.get("resolution_options") or None) if meta_json else None,
        "resolutions": (meta_json.get("resolutions") or None) if meta_json else None,
        "extent": (meta_json.get("extent") or meta_json.get("bbox")) if meta_json else None,
        "grid": meta_json.get("grid") if meta_json else None,
        "timeline": [_timeline_item(entry) for entry in entries],
        "variables": variables,
        "composites": composites,
        "products": composites + variables,
        "webp": _as_posix(webp_path),
        "webp_url": _webp_url(webp_path),
        "webp_files": [_as_posix(path) for path in webp_files],
        "image": _as_posix(webp_path),
        "image_url": _webp_url(webp_path),
        "image_files": [_as_posix(path) for path in webp_files],
    }
