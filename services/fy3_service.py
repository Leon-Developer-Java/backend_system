from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "FY3"
STATIC_PREFIX = "/data/FY3"
BEIJING_OFFSET = timedelta(hours=8)


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


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    observed = _parse_time(meta.get("observation_time"))
    return {"path": meta_path, "meta": meta, "observed": observed, "scene_id": meta.get("scene_id") or meta_path.parents[1].name}


def _entries() -> list[dict[str, Any]]:
    entries = [entry for path in DATA_DIR.glob("*/*/meta/scene.meta.json") if (entry := _read_meta(path))]
    return sorted(entries, key=lambda item: (item["observed"] or datetime.min.replace(tzinfo=timezone.utc), item["scene_id"]))


def _product_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("key") or "").strip()


def _existing_path(value: str | None, meta_path: Path | None) -> Path | None:
    if not value:
        return None
    if Path(value).suffix.lower() != ".webp":
        return None
    path = Path(value)
    if path.exists():
        return path
    if meta_path:
        for folder in ("diff", "latlon", "composites"):
            if folder not in path.parts:
                continue
            relative = path.parts[path.parts.index(folder) :]
            candidate = meta_path.parent.parent.joinpath(*relative)
            if candidate.exists():
                return candidate
    return None


def _with_urls(items: list[dict[str, Any]], meta_path: Path | None) -> list[dict[str, Any]]:
    enriched = []
    for item in items:
        path = _existing_path(item.get("png"), meta_path)
        if not path:
            continue
        item_copy = {**item, "png_url": _png_url(path)}
        assets = item.get("resolution_assets")
        if isinstance(assets, dict):
            item_copy["resolution_assets"] = {
                key: {**asset, "png_url": _png_url(_existing_path(asset.get("png"), meta_path))}
                for key, asset in assets.items()
                if isinstance(asset, dict) and _existing_path(asset.get("png"), meta_path)
            }
        enriched.append(item_copy)
    return enriched


def _default_product(meta: dict[str, Any]) -> dict[str, Any] | None:
    products = list(meta.get("composites") or []) + list(meta.get("variables") or [])
    for key in ("B03", "B01", "B20"):
        for item in products:
            if _product_name(item) == key:
                return item
    return products[0] if products else None


def _timeline_item(entry: dict[str, Any]) -> dict[str, Any]:
    observed = entry["observed"]
    label_time = observed + BEIJING_OFFSET if observed else None
    meta = entry["meta"]
    return {
        "scene_id": entry["scene_id"],
        "time": observed.isoformat().replace("+00:00", "Z") if observed else meta.get("observation_time"),
        "label": label_time.strftime("%m-%d %H:%M") if label_time else entry["scene_id"],
        "label_timezone": "Asia/Shanghai",
        "extent": meta.get("extent"),
        "quality": meta.get("quality"),
    }


def _frame_item(entry: dict[str, Any]) -> dict[str, Any]:
    meta = entry["meta"]
    meta_path = entry["path"]
    product = _default_product(meta)
    png_path = _existing_path(product.get("png") if product else None, meta_path)
    return {
        "scene_id": entry["scene_id"],
        "time": meta.get("observation_time"),
        "extent": meta.get("extent"),
        "png_url": _png_url(png_path),
        "quality": meta.get("quality"),
    }


def _union_extent(frames: list[dict[str, Any]]) -> list[float] | None:
    extents = [frame.get("extent") for frame in frames if isinstance(frame.get("extent"), list) and len(frame["extent"]) == 4]
    if not extents:
        return None
    return [
        min(float(item[0]) for item in extents),
        min(float(item[1]) for item in extents),
        max(float(item[2]) for item in extents),
        max(float(item[3]) for item in extents),
    ]


def get_display_data(scene_id: str | None = None, limit: int = 24) -> dict[str, Any]:
    entries = _entries()
    selected = next((entry for entry in entries if entry["scene_id"] == scene_id), None) if scene_id else None
    if limit > 0:
        entries = entries[-limit:]
    selected = selected or (entries[-1] if entries else None)
    meta = selected["meta"] if selected else {}
    meta_path = selected["path"] if selected else None
    variables = _with_urls(meta.get("variables") or [], meta_path)
    composites = _with_urls(meta.get("composites") or [], meta_path)
    frames = [_frame_item(entry) for entry in entries]
    selected_frame = _frame_item(selected) if selected else None
    return {
        "business_type": "FY3",
        "meta_file": _as_posix(meta_path),
        "meta_json": meta or None,
        "weather_info": meta.get("weather_info"),
        "extent": _union_extent(frames) or meta.get("extent"),
        "grid": meta.get("grid"),
        "resolution": meta.get("resolution"),
        "resolutions": meta.get("resolutions") or {"original": meta.get("resolution")},
        "timeline": [_timeline_item(entry) for entry in entries],
        "frames": frames,
        "variables": variables,
        "composites": composites,
        "products": composites + variables,
        "png": None,
        "png_url": selected_frame.get("png_url") if selected_frame else None,
        "png_files": [frame["png_url"] for frame in frames if frame.get("png_url")],
    }
