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


def _webp_url(path: Path | None) -> str | None:
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


def _normalize_product_fields(item: dict[str, Any]) -> dict[str, Any]:
    item_copy = {**item}
    webp = item_copy.get("webp") or item_copy.get("png")
    item_copy.pop("png", None)
    item_copy.pop("float32", None)
    if webp:
        item_copy["webp"] = webp
        item_copy["image"] = webp
    assets = item_copy.get("resolution_assets")
    if isinstance(assets, dict):
        cleaned_assets = {}
        for key, asset in assets.items():
            if not isinstance(asset, dict):
                continue
            asset_copy = {**asset}
            asset_webp = asset_copy.get("webp") or asset_copy.get("png")
            asset_copy.pop("png", None)
            asset_copy.pop("float32", None)
            if asset_webp:
                asset_copy["webp"] = asset_webp
                asset_copy["image"] = asset_webp
            cleaned_assets[key] = asset_copy
        item_copy["resolution_assets"] = cleaned_assets
    return item_copy


def _normalize_meta_fields(meta: dict[str, Any]) -> dict[str, Any]:
    meta_copy = {**meta}
    meta_copy.pop("png", None)
    meta_copy.pop("png_url", None)
    meta_copy.pop("png_files", None)
    meta_copy.pop("float32", None)
    meta_copy["variables"] = [_normalize_product_fields(item) for item in meta_copy.get("variables") or [] if isinstance(item, dict)]
    meta_copy["composites"] = [_normalize_product_fields(item) for item in meta_copy.get("composites") or [] if isinstance(item, dict)]
    meta_copy["products"] = list(meta_copy.get("composites") or []) + list(meta_copy.get("variables") or [])
    return meta_copy


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    meta = _normalize_meta_fields(meta)
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
        path = _existing_path(item.get("webp") or item.get("png"), meta_path)
        if not path:
            continue
        item_copy = {**item}
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
    webp_path = _existing_path((product.get("webp") or product.get("png")) if product else None, meta_path)
    return {
        "scene_id": entry["scene_id"],
        "time": meta.get("observation_time"),
        "extent": meta.get("extent"),
        "webp_url": _webp_url(webp_path),
        "image_url": _webp_url(webp_path),
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
        "webp": None,
        "webp_url": selected_frame.get("webp_url") if selected_frame else None,
        "webp_files": [frame["webp_url"] for frame in frames if frame.get("webp_url")],
        "image": None,
        "image_url": selected_frame.get("image_url") if selected_frame else None,
        "image_files": [frame["image_url"] for frame in frames if frame.get("image_url")],
    }
