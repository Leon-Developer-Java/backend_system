from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from adapters import fy3_adapter


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "FY3"
STATIC_PREFIX = "/data/FY3"
BEIJING_OFFSET = timedelta(hours=8)
PARSE_TASK_LOCK = Lock()
PARSE_TASK_LIMIT = 100
PARSE_TASK_ACTIVE_STATES = {"queued", "running"}
_PARSE_TASKS: dict[str, dict[str, Any]] = {}


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
    meta_paths = [
        *DATA_DIR.glob("*/*/meta/scene.meta.json"),
        *DATA_DIR.glob("assets/*/*/*/meta/scene.meta.json"),
    ]
    entries = [entry for path in meta_paths if (entry := _read_meta(path))]
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
        "status": meta.get("status") or "parsed",
        "displayable": meta.get("status") != "no_coverage",
    }


def _frame_item(entry: dict[str, Any]) -> dict[str, Any]:
    meta = entry["meta"]
    meta_path = entry["path"]
    product = _default_product(meta)
    webp_path = _existing_path((product.get("webp") or product.get("png")) if product else None, meta_path)
    resolution_assets = product.get("resolution_assets") if isinstance(product, dict) else None
    available_resolutions = ["original"]
    if isinstance(resolution_assets, dict):
        available_resolutions = [
            key
            for key, asset in resolution_assets.items()
            if isinstance(asset, dict) and _existing_path(asset.get("webp") or asset.get("png"), meta_path)
        ]
        if "original" not in available_resolutions and webp_path:
            available_resolutions.insert(0, "original")
    return {
        "scene_id": entry["scene_id"],
        "time": meta.get("observation_time"),
        "extent": meta.get("extent"),
        "webp_url": _webp_url(webp_path),
        "image_url": _webp_url(webp_path),
        "available_resolutions": available_resolutions,
        "quality": meta.get("quality"),
        "status": meta.get("status") or "parsed",
        "displayable": meta.get("status") != "no_coverage",
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
        if selected:
            selected_index = entries.index(selected)
            entries = entries[max(0, selected_index - limit + 1) : selected_index + 1]
        else:
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
        "extent": (selected_frame or {}).get("extent") or meta.get("extent") or _union_extent(frames),
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


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _task_copy(task: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(task)


def _prune_parse_tasks_locked() -> None:
    if len(_PARSE_TASKS) <= PARSE_TASK_LIMIT:
        return
    completed = [
        task_id
        for task_id, task in _PARSE_TASKS.items()
        if task.get("state") not in PARSE_TASK_ACTIVE_STATES
    ]
    for task_id in completed[: max(0, len(_PARSE_TASKS) - PARSE_TASK_LIMIT)]:
        _PARSE_TASKS.pop(task_id, None)


def list_parse_tasks(active_only: bool = False) -> list[dict[str, Any]]:
    with PARSE_TASK_LOCK:
        tasks = [
            _task_copy(task)
            for task in _PARSE_TASKS.values()
            if not active_only or task.get("state") in PARSE_TASK_ACTIVE_STATES
        ]
    return sorted(tasks, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def get_parse_task(task_id: str) -> dict[str, Any]:
    with PARSE_TASK_LOCK:
        task = _PARSE_TASKS.get(task_id)
        if task is None:
            raise ValueError("FY-3 解析任务不存在。")
        return _task_copy(task)


def create_parse_task(
    source_dir: str | Path,
    scene_ids: list[str] | None,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    scenes = fy3_adapter.scan_raw_scenes(source_dir)
    scene_map = {str(scene.get("scene_id")): scene for scene in scenes}
    requested = list(dict.fromkeys(str(item) for item in (scene_ids or []) if str(item)))
    if not requested:
        requested = [str(scene["scene_id"]) for scene in scenes if scene.get("complete")]
    if not requested:
        raise ValueError("没有可解析的 FY-3 完整场景。")

    unknown = [scene_id for scene_id in requested if scene_id not in scene_map]
    if unknown:
        raise ValueError(f"未找到 FY-3 raw 场景：{'、'.join(unknown)}")
    incomplete = [scene_map[scene_id] for scene_id in requested if not scene_map[scene_id].get("complete")]
    if incomplete:
        details = [
            f"{scene['scene_id']} 缺少 {'、'.join(scene.get('missing') or [])}"
            for scene in incomplete
        ]
        raise ValueError("；".join(details))

    requested_set = set(requested)
    with PARSE_TASK_LOCK:
        for task in _PARSE_TASKS.values():
            if task.get("state") not in PARSE_TASK_ACTIVE_STATES:
                continue
            overlap = requested_set.intersection(task.get("scene_ids") or [])
            if not overlap:
                continue
            if requested_set.issubset(set(task.get("scene_ids") or [])):
                return _task_copy(task), False
            raise ValueError(f"场景正在其他任务中解析：{'、'.join(sorted(overlap))}")

        task_id = f"fy3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        task = {
            "task_id": task_id,
            "business_type": "FY3",
            "state": "queued",
            "stage": "queued",
            "progress": 0.0,
            "force": bool(force),
            "scene_ids": requested,
            "scene_total": len(requested),
            "scene_done": 0,
            "current_scene": None,
            "current_band": None,
            "band_index": 0,
            "band_total": len(fy3_adapter.CORE_BANDS),
            "created_at": _utc_iso(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        _PARSE_TASKS[task_id] = task
        _prune_parse_tasks_locked()
        return _task_copy(task), True


def _update_parse_task_progress(task_id: str, event: dict[str, Any]) -> None:
    with PARSE_TASK_LOCK:
        task = _PARSE_TASKS.get(task_id)
        if task is None or task.get("state") not in PARSE_TASK_ACTIVE_STATES:
            return
        scene_total = max(1, int(event.get("scene_total") or task.get("scene_total") or 1))
        scene_index = max(1, int(event.get("scene_index") or 1))
        band_total = max(1, int(event.get("band_total") or task.get("band_total") or 1))
        band_index = max(0, int(event.get("band_index") or 0))
        stage = str(event.get("stage") or task.get("stage") or "running")
        if stage in {"scene_completed", "scene_failed"}:
            progress = scene_index / scene_total * 100
            scene_done = scene_index
        else:
            progress = ((scene_index - 1) + band_index / band_total) / scene_total * 100
            scene_done = max(0, scene_index - 1)
        task.update({
            "stage": stage,
            "progress": round(min(max(progress, 0.0), 99.9), 1),
            "scene_total": scene_total,
            "scene_done": scene_done,
            "current_scene": event.get("scene_id") or task.get("current_scene"),
            "current_band": event.get("band") or task.get("current_band"),
            "band_index": band_index,
            "band_total": band_total,
        })


def run_parse_task(task_id: str, source_dir: str | Path = DATA_DIR) -> None:
    with PARSE_TASK_LOCK:
        task = _PARSE_TASKS.get(task_id)
        if task is None:
            return
        task.update({"state": "running", "stage": "starting", "started_at": _utc_iso(), "error": None})
        scene_ids = list(task.get("scene_ids") or [])
        force = bool(task.get("force"))

    try:
        result = fy3_adapter.update_from_raw(
            source_dir,
            force=force,
            scene_ids=scene_ids,
            progress_callback=lambda event: _update_parse_task_progress(task_id, event),
        )
        rows = list(result.get("results") or [])
        displayable_ids = [
            str(item.get("scene_id"))
            for item in rows
            if item.get("status") in {"ok", "cached"} and item.get("displayable")
        ]
        no_coverage_ids = [
            str(item.get("scene_id"))
            for item in rows
            if item.get("status") in {"ok", "cached"} and not item.get("displayable")
        ]
        failed_rows = [item for item in rows if item.get("status") == "error"]
        result = {
            **result,
            "displayable_scene_ids": displayable_ids,
            "no_coverage_scene_ids": no_coverage_ids,
        }
        with PARSE_TASK_LOCK:
            task = _PARSE_TASKS.get(task_id)
            if task is not None:
                task.update({
                    "state": "partial" if failed_rows else "completed",
                    "stage": "completed",
                    "progress": 100.0,
                    "scene_done": len(rows),
                    "current_band": None,
                    "finished_at": _utc_iso(),
                    "result": result,
                    "error": "；".join(
                        f"{item.get('scene_id')}：{item.get('error') or '解析失败'}"
                        for item in failed_rows
                    ) or None,
                })
    except Exception as exc:
        with PARSE_TASK_LOCK:
            task = _PARSE_TASKS.get(task_id)
            if task is not None:
                task.update({
                    "state": "failed",
                    "stage": "failed",
                    "finished_at": _utc_iso(),
                    "error": str(exc),
                })
