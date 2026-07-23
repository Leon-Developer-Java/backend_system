from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapters import fy3_adapter, himawari_adapter


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DB_PATH = DATA_ROOT / "satellite_tasks.sqlite3"
ACTIVE_STATES = {"queued", "running", "cancelling"}
FINAL_STATES = {"succeeded", "partial", "failed", "cancelled", "interrupted"}
_LOCK = threading.RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="satellite-parse")
_FUTURES: dict[str, Future] = {}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS satellite_parse_tasks (
                task_id TEXT PRIMARY KEY,
                business_type TEXT NOT NULL,
                owner_sub TEXT NOT NULL,
                owner_name TEXT,
                state TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                scene_ids_json TEXT NOT NULL,
                force INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                error TEXT,
                current_scene TEXT,
                current_band TEXT,
                scene_done INTEGER NOT NULL DEFAULT 0,
                scene_total INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    value = dict(row)
    value["scene_ids"] = json.loads(value.pop("scene_ids_json") or "[]")
    result = value.pop("result_json")
    value["result"] = json.loads(result) if result else None
    value["force"] = bool(value["force"])
    value["cancel_requested"] = bool(value["cancel_requested"])
    return value


def _get_unchecked(task_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        return _decode(connection.execute("SELECT * FROM satellite_parse_tasks WHERE task_id=?", (task_id,)).fetchone())


def _update(task_id: str, **values: Any) -> dict[str, Any]:
    allowed = {
        "state", "stage", "progress", "cancel_requested", "result", "error", "current_scene",
        "current_band", "scene_done", "started_at", "finished_at",
    }
    fields: list[str] = []
    params: list[Any] = []
    for key, value in values.items():
        if key not in allowed:
            continue
        column = "result_json" if key == "result" else key
        if key == "result":
            value = json.dumps(value, ensure_ascii=False) if value is not None else None
        if key == "cancel_requested":
            value = int(bool(value))
        fields.append(f"{column}=?")
        params.append(value)
    fields.append("updated_at=?")
    params.append(_utc_iso())
    params.append(task_id)
    with _LOCK, _connect() as connection:
        connection.execute(f"UPDATE satellite_parse_tasks SET {', '.join(fields)} WHERE task_id=?", params)
    task = _get_unchecked(task_id)
    if task is None:
        raise ValueError("卫星解析任务不存在。")
    return task


def _authorized(task: dict[str, Any], owner_sub: str, is_admin: bool) -> bool:
    return is_admin or task.get("owner_sub") == owner_sub


def get_task(task_id: str, owner_sub: str, is_admin: bool = False) -> dict[str, Any]:
    task = _get_unchecked(task_id)
    if task is None or not _authorized(task, owner_sub, is_admin):
        raise ValueError("卫星解析任务不存在。")
    return task


def list_tasks(
    business_type: str,
    owner_sub: str,
    is_admin: bool = False,
    active_only: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    clauses = ["business_type=?"]
    params: list[Any] = [business_type]
    if not is_admin:
        clauses.append("owner_sub=?")
        params.append(owner_sub)
    if active_only:
        clauses.append("state IN ('queued','running','cancelling')")
    params.append(max(1, min(500, int(limit))))
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM satellite_parse_tasks WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
    return [_decode(row) for row in rows]


def _scan_scenes(business_type: str, source_dir: Path) -> list[dict[str, Any]]:
    if business_type == "FY3":
        return fy3_adapter.scan_raw_scenes(source_dir)
    if business_type == "Himawari":
        return himawari_adapter.scan_raw_scenes(source_dir)
    raise ValueError("卫星解析任务仅支持 FY3 和 Himawari。")


def create_task(
    business_type: str,
    source_dir: str | Path,
    scene_ids: list[str] | None,
    force: bool,
    owner_sub: str,
    owner_name: str | None,
) -> tuple[dict[str, Any], bool]:
    source = Path(source_dir)
    scenes = _scan_scenes(business_type, source)
    scene_map = {str(scene.get("scene_id")): scene for scene in scenes}
    requested = list(dict.fromkeys(str(item) for item in (scene_ids or []) if str(item)))
    if not requested:
        requested = [str(scene["scene_id"]) for scene in scenes if scene.get("complete")]
    if not requested:
        raise ValueError(f"没有可解析的 {business_type} 完整场景。")
    unknown = [scene_id for scene_id in requested if scene_id not in scene_map]
    if unknown:
        raise ValueError(f"未找到 {business_type} raw 场景：{'、'.join(unknown)}")
    incomplete = [scene_map[item] for item in requested if not scene_map[item].get("complete")]
    if incomplete:
        raise ValueError("；".join(
            f"{scene['scene_id']} 缺少 {'、'.join(scene.get('missing') or [])}" for scene in incomplete
        ))

    requested_set = set(requested)
    with _connect() as connection:
        active_rows = connection.execute(
            "SELECT * FROM satellite_parse_tasks WHERE business_type=? AND state IN ('queued','running','cancelling')",
            (business_type,),
        ).fetchall()
    for row in active_rows:
        active = _decode(row)
        overlap = requested_set.intersection(active.get("scene_ids") or [])
        if not overlap:
            continue
        if active.get("owner_sub") == owner_sub and requested_set.issubset(set(active.get("scene_ids") or [])):
            return active, False
        raise ValueError(f"场景正在其他任务中解析：{'、'.join(sorted(overlap))}")

    prefix = "fy3" if business_type == "FY3" else "himawari"
    task_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    now = _utc_iso()
    with _LOCK, _connect() as connection:
        connection.execute(
            """
            INSERT INTO satellite_parse_tasks (
                task_id,business_type,owner_sub,owner_name,state,stage,progress,scene_ids_json,force,
                scene_total,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                task_id, business_type, owner_sub, owner_name, "queued", "queued", 0,
                json.dumps(requested, ensure_ascii=False), int(force), len(requested), now, now,
            ),
        )
    task = _get_unchecked(task_id)
    submit_task(task_id, source)
    return task, True


def _progress_callback(task_id: str, scene_index: int, scene_total: int):
    def callback(event: dict[str, Any]) -> None:
        band_index = int(event.get("band_index") or event.get("queue_done") or 0)
        band_total = max(1, int(event.get("band_total") or event.get("queue_total") or 1))
        progress = ((scene_index - 1) + band_index / band_total) / max(1, scene_total) * 100
        _update(
            task_id,
            stage=str(event.get("stage") or "running"),
            progress=round(min(99.9, max(0.0, progress)), 1),
            current_scene=event.get("scene_id"),
            current_band=event.get("band"),
            scene_done=max(0, scene_index - 1),
        )
    return callback


def _run_task(task_id: str, source_dir: Path) -> None:
    task = _get_unchecked(task_id)
    if task is None:
        return
    _update(task_id, state="running", stage="starting", started_at=_utc_iso(), error=None)
    rows: list[dict[str, Any]] = []
    scene_ids = list(task.get("scene_ids") or [])
    try:
        for index, scene_id in enumerate(scene_ids, start=1):
            current = _get_unchecked(task_id)
            if current is None or current.get("cancel_requested"):
                _update(
                    task_id, state="cancelled", stage="cancelled", progress=min(99.9, float(current.get("progress") or 0) if current else 0),
                    finished_at=_utc_iso(), current_band=None,
                )
                return
            _update(task_id, stage="processing_scene", current_scene=scene_id, current_band=None)
            if task["business_type"] == "FY3":
                result = fy3_adapter.update_from_raw(
                    source_dir,
                    force=bool(task.get("force")),
                    scene_ids=[scene_id],
                    progress_callback=_progress_callback(task_id, index, len(scene_ids)),
                )
            else:
                result = himawari_adapter.update_from_raw(
                    source_dir,
                    source_dir,
                    force=bool(task.get("force")),
                    scene_ids=[scene_id],
                    progress_callback=_progress_callback(task_id, index, len(scene_ids)),
                )
            rows.extend(result.get("results") or [])
            _update(
                task_id,
                progress=round(index / len(scene_ids) * 100, 1),
                scene_done=index,
                current_band=None,
            )

        failed = [item for item in rows if item.get("status") == "error"]
        displayable = [
            str(item.get("scene_id")) for item in rows
            if item.get("status") in {"ok", "cached"} and item.get("displayable", True)
        ]
        no_coverage = [
            str(item.get("scene_id")) for item in rows
            if item.get("status") in {"ok", "cached"} and not item.get("displayable", True)
        ]
        result = {
            "scene_count": len(scene_ids),
            "processed": len([item for item in rows if item.get("status") == "ok"]),
            "cached": len([item for item in rows if item.get("status") == "cached"]),
            "failed": len(failed),
            "results": rows,
            "displayable_scene_ids": displayable,
            "no_coverage_scene_ids": no_coverage,
        }
        _update(
            task_id,
            state="partial" if failed else "succeeded",
            stage="completed",
            progress=100,
            result=result,
            error="；".join(f"{item.get('scene_id')}：{item.get('error') or '解析失败'}" for item in failed) or None,
            finished_at=_utc_iso(),
            current_band=None,
        )
    except Exception as exc:
        _update(task_id, state="failed", stage="failed", error=str(exc), finished_at=_utc_iso())
    finally:
        with _LOCK:
            _FUTURES.pop(task_id, None)


def submit_task(task_id: str, source_dir: str | Path) -> None:
    with _LOCK:
        existing = _FUTURES.get(task_id)
        if existing and not existing.done():
            return
        _FUTURES[task_id] = _EXECUTOR.submit(_run_task, task_id, Path(source_dir))


def cancel_task(task_id: str, owner_sub: str, is_admin: bool = False) -> dict[str, Any]:
    task = get_task(task_id, owner_sub, is_admin)
    if task["state"] in FINAL_STATES:
        return task
    future = _FUTURES.get(task_id)
    if future and future.cancel():
        return _update(task_id, state="cancelled", stage="cancelled", cancel_requested=True, finished_at=_utc_iso())
    return _update(task_id, state="cancelling", stage="cancelling", cancel_requested=True)


def retry_task(task_id: str, source_dir: str | Path, owner_sub: str, owner_name: str | None, is_admin: bool = False) -> dict[str, Any]:
    task = get_task(task_id, owner_sub, is_admin)
    if task["state"] in ACTIVE_STATES:
        raise ValueError("任务仍在运行，不能重试。")
    value, _ = create_task(
        task["business_type"], source_dir, task["scene_ids"], bool(task["force"]), owner_sub, owner_name,
    )
    return value


def recover_tasks(source_dirs: dict[str, Path]) -> None:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT task_id,business_type FROM satellite_parse_tasks WHERE state IN ('queued','running','cancelling')"
        ).fetchall()
    for row in rows:
        task_id = str(row["task_id"])
        business = str(row["business_type"])
        current = _get_unchecked(task_id)
        if current and (current.get("state") == "cancelling" or current.get("cancel_requested")):
            _update(task_id, state="cancelled", stage="cancelled", finished_at=_utc_iso())
            continue
        if business not in source_dirs:
            _update(task_id, state="interrupted", stage="interrupted", error="服务重启后无法定位原始数据目录", finished_at=_utc_iso())
            continue
        _update(task_id, state="queued", stage="recovered", cancel_requested=False, error=None)
        submit_task(task_id, source_dirs[business])


def shutdown() -> None:
    _EXECUTOR.shutdown(wait=False, cancel_futures=False)


init_db()
