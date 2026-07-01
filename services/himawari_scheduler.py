import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from adapters import himawari_adapter


FALSE_VALUES = {"0", "false", "no", "off"}
DEFAULT_DOWNLOAD_MAX_JOBS_PER_RUN = 12
DOWNLOAD_STAGES = {"connecting", "listing", "downloading"}
PARSE_STAGES = {"parsing", "processing_band", "compositing", "writing_meta", "cleanup_raw"}
CLEAR_SCENE_STAGES = {"downloaded", "parsed", "failed", "error"}
STATE_LOCK = Lock()


def _worker_state() -> dict[str, Any]:
    return {
        "state": "idle",
        "running": False,
        "stage": "idle",
        "current_phase": None,
        "current_scene": None,
        "current_file": None,
        "current_band": None,
        "current_detail": None,
        "queue_done": 0,
        "queue_total": 0,
        "last_started_at": None,
        "last_finished_at": None,
        "next_run_at": None,
        "last_result": None,
        "last_error": None,
    }


_STATE: dict[str, Any] = {
    "state": "idle",
    "running": False,
    "stage": "idle",
    "current_phase": None,
    "current_scene": None,
    "current_file": None,
    "current_band": None,
    "current_detail": None,
    "queue_done": 0,
    "queue_total": 0,
    "last_started_at": None,
    "last_finished_at": None,
    "next_run_at": None,
    "last_result": None,
    "last_error": None,
    "active_downloads": {},
    "active_parses": {},
    "workers": {
        "download": _worker_state(),
    },
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    errors = result.get("errors", [])
    return {
        "queue": result.get("queue"),
        "downloaded_count": len(result.get("downloaded", [])),
        "processed_raw_count": len(result.get("processed_raw", [])),
        "skipped_count": len(result.get("skipped", [])),
        "target_complete_count": len(result.get("target_complete", [])),
        "error_count": len(errors),
        "error_samples": errors[:3],
        "removed_part_count": len(result.get("removed_part_files", [])),
        "removed_raw_count": len(result.get("removed_raw_dirs", [])),
        "removed_expired_count": len(result.get("removed_expired", [])),
        "phase": result.get("phase"),
        "stopped": result.get("stopped"),
    }


def auto_download_enabled(environ: dict[str, str] | None = None) -> bool:
    env = environ or os.environ
    return env.get("HIMAWARI_AUTO_DOWNLOAD", "1").strip().lower() not in FALSE_VALUES


def get_himawari_auto_status(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = environ or os.environ
    enabled = auto_download_enabled(env)
    credentials_ready = _credentials_ready(env)
    if not enabled:
        state = "disabled"
    elif not credentials_ready:
        state = "waiting_credentials"
    else:
        state = _STATE["state"]
    return {
        "enabled": enabled,
        "credentials_ready": credentials_ready,
        "state": state,
        "running": bool(_STATE["running"]),
        "stage": _STATE["stage"],
        "current_phase": _STATE["current_phase"],
        "current_scene": _STATE["current_scene"],
        "current_file": _STATE["current_file"],
        "current_band": _STATE["current_band"],
        "current_detail": _STATE["current_detail"],
        "queue_done": _STATE["queue_done"],
        "queue_total": _STATE["queue_total"],
        "last_started_at": _STATE["last_started_at"],
        "last_finished_at": _STATE["last_finished_at"],
        "next_run_at": _STATE["next_run_at"],
        "last_result": _STATE["last_result"],
        "last_error": _STATE["last_error"],
        "active_downloads": _active_items("active_downloads"),
        "active_parses": _active_items("active_parses"),
        "workers": _STATE["workers"],
        "config": {
            "window_hours": _window_hours_env(env),
            "backfill_hours": _window_hours_env(env),
            "retention_hours": _retention_hours_env(env),
            "latest_delay_minutes": _latest_delay_minutes_env(env),
            "download_interval_minutes": _int_env(env, "HIMAWARI_DOWNLOAD_INTERVAL_MINUTES", 10),
            "download_interval_seconds": _worker_interval_seconds(env, "download"),
            "download_max_jobs_per_run": _worker_max_jobs(env, "download"),
            "max_workers": _max_workers_env(env),
            "file_workers": _file_workers_env(env),
            "bands": _bands_env(env),
        },
    }


def update_himawari_progress(event: dict[str, Any]) -> None:
    with STATE_LOCK:
        worker_name = event.get("worker")
        worker = _STATE["workers"].get(worker_name) if worker_name else None
        update = {
            "stage": event.get("stage", _STATE["stage"]),
            "current_phase": event.get("phase", _STATE["current_phase"]),
            "current_scene": event.get("scene_id", _STATE["current_scene"]),
            "current_file": event.get("file", _STATE["current_file"]),
            "current_band": event.get("band", _STATE["current_band"]),
            "current_detail": event.get("detail", _STATE["current_detail"]),
            "queue_done": event.get("queue_done", _STATE["queue_done"]),
            "queue_total": event.get("queue_total", _STATE["queue_total"]),
        }
        if worker is not None:
            worker.update(update)
        _update_active_items(event)
        _STATE.update(update)


def _active_items(key: str) -> list[dict[str, Any]]:
    values = list((_STATE.get(key) or {}).values())
    return sorted(values, key=lambda item: str(item.get("scene_id") or ""))


def _active_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": event.get("scene_id"),
        "stage": event.get("stage"),
        "queue_done": event.get("queue_done"),
        "queue_total": event.get("queue_total"),
        "updated_at": _utc_iso(),
    }


def _update_active_items(event: dict[str, Any]) -> None:
    stage = event.get("stage")
    scene_id = event.get("scene_id")
    if stage == "idle":
        _STATE.setdefault("active_downloads", {}).clear()
        _STATE.setdefault("active_parses", {}).clear()
        return
    if not scene_id:
        return
    downloads = _STATE.setdefault("active_downloads", {})
    parses = _STATE.setdefault("active_parses", {})
    if stage in CLEAR_SCENE_STAGES:
        downloads.pop(scene_id, None)
        parses.pop(scene_id, None)
        return
    if stage in DOWNLOAD_STAGES:
        downloads[scene_id] = _active_payload(event)
        return
    if stage in PARSE_STAGES:
        downloads.pop(scene_id, None)
        parses[scene_id] = _active_payload(event)


def _int_env(env: dict[str, str], key: str, default: int) -> int:
    try:
        return int(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _window_hours_env(env: dict[str, str]) -> int:
    if "HIMAWARI_WINDOW_HOURS" in env:
        return max(1, _int_env(env, "HIMAWARI_WINDOW_HOURS", himawari_adapter.DEFAULT_WINDOW_HOURS))
    return max(1, _int_env(env, "HIMAWARI_BACKFILL_HOURS", himawari_adapter.DEFAULT_WINDOW_HOURS))


def _retention_hours_env(env: dict[str, str]) -> int:
    return max(1, _int_env(env, "HIMAWARI_RETENTION_HOURS", _window_hours_env(env)))


def _latest_delay_minutes_env(env: dict[str, str]) -> int:
    return max(0, _int_env(env, "HIMAWARI_LATEST_DELAY_MINUTES", himawari_adapter.DEFAULT_LATEST_DELAY_MINUTES))


def _worker_interval_seconds(env: dict[str, str], queue: str) -> int:
    key = f"HIMAWARI_{queue.upper()}_INTERVAL_SECONDS"
    fallback = max(1, _int_env(env, "HIMAWARI_DOWNLOAD_INTERVAL_MINUTES", 10)) * 60
    default = 60
    return max(1, _int_env(env, key, min(default, fallback)))


def _worker_max_jobs(env: dict[str, str], queue: str) -> int:
    key = f"HIMAWARI_{queue.upper()}_MAX_JOBS_PER_RUN"
    if key in env:
        return max(0, _int_env(env, key, 0))
    if "HIMAWARI_MAX_JOBS_PER_RUN" in env:
        return max(0, _int_env(env, "HIMAWARI_MAX_JOBS_PER_RUN", 0))
    if "HIMAWARI_MAX_SCENES_PER_RUN" in env:
        return max(0, _int_env(env, "HIMAWARI_MAX_SCENES_PER_RUN", 0))
    if queue.lower() == "download":
        return DEFAULT_DOWNLOAD_MAX_JOBS_PER_RUN
    return 0


def _max_workers_env(env: dict[str, str]) -> int:
    return max(1, min(_int_env(env, "HIMAWARI_MAX_WORKERS", 1), 4))


def _file_workers_env(env: dict[str, str]) -> int:
    return max(1, min(_int_env(env, "HIMAWARI_FILE_WORKERS", 4), 4))


def _bands_env(env: dict[str, str]) -> list[str] | None:
    value = env.get("HIMAWARI_BANDS", "")
    bands = [item.strip().upper() for item in value.split(",") if item.strip()]
    target = list(himawari_adapter.HIMAWARI_TARGET_BANDS)
    if not bands:
        return target
    selected = [item for item in bands if item in target]
    return selected or target


def _quick_bands_env(env: dict[str, str]) -> list[str] | None:
    return _bands_env(env)


def _credentials_ready(env: dict[str, str]) -> bool:
    return bool(env.get("HIMAWARI_FTP_USER", "").strip() and env.get("HIMAWARI_FTP_PASSWORD", "").strip())


def _config(env: dict[str, str], queue: str) -> dict[str, Any]:
    return {
        "output_root": Path(env.get("HIMAWARI_OUTPUT_ROOT", himawari_adapter.DATA_DIR.as_posix())),
        "hours": _window_hours_env(env),
        "retention_hours": _retention_hours_env(env),
        "delay_minutes": _latest_delay_minutes_env(env),
        "interval_minutes": _int_env(env, "HIMAWARI_DOWNLOAD_INTERVAL_MINUTES", 10),
        "max_jobs_per_run": _worker_max_jobs(env, queue),
        "max_workers": _max_workers_env(env),
        "queue": queue,
        "bands": _bands_env(env),
        "quick_bands": _bands_env(env),
        "file_workers": _file_workers_env(env),
        "host": env.get("HIMAWARI_FTP_HOST", "").strip() or None,
        "user": env.get("HIMAWARI_FTP_USER", "").strip() or None,
        "password": env.get("HIMAWARI_FTP_PASSWORD", "").strip() or None,
        "remote_root": env.get("HIMAWARI_FTP_ROOT", "").strip() or None,
    }


async def run_himawari_auto_download_once(
    environ: dict[str, str] | None = None,
    recover: Callable[..., dict[str, Any]] = himawari_adapter.recover_himawari_scene_window,
    queue: str = "download",
) -> dict[str, Any] | None:
    env = environ or os.environ
    if not _credentials_ready(env):
        message = "请设置 HIMAWARI_FTP_USER 和 HIMAWARI_FTP_PASSWORD。"
        _STATE.update({"state": "waiting_credentials", "running": False, "stage": "waiting_credentials", "last_error": message})
        print(f"[Himawari] 自动下载未启动：{message}")
        return None
    queue = queue.lower()
    worker = _STATE["workers"].setdefault(queue, _worker_state())
    config = _config(env, queue)
    started_at = _utc_iso()
    reset_fields = {
        "state": "running",
        "running": True,
        "stage": "starting",
        "current_phase": queue,
        "current_scene": None,
        "current_file": None,
        "current_band": None,
        "current_detail": None,
        "queue_done": 0,
        "queue_total": 0,
        "last_started_at": started_at,
        "last_error": None,
    }
    worker.update(reset_fields)
    _STATE.update({**reset_fields, "last_started_at": started_at})

    def progress(event: dict[str, Any]) -> None:
        update_himawari_progress({"worker": queue, **event})

    try:
        result = await asyncio.to_thread(recover, progress_callback=progress, **config) or {}
    except Exception as exc:
        finished_at = _utc_iso()
        worker.update({"state": "error", "running": False, "stage": "error", "last_finished_at": finished_at, "last_error": str(exc)})
        _STATE.update({"state": "error", "running": _any_worker_running(), "stage": "error", "last_finished_at": finished_at, "last_error": str(exc)})
        raise
    summary = _summarize_result(result)
    finished_at = _utc_iso()
    worker.update({
        "state": "completed",
        "running": False,
        "stage": "idle",
        "last_finished_at": finished_at,
        "last_result": summary,
        "last_error": None,
    })
    running = _any_worker_running()
    _STATE.update({
        "state": "running" if running else "completed",
        "running": running,
        "stage": "idle",
        "current_phase": queue,
        "last_finished_at": finished_at,
        "last_result": summary,
        "last_error": None,
    })
    print(
        "[Himawari] 自动下载完成："
        f"下载 {summary['downloaded_count']}，"
        f"解析 raw {summary['processed_raw_count']}，"
        f"跳过 {summary['skipped_count']}，"
        f"错误 {summary['error_count']}"
    )
    return result


async def himawari_auto_download_loop(environ: dict[str, str] | None = None) -> None:
    env = environ or os.environ
    await himawari_auto_download_worker_loop(env, "download")


async def himawari_auto_download_worker_loop(
    environ: dict[str, str] | None = None,
    queue: str = "download",
    recover: Callable[..., dict[str, Any]] = himawari_adapter.recover_himawari_scene_window,
) -> None:
    env = environ or os.environ
    queue = queue.lower()
    while True:
        try:
            await run_himawari_auto_download_once(env, recover=recover, queue=queue)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            finished_at = _utc_iso()
            worker = _STATE["workers"].setdefault(queue, _worker_state())
            worker.update({"state": "error", "running": False, "stage": "error", "last_finished_at": finished_at, "last_error": str(exc)})
            _STATE.update({"state": "error", "running": _any_worker_running(), "stage": "error", "last_finished_at": finished_at, "last_error": str(exc)})
            print(f"[Himawari] 自动下载异常：{exc}")
        interval_seconds = _worker_interval_seconds(env, queue)
        next_run = datetime.now(timezone.utc) + timedelta(seconds=interval_seconds)
        next_run_iso = next_run.isoformat().replace("+00:00", "Z")
        worker = _STATE["workers"].setdefault(queue, _worker_state())
        worker["next_run_at"] = next_run_iso
        next_runs = [
            item.get("next_run_at")
            for item in _STATE["workers"].values()
            if item.get("next_run_at")
        ]
        _STATE["next_run_at"] = min(next_runs) if next_runs else next_run_iso
        await asyncio.sleep(interval_seconds)


def _any_worker_running() -> bool:
    return any(bool(worker.get("running")) for worker in _STATE["workers"].values())


def start_himawari_auto_download(
    environ: dict[str, str] | None = None,
    recover: Callable[..., dict[str, Any]] = himawari_adapter.recover_himawari_scene_window,
) -> list[asyncio.Task] | None:
    env = environ or os.environ
    if not auto_download_enabled(env):
        _STATE.update({"state": "disabled", "running": False})
        print("[Himawari] 自动下载已关闭：HIMAWARI_AUTO_DOWNLOAD=0。")
        return None
    _STATE["workers"] = {"download": _worker_state()}
    config = _config(env, "download")
    output_root = config["output_root"]
    removed_raw = himawari_adapter.cleanup_himawari_raw_dirs(output_root)
    removed_expired = himawari_adapter.cleanup_himawari_retention(
        output_root,
        retention_hours=config["retention_hours"],
        delay_minutes=config["delay_minutes"],
    )
    cleanup_result = {}
    if removed_raw:
        cleanup_result["removed_raw_count"] = len(removed_raw)
        print(f"[Himawari] 启动前清理 raw 原始目录：{len(removed_raw)}")
    if removed_expired:
        cleanup_result["removed_expired_count"] = len(removed_expired)
        print(f"[Himawari] 启动前清理过期自动结果：{len(removed_expired)}")
    if cleanup_result:
        _STATE["last_result"] = cleanup_result
    return [
        asyncio.create_task(himawari_auto_download_worker_loop(env, "download", recover=recover), name="himawari-download"),
    ]
