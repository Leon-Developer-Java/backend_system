import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from adapters import himawari_adapter
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


def _display_window_hours_env() -> int:
    env = os.environ
    key = "HIMAWARI_WINDOW_HOURS" if "HIMAWARI_WINDOW_HOURS" in env else "HIMAWARI_BACKFILL_HOURS"
    try:
        return max(1, int(env.get(key, DEFAULT_WINDOW_HOURS)))
    except ValueError:
        return DEFAULT_WINDOW_HOURS


def _display_latest_delay_minutes_env() -> int:
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
    include_scene_id: str | None = None,
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
    if include_scene_id and not any(entry["scene_id"] == include_scene_id for entry in window_entries):
        selected = next((entry for entry in sorted_entries if entry["scene_id"] == include_scene_id), None)
        if selected:
            window_entries.append(selected)
            window_entries.sort(
                key=lambda item: (item["observed"] or datetime.min.replace(tzinfo=timezone.utc), item["path"].as_posix())
            )
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
    retention_hours = _display_window_hours_env() if retention_hours is None else retention_hours
    delay_minutes = _display_latest_delay_minutes_env() if delay_minutes is None else delay_minutes
    entries = _meta_entries(
        retention_hours=retention_hours,
        now=now,
        delay_minutes=delay_minutes,
        include_scene_id=scene_id,
    )
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


# 自动下载调度与展示服务共享同一份 Himawari 配置和状态。
FALSE_VALUES = {"0", "false", "no", "off"}
DEFAULT_DOWNLOAD_MAX_JOBS_PER_RUN = 7
DOWNLOAD_STAGES = {"connecting", "listing", "downloading"}
PARSE_STAGES = {"parsing", "processing_band", "compositing", "writing_meta", "cleanup_raw"}
CLEAR_SCENE_STAGES = {"downloaded", "parsed", "failed", "error"}
STATE_LOCK = Lock()
LOG_LOCK = Lock()
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "himawari_auto.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
_LOGGER_CACHE: dict[str, logging.Logger] = {}


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


def _log_path(log_file: str | Path | None = None) -> Path:
    return Path(log_file) if log_file else LOG_FILE


def _logger_for(log_file: str | Path | None = None) -> logging.Logger:
    path = _log_path(log_file)
    key = str(path.resolve())
    with LOG_LOCK:
        logger = _LOGGER_CACHE.get(key)
        if logger:
            return logger
        path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"himawari.auto.{abs(hash(key))}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = RotatingFileHandler(
            path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        _LOGGER_CACHE[key] = logger
        return logger


def _format_log_fields(fields: dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        lowered = key.lower()
        if "password" in lowered or "token" in lowered or "secret" in lowered:
            continue
        parts.append(f"{key}={value}")
    return "" if not parts else " | " + " ".join(parts)


def write_himawari_log(
    message: str,
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    **fields: Any,
) -> None:
    logger = _logger_for(log_file)
    logger.log(level, "%s%s", message, _format_log_fields(fields))


def read_himawari_auto_log(lines: int = 200, log_file: str | Path | None = None) -> list[str]:
    path = _log_path(log_file)
    if not path.exists():
        return []
    safe_lines = max(1, min(int(lines or 200), 1000))
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-safe_lines:]]


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
        "removed_expired_count": len(result.get("removed_expired", [])),
        "phase": result.get("phase"),
        "stopped": result.get("stopped"),
    }


def auto_download_enabled(environ: dict[str, str] | None = None) -> bool:
    env = environ or os.environ
    return env.get("HIMAWARI_AUTO_DOWNLOAD", "0").strip().lower() not in FALSE_VALUES


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
    write_himawari_log(
        event.get("detail") or "Himawari 自动处理状态更新",
        worker=worker_name,
        stage=update["stage"],
        phase=update["current_phase"],
        scene_id=update["current_scene"],
        file=update["current_file"],
        band=update["current_band"],
        queue_done=update["queue_done"],
        queue_total=update["queue_total"],
    )


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
    return max(1, min(_int_env(env, "HIMAWARI_FILE_WORKERS", 4), 12))


def _bands_env(env: dict[str, str]) -> list[str] | None:
    raw = env.get("HIMAWARI_BANDS", ",".join(himawari_adapter.HIMAWARI_DEFAULT_BANDS))
    requested = [item.strip().upper() for item in raw.split(",") if item.strip()]
    target = [item for item in requested if item in himawari_adapter.HIMAWARI_TARGET_BANDS]
    return list(dict.fromkeys(target)) or list(himawari_adapter.HIMAWARI_DEFAULT_BANDS)


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
        write_himawari_log("自动下载未启动", level=logging.WARNING, stage="waiting_credentials", detail=message)
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
    write_himawari_log(
        "自动下载轮次开始",
        worker=queue,
        stage="starting",
        phase=queue,
        max_jobs=config["max_jobs_per_run"],
        max_workers=config["max_workers"],
        file_workers=config["file_workers"],
        bands=",".join(config["bands"] or []),
    )

    def progress(event: dict[str, Any]) -> None:
        update_himawari_progress({"worker": queue, **event})

    try:
        result = await asyncio.to_thread(recover, progress_callback=progress, **config) or {}
    except Exception as exc:
        finished_at = _utc_iso()
        worker.update({"state": "error", "running": False, "stage": "error", "last_finished_at": finished_at, "last_error": str(exc)})
        _STATE.update({"state": "error", "running": _any_worker_running(), "stage": "error", "last_finished_at": finished_at, "last_error": str(exc)})
        write_himawari_log("自动下载轮次异常", level=logging.ERROR, worker=queue, stage="error", error=str(exc))
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
    write_himawari_log(
        "自动下载轮次完成",
        worker=queue,
        stage="completed",
        downloaded=summary["downloaded_count"],
        processed_raw=summary["processed_raw_count"],
        skipped=summary["skipped_count"],
        errors=summary["error_count"],
        removed_part=summary["removed_part_count"],
        removed_expired=summary["removed_expired_count"],
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
            write_himawari_log("自动下载循环异常", level=logging.ERROR, worker=queue, stage="error", error=str(exc))
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
        write_himawari_log("自动下载已关闭", stage="disabled")
        print("[Himawari] 自动下载已关闭：HIMAWARI_AUTO_DOWNLOAD=0。")
        return None
    if not _credentials_ready(env):
        message = "请设置 HIMAWARI_FTP_USER 和 HIMAWARI_FTP_PASSWORD，并重启后端。"
        _STATE.update({
            "state": "waiting_credentials",
            "running": False,
            "stage": "waiting_credentials",
            "last_error": message,
        })
        write_himawari_log("自动下载未启动", level=logging.WARNING, stage="waiting_credentials", detail=message)
        print(f"[Himawari] 自动下载未启动：{message}")
        return None
    _STATE["workers"] = {"download": _worker_state()}
    return [
        asyncio.create_task(himawari_auto_download_worker_loop(env, "download", recover=recover), name="himawari-download"),
    ]
