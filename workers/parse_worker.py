from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BACKEND_ROOT.parent
load_dotenv(BACKEND_ROOT / ".env")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from DB.config import create_database_engine
from DB.migrate import init_database
from DB.repository import (
    claim_next_satellite_collection,
    claim_next_task,
    clone_successful_source_task,
    commit_satellite_collection_failure,
    commit_satellite_collection_success,
    commit_task_failure,
    commit_task_success,
    heartbeat_satellite_collection,
    heartbeat_task,
    recover_expired_collections,
    recover_expired_tasks,
)
from services.adapter_runner import cleanup_stage, publish_adapter_output
from services.asset_catalog import build_asset_catalog


RAW_STORAGE_ROOT = Path(
    os.getenv("RAW_STORAGE_ROOT", str(WORKSPACE_ROOT / "storage" / "raw"))
).resolve()
TMP_STORAGE_ROOT = Path(
    os.getenv("TMP_STORAGE_ROOT", str(WORKSPACE_ROOT / "storage" / "tmp"))
).resolve()
PRODUCT_DATA_ROOT = Path(
    os.getenv("PRODUCT_DATA_ROOT", str(BACKEND_ROOT / "data"))
).resolve()
QUEUE_SLEEP_SECONDS = float(os.getenv("ADAPTER_QUEUE_SLEEP_SECONDS", "2"))
LEASE_SECONDS = int(os.getenv("ADAPTER_LEASE_SECONDS", "1800"))
HEARTBEAT_SECONDS = int(os.getenv("ADAPTER_HEARTBEAT_SECONDS", "60"))
MAX_ATTEMPTS = int(os.getenv("ADAPTER_MAX_ATTEMPTS", "3"))
BACKOFF_SECONDS = [
    int(item)
    for item in os.getenv("ADAPTER_RETRY_BACKOFF_SECONDS", "60,300,1800").split(",")
    if item.strip()
] or [60, 300, 1800]
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("ADAPTER_TIMEOUT_SECONDS", "3600"))
TIMEOUTS = {
    "ERA5": int(os.getenv("ADAPTER_TIMEOUT_ERA5", str(DEFAULT_TIMEOUT_SECONDS))),
    "GFS": int(os.getenv("ADAPTER_TIMEOUT_GFS", str(DEFAULT_TIMEOUT_SECONDS))),
    "ECMWF": int(os.getenv("ADAPTER_TIMEOUT_ECMWF", str(DEFAULT_TIMEOUT_SECONDS))),
    "CMA": int(os.getenv("ADAPTER_TIMEOUT_CMA", str(DEFAULT_TIMEOUT_SECONDS))),
    "RADAR": int(os.getenv("ADAPTER_TIMEOUT_RADAR", str(DEFAULT_TIMEOUT_SECONDS))),
    "WRF": int(os.getenv("ADAPTER_TIMEOUT_WRF", str(DEFAULT_TIMEOUT_SECONDS))),
    "FY3": int(os.getenv("ADAPTER_TIMEOUT_FY3", str(DEFAULT_TIMEOUT_SECONDS))),
    "HIMAWARI": int(os.getenv("ADAPTER_TIMEOUT_HIMAWARI", str(DEFAULT_TIMEOUT_SECONDS))),
}

LOG_DIR = BACKEND_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=os.getenv("ADAPTER_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "parse_worker.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("adapter-worker")


class AdapterProcessError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


@contextmanager
def singleton_lock():
    lock_dir = TMP_STORAGE_ROOT / "adapter"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "parse_worker.lock"
    if not lock_path.exists() or lock_path.stat().st_size == 0:
        lock_path.write_text(" ", encoding="ascii")
    handle = lock_path.open("r+")
    acquired = False
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("another Adapter Worker is already running") from exc
            acquired = True
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("another Adapter Worker is already running") from exc
            acquired = True
        handle.seek(0)
        handle.truncate()
        handle.write(f"{socket.gethostname()}:{os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            if acquired and os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif acquired:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _raw_path(task: dict) -> Path:
    path = (RAW_STORAGE_ROOT / str(task["source_path"])).resolve()
    try:
        path.relative_to(RAW_STORAGE_ROOT)
    except ValueError as exc:
        raise AdapterProcessError("public_info.source_path escaped RAW_STORAGE_ROOT", retryable=False) from exc
    if not path.is_file():
        raise AdapterProcessError(f"raw source is missing: {task['source_path']}", retryable=False)
    return path


def _display_type(data_type: str) -> str:
    normalized = data_type.upper()
    return {"RADAR": "Radar", "HIMAWARI": "Himawari"}.get(normalized, normalized)


def _cleanup_published_output(task: dict) -> None:
    final_dir = (
        PRODUCT_DATA_ROOT
        / _display_type(str(task["data_type"]))
        / "assets"
        / str(task["file_uuid"])
    ).resolve()
    try:
        final_dir.relative_to(PRODUCT_DATA_ROOT)
    except ValueError:
        logger.error("refused to clean published path outside product root: %s", final_dir)
        return
    shutil.rmtree(final_dir, ignore_errors=True)


def _terminate_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_child(engine, task: dict, worker_id: str) -> dict:
    file_uuid = task["file_uuid"]
    attempt_id = str(uuid.uuid4())
    job_dir = TMP_STORAGE_ROOT / "adapter" / file_uuid / attempt_id
    job_dir.mkdir(parents=True, exist_ok=False)
    display_type = _display_type(task["data_type"])
    stage_dir = PRODUCT_DATA_ROOT / display_type / ".adapter_staging" / file_uuid / attempt_id
    result_path = job_dir / "result.json"
    error_path = job_dir / "error.json"
    job_path = job_dir / "job.json"
    job = {
        "file_uuid": file_uuid,
        "collection_uuid": task.get("collection_uuid"),
        "data_type": task["data_type"],
        "output_root": PRODUCT_DATA_ROOT.as_posix(),
        "stage_dir": stage_dir.as_posix(),
        "result_path": result_path.as_posix(),
        "error_path": error_path.as_posix(),
    }
    is_collection = task.get("task_kind") == "satellite_collection"
    if is_collection:
        job["task_kind"] = "satellite_collection"
        job["input_dir"] = (job_dir / "raw_input").as_posix()
        job["members"] = [
            {
                **member,
                "source_path": _raw_path(member).as_posix(),
            }
            for member in task.get("members") or []
        ]
    else:
        job["source_path"] = _raw_path(task).as_posix()
        job["original_file_name"] = task.get("original_file_name")
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    env = os.environ.copy()
    python_path = os.pathsep.join([str(BACKEND_ROOT), str(WORKSPACE_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONPATH"] = python_path
    started = time.monotonic()
    last_heartbeat = started
    timeout = TIMEOUTS.get(str(task["data_type"]).upper(), DEFAULT_TIMEOUT_SECONDS)
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-m", "workers.adapter_subprocess", "--job", str(job_path)],
                cwd=BACKEND_ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            while process.poll() is None:
                now = time.monotonic()
                if now - started > timeout:
                    _terminate_process(process)
                    raise AdapterProcessError(f"Adapter timed out after {timeout} seconds", retryable=True)
                if now - last_heartbeat >= HEARTBEAT_SECONDS:
                    heartbeat_ok = (
                        heartbeat_satellite_collection(
                            engine,
                            str(task["collection_uuid"]),
                            worker_id,
                            LEASE_SECONDS,
                        )
                        if is_collection
                        else heartbeat_task(engine, file_uuid, worker_id, LEASE_SECONDS)
                    )
                    if not heartbeat_ok:
                        _terminate_process(process)
                        raise AdapterProcessError("task lease was lost while Adapter was running", retryable=True)
                    last_heartbeat = now
                time.sleep(1)

        if process.returncode != 0:
            if error_path.is_file():
                error = json.loads(error_path.read_text(encoding="utf-8"))
                raise AdapterProcessError(
                    f"{error.get('error_type', 'AdapterError')}: {error.get('message', 'unknown error')}",
                    retryable=bool(error.get("retryable")),
                )
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise AdapterProcessError(
                f"Adapter subprocess exited with code {process.returncode}: {stderr_text}",
                retryable=True,
            )
        if not result_path.is_file():
            raise AdapterProcessError("Adapter subprocess returned no result.json", retryable=True)
        child_result = json.loads(result_path.read_text(encoding="utf-8"))
        meta, meta_file, final_dir = publish_adapter_output(child_result, PRODUCT_DATA_ROOT)
        assets = build_asset_catalog(
            file_uuid=file_uuid,
            data_type=str(task["data_type"]).upper(),
            meta=meta,
            product_root=PRODUCT_DATA_ROOT,
            final_dir=final_dir,
        )
        if not assets:
            raise AdapterProcessError("Adapter output contains no queryable WebP assets", retryable=False)
        webp_files = [path for path in final_dir.rglob("*.webp") if path.is_file()]
        default_asset = next((asset for asset in assets if asset.get("is_default")), assets[0])
        result = {
            "data_type": str(task["data_type"]).upper(),
            "meta_path": meta_file.relative_to(PRODUCT_DATA_ROOT).as_posix(),
            "default_webp_url": default_asset.get("webp_url"),
            "webp_count": len(webp_files),
            "adapter_name": child_result.get("adapter_name"),
            "adapter_version": child_result.get("adapter_version"),
            "meta_schema_version": str(meta.get("schema_version") or "legacy"),
            "assets": assets,
        }
        if result["data_type"] == "WRF":
            try:
                from adapters.wrf_adapter import validate_before_db_write

                validate_before_db_write(
                    meta=meta,
                    meta_file=meta_file,
                    final_dir=final_dir,
                    product_root=PRODUCT_DATA_ROOT,
                    result=result,
                    assets=assets,
                )
            except Exception as exc:
                raise AdapterProcessError(str(exc), retryable=False) from exc
        return result
    except Exception:
        failure_dir = LOG_DIR / "adapter_failures" / file_uuid / attempt_id
        if job_dir.is_dir():
            failure_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(job_dir, failure_dir, dirs_exist_ok=True)
        cleanup_stage(stage_dir, PRODUCT_DATA_ROOT)
        raise
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def process_one(engine, task: dict, worker_id: str) -> None:
    is_collection = task.get("task_kind") == "satellite_collection"
    logger.info(
        "claimed %s=%s data_type=%s attempt=%s",
        "collection_uuid" if is_collection else "file_uuid",
        task.get("collection_uuid") if is_collection else task["file_uuid"],
        task["data_type"],
        task["parse_attempts"],
    )
    try:
        if not is_collection and clone_successful_source_task(engine, task, worker_id):
            logger.info("reused duplicate assets file_uuid=%s source=%s", task["file_uuid"], task["source_file_uuid"])
            return
        result = _run_child(engine, task, worker_id)
        assets = result.pop("assets")
        if is_collection:
            commit_satellite_collection_success(
                engine,
                str(task["collection_uuid"]),
                worker_id,
                result,
                assets,
            )
            logger.info(
                "parsed collection_uuid=%s leader=%s webp_count=%s",
                task["collection_uuid"],
                task["file_uuid"],
                result["webp_count"],
            )
        else:
            commit_task_success(engine, task["file_uuid"], worker_id, result, assets)
            logger.info("parsed file_uuid=%s webp_count=%s", task["file_uuid"], result["webp_count"])
    except AdapterProcessError as exc:
        _cleanup_published_output(task)
        commit_failure = commit_satellite_collection_failure if is_collection else commit_task_failure
        status = commit_failure(
            engine, task, worker_id, str(exc),
            retryable=exc.retryable,
            max_attempts=MAX_ATTEMPTS,
            backoff_seconds=BACKOFF_SECONDS,
        )
        logger.error("parse failed task=%s status=%s error=%s", task.get("collection_uuid") or task["file_uuid"], status, exc)
    except Exception as exc:
        _cleanup_published_output(task)
        commit_failure = commit_satellite_collection_failure if is_collection else commit_task_failure
        status = commit_failure(
            engine, task, worker_id, f"{type(exc).__name__}: {exc}",
            retryable=True,
            max_attempts=MAX_ATTEMPTS,
            backoff_seconds=BACKOFF_SECONDS,
        )
        logger.exception("unexpected parse failure task=%s status=%s", task.get("collection_uuid") or task["file_uuid"], status)


def run(*, once: bool = False, max_tasks: int | None = None) -> int:
    engine, _ = init_database(import_users=True)
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4()}"
    processed = 0
    with singleton_lock():
        recovered = recover_expired_tasks(engine)
        recovered_collections = recover_expired_collections(engine)
        if recovered or recovered_collections:
            logger.warning(
                "recovered %s expired task(s) and %s collection(s)",
                recovered,
                recovered_collections,
            )
        logger.info("worker started id=%s concurrency=1", worker_id)
        while True:
            task = claim_next_satellite_collection(engine, worker_id, LEASE_SECONDS)
            if task is None:
                task = claim_next_task(engine, worker_id, LEASE_SECONDS)
            if task is None:
                if once or max_tasks is not None and processed >= max_tasks:
                    return processed
                time.sleep(QUEUE_SLEEP_SECONDS)
                continue
            process_one(engine, task, worker_id)
            processed += 1
            if once or max_tasks is not None and processed >= max_tasks:
                return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-concurrency Adapter Worker")
    parser.add_argument("--once", action="store_true", help="process at most one task and exit")
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()
    try:
        run(once=args.once, max_tasks=args.max_tasks)
    except KeyboardInterrupt:
        logger.info("worker stopped")


if __name__ == "__main__":
    main()
