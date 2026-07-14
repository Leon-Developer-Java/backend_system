import asyncio
import contextlib
import os
import json
import subprocess
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from adapters import (
    cma_adapter,
    era5_adapter,
    fy3_adapter,
    gfs_adapter,
    himawari_adapter,
    radar_adapter,
    wrf_adapter,
)
from services import (
    cma_service,
    era5_service,
    era5_store,
    fy3_service,
    gfs_service,
    himawari_service,
    radar_service,
    wrf_service,
)

try:
    from services import himawari_scheduler
except Exception:
    himawari_scheduler = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

BUSINESS_DIRS = {
    "CMA": DATA_DIR / "CMA",
    "ERA5": DATA_DIR / "ERA5",
    "GFS": DATA_DIR / "GFS",
    "ECMWF": DATA_DIR / "ECMWF",
    "FY3": DATA_DIR / "FY3",
    "Himawari": DATA_DIR / "Himawari",
    "Radar": DATA_DIR / "Radar",
    "WRF": DATA_DIR / "WRF",
}

ADAPTERS = {
    "CMA": cma_adapter,
    "ERA5": era5_adapter,
    "GFS": gfs_adapter,
    "ECMWF": gfs_adapter,
    "FY3": fy3_adapter,
    "Himawari": himawari_adapter,
    "Radar": radar_adapter,
    "WRF": wrf_adapter,
}

DISPLAY_SERVICES = {
    "CMA": cma_service,
    "ERA5": era5_service,
    "GFS": gfs_service,
    "ECMWF": gfs_service,
    "FY3": fy3_service,
    "HIMAWARI": himawari_service,
    "RADAR": radar_service,
    "WRF": wrf_service,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    himawari_tasks = []
    if himawari_scheduler is not None:
        himawari_tasks = himawari_scheduler.start_himawari_auto_download()
    try:
        yield
    finally:
        if himawari_tasks:
            for task in himawari_tasks:
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*himawari_tasks)


app = FastAPI(title="Weather Data Display Backend", version="0.1.0", lifespan=lifespan)

_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:5174,http://127.0.0.1:5174,"
    "http://localhost:5177,http://127.0.0.1:5177",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 关键：让前端可以访问后端生成的静态图片：
# http://127.0.0.1:8002/data/GFS/xxx.png
# http://127.0.0.1:8002/data/Radar/xxx.webp
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


def ok(data: Any = None, message: str = "success") -> dict[str, Any]:
    return {"code": 0, "data": data, "message": message}


DIFF_TASK_LOG = DATA_DIR / "diff_tasks.jsonl"


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_diff_task(record: dict[str, Any]) -> None:
    DIFF_TASK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DIFF_TASK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_diff_tasks(limit: int = 20) -> list[dict[str, Any]]:
    if not DIFF_TASK_LOG.exists():
        return []

    rows = []
    with DIFF_TASK_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    latest = {}
    for row in rows:
        tid = row.get("task_id")
        if tid:
            latest[tid] = row

    return list(latest.values())[-limit:][::-1]


def _run_diff_build_task(
    task_id: str,
    source: str,
    lead_start: int,
    lead_end: int,
    lead_step: int,
    bbox: str,
    max_pixels: int,
) -> None:
    start = datetime.now()

    _append_diff_task({
        "task_id": task_id,
        "source": source,
        "status": "running",
        "started_at": _now_text(),
        "lead_start": lead_start,
        "lead_end": lead_end,
        "lead_step": lead_step,
        "bbox": bbox,
        "max_pixels": max_pixels,
    })

    env = os.environ.copy()
    env["WEATHER_DIFF_BUILD_SYNC"] = "1"
    env["WEATHER_DIFF_BBOX"] = bbox
    env["WEATHER_DIFF_MAX_PIXELS"] = str(max_pixels)

    cmd = [
        os.sys.executable,
        str(BASE_DIR / "scripts" / "download_gfs_ecmwf.py"),
        "--source", source,
        "--lead-start", str(lead_start),
        "--lead-end", str(lead_end),
        "--lead-step", str(lead_step),
        "--min-success", "4",
        "--parse-after",
        "--overwrite",
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=None,
        )

        duration = (datetime.now() - start).total_seconds()
        log_tail = (completed.stdout or "")[-5000:]

        _append_diff_task({
            "task_id": task_id,
            "source": source,
            "status": "success" if completed.returncode == 0 else "failed",
            "started_at": start.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_text(),
            "duration_seconds": round(duration, 2),
            "exit_code": completed.returncode,
            "lead_start": lead_start,
            "lead_end": lead_end,
            "lead_step": lead_step,
            "bbox": bbox,
            "max_pixels": max_pixels,
            "log_tail": log_tail,
        })

    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        _append_diff_task({
            "task_id": task_id,
            "source": source,
            "status": "failed",
            "started_at": start.strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_text(),
            "duration_seconds": round(duration, 2),
            "lead_start": lead_start,
            "lead_end": lead_end,
            "lead_step": lead_step,
            "bbox": bbox,
            "max_pixels": max_pixels,
            "error": str(e),
        })



def normalize_business_type(value: str) -> str:
    key = str(value or "").strip()
    upper = key.upper()

    if key == "雷达":
        return "Radar"
    if key == "葵花":
        return "Himawari"
    if upper == "FY-3":
        return "FY3"
    if upper == "HIMAWARI":
        return "Himawari"
    if upper == "RADAR":
        return "Radar"
    if upper == "ECMWF":
        return "ECMWF"
    if upper in {"CMA", "ERA5", "FY3", "GFS", "WRF"}:
        return upper

    return key


def infer_business_type(filename: str) -> str:
    name = filename.lower()
    suffix = Path(filename).suffix.lower()

    if himawari_adapter.is_hsd_filename(filename):
        return "Himawari"
    if fy3_adapter.is_fy3_filename(filename):
        return "FY3"
    if name.startswith("z_radr") or "z_radr" in name:
        return "Radar"
    if "ecmwf" in name or "ifs" in name:
        return "ECMWF"
    if "cma" in name:
        return "CMA"
    if "era5" in name:
        return "ERA5"
    if "gfs" in name:
        return "GFS"
    if "fy3" in name or "fy-3" in name:
        return "FY3"
    if "himawari" in name or "hsd" in name:
        return "Himawari"
    if "radar" in name or "cinrad" in name:
        return "Radar"
    if "wrf" in name:
        return "WRF"

    if suffix in {".grib", ".grib2"}:
        return "GFS"
    if suffix == ".hdf" and "fy3" in name:
        return "FY3"
    if suffix == ".hsd":
        return "Himawari"
    if suffix in {".cinrad", ".radar", ".bz2"}:
        return "Radar"
    if suffix == ".nc":
        return "ERA5"

    raise ValueError("无法根据文件名或扩展名识别业务类型，请在文件名中包含 CMA、ERA5、GFS、ECMWF、FY3、Himawari、Radar 或 WRF。")


def save_upload_file(file: UploadFile, target_dir: Path, business_type: str | None = None) -> Path:
    if not file.filename:
        raise ValueError("上传文件名为空。")

    safe_name = Path(file.filename).name
    if business_type and (adapter := ADAPTERS.get(business_type)) and hasattr(adapter, "upload_target_dir"):
        target_dir = adapter.upload_target_dir(safe_name, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name
    if business_type not in {"Radar", "FY3", "Himawari"}:
        target_path = unique_upload_path(target_path)

    with target_path.open("wb") as output:
        output.write(file.file.read())

    return target_path


def unique_upload_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def save_upload_files(files: list[UploadFile], target_dir: Path, business_type: str) -> list[Path]:
    return [save_upload_file(item, target_dir, business_type=business_type) for item in files]


def infer_upload_business_type(files: list[UploadFile]) -> str:
    for item in files:
        if item.filename and himawari_adapter.is_hsd_filename(Path(item.filename).name):
            return "Himawari"
    for item in files:
        if item.filename and fy3_adapter.is_fy3_filename(Path(item.filename).name):
            return "FY3"
    for item in files:
        if not item.filename:
            continue
        try:
            return infer_business_type(item.filename)
        except ValueError:
            continue
    raise ValueError("无法根据文件名或扩展名识别业务类型，请在文件名中包含 CMA、ERA5、GFS、ECMWF、FY3、Himawari、Radar 或 WRF。")


def _upload_files_from_params(
    file: UploadFile | None,
    files: list[UploadFile] | None,
) -> list[UploadFile]:
    return [item for item in ([file] if file else []) + (files or []) if item.filename]


def _raw_upload_files_for_business(upload_files: list[UploadFile], business_type: str) -> list[UploadFile]:
    if business_type == "FY3":
        return [item for item in upload_files if item.filename and fy3_adapter.is_fy3_filename(item.filename)]
    if business_type == "Himawari":
        return [item for item in upload_files if item.filename and himawari_adapter.is_hsd_filename(item.filename)]
    return upload_files


def _raw_upload_file_mismatches(upload_files: list[UploadFile], business_type: str) -> list[str]:
    accepted = _raw_upload_files_for_business(upload_files, business_type)
    accepted_ids = {id(item) for item in accepted}
    return [Path(item.filename or "").name for item in upload_files if id(item) not in accepted_ids]


def _raw_scenes_for_business(business_type: str) -> list[dict[str, Any]]:
    if business_type == "FY3":
        return fy3_adapter.scan_raw_scenes(BUSINESS_DIRS["FY3"])
    if business_type == "Himawari":
        return himawari_adapter.scan_raw_scenes(BUSINESS_DIRS["Himawari"])
    raise ValueError("raw 场景队列仅支持 FY3 和 Himawari。")


@app.get("/")
def root() -> dict[str, Any]:
    return ok({"service": "weather-data-display-backend", "docs": "/docs"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    return ok({"status": "online"})


@app.get("/api/himawari/auto-status")
def himawari_auto_status() -> dict[str, Any]:
    if himawari_scheduler is None:
        return ok({
            "status": "disabled",
            "message": "当前 services 中未配置 himawari_scheduler，已跳过自动下载任务。",
        })
    return ok(himawari_scheduler.get_himawari_auto_status())


@app.get("/api/himawari/auto-log")
def himawari_auto_log(lines: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
    log_lines = himawari_scheduler.read_himawari_auto_log(lines=lines)
    return ok({"lines": log_lines, "count": len(log_lines)})


@app.post("/api/files/parse")
def parse_file(
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    business_type: str | None = Form(default=None),
    data_type: str | None = Form(default=None),
) -> dict[str, Any]:
    try:
        upload_files = _upload_files_from_params(file, files)
        if not upload_files:
            raise ValueError("请选择要解析的文件。")
        requested_type = normalize_business_type(business_type or data_type or "")
        if requested_type and requested_type in ADAPTERS:
            business_type = requested_type
        else:
            business_type = infer_upload_business_type(upload_files)
        adapter = ADAPTERS[business_type]
        if hasattr(adapter, "select_upload_files"):
            upload_files = adapter.select_upload_files(upload_files)
        saved_paths = save_upload_files(upload_files, BUSINESS_DIRS[business_type], business_type)
        saved_path = saved_paths[0]
        if business_type == "Radar" and len(saved_paths) > 1:
            meta = radar_adapter.process_files([str(path) for path in saved_paths], data_type=business_type)
        elif business_type == "CMA" and len(saved_paths) > 1:
            meta = cma_adapter.process_files([str(path) for path in saved_paths], data_type=business_type)
        else:
            meta = ADAPTERS[business_type].process_file(str(saved_path), data_type=business_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ok(
        {
            "file_name": saved_path.name,
            "file_count": len(saved_paths),
            "directory": str(saved_path.parent).replace("\\", "/") + "/",
            "business_type": business_type,
            "meta": meta,
            "weather_info": meta.get("weather_info", {}),
        }
    )


@app.post("/api/files/raw-upload")
def raw_upload_file(
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    business_type: str | None = Form(default=None),
    data_type: str | None = Form(default=None),
) -> dict[str, Any]:
    try:
        upload_files = _upload_files_from_params(file, files)
        if not upload_files:
            raise ValueError("请选择要上传的 raw 文件。")
        requested_type = normalize_business_type(business_type or data_type or "")
        business = requested_type if requested_type in {"FY3", "Himawari"} else infer_upload_business_type(upload_files)
        if business not in {"FY3", "Himawari"}:
            raise ValueError("raw-only 上传当前仅支持 FY3 和 Himawari。")

        mismatches = _raw_upload_file_mismatches(upload_files, business)
        if mismatches:
            raise ValueError(f"{business} raw 上传包含不匹配文件：{'、'.join(mismatches)}")

        raw_files = _raw_upload_files_for_business(upload_files, business)
        if not raw_files:
            raise ValueError(f"未找到可识别的 {business} raw 文件。")

        saved_paths = save_upload_files(raw_files, BUSINESS_DIRS[business], business)
        scenes = _raw_scenes_for_business(business)
        touched_dirs = {path.parent.as_posix() for path in saved_paths}
        touched_scenes = [scene for scene in scenes if scene.get("raw_dir") in touched_dirs]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ok(
        {
            "business_type": business,
            "file_count": len(saved_paths),
            "files": [path.name for path in saved_paths],
            "directories": sorted(touched_dirs),
            "scenes": touched_scenes,
            "all_scene_count": len(scenes),
            "message": "raw 文件已保存，未触发解析。",
        }
    )


@app.get("/api/display/{business_type}/raw-scenes")
def raw_scenes(business_type: str) -> dict[str, Any]:
    business = normalize_business_type(business_type)
    try:
        scenes = _raw_scenes_for_business(business)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ok({"business_type": business, "scene_count": len(scenes), "scenes": scenes})


@app.post("/api/display/{business_type}/update")
def update_display_from_raw(
    business_type: str,
    force: bool = Query(default=False),
) -> dict[str, Any]:
    business = normalize_business_type(business_type)
    try:
        if business == "FY3":
            result = fy3_adapter.update_from_raw(BUSINESS_DIRS["FY3"], force=force)
        elif business == "Himawari":
            result = himawari_adapter.update_from_raw(BUSINESS_DIRS["Himawari"], BUSINESS_DIRS["Himawari"], force=force)
        else:
            raise ValueError("raw update 当前仅支持 FY3 和 Himawari。")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ok({"business_type": business, **result})


@app.post("/api/display/{business_type}/diff-build")
def start_diff_build(
    business_type: str,
    background_tasks: BackgroundTasks,
    bbox: str = Query(default="105,20,125,40"),
    max_pixels: int = Query(default=12000000, ge=100000, le=120000000),
    lead_start: int = Query(default=0, ge=0),
    lead_end: int = Query(default=24, ge=0),
    lead_step: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    source = business_type.upper()

    if source not in {"GFS", "ECMWF"}:
        raise HTTPException(status_code=400, detail="仅支持 GFS / ECMWF 差分构建。")

    if lead_step is None:
        lead_step = 3 if source == "ECMWF" else 1

    task_id = f"diff_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    background_tasks.add_task(
        _run_diff_build_task,
        task_id,
        source,
        lead_start,
        lead_end,
        lead_step,
        bbox,
        max_pixels,
    )

    return ok({
        "task_id": task_id,
        "source": source,
        "status": "submitted",
        "bbox": bbox,
        "max_pixels": max_pixels,
        "lead_start": lead_start,
        "lead_end": lead_end,
        "lead_step": lead_step,
        "message": "差分生成任务已提交，原始分辨率可先展示，1km/3km 生成完成后刷新页面即可切换。",
    })


@app.get("/api/display/diff-tasks")
def list_diff_tasks(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    return ok({
        "tasks": _read_diff_tasks(limit=limit),
    })


@app.get("/api/display/{business_type}")
def display_data(
    business_type: str,
    variable: str | None = Query(default=None),
    level_index: int = Query(default=0, ge=0),
    time_index: int = Query(default=0, ge=0),
    resolution: str | None = Query(default="native"),
    meta_file: str | None = Query(default=None),
    scene_id: str | None = Query(default=None),
) -> dict[str, Any]:
    raw_key = business_type.upper()
    normalized_key = normalize_business_type(business_type)
    service = DISPLAY_SERVICES.get(raw_key)

    if service is None:
        raise HTTPException(status_code=404, detail="不支持的数据类型。")

    if raw_key == "CMA":
        return ok(service.get_display_data(variable=variable, level_index=level_index, time_index=time_index, meta_file=meta_file, resolution=resolution))

    if raw_key == "ERA5":
        return ok(service.get_display_data(variable=variable, level_index=level_index))

    if raw_key == "HIMAWARI":
        return ok(service.get_display_data(scene_id=scene_id))
    if business_type.upper() == "RADAR":
        return ok(service.get_display_data(time_index=time_index))

    if raw_key in {"GFS", "ECMWF"}:
        return ok(gfs_service.get_display_data(data_type=normalized_key))

    return ok(service.get_display_data())


@app.get("/api/ERA5/datasets")
def era5_datasets(
    keyword: str | None = Query(default=None),
    variable: str | None = Query(default=None),
    status: str | None = Query(default=None),
    time_start: str | None = Query(default=None),
    time_end: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return ok(
        era5_store.list_datasets(
            keyword=keyword,
            variable=variable,
            status=status,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
            offset=offset,
        )
    )


@app.get("/api/ERA5/datasets/{dataset_id}")
def era5_dataset_detail(dataset_id: str) -> dict[str, Any]:
    dataset = era5_store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="ERA5 dataset not found.")
    return ok(dataset)


@app.get("/api/ERA5/datasets/{dataset_id}/assets")
def era5_dataset_assets(
    dataset_id: str,
    variable: str | None = Query(default=None),
    resolution: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    return ok(
        era5_store.list_assets(
            dataset_id,
            variable=variable,
            resolution=resolution,
            limit=limit,
            offset=offset,
        )
    )


@app.post("/api/wrf/rescan")
def wrf_rescan() -> dict[str, Any]:
    results = []
    for raw_file in BUSINESS_DIRS["WRF"].glob("wrfout_d*"):
        if raw_file.suffix or not raw_file.is_file():
            continue
        try:
            meta = wrf_adapter.process_file(str(raw_file))
            results.append({
                "file": raw_file.name,
                "status": "ok",
                "webp_count": len(meta.get("webp_files", [])),
            })
        except Exception as exc:
            results.append({"file": raw_file.name, "status": "error", "error": str(exc)})
    return ok({"processed": len(results), "results": results})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=True)
