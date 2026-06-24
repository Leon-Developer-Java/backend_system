import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from adapters import (
    cma_adapter,
    era5_adapter,
    gfs_adapter,
    himawari_adapter,
    radar_adapter,
    wrf_adapter,
)
from services import (
    cma_service,
    era5_service,
    gfs_service,
    himawari_service,
    radar_service,
    wrf_service,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

BUSINESS_DIRS = {
    "CMA": DATA_DIR / "CMA",
    "ERA5": DATA_DIR / "ERA5",
    "GFS": DATA_DIR / "GFS",
    "Himawari": DATA_DIR / "Himawari",
    "Radar": DATA_DIR / "Radar",
    "WRF": DATA_DIR / "WRF",
}

ADAPTERS = {
    "CMA": cma_adapter,
    "ERA5": era5_adapter,
    "GFS": gfs_adapter,
    "Himawari": himawari_adapter,
    "Radar": radar_adapter,
    "WRF": wrf_adapter,
}

DISPLAY_SERVICES = {
    "CMA": cma_service,
    "ERA5": era5_service,
    "GFS": gfs_service,
    "HIMAWARI": himawari_service,
    "RADAR": radar_service,
    "WRF": wrf_service,
}


app = FastAPI(title="Weather Data Display Backend", version="0.1.0")

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

# 关键：让前端可以访问后端生成的 PNG：
# http://127.0.0.1:8002/data/GFS/xxx.png
# http://127.0.0.1:8002/data/GFS/wait_process/xxx.png
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


def ok(data: Any = None, message: str = "success") -> dict[str, Any]:
    return {"code": 0, "data": data, "message": message}


def infer_business_type(filename: str) -> str:
    name = filename.lower()
    suffix = Path(filename).suffix.lower()

    if himawari_adapter.is_hsd_filename(filename):
        return "Himawari"
    if name.startswith("z_radr") or "z_radr" in name:
        return "Radar"
    if "cma" in name:
        return "CMA"
    if "era5" in name:
        return "ERA5"
    if "gfs" in name:
        return "GFS"
    if "himawari" in name or "hsd" in name:
        return "Himawari"
    if "radar" in name or "cinrad" in name:
        return "Radar"
    if "wrf" in name:
        return "WRF"

    if suffix in {".grib", ".grib2"}:
        return "GFS"
    if suffix == ".hsd":
        return "Himawari"
    if suffix in {".cinrad", ".radar", ".bz2"}:
        return "Radar"
    if suffix == ".nc":
        return "ERA5"

    raise ValueError("无法根据文件名或扩展名识别业务类型，请在文件名中包含 CMA、ERA5、GFS、Himawari、Radar 或 WRF。")


def save_upload_file(file: UploadFile, target_dir: Path, business_type: str | None = None) -> Path:
    if not file.filename:
        raise ValueError("上传文件名为空。")

    safe_name = Path(file.filename).name
    if business_type and (adapter := ADAPTERS.get(business_type)) and hasattr(adapter, "upload_target_dir"):
        target_dir = adapter.upload_target_dir(safe_name, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name

    with target_path.open("wb") as output:
        output.write(file.file.read())

    return target_path


def save_upload_files(files: list[UploadFile], target_dir: Path, business_type: str) -> list[Path]:
    return [save_upload_file(item, target_dir, business_type=business_type) for item in files]


def infer_upload_business_type(files: list[UploadFile]) -> str:
    for item in files:
        if item.filename and himawari_adapter.is_hsd_filename(Path(item.filename).name):
            return "Himawari"
    for item in files:
        if not item.filename:
            continue
        try:
            return infer_business_type(item.filename)
        except ValueError:
            continue
    raise ValueError("无法根据文件名或扩展名识别业务类型，请在文件名中包含 CMA、ERA5、GFS、Himawari、Radar 或 WRF。")


@app.get("/")
def root() -> dict[str, Any]:
    return ok({"service": "weather-data-display-backend", "docs": "/docs"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    return ok({"status": "online"})


@app.post("/api/files/parse")
def parse_file(
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
) -> dict[str, Any]:
    try:
        upload_files = [item for item in ([file] if file else []) + (files or []) if item.filename]
        if not upload_files:
            raise ValueError("请选择要解析的文件。")
        business_type = infer_upload_business_type(upload_files)
        adapter = ADAPTERS[business_type]
        if hasattr(adapter, "select_upload_files"):
            upload_files = adapter.select_upload_files(upload_files)
        saved_paths = save_upload_files(upload_files, BUSINESS_DIRS[business_type], business_type)
        saved_path = saved_paths[0]
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


@app.get("/api/display/{business_type}")
def display_data(
    business_type: str,
    variable: str | None = Query(default=None),
    level_index: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    service = DISPLAY_SERVICES.get(business_type.upper())
    if service is None:
        raise HTTPException(status_code=404, detail="不支持的数据类型。")

    if business_type.upper() == "CMA":
        return ok(service.get_display_data(variable=variable, level_index=level_index))

    return ok(service.get_display_data())


@app.get("/api/radar/grid")
def radar_grid(
    file: str | None = Query(default=None),
    product: str = Query(...),
    level: str = Query(default="max"),
) -> Response:
    try:
        grid = radar_service.get_grid_data(file, product, level)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headers = {
        "X-Radar-Nx": str(grid["grid"]["nx"]),
        "X-Radar-Ny": str(grid["grid"]["ny"]),
        "X-Radar-Extent": ",".join(str(item) for item in grid["extent"]),
        "X-Radar-Missing": str(grid["missing"]),
        "X-Radar-Dtype": grid["dtype"],
    }
    return Response(content=grid["bytes"], media_type="application/octet-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
