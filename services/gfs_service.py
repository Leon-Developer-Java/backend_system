from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from adapters.gfs_adapter import process_file


BASE_DIR = Path(__file__).resolve().parents[1]

# 展示后端数据根目录。
# 默认是 backend_system/data。
# 如果老师部署目录是 D:/weather_prediction_system/backend/data，
# 可以通过环境变量 WEATHER_DATA_ROOT 覆盖。
DATA_ROOT = Path(os.getenv("WEATHER_DATA_ROOT", str(BASE_DIR / "data")))

# GFS 主目录
DATA_DIR = DATA_ROOT / "GFS"

# 上传后台落盘目录：data/GFS/wait_process/
WAIT_PROCESS_DIR = DATA_DIR / "wait_process"


def _to_web_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None

    return str(path).replace("\\", "/")


def _unique_sorted_files(files: list[Path]) -> list[Path]:
    """
    去重并按修改时间倒序排列。
    """
    unique: dict[str, Path] = {}

    for file in files:
        try:
            unique[str(file.resolve())] = file
        except Exception:
            unique[str(file)] = file

    return sorted(
        unique.values(),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _list_grib_files() -> list[Path]:
    """
    优先读取上传后台落盘目录：
        data/GFS/wait_process/

    同时兼容本地调试目录：
        data/GFS/
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAIT_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []

    search_dirs = [
        WAIT_PROCESS_DIR,  # 上传后台保存目录，优先
        DATA_DIR,          # 本地测试目录，兜底
    ]

    for folder in search_dirs:
        for suffix in ["*.grib", "*.grb", "*.grib2"]:
            files.extend(folder.glob(suffix))

    return _unique_sorted_files(files)


def _list_meta_files() -> list[Path]:
    """
    同时搜索：
        data/GFS/wait_process/*.meta.json
        data/GFS/*.meta.json
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAIT_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []

    for folder in [WAIT_PROCESS_DIR, DATA_DIR]:
        files.extend(folder.glob("*.meta.json"))

    return _unique_sorted_files(files)


def _list_png_files() -> list[Path]:
    """
    同时搜索：
        data/GFS/wait_process/*.png
        data/GFS/*.png
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAIT_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []

    for folder in [WAIT_PROCESS_DIR, DATA_DIR]:
        files.extend(folder.glob("*.png"))

    return _unique_sorted_files(files)


def _find_meta_for_grib(grib_file: Path) -> Path:
    """
    例如：
        053031.grib
    对应：
        053031.grib.meta.json
    """
    return grib_file.with_name(grib_file.name + ".meta.json")


def _find_png_for_grib(grib_file: Path) -> Path:
    """
    例如：
        053031.grib
    对应：
        053031.grib.png
    """
    return grib_file.with_name(grib_file.name + ".png")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _to_static_url(path: Optional[Path]) -> Optional[str]:
    """
    把本地 png 路径转换成浏览器可访问 URL。

    如果 png 在：
        backend_system/data/GFS/xxx.png
    返回：
        /data/GFS/xxx.png

    如果 png 在：
        backend_system/data/GFS/wait_process/xxx.png
    返回：
        /data/GFS/wait_process/xxx.png
    """
    if path is None:
        return None

    try:
        rel_path = path.resolve().relative_to(DATA_ROOT.resolve())
        rel_url = str(rel_path).replace("\\", "/")
        return f"/data/{rel_url}"
    except Exception:
        return f"/data/GFS/{path.name}"


def _ensure_latest_meta_and_png() -> tuple[Optional[Path], Optional[dict[str, Any]], Optional[Path], Optional[Path]]:
    """
    1. 优先从 data/GFS/wait_process 找最新 GRIB/GRIB2
    2. 如果 wait_process 没有，再从 data/GFS 找
    3. 检查对应 meta.json 和 png 是否存在
    4. 如果缺失或者过期，调用 gfs_adapter.process_file 自动生成
    5. 返回 meta_path, meta_json, latest_grib, png_path
    """
    grib_files = _list_grib_files()

    if not grib_files:
        return None, None, None, None

    latest_grib = grib_files[0]
    expected_meta = _find_meta_for_grib(latest_grib)
    expected_png = _find_png_for_grib(latest_grib)

    need_parse = False

    if not expected_meta.exists():
        need_parse = True

    if not expected_png.exists():
        need_parse = True

    if expected_meta.exists() and expected_meta.stat().st_mtime < latest_grib.stat().st_mtime:
        need_parse = True

    if expected_png.exists() and expected_png.stat().st_mtime < latest_grib.stat().st_mtime:
        need_parse = True

    if need_parse:
        process_file(str(latest_grib), data_type="GFS")

    meta_json = _read_json(expected_meta)

    if meta_json is None:
        meta_files = _list_meta_files()
        if meta_files:
            expected_meta = meta_files[0]
            meta_json = _read_json(expected_meta)

    if not expected_png.exists():
        png_files = _list_png_files()
        expected_png = png_files[0] if png_files else None

    return (
        expected_meta if expected_meta.exists() else None,
        meta_json,
        latest_grib,
        expected_png if expected_png and expected_png.exists() else None,
    )


def get_display_data() -> dict[str, Any]:
    """
    GFS 展示接口数据。

    返回给 main.py 的 /api/display/GFS。
    前端需要重点使用：
        weather_info
        png_url
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAIT_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path, meta_json, latest_grib, png_path = _ensure_latest_meta_and_png()

    grib_files = _list_grib_files()
    meta_files = _list_meta_files()
    png_files = _list_png_files()

    weather_info = None

    if isinstance(meta_json, dict):
        weather_info = meta_json.get("weather_info")

    return {
        "business_type": "GFS",
        "data_type": "GFS",
        "status": "ok" if latest_grib else "no_data",
        "message": "GFS 数据读取成功" if latest_grib else "data/GFS 和 data/GFS/wait_process 目录下暂无 GRIB/GRIB2 文件",

        # 当前实际采用的源文件
        "source_file": _to_web_path(latest_grib),
        "source_files": [_to_web_path(path) for path in grib_files],

        # 当前实际采用的 meta
        "meta_file": _to_web_path(meta_path),
        "meta_files": [_to_web_path(path) for path in meta_files],
        "meta_json": meta_json,

        # 解析后的气象信息
        "weather_info": weather_info,

        # 当前实际采用的 png
        "png": _to_web_path(png_path),
        "png_url": _to_static_url(png_path),

        # 所有 png
        "png_files": [_to_web_path(path) for path in png_files],
        "png_urls": [_to_static_url(path) for path in png_files],

        # 调试信息
        "debug": {
            "data_root": _to_web_path(DATA_ROOT),
            "data_dir": _to_web_path(DATA_DIR),
            "wait_process_dir": _to_web_path(WAIT_PROCESS_DIR),
            "read_priority": [
                _to_web_path(WAIT_PROCESS_DIR),
                _to_web_path(DATA_DIR),
            ],
        },
    }