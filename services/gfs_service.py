import json
from pathlib import Path
from typing import Any, Optional

from adapters.gfs_adapter import process_file


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "GFS"


def _to_web_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None

    return str(path).replace("\\", "/")


def _list_grib_files() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = []

    for suffix in ["*.grib", "*.grb", "*.grib2"]:
        files.extend(DATA_DIR.glob(suffix))

    return sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)


def _list_meta_files() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    return sorted(
        DATA_DIR.glob("*.meta.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _list_png_files() -> list[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    return sorted(
        DATA_DIR.glob("*.png"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _find_meta_for_grib(grib_file: Path) -> Path:
    return grib_file.with_name(grib_file.name + ".meta.json")


def _find_png_for_grib(grib_file: Path) -> Path:
    return grib_file.with_name(grib_file.name + ".png")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _to_static_url(path: Optional[Path]) -> Optional[str]:
    """
    把本地 data/GFS/xxx.png 转成浏览器可以访问的后端 URL。
    main.py 里需要 app.mount("/data", StaticFiles(...))
    """
    if path is None:
        return None

    return f"/data/GFS/{path.name}"


def _ensure_latest_meta_and_png() -> tuple[Optional[Path], Optional[dict[str, Any]], Optional[Path], Optional[Path]]:
    """
    1. 找最新 GRIB/GRIB2
    2. 检查对应 meta.json 和 png 是否存在
    3. 如果缺失或者过期，自动调用 gfs_adapter.process_file 生成
    4. 返回 meta_path, meta_json, latest_grib, png_path
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)

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
        "message": "GFS 数据读取成功" if latest_grib else "data/GFS 目录下暂无 GRIB/GRIB2 文件",

        "source_file": _to_web_path(latest_grib),
        "source_files": [_to_web_path(path) for path in grib_files],

        "meta_file": _to_web_path(meta_path),
        "meta_files": [_to_web_path(path) for path in meta_files],
        "meta_json": meta_json,

        "weather_info": weather_info,

        "png": _to_web_path(png_path),
        "png_url": _to_static_url(png_path),
        "png_files": [_to_web_path(path) for path in png_files],
        "png_urls": [_to_static_url(path) for path in png_files],
    }