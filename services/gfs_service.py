from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from adapters.gfs_adapter import process_file


BASE_DIR = Path(__file__).resolve().parents[1]

# 展示后端数据根目录。
# 默认是 backend_system/data。
# 可通过环境变量 WEATHER_DATA_ROOT 覆盖。
DATA_ROOT = Path(os.getenv("WEATHER_DATA_ROOT", str(BASE_DIR / "data")))


def _normalize_data_type(data_type: str | None = None) -> str:
    key = str(data_type or "GFS").strip().upper()
    if key in {"EC", "ECMWF", "IFS"}:
        return "ECMWF"
    return "GFS"


def _data_dir(data_type: str | None = None) -> Path:
    return DATA_ROOT / _normalize_data_type(data_type)


def _wait_process_dir(data_type: str | None = None) -> Path:
    return _data_dir(data_type) / "wait_process"


def _to_web_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def _unique_sorted_files(files: list[Path]) -> list[Path]:
    """去重并按修改时间倒序排列。"""
    unique: dict[str, Path] = {}

    for file in files:
        try:
            unique[str(file.resolve())] = file
        except Exception:
            unique[str(file)] = file

    return sorted(unique.values(), key=lambda item: item.stat().st_mtime, reverse=True)


def _list_grib_files(data_type: str | None = None) -> list[Path]:
    """
    优先读取：
        data/{GFS|ECMWF}/wait_process/
    其次读取：
        data/{GFS|ECMWF}/
    """
    data_dir = _data_dir(data_type)
    wait_process_dir = _wait_process_dir(data_type)

    data_dir.mkdir(parents=True, exist_ok=True)
    wait_process_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for folder in [wait_process_dir, data_dir]:
        for suffix in ["*.grib", "*.grb", "*.grib2"]:
            files.extend(folder.glob(suffix))

    return _unique_sorted_files(files)


def _list_meta_files(data_type: str | None = None) -> list[Path]:
    data_dir = _data_dir(data_type)
    wait_process_dir = _wait_process_dir(data_type)

    data_dir.mkdir(parents=True, exist_ok=True)
    wait_process_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for folder in [wait_process_dir, data_dir]:
        files.extend(folder.glob("*.meta.json"))

    return _unique_sorted_files(files)


def _list_files_by_suffix(data_type: str | None = None, suffixes: tuple[str, ...] = (".webp", ".png")) -> list[Path]:
    data_dir = _data_dir(data_type)
    wait_process_dir = _wait_process_dir(data_type)

    data_dir.mkdir(parents=True, exist_ok=True)
    wait_process_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for folder in [wait_process_dir, data_dir]:
        for suffix in suffixes:
            files.extend(folder.glob(f"*{suffix}"))

    return _unique_sorted_files(files)


def _list_webp_files(data_type: str | None = None) -> list[Path]:
    return _list_files_by_suffix(data_type, suffixes=(".webp",))


def _list_png_files(data_type: str | None = None) -> list[Path]:
    return _list_files_by_suffix(data_type, suffixes=(".png",))


def _find_meta_for_grib(grib_file: Path) -> Path:
    return grib_file.with_name(grib_file.name + ".meta.json")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _to_static_url(path: Optional[Path], data_type: str | None = None) -> Optional[str]:
    """
    把本地图片 / 格点路径转换成浏览器可访问 URL。

    例如：
        backend_system/data/GFS/xxx.webp
        backend_system/data/ECMWF/wait_process/xxx.webp

    返回：
        /data/GFS/xxx.webp
        /data/ECMWF/wait_process/xxx.webp
    """
    if path is None:
        return None

    try:
        rel_path = path.resolve().relative_to(DATA_ROOT.resolve())
        rel_url = str(rel_path).replace("\\", "/")
        return f"/data/{rel_url}"
    except Exception:
        return f"/data/{_normalize_data_type(data_type)}/{path.name}"


def _normalize_url(value: Any, data_type: str | None = None) -> str:
    if not value:
        return ""

    text = str(value).replace("\\", "/")

    if text.startswith("http://") or text.startswith("https://") or text.startswith("data:"):
        return text

    if text.startswith("/data/"):
        return text

    idx = text.find("/data/")
    if idx >= 0:
        return text[idx:]

    return f"/data/{_normalize_data_type(data_type)}/{Path(text).name}"


def _normalize_urls(values: Any, data_type: str | None = None) -> list[str]:
    if not isinstance(values, list):
        return []

    out: list[str] = []
    for item in values:
        url = _normalize_url(item, data_type)
        if url:
            out.append(url)

    return out


def _first_nonempty_string(*items: Any) -> str:
    for item in items:
        if isinstance(item, str) and item.strip():
            return item
    return ""


def _first_nonempty_list(*items: Any) -> list[Any]:
    for item in items:
        if isinstance(item, list) and item:
            return item
    return []


def _weather_info(meta_json: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(meta_json, dict):
        info = meta_json.get("weather_info")
        if isinstance(info, dict):
            return info
    return {}


def _nested_dict(meta_json: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(meta_json, dict):
        return {}
    value = meta_json.get(key)
    return value if isinstance(value, dict) else {}


def _extract_image_fields(meta_json: dict[str, Any] | None, data_type: str) -> dict[str, Any]:
    """
    从 meta_json / weather_info / meta / extra 中提取图像字段。
    优先级：
        webp > image > png

    说明：
    - 新前端优先读 image_url / image_urls；
    - 这里保证 image_* 一定优先指向 webp；
    - png_* 仅保留兼容旧前端。
    """
    if not isinstance(meta_json, dict):
        return {
            "image_url": None,
            "image_urls": [],
            "image_format": None,
            "webp_url": None,
            "webp_urls": [],
            "png_url": None,
            "png_urls": [],
        }

    info = _nested_dict(meta_json, "weather_info")
    panel_meta = _nested_dict(meta_json, "meta")
    extra = _nested_dict(meta_json, "extra")

    webp_url = _first_nonempty_string(
        meta_json.get("webp_url"),
        info.get("webp_url"),
        panel_meta.get("webp_url"),
        extra.get("webp_url"),
    )
    image_url = _first_nonempty_string(
        meta_json.get("image_url"),
        info.get("image_url"),
        panel_meta.get("image_url"),
        extra.get("image_url"),
    )
    png_url = _first_nonempty_string(
        meta_json.get("png_url"),
        info.get("png_url"),
        panel_meta.get("png_url"),
        extra.get("png_url"),
    )

    webp_urls = _first_nonempty_list(
        meta_json.get("webp_urls"),
        info.get("webp_urls"),
        panel_meta.get("webp_urls"),
        extra.get("webp_urls"),
    )
    image_urls = _first_nonempty_list(
        meta_json.get("image_urls"),
        info.get("image_urls"),
        panel_meta.get("image_urls"),
        extra.get("image_urls"),
    )
    png_urls = _first_nonempty_list(
        meta_json.get("png_urls"),
        info.get("png_urls"),
        panel_meta.get("png_urls"),
        extra.get("png_urls"),
    )

    preferred_url = webp_url or image_url or png_url
    preferred_urls = webp_urls or image_urls or png_urls

    preferred_url = _normalize_url(preferred_url, data_type) if preferred_url else None
    preferred_urls = _normalize_urls(preferred_urls, data_type)

    webp_url = _normalize_url(webp_url, data_type) if webp_url else None
    webp_urls = _normalize_urls(webp_urls, data_type)

    png_url = _normalize_url(png_url, data_type) if png_url else None
    png_urls = _normalize_urls(png_urls, data_type)

    image_format = "webp" if (webp_url or webp_urls) else ("png" if (png_url or png_urls) else None)

    return {
        "image_url": preferred_url,
        "image_urls": preferred_urls,
        "image_format": image_format,
        "webp_url": webp_url,
        "webp_urls": webp_urls,
        "png_url": png_url,
        "png_urls": png_urls,
    }


def _meta_has_preferred_image(meta_json: dict[str, Any] | None, data_type: str) -> bool:
    fields = _extract_image_fields(meta_json, data_type)
    return bool(fields.get("image_url") or fields.get("image_urls"))


def _ensure_latest_meta(data_type: str | None = None) -> tuple[Optional[Path], Optional[dict[str, Any]], Optional[Path]]:
    """
    1. 找 data/{GFS|ECMWF}/wait_process 最新 GRIB；
    2. 找不到再找 data/{GFS|ECMWF}；
    3. meta 缺失 / 过期 / 没有 webp/image 信息时，调用 adapter 重新解析；
    4. 返回 meta_path, meta_json, latest_grib。
    """
    data_type = _normalize_data_type(data_type)
    grib_files = _list_grib_files(data_type)

    if not grib_files:
        meta_files = _list_meta_files(data_type)
        if meta_files:
            meta_path = meta_files[0]
            return meta_path, _read_json(meta_path), None
        return None, None, None

    latest_grib = grib_files[0]
    expected_meta = _find_meta_for_grib(latest_grib)
    need_parse = False

    if not expected_meta.exists():
        need_parse = True
    else:
        try:
            if expected_meta.stat().st_mtime < latest_grib.stat().st_mtime:
                need_parse = True
        except Exception:
            need_parse = True

        old_meta = _read_json(expected_meta)
        if not _meta_has_preferred_image(old_meta, data_type):
            need_parse = True

    if need_parse:
        process_file(str(latest_grib), data_type=data_type)

    meta_json = _read_json(expected_meta)

    if meta_json is None:
        meta_files = _list_meta_files(data_type)
        if meta_files:
            expected_meta = meta_files[0]
            meta_json = _read_json(expected_meta)

    return (
        expected_meta if expected_meta.exists() else None,
        meta_json,
        latest_grib,
    )


def _fallback_image_urls_from_files(data_type: str) -> dict[str, Any]:
    webp_files = _list_webp_files(data_type)
    png_files = _list_png_files(data_type)

    webp_urls = [_to_static_url(path, data_type) for path in webp_files]
    png_urls = [_to_static_url(path, data_type) for path in png_files]

    webp_urls = [url for url in webp_urls if url]
    png_urls = [url for url in png_urls if url]

    image_urls = webp_urls or png_urls
    image_url = image_urls[0] if image_urls else None

    return {
        "image_url": image_url,
        "image_urls": image_urls,
        "image_format": "webp" if webp_urls else ("png" if png_urls else None),
        "webp_url": webp_urls[0] if webp_urls else None,
        "webp_urls": webp_urls,
        "png_url": png_urls[0] if png_urls else None,
        "png_urls": png_urls,
    }


def _extract_top_level(meta_json: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not isinstance(meta_json, dict):
        return default

    if key in meta_json:
        return meta_json.get(key)

    for nested_key in ("weather_info", "meta", "extra"):
        nested = meta_json.get(nested_key)
        if isinstance(nested, dict) and key in nested:
            return nested.get(key)

    return default


def get_display_data(data_type: str | None = "GFS") -> dict[str, Any]:
    """
    GFS / ECMWF 统一 GRIB 展示接口数据。

    关键原则：
    - GFS 和 ECMWF 是两个独立业务数据源；
    - 目录、接口、source/data_type 保持分开；
    - 二者底层复用同一个 GRIB adapter；
    - 图像展示优先 WebP，PNG 仅作旧前端兼容兜底。
    """
    data_type = _normalize_data_type(data_type)
    data_dir = _data_dir(data_type)
    wait_process_dir = _wait_process_dir(data_type)

    data_dir.mkdir(parents=True, exist_ok=True)
    wait_process_dir.mkdir(parents=True, exist_ok=True)

    meta_path, meta_json, latest_grib = _ensure_latest_meta(data_type)

    grib_files = _list_grib_files(data_type)
    meta_files = _list_meta_files(data_type)

    image_fields = _extract_image_fields(meta_json, data_type)
    if not image_fields.get("image_url") and not image_fields.get("image_urls"):
        image_fields = _fallback_image_urls_from_files(data_type)

    weather_info = _weather_info(meta_json)

    message = (
        f"{data_type} 数据读取成功"
        if latest_grib or meta_json
        else f"data/{data_type} 和 data/{data_type}/wait_process 目录下暂无 GRIB/GRIB2 文件"
    )

    return {
        "business_type": data_type,
        "data_type": data_type,
        "source": data_type,
        "status": "ok" if latest_grib or meta_json else "no_data",
        "message": message,

        # 当前实际采用的源文件
        "source_file": _to_web_path(latest_grib),
        "source_files": [_to_web_path(path) for path in grib_files],

        # 当前实际采用的 meta
        "meta_file": _to_web_path(meta_path),
        "meta_files": [_to_web_path(path) for path in meta_files],
        "meta_json": meta_json,

        # 解析后的气象信息
        "weather_info": weather_info,

        # 新前端主字段：WebP 优先
        "image": image_fields.get("image_url"),
        "image_url": image_fields.get("image_url"),
        "image_urls": image_fields.get("image_urls", []),
        "image_format": image_fields.get("image_format"),

        # 明确暴露 WebP
        "webp_url": image_fields.get("webp_url"),
        "webp_urls": image_fields.get("webp_urls", []),

        # PNG 仅作为兼容字段
        "png": image_fields.get("png_url"),
        "png_url": image_fields.get("png_url"),
        "png_urls": image_fields.get("png_urls", []),
        "png_files": [_to_web_path(path) for path in _list_png_files(data_type)],

        # 多变量与时间信息
        "extent": _extract_top_level(meta_json, "extent", None),
        "times": _extract_top_level(meta_json, "times", []),
        "forecast_hours": _extract_top_level(meta_json, "forecast_hours", []),
        "forecast_labels": _extract_top_level(meta_json, "forecast_labels", []),
        "valid_times": _extract_top_level(meta_json, "valid_times", []),
        "valid_hours": _extract_top_level(meta_json, "valid_hours", []),
        "valid_time_hours": _extract_top_level(meta_json, "valid_time_hours", []),

        "variable_options": _extract_top_level(meta_json, "variable_options", []),
        "variable_layers": _extract_top_level(meta_json, "variable_layers", {}),
        "default_variable": _extract_top_level(meta_json, "default_variable", None),

        "binary_layer": _extract_top_level(meta_json, "binary_layer", {}),
        "binary_layers": _extract_top_level(meta_json, "binary_layers", {}),
        "render_mode": "webp" if image_fields.get("image_format") == "webp" else "png",

        # 调试信息
        "debug": {
            "data_root": _to_web_path(DATA_ROOT),
            "data_dir": _to_web_path(data_dir),
            "wait_process_dir": _to_web_path(wait_process_dir),
            "read_priority": [
                _to_web_path(wait_process_dir),
                _to_web_path(data_dir),
            ],
            "preferred_image_format": image_fields.get("image_format"),
            "preferred_image_url": image_fields.get("image_url"),
        },
    }
