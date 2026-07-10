from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from adapters.gfs_adapter import process_file


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("WEATHER_DATA_ROOT", str(BASE_DIR / "data")))


def _normalize_data_type(data_type: str | None) -> str:
    return "ECMWF" if str(data_type or "").strip().upper() in {"EC", "ECMWF", "IFS"} else "GFS"


def _data_dir(data_type: str) -> Path:
    return DATA_ROOT / _normalize_data_type(data_type)


def _wait_dir(data_type: str) -> Path:
    return _data_dir(data_type) / "wait_process"


def _sorted_unique(paths: list[Path]) -> list[Path]:
    unique = {str(path.resolve()): path for path in paths if path.exists()}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime, reverse=True)


def _list_files(data_type: str, patterns: tuple[str, ...]) -> list[Path]:
    folders = [_wait_dir(data_type), _data_dir(data_type)]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for folder in folders:
        for pattern in patterns:
            files.extend(folder.glob(pattern))
    return _sorted_unique(files)


def _list_grib_files(data_type: str) -> list[Path]:
    return _list_files(data_type, ("*.grib", "*.grb", "*.grib2"))


def _list_meta_files(data_type: str) -> list[Path]:
    return _list_files(data_type, ("*.meta.json",))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _to_static_url(path: Path) -> str:
    relative = path.resolve().relative_to(DATA_ROOT.resolve())
    return "/data/" + str(relative).replace("\\", "/")


def _meta_is_current(meta: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(meta, dict)
        and meta.get("schema_version") == "2.0"
        and meta.get("image_format") == "webp"
        and meta.get("variable_layers")
        and not any(key in meta for key in ("png_url", "png_urls", "binary_layer", "binary_layers", "grid_urls"))
    )


def _ensure_latest_meta(data_type: str) -> tuple[Path | None, dict[str, Any] | None, Path | None]:
    grib_files = _list_grib_files(data_type)
    if not grib_files:
        meta_files = _list_meta_files(data_type)
        if not meta_files:
            return None, None, None
        return meta_files[0], _read_json(meta_files[0]), None

    source_file = grib_files[0]
    meta_file = source_file.with_name(f"{source_file.name}.meta.json")
    meta = _read_json(meta_file) if meta_file.exists() else None

    need_parse = (
        not meta_file.exists()
        or meta_file.stat().st_mtime < source_file.stat().st_mtime
        or not _meta_is_current(meta)
    )
    if need_parse:
        meta = process_file(str(source_file), data_type=data_type)

    return meta_file, meta, source_file


def get_display_data(data_type: str | None = "GFS") -> dict[str, Any]:
    """
    轻量展示接口：
    - 不返回完整 meta_json 的重复副本；
    - 不返回服务器绝对路径；
    - 不返回 PNG、float32、binary、3km/1km 字段；
    - 只返回前端当前需要的一份 WebP 图层结构。
    """
    source = _normalize_data_type(data_type)
    meta_file, meta, source_file = _ensure_latest_meta(source)

    if not meta:
        return {
            "business_type": source,
            "data_type": source,
            "source": source,
            "status": "no_data",
            "message": f"data/{source} 目录下暂无可展示的 GRIB/GRIB2 数据。",
            "image_format": "webp",
            "render_mode": "webp",
            "variable_options": [],
            "variable_layers": {},
        }

    default_variable = meta.get("default_variable")
    variable_layers = meta.get("variable_layers") or {}
    default_layer = variable_layers.get(default_variable) or next(iter(variable_layers.values()), {})

    return {
        "business_type": source,
        "data_type": source,
        "source": source,
        "status": "ok",
        "message": f"{source} 数据读取成功",
        "file_name": source_file.name if source_file else meta.get("file_name"),
        "meta_url": _to_static_url(meta_file) if meta_file else None,
        "schema_version": meta.get("schema_version", "2.0"),
        "run_time": meta.get("run_time"),
        "extent": meta.get("extent"),
        "bbox": meta.get("bbox"),
        "grid": meta.get("grid"),
        "resolution": meta.get("resolution"),
        "default_variable": default_variable,
        "variable_options": meta.get("variable_options", []),
        "variable_layers": variable_layers,
        "weather_info": meta.get("weather_info", {}),
        "times": default_layer.get("times", []),
        "forecast_hours": default_layer.get("forecast_hours", []),
        "forecast_labels": default_layer.get("forecast_labels", []),
        "image_format": "webp",
        "image_url": default_layer.get("image_url") or meta.get("image_url"),
        "image_urls": default_layer.get("image_urls") or meta.get("image_urls", []),
        "render_mode": "webp",
    }
