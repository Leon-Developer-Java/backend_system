import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from adapters import radar_adapter


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Radar"


def _as_posix(path: Path | None) -> str | None:
    return str(path).replace("\\", "/") if path else None


def _png_data_url(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _png_from_meta(meta_json: dict[str, Any] | None) -> Path | None:
    if not meta_json:
        return None
    for key in ("default_png", "png"):
        value = meta_json.get(key)
        if value:
            path = Path(value)
            if path.exists():
                return path
    for value in meta_json.get("png_files", []):
        path = Path(value)
        if path.exists():
            return path
    weather_png = meta_json.get("weather_info", {}).get("png")
    if weather_png:
        path = Path(weather_png)
        if path.exists():
            return path
    return None


def _source_from_meta(meta_json: dict[str, Any] | None) -> Path | None:
    if not meta_json:
        return None
    value = meta_json.get("source_file") or meta_json.get("file_detail", {}).get("path")
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _frame_source(frame: dict[str, Any] | None) -> Path | None:
    if not frame:
        return None
    value = frame.get("source_file")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _frame_png(frame: dict[str, Any] | None) -> Path | None:
    if not frame:
        return None
    for key in ("default_png", "png"):
        value = frame.get(key)
        if value:
            path = Path(value)
            if path.exists():
                return path
    return None


def _frame_from_meta(meta_json: dict[str, Any]) -> dict[str, Any] | None:
    source = _source_from_meta(meta_json)
    if not source:
        return None
    weather_info = meta_json.get("weather_info", {})
    times = meta_json.get("times") if isinstance(meta_json.get("times"), list) else []
    time_value = str(times[0]) if times else str(weather_info.get("time") or "")
    png = _png_from_meta(meta_json)
    return {
        "index": 0,
        "file": source.name,
        "source_file": source.as_posix(),
        "meta_file": meta_json.get("meta_file"),
        "time": time_value,
        "time_label": weather_info.get("time") or time_value,
        "extent": meta_json.get("bbox") or meta_json.get("extent") or weather_info.get("extent"),
        "png": _as_posix(png),
        "default_png": _as_posix(png),
        "weather_info": weather_info,
    }


def _frames_from_meta(meta_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not meta_json:
        return []
    frames = meta_json.get("frames")
    if isinstance(frames, list) and frames:
        normalized = []
        for index, item in enumerate(frames):
            if not isinstance(item, dict):
                continue
            source = _frame_source(item)
            frame = dict(item)
            frame["index"] = index
            if source:
                frame["file"] = frame.get("file") or source.name
                frame["source_file"] = source.as_posix()
            normalized.append(frame)
        normalized = sorted(normalized, key=lambda item: item.get("time") or item.get("file") or "")
        for index, item in enumerate(normalized):
            item["index"] = index
        return normalized
    frame = _frame_from_meta(meta_json)
    return [frame] if frame else []


def _latest_source_file() -> Path | None:
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if meta_files:
        with meta_files[0].open("r", encoding="utf-8") as file:
            source = _source_from_meta(json.load(file))
            if source:
                return source

    nc_files = sorted(DATA_DIR.glob("*.nc"), key=lambda item: item.stat().st_mtime, reverse=True)
    return nc_files[0] if nc_files else None


def _resolve_source_file(file_name: str | None = None) -> Path:
    if file_name:
        path = DATA_DIR / Path(file_name).name
        if path.exists() and path.suffix.lower() == ".nc":
            return path
        raise ValueError("雷达源文件不存在。")

    source = _latest_source_file()
    if source:
        return source
    raise ValueError("未找到可用雷达 NetCDF 文件。")


def _with_grid_urls(grid_catalog: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not grid_catalog:
        return []

    products = []
    source_name = Path(grid_catalog["source_file"]).name
    for product in grid_catalog.get("products", []):
        levels = []
        for level in product.get("levels", []):
            query = urlencode(
                {
                    "file": source_name,
                    "product": product["key"],
                    "level": level["key"],
                }
            )
            levels.append(
                {
                    **level,
                    "grid_url": f"/api/radar/grid?{query}",
                    "grid": product["grid"],
                    "extent": product["extent"],
                    "missing": product["missing"],
                    "dtype": "float32",
                }
            )
        products.append({**product, "levels": levels})
    return products


def get_grid_data(file_name: str | None, product: str, level: str) -> dict[str, Any]:
    source_path = _resolve_source_file(file_name)
    return radar_adapter.read_grid_values(source_path, product, level)


def get_display_data() -> dict[str, Any]:
    # 前端点击 Radar 类型时调用该函数，读取 Radar 目录下的 meta.json 和 PNG。
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    png_files = sorted(DATA_DIR.glob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True)

    meta_json = None
    if meta_files:
        with meta_files[0].open("r", encoding="utf-8") as file:
            meta_json = json.load(file)

    png_path = _png_from_meta(meta_json) or (png_files[0] if png_files else None)
    weather_info = meta_json.get("weather_info", {}) if meta_json else {}
    frames = _frames_from_meta(meta_json)
    current_frame = frames[0] if frames else None
    frame_png = _frame_png(current_frame)
    if frame_png:
        png_path = frame_png
    extent = None
    if meta_json:
        extent = meta_json.get("extent") or meta_json.get("bbox") or weather_info.get("extent")
    if current_frame and current_frame.get("extent"):
        extent = current_frame.get("extent")

    grid_error = None
    grid_products: list[dict[str, Any]] = []
    source_path = _frame_source(current_frame) or _source_from_meta(meta_json)
    if source_path:
        try:
            grid_products = _with_grid_urls(radar_adapter.build_grid_catalog(source_path))
        except Exception as exc:  # pragma: no cover - surfaced to frontend for diagnostics
            grid_error = str(exc)

    times = [str(frame.get("time")) for frame in frames if frame.get("time")]

    return {
        "business_type": "Radar",
        "meta_file": _as_posix(meta_files[0] if meta_files else None),
        "meta_json": meta_json,
        "weather_info": weather_info,
        "extent": extent,
        "png": _as_posix(png_path),
        "png_data_url": _png_data_url(png_path),
        "png_files": [_as_posix(path) for path in png_files],
        "frames": frames,
        "times": times,
        "frame_count": len(frames),
        "grid_products": grid_products,
        "grid_error": grid_error,
    }
