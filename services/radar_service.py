import json
from pathlib import Path
from typing import Any

from adapters import radar_adapter


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATA_DIR = DATA_ROOT / "Radar"


def _as_posix(path: Path | None) -> str | None:
    return str(path).replace("\\", "/") if path else None


def _load_meta(meta_file: Path) -> dict[str, Any] | None:
    try:
        with meta_file.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _existing_path(value: Any) -> Path | None:
    if isinstance(value, list):
        for item in value:
            path = _existing_path(item)
            if path:
                return path
        return None
    if not value:
        return None

    text = str(value).replace("\\", "/")
    if text.startswith("/data/"):
        candidate = DATA_ROOT / text.removeprefix("/data/")
        return candidate if candidate.exists() else None

    path = Path(text)
    if path.exists():
        return path

    fallback = DATA_DIR / path.name
    if fallback.exists():
        return fallback
    return None


def _public_url(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            url = _public_url(item)
            if url:
                return url
        return None
    if not value:
        return None

    text = str(value).replace("\\", "/")
    if text.startswith(("http://", "https://", "data:", "/data/")):
        return text

    data_index = text.find("/data/")
    if data_index >= 0:
        return text[data_index:]

    path = _existing_path(text)
    if path:
        return radar_adapter.public_data_path(path)
    return text


def _webp_from_item(item: dict[str, Any] | None) -> str | None:
    if not item:
        return None

    for key in ("default_webp", "webp_url", "webp", "image_url"):
        url = _public_url(item.get(key))
        if url:
            return url

    url = _public_url(item.get("webp_files"))
    if url:
        return url

    weather_info = item.get("weather_info")
    if isinstance(weather_info, dict):
        for key in ("webp_url", "webp", "image_url"):
            url = _public_url(weather_info.get(key))
            if url:
                return url
    return None


def _source_from_meta(meta_json: dict[str, Any] | None) -> Path | None:
    if not meta_json:
        return None
    return (
        _existing_path(meta_json.get("source_file"))
        or _existing_path(meta_json.get("source_files"))
        or _existing_path(meta_json.get("file_detail", {}).get("path"))
    )


def _frame_source(frame: dict[str, Any] | None) -> Path | None:
    if not frame:
        return None
    return _existing_path(frame.get("source_file")) or _existing_path(frame.get("file"))


def _frame_from_meta(meta_json: dict[str, Any]) -> dict[str, Any] | None:
    source = _source_from_meta(meta_json)
    webp = _webp_from_item(meta_json)
    if not source and not webp:
        return None

    weather_info = meta_json.get("weather_info", {})
    times = meta_json.get("times") if isinstance(meta_json.get("times"), list) else []
    time_value = str(times[0]) if times else str(weather_info.get("time") or "")
    return {
        "index": 0,
        "file": source.name if source else str(meta_json.get("file") or ""),
        "source_file": source.as_posix() if source else meta_json.get("source_file"),
        "meta_file": meta_json.get("meta_file"),
        "time": time_value,
        "time_label": weather_info.get("time") or time_value,
        "extent": meta_json.get("bbox") or meta_json.get("extent") or weather_info.get("extent"),
        "webp": webp,
        "webp_url": webp,
        "default_webp": webp,
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
            webp = _webp_from_item(item)
            frame = dict(item)
            frame["index"] = index
            if source:
                frame["file"] = frame.get("file") or source.name
                frame["source_file"] = source.as_posix()
            if webp:
                frame["webp"] = webp
                frame["webp_url"] = webp
                frame["default_webp"] = webp
            if source or webp or frame.get("file"):
                normalized.append(frame)
        normalized = sorted(normalized, key=lambda item: item.get("time") or item.get("file") or "")
        for index, item in enumerate(normalized):
            item["index"] = index
        return normalized
    frame = _frame_from_meta(meta_json)
    return [frame] if frame else []


def _meta_has_source(meta_json: dict[str, Any] | None) -> bool:
    if _source_from_meta(meta_json):
        return True
    frames = meta_json.get("frames") if isinstance(meta_json, dict) else None
    if isinstance(frames, list):
        return any(_frame_source(item) for item in frames if isinstance(item, dict))
    return False


def _select_meta_file() -> tuple[Path, dict[str, Any]] | None:
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    display_fallback: tuple[Path, dict[str, Any]] | None = None
    newest: tuple[Path, dict[str, Any]] | None = None

    for meta_file in meta_files:
        meta_json = _load_meta(meta_file)
        if not meta_json:
            continue
        newest = newest or (meta_file, meta_json)
        if _meta_has_source(meta_json):
            return meta_file, meta_json
        if display_fallback is None and (_webp_from_item(meta_json) or _frames_from_meta(meta_json)):
            display_fallback = (meta_file, meta_json)

    return display_fallback or newest


def _latest_source_file() -> Path | None:
    selected = _select_meta_file()
    if selected:
        _, meta_json = selected
        source = _source_from_meta(meta_json)
        if source:
            return source
        for frame in _frames_from_meta(meta_json):
            source = _frame_source(frame)
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


def _clamp_index(index: int, size: int) -> int:
    if size <= 0:
        return 0
    return min(max(int(index or 0), 0), size - 1)


def _active_level(products: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not products:
        return None
    levels = products[0].get("levels") or []
    return levels[0] if levels else None


def get_display_data(time_index: int = 0) -> dict[str, Any]:
    meta_file: Path | None = None
    meta_json: dict[str, Any] | None = None
    display_error = None

    selected = _select_meta_file()
    if selected:
        meta_file, meta_json = selected
    else:
        source_path = _latest_source_file()
        if source_path:
            try:
                meta_json = radar_adapter.process_file(source_path, data_type="Radar")
                candidate = Path(str(meta_json.get("meta_file") or ""))
                meta_file = candidate if candidate.exists() else None
            except Exception as exc:  # pragma: no cover - surfaced to frontend for diagnostics
                display_error = str(exc)

    weather_info = meta_json.get("weather_info", {}) if meta_json else {}
    frames = _frames_from_meta(meta_json)
    current_frame = frames[_clamp_index(time_index, len(frames))] if frames else None
    source_path = _frame_source(current_frame) or _source_from_meta(meta_json)

    products: list[dict[str, Any]] = []
    if source_path:
        try:
            catalog = radar_adapter.build_webp_catalog(source_path)
            products = catalog.get("products", [])
        except Exception as exc:  # pragma: no cover - surfaced to frontend for diagnostics
            display_error = str(exc) if display_error is None else f"{display_error}; {exc}"

    level = _active_level(products)
    webp_url = _public_url(level.get("webp_url") if level else None) or _webp_from_item(current_frame) or _webp_from_item(meta_json)
    extent = (
        level.get("extent") if level else None
    ) or (
        current_frame.get("extent") if current_frame else None
    ) or (
        meta_json.get("extent") or meta_json.get("bbox") if meta_json else None
    ) or weather_info.get("extent")
    times = [str(frame.get("time")) for frame in frames if frame.get("time")]
    if not times and isinstance(meta_json, dict) and isinstance(meta_json.get("times"), list):
        times = [str(item) for item in meta_json["times"]]

    return {
        "business_type": "Radar",
        "meta_file": _as_posix(meta_file),
        "meta_json": meta_json,
        "weather_info": weather_info,
        "extent": extent,
        "webp": webp_url,
        "webp_url": webp_url,
        "webp_files": [webp_url] if webp_url else [],
        "products": products,
        "frames": frames,
        "frame": current_frame,
        "times": times,
        "frame_count": len(frames),
        "display_error": display_error,
    }
