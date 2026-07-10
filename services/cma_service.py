import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from PIL import Image

from adapters import cma_adapter


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "CMA"
CMA_SOURCE_DIRS = (DATA_DIR,)
NODATA = -999999.0
DEFAULT_RECENT_FRAMES = 5


def get_display_data(
    variable: str | None = None,
    level_index: int = 0,
    time_index: int = 0,
    meta_file: str | None = None,
    resolution: str | None = None,
) -> dict[str, Any]:
    resolution_key = cma_adapter.normalize_resolution_key(resolution)
    meta_path = _selected_meta_file(meta_file)
    meta_json = _load_meta_file(meta_path) if meta_path else None
    if meta_json:
        meta_json = _with_recent_series_fallback(meta_json)
    variables = _display_variables(meta_json)
    frames = _frames_from_meta(meta_json) or (_series_frames(_source_file(meta_json)) if meta_json else [])
    grid = None
    weather_info = meta_json.get("weather_info", {}) if isinstance(meta_json, dict) else {}
    warnings: list[str] = []

    if meta_json:
        try:
            frame = _active_frame(frames, time_index)
            variable_item = _variable_item(variables, variable or _primary_variable(meta_json)) or {}
            variable_name = variable_item.get("name") or variable or _primary_variable(meta_json)
            source_file = _frame_source(frame or {}, meta_json) or _source_file(meta_json)
            resolution_options = _resolution_options_from_meta(meta_json)
            meta_json = _merge_resolution_options(meta_json, resolution_options)
            grid = _cached_display_grid(meta_json, source_file, variable_item, variable_name, level_index, resolution_key, resolution_options)
            if grid is None:
                grid = get_grid_data(
                    variable=variable,
                    level_index=level_index,
                    file_name=frame.get("file") if frame else None,
                    meta=meta_json,
                )
                source_file = _resolve_source_file(grid["file"])
                resolution_options = _resolution_options(meta_json, grid)
                meta_json = _merge_resolution_options(meta_json, resolution_options)
                resolution_result = cma_adapter.resample_grid(
                    _grid_values_array(grid),
                    grid["extent"],
                    resolution_key,
                    grid["variable"],
                )
                warnings.extend(resolution_result.get("warnings", []))
                _apply_resolution_result(grid, resolution_result, resolution_options)
                webp_path = cma_adapter.write_grid_webp(
                    source_file,
                    resolution_result["data"],
                    grid["variable"],
                    grid.get("level_index", 0),
                    resolution=resolution_key,
                    stats=variable_item.get("stats"),
                )
                grid["webp_url"] = cma_adapter.public_data_path(webp_path)
            stats = variable_item.get("stats") if isinstance(variable_item, dict) else None
            if isinstance(stats, dict):
                grid["scale_min"] = stats.get("min")
                grid["scale_max"] = stats.get("max")
            warnings.extend(_quality_warnings(grid, meta_json))
            warnings = _unique_warnings(warnings)
            grid["warnings"] = warnings
            grid["values"] = []
            weather_info = _build_display_weather_info(meta_json, grid, variables, frame, warnings)
            meta_json = _merge_display_meta(meta_json, weather_info)
        except ValueError:
            grid = None

    return {
        "business_type": "CMA",
        "meta_file": str(meta_path).replace("\\", "/") if meta_path else None,
        "meta_json": meta_json,
        "webp": grid.get("webp_url") if grid else (meta_json.get("default_webp") if isinstance(meta_json, dict) else None),
        "webp_files": meta_json.get("webp_files", []) if isinstance(meta_json, dict) else [],
        "variables": variables,
        "grid": grid,
        "frames": frames,
        "times": [frame["time"] for frame in frames],
        "frame_count": len(frames),
        "weather_info": weather_info,
        "resolution": resolution_key,
        "resolution_options": meta_json.get("resolution_options", []) if isinstance(meta_json, dict) else [],
        "warnings": warnings,
    }


def get_grid_data(
    variable: str | None = None,
    level_index: int = 0,
    file_name: str | None = None,
    meta: dict[str, Any] | None = None,
    meta_file: str | None = None,
) -> dict[str, Any]:
    meta = meta or _latest_meta(meta_file)
    source_file = _resolve_source_file(file_name) if file_name else _source_file(meta)
    suffix = source_file.suffix.lower()
    file_format = "NC" if suffix == ".nc" else "GRIB" if suffix in {".grib", ".grib2"} else str(meta.get("file_format") or suffix.lstrip(".")).upper()
    variable_name = variable or _primary_variable(meta)

    if file_format == "NC" or suffix == ".nc":
        payload = _read_nc_grid(source_file, meta, variable_name, level_index)
    elif suffix in {".grib", ".grib2"}:
        payload = _read_grib_grid(source_file, meta, variable_name)
    else:
        raise ValueError(f"Unsupported CMA grid file: {source_file.name}")

    return {
        "business_type": "CMA",
        "dataset_id": meta.get("dataset_id"),
        "file": source_file.name,
        "variable": payload["variable"],
        "label": payload["label"],
        "unit": _clean_unit(payload["unit"]),
        "level_index": payload.get("level_index", 0),
        "width": payload["width"],
        "height": payload["height"],
        "extent": payload["extent"],
        "min": payload["min"],
        "max": payload["max"],
        "mean": payload["mean"],
        "nodata": NODATA,
        "values": payload["values"],
        "variables": _display_variables(meta),
        "meta": _grid_meta(meta, payload),
    }


def _grid_values_array(grid: dict[str, Any]) -> np.ndarray:
    values = np.array(grid.get("values") or [], dtype="float32")
    width = int(grid.get("width") or 0)
    height = int(grid.get("height") or 0)
    if width <= 0 or height <= 0 or values.size != width * height:
        raise ValueError("CMA grid values do not match grid dimensions.")
    data = values.reshape((height, width))
    nodata = float(grid.get("nodata", NODATA))
    data[data == nodata] = np.nan
    return data


def _resolution_options_from_meta(meta: dict[str, Any]) -> list[dict[str, Any]]:
    existing = meta.get("resolution_options") or meta.get("extra", {}).get("cma", {}).get("resolutions")
    if isinstance(existing, list) and existing:
        return existing

    spatial = meta.get("spatial") if isinstance(meta.get("spatial"), dict) else {}
    extent = meta.get("extent") or meta.get("bbox")
    width = int(spatial.get("nx") or 0)
    height = int(spatial.get("ny") or 0)
    native_resolution = {
        "lon": spatial.get("resolution_lon"),
        "lat": spatial.get("resolution_lat"),
    }
    return cma_adapter.build_resolution_options(
        extent,
        width,
        height,
        native_resolution,
        meta.get("weather_info", {}).get("coverage"),
    )


def _cached_display_grid(
    meta: dict[str, Any],
    source_file: Path,
    variable_item: dict[str, Any],
    variable_name: str,
    level_index: int,
    resolution_key: str,
    resolution_options: list[dict[str, Any]],
) -> dict[str, Any] | None:
    webp_path = cma_adapter.grid_webp_path(source_file, variable_name, level_index, resolution_key)
    if not webp_path.exists():
        return None

    if _is_zero_to_360_extent(meta.get("extent") or meta.get("bbox")):
        return None

    extent = _display_extent(meta.get("extent") or meta.get("bbox"))
    if not extent:
        return None

    spatial = meta.get("spatial") if isinstance(meta.get("spatial"), dict) else {}
    option = _selected_resolution_option(resolution_options, resolution_key)
    width = int(option.get("width") or spatial.get("nx") or 0)
    height = int(option.get("height") or spatial.get("ny") or 0)
    if width <= 0 or height <= 0:
        with Image.open(webp_path) as image:
            width, height = image.size

    stats = variable_item.get("stats") if isinstance(variable_item, dict) else {}
    min_value = _finite_float(stats.get("min"), 0.0)
    max_value = _finite_float(stats.get("max"), 1.0)
    mean_value = _finite_float(stats.get("mean"), min_value)
    webp_url = cma_adapter.public_data_path(webp_path)
    unit = _clean_unit(variable_item.get("unit") or variable_item.get("display_unit") or "")
    label = _strip_label_unit(variable_item.get("label") or variable_name, unit)
    return {
        "business_type": "CMA",
        "dataset_id": meta.get("dataset_id"),
        "file": source_file.name,
        "variable": variable_name,
        "label": label,
        "unit": unit,
        "level_index": level_index,
        "width": width,
        "height": height,
        "extent": extent,
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "nodata": NODATA,
        "values": [],
        "variables": _display_variables(meta),
        "resolution_key": resolution_key,
        "resolution": option.get("label") or resolution_key,
        "is_native_resolution": bool(option.get("is_native")),
        "playable": bool(option.get("playable")),
        "resampling": option.get("resampling"),
        "target_resolution_km": option.get("target_resolution_km"),
        "resolution_options": resolution_options,
        "valid_count": int(width * height),
        "total_count": int(width * height),
        "valid_ratio": 1.0,
        "webp_url": webp_url,
        "cache_hit": True,
    }


def _finite_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if np.isfinite(number) else fallback


def _display_extent(extent: Any) -> list[float] | None:
    if not isinstance(extent, list) or len(extent) != 4:
        return None
    try:
        west, south, east, north = [float(item) for item in extent]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite([west, south, east, north])) or west >= east or south >= north:
        return None
    if _is_zero_to_360_extent([west, south, east, north]):
        return [-180.0, max(-90.0, south), 180.0, min(90.0, north)]
    return [
        max(-180.0, west),
        max(-90.0, south),
        min(180.0, east),
        min(90.0, north),
    ]


def _is_zero_to_360_extent(extent: Any) -> bool:
    if not isinstance(extent, list) or len(extent) != 4:
        return False
    try:
        west, _, east, _ = [float(item) for item in extent]
    except (TypeError, ValueError):
        return False
    return west >= 0 and east > 180


def _resolution_options(meta: dict[str, Any], grid: dict[str, Any]) -> list[dict[str, Any]]:
    existing = meta.get("resolution_options") or meta.get("extra", {}).get("cma", {}).get("resolutions")
    if isinstance(existing, list) and existing:
        return existing

    spatial = meta.get("spatial") if isinstance(meta.get("spatial"), dict) else {}
    native_resolution = {
        "lon": spatial.get("resolution_lon"),
        "lat": spatial.get("resolution_lat"),
    }
    return cma_adapter.build_resolution_options(
        grid.get("extent") or meta.get("bbox") or meta.get("extent"),
        int(grid.get("width") or spatial.get("nx") or 0),
        int(grid.get("height") or spatial.get("ny") or 0),
        native_resolution,
        meta.get("weather_info", {}).get("coverage"),
    )


def _merge_resolution_options(meta: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(meta)
    extra = dict(merged.get("extra") or {})
    cma = dict(extra.get("cma") or {})
    cma["resolutions"] = options
    extra["cma"] = cma
    merged["extra"] = extra
    merged["resolution_options"] = options
    return merged


def _selected_resolution_option(options: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for item in options:
        if item.get("key") == key:
            return item
    return options[0] if options else {"key": key, "label": key, "playable": key == "native"}


def _apply_resolution_result(
    grid: dict[str, Any],
    resolution_result: dict[str, Any],
    options: list[dict[str, Any]],
) -> None:
    data = np.array(resolution_result["data"], dtype="float32")
    finite = data[np.isfinite(data)]
    option = _selected_resolution_option(options, resolution_result["resolution_key"])
    grid.update(
        {
            "width": int(data.shape[1]),
            "height": int(data.shape[0]),
            "resolution_key": resolution_result["resolution_key"],
            "resolution": option.get("label") or resolution_result["resolution_key"],
            "is_native_resolution": bool(resolution_result.get("is_native_resolution")),
            "playable": bool(resolution_result.get("playable")),
            "resampling": resolution_result.get("resampling"),
            "target_resolution_km": resolution_result.get("target_resolution_km"),
            "resolution_options": options,
            "valid_count": int(finite.size),
            "total_count": int(data.size),
            "valid_ratio": round(float(finite.size / data.size), 6) if data.size else 0.0,
        }
    )
    if finite.size:
        grid["min"] = round(float(np.nanmin(finite)), 6)
        grid["max"] = round(float(np.nanmax(finite)), 6)
        grid["mean"] = round(float(np.nanmean(finite)), 6)
    else:
        grid["min"] = 0.0
        grid["max"] = 1.0
        grid["mean"] = 0.0
    grid["values"] = np.where(np.isfinite(data), data, NODATA).astype("float32").reshape(-1).round(6).tolist()


def _quality_warnings(grid: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    width = int(grid.get("width") or 0)
    height = int(grid.get("height") or 0)
    if width <= 0 or height <= 0:
        warnings.append("Invalid grid dimensions.")

    extent = grid.get("extent")
    if not _valid_extent(extent):
        warnings.append("Invalid map extent.")
    elif meta.get("bbox") and _valid_extent(meta.get("bbox")):
        if any(abs(float(a) - float(b)) > 1e-4 for a, b in zip(extent, meta["bbox"])):
            warnings.append("Grid extent differs from meta bbox.")

    valid_count = int(grid.get("valid_count") or 0)
    total_count = int(grid.get("total_count") or 0)
    if total_count and valid_count == 0:
        warnings.append("Grid contains no valid values.")
    elif total_count and valid_count / total_count < 0.01:
        warnings.append("Grid valid value ratio is below 1%.")

    for key in ("min", "max", "mean"):
        try:
            value = float(grid.get(key))
        except (TypeError, ValueError):
            warnings.append(f"Grid {key} is not numeric.")
            continue
        if not np.isfinite(value):
            warnings.append(f"Grid {key} is not finite.")
    if float(grid.get("max", 0)) < float(grid.get("min", 0)):
        warnings.append("Grid max is smaller than min.")
    return warnings


def _valid_extent(extent: Any) -> bool:
    if not isinstance(extent, list) or len(extent) != 4:
        return False
    try:
        west, south, east, north = [float(item) for item in extent]
    except (TypeError, ValueError):
        return False
    return (
        all(np.isfinite([west, south, east, north]))
        and -180 <= west < east <= 180
        and -90 <= south < north <= 90
    )


def _unique_warnings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _with_recent_series_fallback(meta: dict[str, Any]) -> dict[str, Any]:
    frames = _frames_from_meta(meta)
    if len(frames) > 1:
        return meta
    return _synthetic_series_from_recent_metas(meta) or meta


def _synthetic_series_from_recent_metas(base_meta: dict[str, Any], limit: int = DEFAULT_RECENT_FRAMES) -> dict[str, Any] | None:
    group_key = _series_group_key(base_meta)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_times: set[str] = set()

    for meta_file in _meta_files():
        meta = _load_meta_file(meta_file)
        if _series_group_key(meta) != group_key:
            continue
        frame = _single_frame_from_meta(meta)
        if not frame:
            continue
        time_key = str(frame.get("time") or frame.get("time_label") or frame.get("file") or "")
        if not time_key or time_key in seen_times:
            continue
        seen_times.add(time_key)
        selected.append((meta, frame))
        if len(selected) >= limit:
            break

    if len(selected) < 2:
        return None

    selected.sort(key=lambda item: item[1].get("time") or item[1].get("file") or "")
    if not _is_continuous_frame_series([frame for _, frame in selected]):
        return None
    frames = []
    for index, (_, frame) in enumerate(selected):
        item = dict(frame)
        item["index"] = index
        frames.append(item)

    times = [str(frame.get("time")) for frame in frames if frame.get("time")]
    extent = _merge_extents([frame.get("extent") for frame in frames]) or base_meta.get("extent") or base_meta.get("bbox")
    weather_info = dict(base_meta.get("weather_info") or {})
    if times:
        weather_info["time"] = f"{_format_time(times[0])} - {_format_time(times[-1])}"
        weather_info["times"] = times
    weather_info["step_count"] = len(frames)
    weather_info["steps"] = str(len(frames))
    weather_info["status"] = "loaded_recent_series"

    merged = dict(base_meta)
    extra = dict(merged.get("extra") or {})
    cma = dict(extra.get("cma") or {})
    cma["status"] = "loaded_recent_series"
    extra["cma"] = cma
    merged.update(
        {
            "file": f"{len(frames)} CMA files",
            "source_file": frames[0].get("source_file"),
            "source_files": [frame.get("source_file") for frame in frames if frame.get("source_file")],
            "webp_files": [frame.get("webp_url") for frame in frames if frame.get("webp_url")],
            "default_webp": frames[0].get("webp_url") or base_meta.get("default_webp"),
            "times": times,
            "extent": extent,
            "bbox": extent,
            "frames": frames,
            "weather_info": weather_info,
            "extra": extra,
        }
    )
    return merged


def _is_continuous_frame_series(frames: list[dict[str, Any]], max_gap_hours: float = 24.0) -> bool:
    parsed_times = [_parse_frame_datetime(frame) for frame in frames]
    if any(item is None for item in parsed_times):
        return True
    for previous, current in zip(parsed_times, parsed_times[1:]):
        gap_hours = (current - previous).total_seconds() / 3600
        if gap_hours <= 0 or gap_hours > max_gap_hours:
            return False
    return True


def _parse_frame_datetime(frame: dict[str, Any]) -> datetime | None:
    text = str(frame.get("time") or frame.get("time_label") or "").strip()
    if not text:
        return None
    if len(text) == 10 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d%H")
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _single_frame_from_meta(meta: dict[str, Any]) -> dict[str, Any] | None:
    frames = _frames_from_meta(meta)
    if len(frames) > 1:
        return None
    frame = dict(frames[0]) if frames else {}
    source = _frame_source(frame, meta)
    if not source:
        return None
    time_value = str(frame.get("time") or meta.get("time") or _parse_time(source) or "")
    extent = frame.get("extent") or meta.get("extent") or meta.get("bbox")
    return {
        **frame,
        "file": frame.get("file") or source.name,
        "source_file": source.as_posix(),
        "time": time_value,
        "time_label": frame.get("time_label") or _format_time(time_value),
        "extent": extent,
        "webp": frame.get("webp") or meta.get("default_webp"),
        "webp_url": frame.get("webp_url") or frame.get("webp") or meta.get("default_webp"),
    }


def _frame_source(frame: dict[str, Any], meta: dict[str, Any]) -> Path | None:
    values = (frame.get("source_file"), frame.get("file"), meta.get("source_file"), meta.get("file"))
    for value in values:
        if isinstance(value, list):
            value = value[0] if value else ""
        if not value:
            continue
        path = Path(str(value))
        if path.exists():
            return path
        name = path.name
        match = next((item for item in _source_files() if item.name == name), None)
        if match:
            return match
    try:
        return _source_file(meta)
    except ValueError:
        return None


def _series_group_key(meta: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(meta.get("file_format") or ""),
        _primary_variable(meta),
        _extent_key(meta.get("extent") or meta.get("bbox")),
    )


def _extent_key(extent: Any) -> str:
    if not _valid_extent(extent):
        return ""
    return ",".join(f"{float(item):.3f}" for item in extent)


def _merge_extents(extents: list[Any]) -> list[float] | None:
    values: list[list[float]] = []
    for extent in extents:
        if not _valid_extent(extent):
            continue
        values.append([float(item) for item in extent])
    if not values:
        return None
    return [
        round(min(item[0] for item in values), 6),
        round(min(item[1] for item in values), 6),
        round(max(item[2] for item in values), 6),
        round(max(item[3] for item in values), 6),
    ]


def _latest_meta(meta_file: str | None = None) -> dict[str, Any]:
    meta_path = _selected_meta_file(meta_file)
    if meta_path is None:
        raise ValueError("No CMA meta.json found. Parse a CMA file first.")
    return _load_meta_file(meta_path)


def _selected_meta_file(meta_file: str | None = None) -> Path | None:
    if meta_file:
        raw = str(meta_file).replace("\\", "/").strip()
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            relative = Path(raw.lstrip("/"))
            options = [
                (Path.cwd() / relative).resolve(),
                (DATA_DIR / relative).resolve(),
                (DATA_DIR.parent / relative).resolve(),
            ]
            candidate = next((item for item in options if item.exists()), options[0])
        else:
            candidate = candidate.resolve()

        if not candidate.exists():
            raise ValueError("CMA meta.json not found.")
        if not candidate.name.endswith(".meta.json"):
            raise ValueError("Invalid CMA meta file.")
        if not _is_under_cma_dir(candidate):
            raise ValueError("CMA meta file is outside the data directory.")
        if not _is_renderable_meta_file(candidate):
            raise ValueError("CMA meta file is not renderable.")
        return candidate

    _ensure_latest_meta()
    meta_files = _meta_files()
    if not meta_files:
        return None

    newest = meta_files[0]
    newest_meta = _load_meta_file(newest)
    if len(_frames_from_meta(newest_meta)) > 1:
        return newest

    newest_group = _series_group_key(newest_meta)
    newest_mtime = newest.stat().st_mtime
    for candidate in meta_files[1:]:
        if newest_mtime - candidate.stat().st_mtime > 600:
            break
        candidate_meta = _load_meta_file(candidate)
        if _series_group_key(candidate_meta) == newest_group and len(_frames_from_meta(candidate_meta)) > 1:
            return candidate

    return newest


def _is_under_cma_dir(path: Path) -> bool:
    resolved = path.resolve()
    for directory in CMA_SOURCE_DIRS:
        try:
            resolved.relative_to(directory.resolve())
            return True
        except ValueError:
            continue
    return False


def _source_file(meta: dict[str, Any]) -> Path:
    by_name = DATA_DIR / str(meta.get("file", ""))
    if by_name.exists():
        return by_name

    source_value = meta.get("source_file", "")
    if isinstance(source_value, list):
        source_value = source_value[0] if source_value else ""
    source = Path(str(source_value))
    if source.exists():
        return source

    candidates = sorted(_source_files(), key=_source_sort_key)
    if not candidates:
        raise ValueError("No CMA source data file found.")
    return candidates[0]


def _resolve_source_file(file_name: str | None) -> Path:
    if file_name:
        safe_name = Path(file_name).name
        matches = [path for path in _source_files() if path.name == safe_name]
        if matches:
            return sorted(matches, key=lambda item: item.stat().st_mtime, reverse=True)[0]
        raise ValueError("CMA source file not found.")
    return _source_file(_latest_meta())


def _meta_files() -> list[Path]:
    files = sorted(
        {
            path
            for directory in CMA_SOURCE_DIRS
            if directory.exists()
            for path in directory.rglob("*.meta.json")
        },
        key=_source_sort_key,
    )
    fallback = DATA_DIR / "meta.json"
    if fallback.exists() and fallback not in files:
        files.append(fallback)
    return [path for path in files if _is_renderable_meta_file(path)]


def _load_meta_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def _is_renderable_meta_file(path: Path) -> bool:
    try:
        return bool(_primary_variable(_load_meta_file(path)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in CMA_SOURCE_DIRS:
        if not directory.exists():
            continue
        for pattern in ("*.nc", "*.grib", "*.grib2"):
            files.extend(path for path in directory.rglob(pattern) if path.is_file())
    return files


def _source_sort_key(path: Path) -> tuple[int, float]:
    resolved = path.resolve()
    for index, directory in enumerate(CMA_SOURCE_DIRS):
        try:
            resolved.relative_to(directory.resolve())
            return (index, -path.stat().st_mtime)
        except ValueError:
            continue
    return (len(CMA_SOURCE_DIRS), -path.stat().st_mtime)


def _series_frames(source_file: Path) -> list[dict[str, Any]]:
    frames = []
    for index, path in enumerate(_series_files(source_file)):
        time_value = _parse_time(path)
        frames.append(
            {
                "index": index,
                "file": path.name,
                "source_file": path.as_posix(),
                "time": time_value,
                "time_label": _format_time(time_value),
                "extent": None,
            }
        )
    return frames


def _frames_from_meta(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    frames = meta.get("frames") if isinstance(meta, dict) else None
    if not isinstance(frames, list) or not frames:
        return []

    normalized = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        source_value = str(frame.get("source_file") or "")
        source = Path(source_value) if source_value else None
        normalized.append(
            {
                **frame,
                "index": index,
                "file": frame.get("file") or (source.name if source else ""),
                "source_file": source.as_posix() if source else frame.get("source_file"),
                "time": str(frame.get("time") or ""),
                "time_label": frame.get("time_label") or _format_time(str(frame.get("time") or "")),
            }
        )
    return sorted(normalized, key=lambda item: item.get("time") or item.get("file") or "")


def _series_files(source_file: Path) -> list[Path]:
    suffix = source_file.suffix.lower()
    patterns = ("*.grib", "*.grib2") if suffix in {".grib", ".grib2"} else (f"*{suffix}",)
    files = [
        path
        for pattern in patterns
        for path in source_file.parent.glob(pattern)
        if path.is_file() and path.suffix.lower() in ({".grib", ".grib2"} if suffix in {".grib", ".grib2"} else {suffix})
    ]
    return sorted(files, key=lambda item: _parse_time(item) or item.name) or [source_file]


def _parse_time(path: Path) -> str:
    import re

    match = re.search(r"_(\d{10})_", path.name)
    return match.group(1) if match else ""


def _format_time(value: str) -> str:
    if len(value) != 10:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]} {value[8:10]}:00"


def _ensure_latest_meta() -> None:
    sources = sorted(_source_files(), key=_source_sort_key)
    if not sources:
        return

    latest_source = sources[0]
    expected_meta = latest_source.with_name(f"{latest_source.name}.meta.json")
    if (
        expected_meta.exists()
        and expected_meta.stat().st_mtime >= latest_source.stat().st_mtime
        and _is_renderable_meta_file(expected_meta)
    ):
        return

    cma_adapter.process_file(str(latest_source), data_type="CMA")


def _primary_variable(meta: dict[str, Any]) -> str:
    cma = meta.get("extra", {}).get("cma", {})
    variables = meta.get("variables", [])
    first_variable = None
    if variables and isinstance(variables[0], dict):
        first_variable = variables[0].get("name")
    elif variables:
        first_variable = variables[0]
    primary = meta.get("default_variable") or cma.get("primary_variable") or first_variable
    if not primary:
        raise ValueError("No CMA variable found in meta.json.")
    return str(primary)


def _display_variables(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not meta:
        return []

    variable_docs = cma_adapter.load_information_map()
    top_variables = meta.get("variables", [])
    if top_variables and isinstance(top_variables[0], dict):
        return [_normalize_variable_item(item, variable_docs) for item in top_variables]

    return [{"name": name, "label": name, "unit": ""} for name in top_variables]


def _normalize_variable_item(item: dict[str, Any], variable_docs: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    variable_docs = variable_docs or {}
    name = item.get("name")
    doc = variable_docs.get(name, {})
    label = doc.get("label") or item.get("long_name") or item.get("label") or name
    return {
        "name": name,
        "label": label,
        "unit": item.get("display_unit") or item.get("unit", ""),
        "dims": item.get("dims", []),
        "shape": item.get("shape", []),
        "stats": item.get("stats"),
        "name_cn": doc.get("desc_zh") or item.get("name_cn"),
        "description": doc.get("desc_zh") or item.get("description"),
        "description_en": doc.get("desc_en") or item.get("description_en"),
        "webp": item.get("webp"),
    }


def _is_grid_variable(item: dict[str, Any]) -> bool:
    dims = item.get("dims") or []
    shape = item.get("shape") or []
    return bool(item.get("band")) or dims[-2:] == ["lat", "lon"] or len(shape) in {2, 3}


def _read_nc_grid(source_file: Path, meta: dict[str, Any], variable: str, level_index: int) -> dict[str, Any]:
    with h5py.File(io.BytesIO(source_file.read_bytes()), "r") as dataset:
        if variable not in dataset:
            variable = _first_available_nc_variable(dataset)
        item = dataset[variable]
        attrs = {key: _decode_attr(value) for key, value in item.attrs.items()}
        raw = item[:]
        if raw.ndim == 3:
            safe_level = min(max(level_index, 0), raw.shape[0] - 1)
            raw = raw[safe_level, :, :]
        elif raw.ndim == 2:
            safe_level = 0
        else:
            raise ValueError(f"CMA variable {variable} is not a 2D grid.")
        data = _clean_grid(raw, attrs.get("_FillValue") or attrs.get("missing_value"))
        data = _orient_nc_grid(dataset, data)
        extent = _nc_extent(dataset, meta)

    return _grid_payload(
        variable=variable,
        label=str(attrs.get("long_name") or variable),
        unit=_clean_unit(attrs.get("units") or ""),
        data=data,
        extent=extent,
        level_index=safe_level,
    )


def _first_available_nc_variable(dataset: h5py.File) -> str:
    for name, item in dataset.items():
        if isinstance(item, h5py.Dataset) and name not in {"lat", "lon"} and len(item.shape) in {2, 3}:
            return name
    raise ValueError("No renderable CMA grid variable found.")


def _read_grib_grid(source_file: Path, meta: dict[str, Any], variable: str) -> dict[str, Any]:
    with rasterio.open(source_file) as dataset:
        band_index = 1
        tags = dataset.tags(1)
        for band in range(1, dataset.count + 1):
            band_tags = dataset.tags(band)
            if band_tags.get("GRIB_ELEMENT") == variable:
                band_index = band
                tags = band_tags
                break
        data = _clean_grid(dataset.read(band_index), dataset.nodata)
        extent = [float(dataset.bounds.left), float(dataset.bounds.bottom), float(dataset.bounds.right), float(dataset.bounds.top)]
    return _grid_payload(
        variable=tags.get("GRIB_ELEMENT", variable),
        label=tags.get("GRIB_COMMENT") or tags.get("GRIB_ELEMENT") or variable,
        unit=_clean_unit(tags.get("GRIB_UNIT", "")),
        data=data,
        extent=extent or meta.get("extent"),
        level_index=0,
    )


def _nc_extent(dataset: h5py.File, meta: dict[str, Any]) -> list[float]:
    if "lon" in dataset and "lat" in dataset:
        lon = np.array(dataset["lon"][:], dtype="float64")
        lat = np.array(dataset["lat"][:], dtype="float64")
        west, east = _coord_edges(lon)
        south, north = _coord_edges(lat)
        if _is_zero_to_360_lon(lon):
            return [-180.0, south, 180.0, north]
        return [west, south, east, north]
    return list(meta.get("extent") or meta.get("bbox") or [73, 15, 135, 55])


def _coord_edges(values: np.ndarray) -> tuple[float, float]:
    flat = np.array(values, dtype="float64").reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 0.0

    unique = np.unique(flat)
    if unique.size == 1:
        center = float(unique[0])
        return center - 0.5, center + 0.5

    ordered = unique if unique[0] <= unique[-1] else unique[::-1]
    deltas = np.diff(ordered)
    step_start = float(deltas[0])
    step_end = float(deltas[-1])
    return float(ordered[0] - step_start / 2), float(ordered[-1] + step_end / 2)


def _orient_nc_grid(dataset: h5py.File, data: np.ndarray) -> np.ndarray:
    oriented = data
    if "lat" in dataset:
        lat = np.array(dataset["lat"][:], dtype="float64")
        if lat.ndim == 1 and lat.size > 1 and lat[0] < lat[-1]:
            oriented = np.flipud(oriented)
    if "lon" in dataset:
        lon = np.array(dataset["lon"][:], dtype="float64")
        if lon.ndim == 1 and lon.size > 1 and lon[0] > lon[-1]:
            oriented = np.fliplr(oriented)
        elif _is_zero_to_360_lon(lon):
            split = int(np.searchsorted(lon, 180.0))
            oriented = np.concatenate([oriented[:, split:], oriented[:, :split]], axis=1)
    return oriented


def _is_zero_to_360_lon(lon: np.ndarray) -> bool:
    values = np.array(lon, dtype="float64").reshape(-1)
    values = values[np.isfinite(values)]
    return bool(values.size and np.nanmin(values) >= 0 and np.nanmax(values) > 180)


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _decode_attr(value.reshape(-1)[0])
    return value


def _clean_unit(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text.startswith("[") and text.endswith("]"):
        return text[1:-1].strip()
    return text


def _clean_grid(array: np.ndarray, missing: Any) -> np.ndarray:
    data = np.array(array, dtype="float32")
    try:
        missing_value = float(np.array(missing).reshape(-1)[0])
        data[data == missing_value] = np.nan
    except Exception:
        pass
    data[np.isinf(data)] = np.nan
    return data


def _grid_payload(variable: str, label: str, unit: str, data: np.ndarray, extent: list[float], level_index: int) -> dict[str, Any]:
    finite = data[np.isfinite(data)]
    min_value = float(np.nanmin(finite)) if finite.size else 0.0
    max_value = float(np.nanmax(finite)) if finite.size else 1.0
    mean_value = float(np.nanmean(finite)) if finite.size else 0.0
    values = np.where(np.isfinite(data), data, NODATA).astype("float32")
    return {
        "variable": variable,
        "label": label,
        "unit": unit,
        "level_index": level_index,
        "width": int(values.shape[1]),
        "height": int(values.shape[0]),
        "extent": [float(item) for item in extent],
        "min": round(min_value, 6),
        "max": round(max_value, 6),
        "mean": round(mean_value, 6),
        "values": values.reshape(-1).round(6).tolist(),
    }


def _grid_meta(meta: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    variable_item = _variable_item(_display_variables(meta), payload["variable"])
    label = _strip_label_unit(variable_item.get("label") if variable_item else payload["label"], _clean_unit(payload["unit"]))
    element = payload["variable"] if not label or label == payload["variable"] else f"{payload['variable']} - {label}"
    weather_info = meta.get("weather_info", {})
    return {
        **{key: value for key, value in meta.items() if key in {"file", "time", "range", "grid", "missing", "vars", "steps"}},
        "element": element,
        "variable_key": payload["variable"],
        "element_desc_zh": variable_item.get("description") if variable_item else weather_info.get("element_desc_zh"),
        "element_desc_en": variable_item.get("description_en") if variable_item else weather_info.get("element_desc_en"),
        "unit": _clean_unit(payload["unit"]),
        "extent": payload["extent"],
        "grid": f"{payload['width']} x {payload['height']}",
        "min": payload["min"],
        "max": payload["max"],
        "mean": payload["mean"],
    }


def _variable_item(variables: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((item for item in variables if item.get("name") == name), None)


def _strip_label_unit(label: Any, unit: str) -> str:
    text = str(label or "").strip()
    if not text or not unit:
        return text
    compact_unit = unit.replace(" ", "").lower()
    import re
    match = re.search(r"\[(.*?)\]\s*$", text)
    if not match:
        return text
    bracket_unit = match.group(1).replace(" ", "").lower()
    if bracket_unit != compact_unit:
        return text
    return text[: match.start()].strip()


def _active_frame(frames: list[dict[str, Any]], time_index: int) -> dict[str, Any] | None:
    if not frames:
        return None
    index = min(max(int(time_index or 0), 0), len(frames) - 1)
    return frames[index]


def _build_display_weather_info(
    meta: dict[str, Any],
    grid: dict[str, Any],
    variables: list[dict[str, Any]],
    frame: dict[str, Any] | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    item = _variable_item(variables, grid["variable"]) or {}
    label = _strip_label_unit(item.get("label") or grid["label"] or grid["variable"], _clean_unit(grid["unit"]))
    element = grid["variable"] if label == grid["variable"] else f"{grid['variable']} - {label}"
    base = dict(meta.get("weather_info", {}))
    base.update(
        {
            "file": grid["file"],
            "element": element,
            "variable_key": grid["variable"],
            "element_desc_zh": item.get("description") or base.get("element_desc_zh") or "",
            "element_desc_en": item.get("description_en") or base.get("element_desc_en") or "",
            "time": frame.get("time_label") if frame else base.get("time"),
            "range": _format_range_text(grid["extent"]),
            "resolution": grid.get("resolution") or base.get("resolution"),
            "resolution_key": grid.get("resolution_key"),
            "grid": f"{grid['width']} x {grid['height']}",
            "unit": _clean_unit(grid["unit"]),
            "missing": str(grid.get("nodata", NODATA)),
            "status": base.get("status") or "解析成功",
            "min": _format_number(grid["min"]),
            "mean": _format_number(grid["mean"]),
            "max": _format_number(grid["max"]),
            "webp_url": grid.get("webp_url"),
            "resampling": grid.get("resampling"),
            "playable": grid.get("playable"),
            "warnings": warnings or [],
        }
    )
    return base


def _merge_display_meta(meta: dict[str, Any], weather_info: dict[str, Any]) -> dict[str, Any]:
    merged = dict(meta)
    merged.update(
        {
            "file": weather_info.get("file") or meta.get("file"),
            "element": weather_info.get("element") or meta.get("element"),
            "variable_key": weather_info.get("variable_key"),
            "element_desc_zh": weather_info.get("element_desc_zh"),
            "element_desc_en": weather_info.get("element_desc_en"),
            "time": weather_info.get("time") or meta.get("time"),
            "range": weather_info.get("range") or meta.get("range"),
            "resolution": weather_info.get("resolution") or meta.get("resolution"),
            "grid": weather_info.get("grid") or meta.get("grid"),
            "missing": weather_info.get("missing") or meta.get("missing"),
            "unit": weather_info.get("unit") or meta.get("unit"),
            "weather_info": weather_info,
        }
    )
    return merged


def _format_range_text(extent: list[float]) -> str:
    west, south, east, north = extent
    return f"{west:.4f}E-{east:.4f}E, {south:.4f}N-{north:.4f}N"


def _format_number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)
