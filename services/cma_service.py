import io
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio

from adapters import cma_adapter


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "CMA"
CMA_SOURCE_DIRS = (DATA_DIR,)
NODATA = -999999.0


def get_display_data(
    variable: str | None = None,
    level_index: int = 0,
    time_index: int = 0,
    meta_file: str | None = None,
) -> dict[str, Any]:
    meta_path = _selected_meta_file(meta_file)
    meta_json = _load_meta_file(meta_path) if meta_path else None
    variables = _display_variables(meta_json)
    frames = _frames_from_meta(meta_json) or (_series_frames(_source_file(meta_json)) if meta_json else [])
    grid = None
    weather_info = meta_json.get("weather_info", {}) if isinstance(meta_json, dict) else {}

    if meta_json:
        try:
            frame = _active_frame(frames, time_index)
            variable_item = _variable_item(variables, variable or _primary_variable(meta_json)) or {}
            grid = get_grid_data(
                variable=variable,
                level_index=level_index,
                file_name=frame.get("file") if frame else None,
                meta=meta_json,
            )
            source_file = _resolve_source_file(grid["file"])
            webp_path = cma_adapter.ensure_variable_webp(
                source_file,
                grid["variable"],
                grid.get("level_index", 0),
                stats=variable_item.get("stats"),
            )
            grid["values"] = []
            grid["webp_url"] = cma_adapter.public_data_path(webp_path)
            grid["image_url"] = grid["webp_url"]
            stats = variable_item.get("stats") if isinstance(variable_item, dict) else None
            if isinstance(stats, dict):
                grid["scale_min"] = stats.get("min")
                grid["scale_max"] = stats.get("max")
            weather_info = _build_display_weather_info(meta_json, grid, variables, frame)
            meta_json = _merge_display_meta(meta_json, weather_info)
        except ValueError:
            grid = None

    return {
        "business_type": "CMA",
        "meta_file": str(meta_path).replace("\\", "/") if meta_path else None,
        "meta_json": meta_json,
        "png": meta_json.get("default_png") if isinstance(meta_json, dict) else None,
        "png_files": meta_json.get("png_files", []) if isinstance(meta_json, dict) else [],
        "webp": grid.get("webp_url") if grid else (meta_json.get("default_webp") if isinstance(meta_json, dict) else None),
        "webp_files": meta_json.get("webp_files", []) if isinstance(meta_json, dict) else [],
        "variables": variables,
        "grid": grid,
        "frames": frames,
        "times": [frame["time"] for frame in frames],
        "frame_count": len(frames),
        "weather_info": weather_info,
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
    return meta_files[0] if meta_files else None


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
    return oriented


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
            "grid": f"{grid['width']} x {grid['height']}",
            "unit": _clean_unit(grid["unit"]),
            "missing": str(grid.get("nodata", NODATA)),
            "status": "解析成功",
            "min": _format_number(grid["min"]),
            "mean": _format_number(grid["mean"]),
            "max": _format_number(grid["max"]),
            "webp_url": grid.get("webp_url"),
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
