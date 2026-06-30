import json
import io
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import h5py
import numpy as np
import rasterio

from adapters import cma_adapter


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "CMA"
PROJECT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CMA_SOURCE_DIRS = (
    PROJECT_DATA_DIR / "NC",
    PROJECT_DATA_DIR / "GRIB",
    DATA_DIR,
)
NODATA = -999999.0
COMMON_NC_DISPLAY_VARIABLES = (
    "Tair_f_inst",
    "Rainf_tavg",
    "TotalPrecip_tavg",
    "Wind_f_inst",
    "Qair_f_inst",
    "Psurf_f_inst",
    "AvgSurfT_inst",
    "SoilMoist_inst",
    "SoilTemp_inst",
    "SWdown_f_tavg",
)


def get_display_data(variable: str | None = None, level_index: int = 0) -> dict[str, Any]:
    # 前端点击 CMA 类型时调用该函数，读取 CMA 目录下的 meta.json 和 PNG。
    _ensure_latest_meta()
    meta_files = _meta_files()
    png_files = sorted(DATA_DIR.rglob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True)

    meta_json = None
    if meta_files:
        with meta_files[0].open("r", encoding="utf-8") as file:
            meta_json = json.load(file)

    variables = _display_variables(meta_json)
    grid = None
    frames: list[dict[str, Any]] = []
    if meta_json:
        try:
            grid = get_grid_data(variable=variable, level_index=level_index)
            frames = _frames_from_meta(meta_json) or _series_frames(_source_file(meta_json))
            grid["values"] = []
            grid["binary_url"] = _grid_url(grid["file"], grid["variable"], level_index)
        except ValueError:
            grid = None

    return {
        "business_type": "CMA",
        "meta_file": str(meta_files[0]).replace("\\", "/") if meta_files else None,
        "meta_json": meta_json,
        "png": str(png_files[0]).replace("\\", "/") if png_files else None,
        "png_files": [str(path).replace("\\", "/") for path in png_files],
        "variables": variables,
        "grid": grid,
        "frames": frames,
        "times": [frame["time"] for frame in frames],
        "frame_count": len(frames),
    }


def get_grid_data(variable: str | None = None, level_index: int = 0, file_name: str | None = None) -> dict[str, Any]:
    meta = _latest_meta()
    source_file = _resolve_source_file(file_name) if file_name else _source_file(meta)
    suffix = source_file.suffix.lower()
    file_format = "NC" if suffix == ".nc" else "GRIB" if suffix in {".grib", ".grib2"} else str(meta.get("file_format") or suffix.lstrip(".")).upper()
    variable_name = variable or _primary_variable(meta)

    if file_format == "NC" or source_file.suffix.lower() == ".nc":
        payload = _read_nc_grid(source_file, meta, variable_name, level_index)
    elif source_file.suffix.lower() in {".grib", ".grib2"}:
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


def get_binary_grid_data(file_name: str | None = None, variable: str | None = None, level_index: int = 0) -> dict[str, Any]:
    grid = get_grid_data(variable=variable, level_index=level_index, file_name=file_name)
    values = np.asarray(grid.pop("values"), dtype="float32")
    return {**grid, "bytes": values.tobytes(), "dtype": "float32"}


def _latest_meta() -> dict[str, Any]:
    _ensure_latest_meta()
    meta_files = _meta_files()
    if not meta_files:
        raise ValueError("No CMA meta.json found. Parse a CMA file first.")
    with meta_files[0].open("r", encoding="utf-8") as file:
        return json.load(file)


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
    return files


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


def _grid_url(file_name: str, variable: str, level_index: int) -> str:
    return f"/api/cma/grid?{urlencode({'file': file_name, 'variable': variable, 'level_index': level_index})}"


def _ensure_latest_meta() -> None:
    sources = sorted(_source_files(), key=_source_sort_key)
    if not sources:
        return

    latest_source = sources[0]
    expected_meta = latest_source.with_name(f"{latest_source.name}.meta.json")
    if expected_meta.exists() and expected_meta.stat().st_mtime >= latest_source.stat().st_mtime:
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
    top_variables = meta.get("variables", [])
    if top_variables and isinstance(top_variables[0], dict):
        display_variables = [
            {
                "name": item.get("name"),
                "label": item.get("long_name") or item.get("name"),
                "unit": item.get("display_unit") or item.get("unit", ""),
                "dims": item.get("dims", []),
                "shape": item.get("shape", []),
            }
            for item in top_variables
            if _is_grid_variable(item)
        ]
        if meta.get("extra", {}).get("cma", {}).get("product_type") == "LAND_NC":
            common = _common_nc_variables(display_variables)
            if common:
                return common
        return display_variables

    cma = meta.get("extra", {}).get("cma", {})
    products = cma.get("products", {})
    for product in products.values():
        variables = product.get("variables", [])
        if variables:
            display_variables = [
                {
                    "name": item.get("name"),
                    "label": item.get("long_name") or item.get("name"),
                    "unit": item.get("unit", ""),
                    "dims": item.get("dims", []),
                    "shape": item.get("shape", []),
                }
                for item in variables
                if _is_grid_variable(item)
            ]
            if product.get("product_type") == "LAND_NC":
                common = _common_nc_variables(display_variables)
                if common:
                    return common
            return display_variables
    return [{"name": name, "label": name, "unit": ""} for name in top_variables]


def _common_nc_variables(variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(item.get("name")): item for item in variables if item.get("name")}
    return [by_name[name] for name in COMMON_NC_DISPLAY_VARIABLES if name in by_name]


def _is_grid_variable(item: dict[str, Any]) -> bool:
    dims = item.get("dims") or []
    shape = item.get("shape") or []
    return bool(item.get("float32")) or bool(item.get("band")) or dims[-2:] == ["lat", "lon"] or len(shape) in {2, 3}


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
    variable_names = [item.get("name") for item in _display_variables(meta) if item.get("name")]
    return {
        **{key: value for key, value in meta.items() if key in {"file", "time", "range", "grid", "missing", "vars", "steps"}},
        "element": ", ".join(variable_names) or str(payload["variable"]),
        "unit": _clean_unit(payload["unit"]),
        "extent": payload["extent"],
        "grid": f"{payload['width']} x {payload['height']}",
        "min": payload["min"],
        "max": payload["max"],
        "mean": payload["mean"],
    }
