from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from PIL import Image

from adapters.base import build_dataset_id, write_meta


LAT_NAMES = ("latitude", "lat", "y")
LON_NAMES = ("longitude", "lon", "x")
TIME_NAMES = ("valid_time", "time")
LEVEL_NAMES = ("pressure_level", "level", "isobaricInhPa", "plev")
PREFERRED_VARIABLES = ("t2m", "tp", "sp", "u10", "v10", "ssrd")
NODATA = -999999.0
WEBP_ALPHA = 200
KM_PER_DEGREE_LAT = 111.32
TARGET_RESOLUTIONS_KM = (1, 3)
MAX_RESAMPLED_CELLS = 5_000_000
COLOR_STOPS = np.asarray(
    [
        [37, 99, 235],
        [8, 145, 178],
        [22, 163, 74],
        [250, 204, 21],
        [220, 38, 38],
    ],
    dtype=np.float32,
)

VARIABLE_LABELS: dict[str, str] = {
    "t2m": "2 metre temperature",
    "tp": "Total precipitation",
    "sp": "Surface pressure",
    "u10": "10 metre U wind component",
    "v10": "10 metre V wind component",
    "ssrd": "Surface solar radiation downwards",
}


class ResampleSkipped(ValueError):
    def __init__(self, reason: str, detail: dict[str, Any]):
        super().__init__(reason)
        self.detail = detail


def _open_dataset(file_path: str) -> xr.Dataset:
    last_error: Exception | None = None
    for engine in ("netcdf4", "h5netcdf", "scipy", None):
        try:
            if engine is None:
                return xr.open_dataset(file_path)
            return xr.open_dataset(file_path, engine=engine)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"ERA5 NetCDF could not be opened: {last_error}") from last_error


def _coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.variables:
            return name
    raise KeyError(f"ERA5 NetCDF missing coordinate: one of {', '.join(candidates)}")


def _lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    return _coord_name(ds, LAT_NAMES), _coord_name(ds, LON_NAMES)


def _time_name(ds: xr.Dataset) -> str | None:
    for name in TIME_NAMES:
        if name in ds.coords or name in ds.variables:
            return name
    return None


def _level_names(ds: xr.Dataset) -> list[str]:
    return [name for name in LEVEL_NAMES if name in ds.coords or name in ds.variables]


def _normalize_longitude(ds: xr.Dataset) -> xr.Dataset:
    _, lon_name = _lat_lon_names(ds)
    lon = ds[lon_name].values
    if np.any(lon > 180):
        ds = ds.assign_coords({lon_name: (((lon + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)
    return ds


def _format_times(ds: xr.Dataset) -> list[str]:
    time_name = _time_name(ds)
    if not time_name:
        return ["static"]
    values = np.asarray(ds[time_name].values).reshape(-1)
    if np.issubdtype(values.dtype, np.datetime64):
        return [np.datetime_as_string(item, unit="m") for item in values]
    return [str(item) for item in values]


def _renderable_variables(ds: xr.Dataset) -> list[str]:
    lat_name, lon_name = _lat_lon_names(ds)
    names: list[str] = []
    for name, data_array in ds.data_vars.items():
        if {lat_name, lon_name}.issubset(set(data_array.dims)):
            names.append(name)
    preferred = [name for name in PREFERRED_VARIABLES if name in names]
    return preferred + [name for name in names if name not in preferred]


def _default_variable(names: list[str]) -> str | None:
    lowered = {name.lower(): name for name in names}
    for preferred in PREFERRED_VARIABLES:
        if preferred in lowered:
            return lowered[preferred]
    return names[0] if names else None


def _time_count(ds: xr.Dataset) -> int:
    time_name = _time_name(ds)
    if not time_name:
        return 1
    return int(ds.sizes.get(time_name, np.asarray(ds[time_name].values).size) or 1)


def _select_grid_slice(ds: xr.Dataset, var_name: str, time_index: int = 0) -> xr.DataArray:
    lat_name, lon_name = _lat_lon_names(ds)
    data_array = ds[var_name]
    selectors: dict[str, int] = {}
    time_name = _time_name(ds)

    for dim in data_array.dims:
        if dim in {lat_name, lon_name}:
            continue
        if dim == time_name:
            selectors[dim] = min(max(time_index, 0), data_array.sizes[dim] - 1)
        else:
            selectors[dim] = 0

    if selectors:
        data_array = data_array.isel(selectors)

    if not {lat_name, lon_name}.issubset(set(data_array.dims)):
        raise ValueError(f"ERA5 variable {var_name} is not a latitude/longitude grid.")

    return data_array.transpose(lat_name, lon_name)


def _grid_values(ds: xr.Dataset, var_name: str, time_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_array = _select_grid_slice(ds, var_name, time_index)
    lat_name, lon_name = _lat_lon_names(ds)
    data = data_array.values.astype(np.float32)
    lat = data_array[lat_name].values.astype(np.float64)
    lon = data_array[lon_name].values.astype(np.float64)

    if lat[0] < lat[-1]:
        data = np.flip(data, axis=0)
        lat = np.flip(lat)

    data = np.where(np.isfinite(data), data, NODATA).astype("<f4", copy=False)
    return data, lat, lon


def _native_resolution_km(lat: np.ndarray, lon: np.ndarray) -> float:
    if len(lat) < 2 or len(lon) < 2:
        return 0.0
    lat_res_km = abs(float(lat[1] - lat[0])) * KM_PER_DEGREE_LAT
    mid_lat = float((np.nanmin(lat) + np.nanmax(lat)) / 2)
    lon_scale = max(math.cos(math.radians(mid_lat)), 0.05)
    lon_res_km = abs(float(lon[1] - lon[0])) * KM_PER_DEGREE_LAT * lon_scale
    return max(lat_res_km, lon_res_km)


def _target_coords(lat: np.ndarray, lon: np.ndarray, target_km: int) -> tuple[np.ndarray, np.ndarray]:
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    mid_lat = (south + north) / 2
    lon_scale = max(math.cos(math.radians(mid_lat)), 0.05)
    lat_step = target_km / KM_PER_DEGREE_LAT
    lon_step = target_km / (KM_PER_DEGREE_LAT * lon_scale)
    lat_count = max(2, int(round((north - south) / lat_step)) + 1)
    lon_count = max(2, int(round((east - west) / lon_step)) + 1)
    return (
        np.linspace(south, north, lat_count, dtype=np.float64),
        np.linspace(west, east, lon_count, dtype=np.float64),
    )


def _resampled_grid_values(
    ds: xr.Dataset,
    var_name: str,
    time_index: int,
    target_km: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data_array = _select_grid_slice(ds, var_name, time_index)
    lat_name, lon_name = _lat_lon_names(ds)
    original_lat = data_array[lat_name].values.astype(np.float64)
    original_lon = data_array[lon_name].values.astype(np.float64)
    native_km = _native_resolution_km(original_lat, original_lon)
    target_lat, target_lon = _target_coords(original_lat, original_lon, target_km)
    cells = int(target_lat.size * target_lon.size)

    detail = {
        "target_km": target_km,
        "native_km": round(float(native_km), 4),
        "width": int(target_lon.size),
        "height": int(target_lat.size),
        "cells": cells,
    }
    if native_km and native_km <= target_km:
        detail["skip_reason"] = "native_resolution_is_finer_or_equal"
        raise ResampleSkipped(detail["skip_reason"], detail)
    if cells > MAX_RESAMPLED_CELLS:
        detail["skip_reason"] = "target_grid_too_large"
        detail["max_cells"] = MAX_RESAMPLED_CELLS
        raise ResampleSkipped(detail["skip_reason"], detail)

    sorted_array = data_array.sortby(lat_name).sortby(lon_name)
    resampled = sorted_array.interp(
        {lat_name: target_lat, lon_name: target_lon},
        method="linear",
        kwargs={"fill_value": None},
    )
    data = resampled.transpose(lat_name, lon_name).values.astype(np.float32)
    lat = resampled[lat_name].values.astype(np.float64)
    lon = resampled[lon_name].values.astype(np.float64)
    if lat[0] < lat[-1]:
        data = np.flip(data, axis=0)
        lat = np.flip(lat)
    data = np.where(np.isfinite(data), data, NODATA).astype("<f4", copy=False)
    return data, lat, lon, detail


def _stats_from_array(data: np.ndarray) -> dict[str, float | int]:
    valid = data[np.isfinite(data) & (data > NODATA + 1)]
    total_count = int(data.size)
    valid_count = int(valid.size)
    missing_count = total_count - valid_count
    missing_ratio = round(missing_count / total_count, 6) if total_count else 1.0
    if valid.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "valid_count": valid_count,
            "missing_count": missing_count,
            "total_count": total_count,
            "missing_ratio": missing_ratio,
        }
    return {
        "min": round(float(valid.min()), 6),
        "max": round(float(valid.max()), 6),
        "mean": round(float(valid.mean()), 6),
        "std": round(float(valid.std()), 6),
        "valid_count": valid_count,
        "missing_count": missing_count,
        "total_count": total_count,
        "missing_ratio": missing_ratio,
    }


def _quality_status(issue_count: int, warning_count: int) -> str:
    if issue_count:
        return "error"
    if warning_count:
        return "warning"
    return "ok"


def _variable_expected_range(var_name: str, unit: str) -> tuple[float | None, float | None] | None:
    key = var_name.lower()
    normalized_unit = unit.lower()
    if key in {"t2m", "d2m", "temperature"} or "temperature" in key:
        if "c" in normalized_unit and "k" not in normalized_unit:
            return (-90.0, 70.0)
        return (180.0, 340.0)
    if key in {"tp", "cp", "lsp", "precipitation"} or "precip" in key:
        return (0.0, None)
    if key in {"sp", "msl", "pressure"} or "pressure" in key:
        return (30000.0, 110000.0)
    if key in {"u10", "v10", "u", "v"} or "wind" in key:
        return (-150.0, 150.0)
    if key in {"ssrd", "strd", "surface_radiation"} or "radiation" in key:
        return (0.0, None)
    return None


def _variable_quality(
    var_name: str,
    unit: str,
    stats: dict[str, float | int],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    missing_ratio = float(stats.get("missing_ratio") or 0.0)
    valid_count = int(stats.get("valid_count") or 0)

    if valid_count == 0:
        issues.append({"code": "no_valid_value", "message": "No valid grid value found."})
    elif missing_ratio > 0.5:
        issues.append({
            "code": "missing_ratio_too_high",
            "message": "More than 50% of grid values are missing.",
            "missing_ratio": missing_ratio,
        })
    elif missing_ratio > 0.1:
        warnings.append({
            "code": "missing_ratio_high",
            "message": "More than 10% of grid values are missing.",
            "missing_ratio": missing_ratio,
        })

    expected = _variable_expected_range(var_name, unit)
    if expected and valid_count:
        min_expected, max_expected = expected
        min_value = float(stats.get("min") or 0.0)
        max_value = float(stats.get("max") or 0.0)
        if min_expected is not None and min_value < min_expected:
            warnings.append({
                "code": "below_expected_range",
                "message": "Variable minimum is below the expected physical range.",
                "min_value": min_value,
                "expected_min": min_expected,
            })
        if max_expected is not None and max_value > max_expected:
            warnings.append({
                "code": "above_expected_range",
                "message": "Variable maximum is above the expected physical range.",
                "max_value": max_value,
                "expected_max": max_expected,
            })

    return {
        "status": _quality_status(len(issues), len(warnings)),
        "issues": issues,
        "warnings": warnings,
    }


def _dataset_quality(
    bbox: list[float],
    lat: np.ndarray,
    lon: np.ndarray,
    times: list[str],
    variables: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    west, south, east, north = bbox

    if south < -90 or north > 90 or south >= north:
        issues.append({"code": "invalid_latitude_range", "bbox": bbox})
    else:
        checks.append({"name": "latitude_range", "status": "ok", "south": south, "north": north})

    if west < -180 or east > 180 or west >= east:
        issues.append({"code": "invalid_longitude_range", "bbox": bbox})
    else:
        checks.append({"name": "longitude_range", "status": "ok", "west": west, "east": east})

    if len(lat) < 2 or len(lon) < 2:
        issues.append({"code": "grid_too_small", "lat_count": int(len(lat)), "lon_count": int(len(lon))})
    else:
        checks.append({"name": "grid_size", "status": "ok", "lat_count": int(len(lat)), "lon_count": int(len(lon))})

    if not times:
        issues.append({"code": "missing_time_axis"})
    elif len(times) != len(set(times)):
        warnings.append({"code": "duplicate_time_steps"})
    else:
        checks.append({"name": "time_axis", "status": "ok", "time_count": len(times)})

    variable_summary = {
        item["name"]: item.get("quality", {"status": "unknown"})
        for item in variables
        if item.get("name")
    }
    for name, quality in variable_summary.items():
        for issue in quality.get("issues") or []:
            issues.append({"code": "variable_issue", "variable": name, "detail": issue})
        for warning in quality.get("warnings") or []:
            warnings.append({"code": "variable_warning", "variable": name, "detail": warning})

    return {
        "status": _quality_status(len(issues), len(warnings)),
        "checks": checks,
        "issues": issues,
        "warnings": warnings,
        "summary": {
            "issue_count": len(issues),
            "warning_count": len(warnings),
            "variable_count": len(variables),
            "time_count": len(times),
        },
        "variables": variable_summary,
    }


def _variable_label(data_array: xr.DataArray, fallback: str) -> str:
    return str(
        data_array.attrs.get("GRIB_name")
        or data_array.attrs.get("long_name")
        or VARIABLE_LABELS.get(fallback.lower())
        or fallback
    )


def _levels(ds: xr.Dataset) -> list[str]:
    names = _level_names(ds)
    if not names:
        return ["surface"]

    values: list[str] = []
    for name in names:
        coord = ds[name]
        unit = str(coord.attrs.get("units") or "")
        for value in np.asarray(coord.values).reshape(-1)[:20]:
            text = f"{float(value):g}" if np.issubdtype(np.asarray(value).dtype, np.number) else str(value)
            values.append(f"{text}{unit}")
    return values or ["surface"]


def _selected_level(ds: xr.Dataset, data_array: xr.DataArray) -> str:
    lat_name, lon_name = _lat_lon_names(ds)
    time_name = _time_name(ds)
    for dim in data_array.dims:
        if dim in {lat_name, lon_name, time_name}:
            continue
        if dim not in ds.coords and dim not in ds.variables:
            continue
        coord = ds[dim]
        values = np.asarray(coord.values).reshape(-1)
        if values.size == 0:
            continue
        value = values[0]
        text = f"{float(value):g}" if np.issubdtype(np.asarray(value).dtype, np.number) else str(value)
        unit = str(coord.attrs.get("units") or "")
        return f"{dim}={text}{unit}"
    return "surface"


def _public_data_path(path: Path) -> str:
    normalized = path.resolve().as_posix()
    marker = "/data/"
    idx = normalized.rfind(marker)
    return normalized[idx:] if idx >= 0 else normalized


def _rgba_from_grid(data: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    valid_mask = np.isfinite(data) & (data > NODATA + 1)
    min_value = float(stats.get("min", 0.0))
    max_value = float(stats.get("max", 1.0))
    span = max(max_value - min_value, 0.000001)
    normalized = np.clip((data.astype(np.float32) - min_value) / span, 0.0, 1.0)

    scaled = normalized * (len(COLOR_STOPS) - 1)
    low = np.floor(scaled).astype(np.int16)
    low = np.clip(low, 0, len(COLOR_STOPS) - 2)
    high = low + 1
    local = (scaled - low)[..., None]
    rgb = COLOR_STOPS[low] + (COLOR_STOPS[high] - COLOR_STOPS[low]) * local

    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid_mask, WEBP_ALPHA, 0).astype(np.uint8)
    return rgba


def _generate_webp(data: np.ndarray, output_path: Path, stats: dict[str, float]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(_rgba_from_grid(data, stats), mode="RGBA")
    image.save(str(output_path), format="WEBP", quality=88, method=6)
    return output_path


def _build_variable_meta(
    ds: xr.Dataset,
    source_file: Path,
    var_name: str,
    times: list[str],
    bbox: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_array = ds[var_name]
    label = _variable_label(data_array, var_name)
    unit = str(data_array.attrs.get("units") or "")
    step_count = max(_time_count(ds), 1)
    width = int(ds.sizes[_lat_lon_names(ds)[1]])
    height = int(ds.sizes[_lat_lon_names(ds)[0]])

    webp_urls: list[str] = []
    step_stats: list[dict[str, float | int]] = []
    resolution_layers: dict[str, Any] = {}
    resolution_status: dict[str, Any] = {}

    for step_index in range(step_count):
        data, _, _ = _grid_values(ds, var_name, step_index)
        stats = _stats_from_array(data)
        step_stats.append(stats)

        webp_path = source_file.with_name(f"{source_file.stem}_{var_name}_step{step_index:03d}.webp")
        _generate_webp(data, webp_path, stats)
        webp_urls.append(_public_data_path(webp_path))

    native_layer = {
        "key": "native",
        "label": "native",
        "target_km": None,
        "width": width,
        "height": height,
        "extent": bbox,
        "webp_urls": webp_urls,
        "image_urls": webp_urls,
        "png_urls": webp_urls,
        "times": times[:step_count],
        "stats": step_stats,
        "nodata": NODATA,
        "resolution": "native",
        "resolution_km": None,
    }
    resolution_layers["native"] = native_layer

    for target_km in TARGET_RESOLUTIONS_KM:
        key = f"{target_km}km"
        target_urls: list[str] = []
        target_stats: list[dict[str, float | int]] = []
        target_detail: dict[str, Any] | None = None
        try:
            for step_index in range(step_count):
                data, _, _, detail = _resampled_grid_values(ds, var_name, step_index, target_km)
                target_detail = detail
                stats = _stats_from_array(data)
                target_stats.append(stats)
                webp_path = source_file.with_name(
                    f"{source_file.stem}_{var_name}_{key}_step{step_index:03d}.webp"
                )
                _generate_webp(data, webp_path, stats)
                target_urls.append(_public_data_path(webp_path))
        except ResampleSkipped as exc:
            resolution_status[key] = {
                "status": "skipped",
                "reason": str(exc),
                "detail": exc.detail,
            }
            continue

        if target_urls and target_detail:
            resolution_layers[key] = {
                "key": key,
                "label": key,
                "target_km": target_km,
                "width": int(target_detail["width"]),
                "height": int(target_detail["height"]),
                "extent": bbox,
                "webp_urls": target_urls,
                "image_urls": target_urls,
                "png_urls": target_urls,
                "times": times[:step_count],
                "stats": target_stats,
                "nodata": NODATA,
                "resolution": key,
                "resolution_km": target_km,
                "native_resolution_km": target_detail.get("native_km"),
            }
            resolution_status[key] = {"status": "generated", "detail": target_detail}

    combined = {
        "min": round(float(min(item["min"] for item in step_stats)), 6),
        "max": round(float(max(item["max"] for item in step_stats)), 6),
        "mean": round(float(np.mean([item["mean"] for item in step_stats])), 6),
        "std": round(float(np.mean([item["std"] for item in step_stats])), 6),
        "valid_count": int(sum(int(item.get("valid_count", 0)) for item in step_stats)),
        "missing_count": int(sum(int(item.get("missing_count", 0)) for item in step_stats)),
        "total_count": int(sum(int(item.get("total_count", 0)) for item in step_stats)),
    }
    combined["missing_ratio"] = round(
        float(combined["missing_count"]) / float(combined["total_count"]),
        6,
    ) if combined["total_count"] else 1.0
    quality = _variable_quality(var_name, unit, combined)

    variable_meta = {
        "name": var_name,
        "long_name": label,
        "short_name": var_name,
        "raw_name": var_name,
        "name_cn": label,
        "unit": unit,
        "display_unit": unit,
        "shape": [int(value) for value in data_array.shape],
        "dims": list(data_array.dims),
        "level": _selected_level(ds, data_array),
        "missing": NODATA,
        "stats": combined,
        "quality": quality,
        "category": "era5",
        "description": label,
        "wavelength": None,
        "float32": {
            "path": None,
            "paths": [],
            "dtype": "float32",
            "byte_order": "little",
            "width": width,
            "height": height,
            "nodata": NODATA,
        },
        "netcdf": {
            "variable": var_name,
            "time_coord": _time_name(ds),
            "lat_coord": _lat_lon_names(ds)[0],
            "lon_coord": _lat_lon_names(ds)[1],
        },
        "webp": {
            "path": webp_urls[0] if webp_urls else None,
            "paths": webp_urls,
            "width": width,
            "height": height,
            "alpha": WEBP_ALPHA / 255,
        },
        "available_resolutions": list(resolution_layers.keys()),
        "resolution_layers": resolution_layers,
        "resolution_status": resolution_status,
        "quality": quality,
        "png": webp_urls[0] if webp_urls else None,
    }

    layer_meta = {
        "name": var_name,
        "label": label,
        "unit": unit,
        "width": width,
        "height": height,
        "extent": bbox,
        "grid_urls": [],
        "float32_urls": [],
        "webp_urls": webp_urls,
        "image_urls": webp_urls,
        "png_urls": webp_urls,
        "times": times[:step_count],
        "stats": step_stats,
        "nodata": NODATA,
        "resolution": "native",
        "available_resolutions": list(resolution_layers.keys()),
        "resolution_layers": resolution_layers,
        "resolution_status": resolution_status,
        "quality": quality,
    }
    return variable_meta, layer_meta


def process_file(file_path: str, data_type: str = "ERA5") -> dict:
    source_file = Path(file_path).resolve()
    ds = _open_dataset(str(source_file))

    try:
        ds = _normalize_longitude(ds)
        lat_name, lon_name = _lat_lon_names(ds)
        times = _format_times(ds)
        var_names = _renderable_variables(ds)
        default_var = _default_variable(var_names)

        lat = ds[lat_name].values.astype(np.float64)
        lon = ds[lon_name].values.astype(np.float64)
        west, east = float(lon.min()), float(lon.max())
        south, north = float(lat.min()), float(lat.max())
        bbox = [west, south, east, north]
        lat_res = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 0.0
        lon_res = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0.0
        grid_str = f"{len(lon)} x {len(lat)}"

        variables: list[dict[str, Any]] = []
        variable_layers: dict[str, Any] = {}
        for var_name in var_names:
            variable_meta, layer_meta = _build_variable_meta(ds, source_file, var_name, times, bbox)
            variables.append(variable_meta)
            variable_layers[var_name] = layer_meta

        default_layer = variable_layers.get(default_var or "") or next(iter(variable_layers.values()), {})
        default_stats = (default_layer.get("stats") or [{}])[0]
        default_label = default_layer.get("label") or default_var or "ERA5"
        default_unit = default_layer.get("unit") or ""
        default_webp = (default_layer.get("webp_urls") or default_layer.get("image_urls") or [None])[0]
        quality_report = _dataset_quality(bbox, lat, lon, times, variables)

        level_list = _levels(ds)
        weather_info: dict[str, Any] = {
            "source": "ERA5",
            "product": "ERA5 reanalysis",
            "element": f"{default_label} ({default_var})" if default_var else "ERA5",
            "time": f"{times[0]} ~ {times[-1]}" if len(times) > 1 else times[0],
            "level": " / ".join(level_list[:3]),
            "range": f"{west:.1f}E-{east:.1f}E, {south:.1f}N-{north:.1f}N",
            "resolution": f"{lat_res:.3f} x {lon_res:.3f} deg",
            "available_resolutions": list(default_layer.get("available_resolutions") or ["native"]),
            "grid": grid_str,
            "validGrid": str(int(len(lon) * len(lat))),
            "coverage": "latitude/longitude grid",
            "missing": str(NODATA),
            "unit": default_unit,
            "variables": ", ".join(var_names),
            "steps": str(len(times)),
            "step_count": len(times),
            "status": "parsed",
            "quality": quality_report["status"],
            "quality_issue_count": quality_report["summary"]["issue_count"],
            "quality_warning_count": quality_report["summary"]["warning_count"],
            "max": f"{float(default_stats.get('max', 0.0)):.4f}",
            "min": f"{float(default_stats.get('min', 0.0)):.4f}",
            "mean": f"{float(default_stats.get('mean', 0.0)):.4f}",
            "alert": "",
            "update": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bars": [0, 0, 0, 0, 0],
            "trend": [0, 0, 0, 0, 0, 0, 0, 0],
        }

        meta_file = source_file.with_name(f"{source_file.name}.meta.json")
        webp_files = [
            path
            for layer in variable_layers.values()
            for res_layer in (layer.get("resolution_layers") or {"native": layer}).values()
            for path in (res_layer.get("webp_urls") or [])
            if path
        ]
        meta: dict[str, Any] = {
            "schema_version": "1.0",
            "dataset_id": build_dataset_id(source_file),
            "data_type": data_type,
            "file_format": "NC",
            "source_file": source_file.as_posix(),
            "meta_file": meta_file.as_posix(),
            "webp_files": webp_files,
            "default_webp": default_webp,
            "png_files": webp_files,
            "default_png": default_webp,
            "default_variable": default_var,
            "times": times,
            "levels": level_list,
            "bbox": bbox,
            "extent": bbox,
            "quality_report": quality_report,
            "variables": variables,
            "variable_options": [
                {
                    "name": item["name"],
                    "label": item["long_name"],
                    "unit": item["unit"],
                }
                for item in variables
            ],
            "variable_layers": variable_layers,
            "available_resolutions": sorted({
                key
                for layer in variable_layers.values()
                for key in (layer.get("available_resolutions") or ["native"])
            }, key=lambda item: (item != "native", item)),
            "composites": [],
            "weather_info": weather_info,
            "extra": {
                "status": "parsed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "era5": {
                    "default_variable": default_var,
                    "lat_coord": lat_name,
                    "lon_coord": lon_name,
                    "time_coord": _time_name(ds),
                    "preferred_variables": list(PREFERRED_VARIABLES),
                    "quality_report": quality_report,
                },
            },
        }

        try:
            from services.era5_store import sync_meta

            meta["extra"]["era5"]["db_sync"] = sync_meta(meta)
        except Exception as exc:
            meta["extra"]["era5"]["db_sync_error"] = str(exc)

        write_meta(meta_file, meta)
        return meta
    finally:
        ds.close()
