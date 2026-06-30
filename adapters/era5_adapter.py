from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adapters.base import build_dataset_id, write_meta


LAT_NAMES = ("latitude", "lat", "y")
LON_NAMES = ("longitude", "lon", "x")
TIME_NAMES = ("valid_time", "time")
LEVEL_NAMES = ("pressure_level", "level", "isobaricInhPa", "plev")
PREFERRED_VARIABLES = ("t2m", "tp", "sp", "u10", "v10", "ssrd")
NODATA = -999999.0

VARIABLE_LABELS: dict[str, str] = {
    "t2m": "2 metre temperature",
    "tp": "Total precipitation",
    "sp": "Surface pressure",
    "u10": "10 metre U wind component",
    "v10": "10 metre V wind component",
    "ssrd": "Surface solar radiation downwards",
}


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
    return [np.datetime_as_string(item, unit="m") for item in values]


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


def _stats_from_array(data: np.ndarray) -> dict[str, float]:
    valid = data[np.isfinite(data) & (data > NODATA + 1)]
    if valid.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": round(float(valid.min()), 6),
        "max": round(float(valid.max()), 6),
        "mean": round(float(valid.mean()), 6),
        "std": round(float(valid.std()), 6),
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


def _public_data_path(path: Path) -> str:
    normalized = path.resolve().as_posix()
    marker = "/data/"
    idx = normalized.rfind(marker)
    return normalized[idx:] if idx >= 0 else normalized


def _generate_png(data: np.ndarray, lon: np.ndarray, lat: np.ndarray, output_path: Path, label: str, unit: str) -> Path:
    masked = np.ma.masked_where(data <= NODATA + 1, data)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(masked, origin="upper", extent=extent, cmap="viridis", aspect="auto")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, linestyle="--", alpha=0.25)
    cbar = plt.colorbar(img, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label(f"{label} ({unit})" if unit else label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_float32_grid(source_file: Path, var_name: str, step_index: int, data: np.ndarray) -> Path:
    output_path = source_file.with_name(f"{source_file.name}_{var_name}_step{step_index:03d}.float32")
    output_path.write_bytes(np.asarray(data, dtype="<f4").tobytes(order="C"))
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

    grid_urls: list[str] = []
    step_stats: list[dict[str, float]] = []
    png_urls: list[str] = []

    for step_index in range(step_count):
        data, lat, lon = _grid_values(ds, var_name, step_index)
        stats = _stats_from_array(data)
        grid_path = _write_float32_grid(source_file, var_name, step_index, data)
        grid_urls.append(_public_data_path(grid_path))
        step_stats.append(stats)

        if step_index == 0:
            png_path = source_file.with_name(f"{source_file.stem}_{var_name}.png")
            _generate_png(data, lon, lat, png_path, label, unit)
            png_urls.append(_public_data_path(png_path))

    combined = {
        "min": round(float(min(item["min"] for item in step_stats)), 6),
        "max": round(float(max(item["max"] for item in step_stats)), 6),
        "mean": round(float(np.mean([item["mean"] for item in step_stats])), 6),
        "std": round(float(np.mean([item["std"] for item in step_stats])), 6),
    }

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
        "level": "surface",
        "missing": NODATA,
        "stats": combined,
        "category": "era5",
        "description": label,
        "wavelength": None,
        "float32": {
            "path": grid_urls[0] if grid_urls else None,
            "paths": grid_urls,
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
        "png": png_urls[0] if png_urls else None,
    }

    layer_meta = {
        "name": var_name,
        "label": label,
        "unit": unit,
        "width": width,
        "height": height,
        "extent": bbox,
        "grid_urls": grid_urls,
        "png_urls": png_urls,
        "times": times[:step_count],
        "stats": step_stats,
        "nodata": NODATA,
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
        default_png = (default_layer.get("png_urls") or [None])[0]

        level_list = _levels(ds)
        weather_info: dict[str, Any] = {
            "source": "ERA5",
            "product": "ERA5 reanalysis",
            "element": f"{default_label} ({default_var})" if default_var else "ERA5",
            "time": f"{times[0]} ~ {times[-1]}" if len(times) > 1 else times[0],
            "level": " / ".join(level_list[:3]),
            "range": f"{west:.1f}E-{east:.1f}E, {south:.1f}N-{north:.1f}N",
            "resolution": f"{lat_res:.3f} x {lon_res:.3f} deg",
            "grid": grid_str,
            "validGrid": str(int(len(lon) * len(lat))),
            "coverage": "latitude/longitude grid",
            "missing": str(NODATA),
            "unit": default_unit,
            "variables": ", ".join(var_names),
            "steps": str(len(times)),
            "step_count": len(times),
            "status": "parsed",
            "quality": "good" if var_names else "no renderable variable",
            "max": f"{float(default_stats.get('max', 0.0)):.4f}",
            "min": f"{float(default_stats.get('min', 0.0)):.4f}",
            "mean": f"{float(default_stats.get('mean', 0.0)):.4f}",
            "alert": "",
            "update": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bars": [0, 0, 0, 0, 0],
            "trend": [0, 0, 0, 0, 0, 0, 0, 0],
        }

        meta_file = source_file.with_name(f"{source_file.name}.meta.json")
        png_files = [item.get("png") for item in variables if item.get("png")]
        meta: dict[str, Any] = {
            "schema_version": "1.0",
            "dataset_id": build_dataset_id(source_file),
            "data_type": data_type,
            "file_format": "NC",
            "source_file": source_file.as_posix(),
            "meta_file": meta_file.as_posix(),
            "png_files": png_files,
            "default_png": default_png,
            "default_variable": default_var,
            "times": times,
            "levels": level_list,
            "bbox": bbox,
            "extent": bbox,
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
                },
            },
        }

        write_meta(meta_file, meta)
        return meta
    finally:
        ds.close()
