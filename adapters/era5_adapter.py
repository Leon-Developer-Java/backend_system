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
        if name in ds.coords:
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
        return ["静态场"]
    return [np.datetime_as_string(item, unit="m") for item in ds[time_name].values]


def _renderable_variables(ds: xr.Dataset) -> list[str]:
    lat_name, lon_name = _lat_lon_names(ds)
    names: list[str] = []
    for name, data_array in ds.data_vars.items():
        if {lat_name, lon_name}.issubset(set(data_array.dims)):
            names.append(name)
    return names


def _default_variable(names: list[str]) -> str | None:
    lowered = {name.lower(): name for name in names}
    for preferred in PREFERRED_VARIABLES:
        if preferred in lowered:
            return lowered[preferred]
    return names[0] if names else None


def _select_first_grid_slice(ds: xr.Dataset, var_name: str, time_index: int = 0) -> xr.DataArray:
    lat_name, lon_name = _lat_lon_names(ds)
    data_array = ds[var_name]
    selectors: dict[str, int] = {}

    for dim in data_array.dims:
        if dim in {lat_name, lon_name}:
            continue
        if dim == _time_name(ds):
            selectors[dim] = min(max(time_index, 0), data_array.sizes[dim] - 1)
        else:
            selectors[dim] = 0

    if selectors:
        data_array = data_array.isel(selectors)

    if not {lat_name, lon_name}.issubset(set(data_array.dims)):
        raise ValueError(f"ERA5 variable {var_name} is not a latitude/longitude grid.")

    return data_array.transpose(lat_name, lon_name)


def _plain_png_axes(data: np.ndarray, extent: list[float]):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.imshow(data, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    return fig, ax


def _generate_png(ds: xr.Dataset, var_name: str, output_path: Path, time_index: int = 0) -> tuple[Path, list[float]]:
    data_array = _select_first_grid_slice(ds, var_name, time_index)
    lat_name, lon_name = _lat_lon_names(ds)
    data = data_array.values.astype(np.float32)
    lat = data_array[lat_name].values.astype(np.float64)
    lon = data_array[lon_name].values.astype(np.float64)

    if lat[0] > lat[-1]:
        data = np.flip(data, axis=0)
        lat = np.flip(lat)

    data = np.ma.masked_invalid(data)
    extent_for_png = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    units = str(data_array.attrs.get("units") or "")
    long_name = _variable_label(data_array, var_name)

    fig, ax = _plain_png_axes(data, extent_for_png)
    img = ax.images[0]
    cbar = plt.colorbar(img, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label(f"{long_name} ({units})" if units else long_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path, [float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())]


def _compute_stats(data_array: xr.DataArray) -> tuple[float, float, float, int]:
    arr = data_array.values.astype(np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0, 0
    return float(finite.min()), float(finite.max()), float(finite.mean()), int(finite.size)


def _variable_label(data_array: xr.DataArray, fallback: str) -> str:
    return str(data_array.attrs.get("GRIB_name") or data_array.attrs.get("long_name") or fallback)


def _variables_meta(ds: xr.Dataset, names: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in names:
        item = ds[name]
        result.append(
            {
                "name": name,
                "long_name": _variable_label(item, name),
                "units": str(item.attrs.get("units") or ""),
                "dims": list(item.dims),
                "shape": [int(value) for value in item.shape],
            }
        )
    return result


def _levels(ds: xr.Dataset) -> list[str]:
    names = _level_names(ds)
    if not names:
        return ["地表"]

    values: list[str] = []
    for name in names:
        coord = ds[name]
        unit = str(coord.attrs.get("units") or "")
        for value in np.asarray(coord.values).reshape(-1)[:20]:
            text = f"{float(value):g}" if np.issubdtype(np.asarray(value).dtype, np.number) else str(value)
            values.append(f"{text}{unit}")
    return values or ["地表"]


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
        lat_res = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 0.0
        lon_res = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0.0
        grid_str = f"{len(lon)} x {len(lat)}"

        png_path = source_file.parent / f"{source_file.stem}_{default_var}.png" if default_var else None
        extent = [west, south, east, north]

        if default_var:
            selected = _select_first_grid_slice(ds, default_var)
            vmin, vmax, vmean, valid_count = _compute_stats(selected)
            var_units = str(ds[default_var].attrs.get("units") or "")
            var_element = f"{_variable_label(ds[default_var], default_var)} ({default_var})"
            if png_path:
                _, extent = _generate_png(ds, default_var, png_path)
        else:
            vmin = vmax = vmean = 0.0
            valid_count = 0
            var_units = ""
            var_element = "无可渲染变量"

        level_list = _levels(ds)
        weather_info: dict[str, Any] = {
            "source": "ERA5",
            "product": "ERA5 再分析数据",
            "element": var_element,
            "time": f"{times[0]} ~ {times[-1]}" if len(times) > 1 else times[0],
            "level": " / ".join(level_list[:3]),
            "range": f"{west:.1f}E-{east:.1f}E, {south:.1f}N-{north:.1f}N",
            "resolution": f"{lat_res:.3f} x {lon_res:.3f} deg",
            "grid": grid_str,
            "validGrid": str(valid_count),
            "coverage": "经纬度网格",
            "missing": "NaN",
            "unit": var_units,
            "variables": ", ".join(var_names),
            "steps": str(len(times)),
            "status": "已解析",
            "quality": "良好" if valid_count else "无有效格点",
            "max": f"{vmax:.4f}",
            "min": f"{vmin:.4f}",
            "mean": f"{vmean:.4f}",
            "alert": "",
            "update": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bars": [0, 0, 0, 0, 0],
            "trend": [0, 0, 0, 0, 0, 0, 0, 0],
        }

        meta_file = source_file.with_name(f"{source_file.name}.meta.json")
        meta: dict[str, Any] = {
            "dataset_id": build_dataset_id(source_file),
            "data_type": data_type,
            "file_format": "NC",
            "source_file": source_file.as_posix(),
            "meta_file": meta_file.as_posix(),
            "png_files": [png_path.as_posix()] if png_path else [],
            "variables": _variables_meta(ds, var_names),
            "times": times,
            "levels": level_list,
            "bbox": [west, south, east, north],
            "extent": extent,
            "weather_info": weather_info,
            "extra": {
                "status": "parsed",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "default_variable": default_var,
                "lat_coord": lat_name,
                "lon_coord": lon_name,
                "time_coord": _time_name(ds),
            },
        }

        write_meta(meta_file, meta)
        return meta
    finally:
        ds.close()
