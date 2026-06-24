from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adapters.base import build_dataset_id, write_meta


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


def _normalize_longitude(ds: xr.Dataset) -> xr.Dataset:
    """ERA5 longitude is sometimes 0-360; convert to -180-180."""
    if "longitude" not in ds.coords:
        return ds
    lon = ds.longitude.values
    if np.any(lon > 180):
        ds = ds.assign_coords(longitude=(((lon + 180) % 360) - 180))
        ds = ds.sortby("longitude")
    return ds


def _detect_time_var(ds: xr.Dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in ds.coords:
            return candidate
    raise KeyError("未找到时间坐标，请检查 NC 文件。")


def _format_times(ds: xr.Dataset, time_var: str) -> list[str]:
    return [np.datetime_as_string(t, unit="m") for t in ds[time_var].values]


def _generate_png(
    ds: xr.Dataset,
    var_name: str,
    output_path: Path,
    time_index: int = 0,
) -> tuple[Path, list[float]]:
    """Render a single time-step fallback PNG, then return (path, extent)."""
    time_var = _detect_time_var(ds)
    data = ds[var_name].isel({time_var: time_index}).values.astype(np.float32)

    lat = ds.latitude.values.astype(np.float64)
    lon = ds.longitude.values.astype(np.float64)

    # Flip to south→north for imshow(origin='lower')
    if lat[0] > lat[-1]:
        data = np.flip(data, axis=0)
        lat = np.flip(lat)

    # Mask NaN (actual NaN values, not the GRIB sentinel)
    data = np.ma.masked_invalid(data)

    # extent: [west, east, south, north] for matplotlib
    extent_cartopy = [float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])]

    units = ds[var_name].attrs.get("units", "")
    long_name = ds[var_name].attrs.get("GRIB_name", var_name)
    fig, ax = _plain_png_axes(data, extent_cartopy)
    img = ax.images[0]

    cbar = plt.colorbar(img, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label(f"{long_name} ({units})" if units else long_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # extent for frontend: [west, south, east, north]
    extent_fe = [float(lon[0]), float(lat[0]), float(lon[-1]), float(lat[-1])]
    return output_path, extent_fe


def _plain_png_axes(data: np.ndarray, extent_cartopy: list[float]):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xlim(extent_cartopy[0], extent_cartopy[1])
    ax.set_ylim(extent_cartopy[2], extent_cartopy[3])
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.imshow(data, origin="lower", extent=extent_cartopy, cmap="viridis", aspect="auto")
    return fig, ax


def _compute_stats(
    ds: xr.Dataset, var_name: str
) -> tuple[float, float, float, int]:
    """Return (min, max, mean, valid_count) across all time steps, skipping NaN."""
    arr = ds[var_name].values.astype(np.float64)
    valid = ~np.isnan(arr)
    valid_count = int(np.sum(valid))
    if valid_count == 0:
        return 0.0, 0.0, 0.0, 0
    return (
        float(arr[valid].min()),
        float(arr[valid].max()),
        float(arr[valid].mean()),
        valid_count,
    )


def process_file(file_path: str, data_type: str = "ERA5") -> dict:
    source_file = Path(file_path).resolve()

    # ---- 1. 打开 & 标准化 ----
    ds = _open_dataset(str(source_file))
    ds = _normalize_longitude(ds)

    time_var = _detect_time_var(ds)
    times_str = _format_times(ds, time_var)
    var_names = list(ds.data_vars.keys())

    lat = ds.latitude.values.astype(np.float64)
    lon = ds.longitude.values.astype(np.float64)
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())

    lat_res = abs(float(lat[1] - lat[0])) if len(lat) > 1 else 0
    lon_res = abs(float(lon[1] - lon[0])) if len(lon) > 1 else 0
    grid_str = f"{len(lon)} × {len(lat)}"

    # ---- 2. 默认变量 & 统计 ----
    default_var = var_names[0] if var_names else None

    if default_var:
        vmin, vmax, vmean, valid_count = _compute_stats(ds, default_var)
        var_units = ds[default_var].attrs.get("units", "")
        var_long_name = ds[default_var].attrs.get("GRIB_name", default_var)
        var_element = f"{var_long_name} ({default_var})"
    else:
        vmin = vmax = vmean = 0.0
        valid_count = 0
        var_units = "无"
        var_element = "无"

    # ---- 3. 生成 PNG（第一个时次） ----
    png_dir = source_file.parent
    png_path = png_dir / f"{source_file.stem}_{default_var}.png" if default_var else None
    extent_fe = [west, south, east, north]
    if png_path and default_var:
        _, extent_fe = _generate_png(ds, default_var, png_path)

    # ---- 4. weather_info ----
    weather_info: dict[str, Any] = {
        "source": "ERA5",
        "product": "ERA5 再分析数据",
        "element": var_element,
        "time": f"{times_str[0]} ~ {times_str[-1]}" if len(times_str) > 1 else times_str[0],
        "level": "地表",
        "range": f"{west:.1f}°E-{east:.1f}°E, {south:.1f}°N-{north:.1f}°N",
        "resolution": f"{lat_res:.3f}° × {lon_res:.3f}°",
        "grid": grid_str,
        "validGrid": str(valid_count),
        "coverage": "中国区域",
        "missing": "NaN",
        "unit": var_units,
        "variables": ", ".join(var_names),
        "steps": str(len(times_str)),
        "status": "已解析",
        "quality": "良好",
        "max": f"{vmax:.4f}",
        "min": f"{vmin:.4f}",
        "mean": f"{vmean:.4f}",
        "alert": "无",
        "update": str(len(times_str)),
        "bars": [0, 0, 0, 0, 0],
        "trend": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    # ---- 5. 组装 meta & 落盘 ----
    meta_file = source_file.with_name(f"{source_file.name}.meta.json")

    meta: dict[str, Any] = {
        "dataset_id": build_dataset_id(source_file),
        "data_type": data_type,
        "file_format": "NC",
        "source_file": source_file.as_posix(),
        "meta_file": meta_file.as_posix(),
        "png_files": [png_path.as_posix()] if png_path else [],
        "variables": [
            {"name": v, "long_name": ds[v].attrs.get("GRIB_name", v), "units": ds[v].attrs.get("units", "")}
            for v in var_names
        ],
        "times": times_str,
        "levels": ["地表"],
        "bbox": [west, south, east, north],
        "extent": extent_fe,
        "weather_info": weather_info,
        "extra": {
            "status": "parsed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "default_variable": default_var,
        },
    }

    write_meta(meta_file, meta)
    return meta
