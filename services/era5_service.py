import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ERA5"
NODATA = -999999.0


def get_display_data(variable: str | None = None, level_index: int = 0) -> dict[str, Any]:
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    png_files = sorted(DATA_DIR.glob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True)

    meta_json = None
    if meta_files:
        with meta_files[0].open("r", encoding="utf-8") as file:
            meta_json = json.load(file)

    variables = _display_variables(meta_json)
    grid = None
    if meta_json:
        try:
            grid = get_grid_data(variable=variable, meta=meta_json)
        except ValueError:
            grid = None

    return {
        "business_type": "ERA5",
        "meta_file": str(meta_files[0]).replace("\\", "/") if meta_files else None,
        "meta_json": meta_json,
        "png": str(png_files[0]).replace("\\", "/") if png_files else None,
        "png_files": [str(path).replace("\\", "/") for path in png_files],
        "variables": variables,
        "grid": grid,
    }


def get_grid_data(variable: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or _latest_meta()
    source_file = _source_file(meta)
    variable = _primary_variable(meta, variable)

    with _open_dataset(source_file) as dataset:
        dataset = _normalize_longitude(dataset)
        if variable not in dataset.data_vars:
            variable = _first_grid_variable(dataset)

        data_array = _select_first_slice(dataset[variable])
        data_array = data_array.transpose("latitude", "longitude")

        lat = data_array["latitude"].values.astype("float64")
        lon = data_array["longitude"].values.astype("float64")
        values = data_array.values.astype("float32")

        if lat[0] < lat[-1]:
            values = np.flip(values, axis=0)
            lat = np.flip(lat)

        values = np.where(np.isfinite(values), values, np.nan).astype("float32")
        finite = values[np.isfinite(values)]
        min_value = float(np.nanmin(finite)) if finite.size else 0.0
        max_value = float(np.nanmax(finite)) if finite.size else 1.0
        mean_value = float(np.nanmean(finite)) if finite.size else 0.0
        output = np.where(np.isfinite(values), values, NODATA).astype("float32")

        attrs = data_array.attrs
        label = str(attrs.get("GRIB_name") or attrs.get("long_name") or variable)
        unit = str(attrs.get("units") or "")
        extent = [
            float(np.nanmin(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lon)),
            float(np.nanmax(lat)),
        ]

    return {
        "business_type": "ERA5",
        "dataset_id": meta.get("dataset_id"),
        "file": source_file.name,
        "variable": variable,
        "label": label,
        "unit": unit,
        "width": int(output.shape[1]),
        "height": int(output.shape[0]),
        "extent": extent,
        "min": round(min_value, 6),
        "max": round(max_value, 6),
        "mean": round(mean_value, 6),
        "nodata": NODATA,
        "values": output.reshape(-1).round(6).tolist(),
        "variables": _display_variables(meta),
        "meta": _grid_meta(meta, variable, label, unit, extent, output.shape, min_value, max_value, mean_value),
    }


def _latest_meta() -> dict[str, Any]:
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not meta_files:
        raise ValueError("No ERA5 meta.json found. Parse an ERA5 file first.")
    with meta_files[0].open("r", encoding="utf-8") as file:
        return json.load(file)


def _source_file(meta: dict[str, Any]) -> Path:
    source = Path(str(meta.get("source_file", "")))
    if source.exists():
        return source

    if source.name:
        by_name = DATA_DIR / source.name
        if by_name.exists():
            return by_name

    candidates = sorted(DATA_DIR.glob("*.nc"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise ValueError("No ERA5 source NetCDF file found.")
    return candidates[0]


def _primary_variable(meta: dict[str, Any], requested: str | None = None) -> str:
    if requested:
        return str(requested)

    primary = meta.get("extra", {}).get("default_variable")
    if primary:
        return str(primary)

    variables = meta.get("variables") or []
    if variables:
        first = variables[0]
        if isinstance(first, dict):
            return str(first.get("name") or "")
        return str(first)

    raise ValueError("No ERA5 variable found in meta.json.")


def _display_variables(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not meta:
        return []

    result: list[dict[str, Any]] = []
    for item in meta.get("variables") or []:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                result.append({
                    "name": name,
                    "label": item.get("long_name") or name,
                    "unit": item.get("units") or item.get("unit") or "",
                })
        elif item:
            result.append({"name": str(item), "label": str(item), "unit": ""})
    return result


def _normalize_longitude(dataset: xr.Dataset) -> xr.Dataset:
    if "longitude" not in dataset.coords:
        return dataset

    lon = dataset.longitude.values
    if np.any(lon > 180):
        dataset = dataset.assign_coords(longitude=(((lon + 180) % 360) - 180))
        dataset = dataset.sortby("longitude")
    return dataset


def _open_dataset(source_file: Path) -> xr.Dataset:
    last_error: Exception | None = None
    for engine in ("netcdf4", "h5netcdf", "scipy", None):
        try:
            if engine is None:
                return xr.open_dataset(source_file)
            return xr.open_dataset(source_file, engine=engine)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"ERA5 NetCDF could not be opened: {last_error}") from last_error


def _first_grid_variable(dataset: xr.Dataset) -> str:
    for name, data_array in dataset.data_vars.items():
        dims = set(data_array.dims)
        if {"latitude", "longitude"}.issubset(dims):
            return name
    raise ValueError("No renderable ERA5 grid variable found.")


def _select_first_slice(data_array: xr.DataArray) -> xr.DataArray:
    selectors = {}
    for dim in data_array.dims:
        if dim not in {"latitude", "longitude"}:
            selectors[dim] = 0

    if selectors:
        data_array = data_array.isel(selectors)

    if not {"latitude", "longitude"}.issubset(set(data_array.dims)):
        raise ValueError(f"ERA5 variable {data_array.name} is not a latitude/longitude grid.")

    return data_array


def _grid_meta(
    meta: dict[str, Any],
    variable: str,
    label: str,
    unit: str,
    extent: list[float],
    shape: tuple[int, int],
    min_value: float,
    max_value: float,
    mean_value: float,
) -> dict[str, Any]:
    weather_info = meta.get("weather_info", {})
    times = meta.get("times") or [""]
    return {
        "source": "ERA5",
        "element": f"{label} ({variable})",
        "unit": unit,
        "extent": extent,
        "grid": f"{int(shape[1])} x {int(shape[0])}",
        "time": weather_info.get("time") or times[0],
        "min": round(min_value, 6),
        "max": round(max_value, 6),
        "mean": round(mean_value, 6),
    }
