import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ERA5"
NODATA = -999999.0
LAT_NAMES = ("latitude", "lat", "y")
LON_NAMES = ("longitude", "lon", "x")


def get_display_data(variable: str | None = None, level_index: int = 0) -> dict[str, Any]:
    meta_json = _latest_meta(allow_empty=True)
    png_files = sorted(DATA_DIR.glob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True)

    variables = _display_variables(meta_json)
    selected = _primary_variable(meta_json, variable) if meta_json else ""
    layer = _layer_for_variable(meta_json, selected) if meta_json else None

    return {
        "business_type": "ERA5",
        "meta_file": _path_string(_meta_path(meta_json)) if meta_json else None,
        "meta_json": meta_json,
        "png": _first_png(meta_json, layer, png_files),
        "png_files": _all_pngs(meta_json, png_files),
        "variables": variables,
        "variable_options": meta_json.get("variable_options", variables) if meta_json else [],
        "variable_layers": meta_json.get("variable_layers", {}) if meta_json else {},
        "default_variable": selected,
        "times": meta_json.get("times", []) if meta_json else [],
        "extent": meta_json.get("extent") or meta_json.get("bbox") if meta_json else None,
        "grid": _grid_descriptor(meta_json, selected, layer) if meta_json and layer else None,
    }


def get_grid_data(variable: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or _latest_meta()
    variable = _primary_variable(meta, variable)
    layer = _layer_for_variable(meta, variable)
    if layer:
        return _grid_descriptor(meta, variable, layer)
    return _legacy_grid_data(variable, meta)


def _latest_meta(allow_empty: bool = False) -> dict[str, Any] | None:
    meta_files = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for meta_file in meta_files:
        with meta_file.open("r", encoding="utf-8") as file:
            meta = json.load(file)
        if meta.get("variable_layers"):
            return meta
    if meta_files:
        with meta_files[0].open("r", encoding="utf-8") as file:
            return json.load(file)
    if allow_empty:
        return None
    raise ValueError("No ERA5 meta.json found. Parse an ERA5 file first.")


def _meta_path(meta: dict[str, Any] | None) -> Path | None:
    if not meta:
        return None
    value = meta.get("meta_file")
    if value:
        path = Path(str(value))
        if path.exists():
            return path
        fallback = DATA_DIR / path.name
        if fallback.exists():
            return fallback
    dataset_id = meta.get("dataset_id")
    if dataset_id:
        candidates = sorted(DATA_DIR.glob("*.meta.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for candidate in candidates:
            try:
                with candidate.open("r", encoding="utf-8") as file:
                    if json.load(file).get("dataset_id") == dataset_id:
                        return candidate
            except Exception:
                continue
    return None


def _path_string(path: Path | None) -> str | None:
    return str(path).replace("\\", "/") if path else None


def _public_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    normalized = str(path).replace("\\", "/")
    marker = "/data/"
    idx = normalized.rfind(marker)
    return normalized[idx:] if idx >= 0 else normalized


def _first_png(meta: dict[str, Any] | None, layer: dict[str, Any] | None, png_files: list[Path]) -> str | None:
    if layer:
        urls = layer.get("png_urls") or []
        if urls:
            return _public_url(urls[0])
    if meta and meta.get("default_png"):
        return _public_url(meta.get("default_png"))
    if png_files:
        return _path_string(png_files[0])
    return None


def _all_pngs(meta: dict[str, Any] | None, png_files: list[Path]) -> list[str]:
    result: list[str] = []
    if meta:
        for item in meta.get("png_files") or []:
            public = _public_url(item)
            if public and public not in result:
                result.append(public)
        for layer in (meta.get("variable_layers") or {}).values():
            for item in layer.get("png_urls") or []:
                public = _public_url(item)
                if public and public not in result:
                    result.append(public)
    for path in png_files:
        value = _path_string(path)
        if value and value not in result:
            result.append(value)
    return result


def _display_variables(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not meta:
        return []

    options = meta.get("variable_options")
    if isinstance(options, list) and options:
        return [
            {
                "name": item.get("name"),
                "label": item.get("label") or item.get("long_name") or item.get("name"),
                "unit": item.get("unit") or item.get("display_unit") or "",
            }
            for item in options
            if isinstance(item, dict) and item.get("name")
        ]

    result: list[dict[str, Any]] = []
    for item in meta.get("variables") or []:
        if isinstance(item, dict):
            name = item.get("name")
            if name:
                result.append({
                    "name": name,
                    "label": item.get("long_name") or item.get("label") or name,
                    "unit": item.get("unit") or item.get("units") or item.get("display_unit") or "",
                })
        elif item:
            result.append({"name": str(item), "label": str(item), "unit": ""})
    return result


def _primary_variable(meta: dict[str, Any] | None, requested: str | None = None) -> str:
    if requested:
        return str(requested)
    if not meta:
        return ""

    primary = meta.get("default_variable") or meta.get("extra", {}).get("default_variable")
    if not primary:
        primary = meta.get("extra", {}).get("era5", {}).get("default_variable")
    if primary:
        return str(primary)

    variables = _display_variables(meta)
    if variables:
        return str(variables[0]["name"])

    raise ValueError("No ERA5 variable found in meta.json.")


def _layer_for_variable(meta: dict[str, Any] | None, variable: str) -> dict[str, Any] | None:
    if not meta or not variable:
        return None
    layers = meta.get("variable_layers") or {}
    if variable in layers:
        return layers[variable]
    lowered = {str(key).lower(): value for key, value in layers.items()}
    return lowered.get(variable.lower())


def _grid_descriptor(meta: dict[str, Any], variable: str, layer: dict[str, Any]) -> dict[str, Any]:
    stats = layer.get("stats") or []
    first_stats = stats[0] if stats else {}
    grid_urls = layer.get("grid_urls") or layer.get("float32_urls") or []
    return {
        "business_type": "ERA5",
        "dataset_id": meta.get("dataset_id"),
        "file": Path(str(meta.get("source_file", ""))).name,
        "variable": variable,
        "label": layer.get("label") or variable,
        "unit": layer.get("unit") or "",
        "width": int(layer.get("width") or 0),
        "height": int(layer.get("height") or 0),
        "extent": layer.get("extent") or meta.get("extent") or meta.get("bbox"),
        "min": first_stats.get("min", 0.0),
        "max": first_stats.get("max", 1.0),
        "mean": first_stats.get("mean", 0.0),
        "nodata": layer.get("nodata", NODATA),
        "grid_urls": grid_urls,
        "png_urls": layer.get("png_urls") or [],
        "times": layer.get("times") or meta.get("times") or [],
        "stats": stats,
        "variables": _display_variables(meta),
        "meta": {
            "source": "ERA5",
            "element": f"{layer.get('label') or variable} ({variable})",
            "unit": layer.get("unit") or "",
            "extent": layer.get("extent") or meta.get("extent") or meta.get("bbox"),
            "grid": f"{int(layer.get('width') or 0)} x {int(layer.get('height') or 0)}",
            "time": (layer.get("times") or meta.get("times") or [""])[0],
            "min": first_stats.get("min", 0.0),
            "max": first_stats.get("max", 1.0),
            "mean": first_stats.get("mean", 0.0),
        },
    }


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


def _legacy_grid_data(variable: str | None, meta: dict[str, Any]) -> dict[str, Any]:
    source_file = _source_file(meta)
    variable = _primary_variable(meta, variable)

    with _open_dataset(source_file) as dataset:
        dataset = _normalize_longitude(dataset)
        if variable not in dataset.data_vars:
            variable = _first_grid_variable(dataset)

        data_array = _select_first_slice(dataset[variable])
        lat_name, lon_name = _lat_lon_names(dataset)
        data_array = data_array.transpose(lat_name, lon_name)

        lat = data_array[lat_name].values.astype("float64")
        lon = data_array[lon_name].values.astype("float64")
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
    }


def _normalize_longitude(dataset: xr.Dataset) -> xr.Dataset:
    try:
        _, lon_name = _lat_lon_names(dataset)
    except KeyError:
        return dataset

    lon = dataset[lon_name].values
    if np.any(lon > 180):
        dataset = dataset.assign_coords({lon_name: (((lon + 180) % 360) - 180)})
        dataset = dataset.sortby(lon_name)
    return dataset


def _coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.variables:
            return name
    raise KeyError(f"ERA5 NetCDF missing coordinate: one of {', '.join(candidates)}")


def _lat_lon_names(dataset: xr.Dataset) -> tuple[str, str]:
    return _coord_name(dataset, LAT_NAMES), _coord_name(dataset, LON_NAMES)


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
    lat_name, lon_name = _lat_lon_names(dataset)
    for name, data_array in dataset.data_vars.items():
        dims = set(data_array.dims)
        if {lat_name, lon_name}.issubset(dims):
            return name
    raise ValueError("No renderable ERA5 grid variable found.")


def _select_first_slice(data_array: xr.DataArray) -> xr.DataArray:
    lat_name = next((name for name in LAT_NAMES if name in data_array.dims), "latitude")
    lon_name = next((name for name in LON_NAMES if name in data_array.dims), "longitude")
    selectors = {}
    for dim in data_array.dims:
        if dim not in {lat_name, lon_name}:
            selectors[dim] = 0

    if selectors:
        data_array = data_array.isel(selectors)

    if not {lat_name, lon_name}.issubset(set(data_array.dims)):
        raise ValueError(f"ERA5 variable {data_array.name} is not a latitude/longitude grid.")

    return data_array
