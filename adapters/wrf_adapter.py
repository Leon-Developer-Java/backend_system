from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from adapters.base import build_dataset_id, write_meta


PRODUCT_VARIABLES = [
    "PM2_5_DRY",
    "PM10",
    "AOD2D_OUT",
    "T2",
    "U10",
    "V10",
    "PSFC",
    "PBLH",
    "RAINC",
    "RAINNC",
]

TARGET_RESOLUTIONS = {
    "1km": 1000.0,
    "3km": 3000.0,
}
DEFAULT_RESOLUTION_KEY = "3km"

VARIABLE_LABELS = {
    "PM2_5_DRY": ("PM2.5 dry mass concentration", "Near-surface fine particulate matter concentration."),
    "PM10": ("PM10 mass concentration", "Near-surface inhalable particulate matter concentration."),
    "AOD2D_OUT": ("Aerosol optical depth", "Column-integrated aerosol optical depth."),
    "T2": ("2m temperature", "Air temperature at 2 metres above ground."),
    "U10": ("10m U wind", "East-west wind component at 10 metres."),
    "V10": ("10m V wind", "North-south wind component at 10 metres."),
    "PSFC": ("Surface pressure", "Model surface pressure."),
    "PBLH": ("Planetary boundary layer height", "Boundary layer height used to assess mixing and dispersion."),
    "RAINC": ("Accumulated convective precipitation", "Accumulated precipitation from convective processes."),
    "RAINNC": ("Accumulated non-convective precipitation", "Accumulated precipitation from non-convective microphysics."),
}
SKIP_NAMES = {
    "Times",
    "XLAT",
    "XLONG",
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "CLAT",
}


_BACKEND_DIR = Path(__file__).resolve().parents[1]


class WrfAdapterError(ValueError):
    """Non-retryable WRF validation or rendering failure with a business phase."""

    def __init__(self, phase: str, message: str):
        self.phase = phase
        super().__init__(f"[WRF:{phase}] {message}")


def _wrf_error(phase: str, message: str, exc: Exception | None = None) -> WrfAdapterError:
    if exc is not None:
        return WrfAdapterError(phase, f"{message}: {type(exc).__name__}: {exc}")
    return WrfAdapterError(phase, message)


def _to_relative(path: Path) -> str:
    try:
        return path.relative_to(_BACKEND_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else _BACKEND_DIR / p

VARIABLE_INFORMATION = {
    "PM2_5_DRY": (
        "PM2.5 dry mass concentration",
        "Near-surface fine particulate matter concentration for air-quality analysis.",
        "Dry mass concentration of fine particulate matter with aerodynamic diameter below 2.5 micrometres.",
    ),
    "PM10": (
        "PM10 mass concentration",
        "Near-surface inhalable particulate matter concentration.",
        "Mass concentration of inhalable particulate matter with aerodynamic diameter below 10 micrometres.",
    ),
    "AOD2D_OUT": (
        "Aerosol optical depth",
        "Column aerosol optical depth for haze and aerosol loading analysis.",
        "Aerosol optical depth indicating column-integrated aerosol extinction of solar radiation.",
    ),
    "T2": (
        "2 metre temperature",
        "Air temperature at 2 metres above the surface.",
        "Air temperature at about 2 metres above the surface, describing near-surface thermal conditions.",
    ),
    "U10": (
        "10 metre U wind component",
        "East-west wind component at 10 metres above the surface.",
        "East-west wind component at 10 metres above the surface; positive values usually indicate eastward flow.",
    ),
    "V10": (
        "10 metre V wind component",
        "North-south wind component at 10 metres above the surface.",
        "North-south wind component at 10 metres above the surface; positive values usually indicate northward flow.",
    ),
    "PSFC": (
        "Surface pressure",
        "Atmospheric pressure near the model surface.",
        "Surface pressure, representing atmospheric pressure near the model surface.",
    ),
    "PBLH": (
        "Planetary boundary layer height",
        "Boundary layer height for mixing and pollutant dispersion assessment.",
        "Planetary boundary layer height, used to assess near-surface mixing and pollutant dispersion conditions.",
    ),
    "RAINC": (
        "Accumulated convective precipitation",
        "Accumulated precipitation produced by convective processes.",
        "Accumulated precipitation produced by convective parameterization processes.",
    ),
    "RAINNC": (
        "Accumulated non-convective precipitation",
        "Accumulated precipitation produced by non-convective cloud microphysics.",
        "Accumulated precipitation produced by non-convective cloud microphysics processes.",
    ),
    "XLAT": ("Latitude", "Latitude coordinate of each model grid point.", "Latitude coordinate of each model grid point."),
    "XLONG": ("Longitude", "Longitude coordinate of each model grid point.", "Longitude coordinate of each model grid point."),
    "Times": ("Valid time", "Valid time of the WRF model output.", "Valid time of the WRF model output."),
}
def _load_runtime():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import colormaps
        from netCDF4 import Dataset
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "WRF adapter requires netCDF4, numpy, matplotlib, and Pillow to parse wrfout and generate WebP."
        ) from exc
    return Dataset, Image, colormaps

def _time_label(ds: Any) -> str:
    if "Times" not in ds.variables:
        return ""
    raw = ds.variables["Times"][0]
    return b"".join(raw).decode("ascii", errors="ignore")


def _lat_lon(ds: Any) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(ds.variables["XLAT"][0])
    lon = np.asarray(ds.variables["XLONG"][0])
    return lat, lon


def _destagger_to_mass_grid(data: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
    if "west_east_stag" in dims:
        axis = dims.index("west_east_stag") - (len(dims) - data.ndim)
        data = 0.5 * (
            np.take(data, range(data.shape[axis] - 1), axis=axis)
            + np.take(data, range(1, data.shape[axis]), axis=axis)
        )
    if "south_north_stag" in dims:
        axis = dims.index("south_north_stag") - (len(dims) - data.ndim)
        data = 0.5 * (
            np.take(data, range(data.shape[axis] - 1), axis=axis)
            + np.take(data, range(1, data.shape[axis]), axis=axis)
        )
    if "bottom_top_stag" in dims:
        axis = dims.index("bottom_top_stag") - (len(dims) - data.ndim)
        data = 0.5 * (
            np.take(data, range(data.shape[axis] - 1), axis=axis)
            + np.take(data, range(1, data.shape[axis]), axis=axis)
        )
    return data


def _field_from_var(ds: Any, variable: str, level: int = 0) -> tuple[np.ndarray, str, str, str]:
    var = ds.variables[variable]
    arr = _destagger_to_mass_grid(np.asarray(var[:]), var.dimensions)

    if arr.ndim == 4:
        data = arr[0, min(level, arr.shape[1] - 1), :, :]
    elif arr.ndim == 3 and var.dimensions[0] == "Time":
        data = arr[0, :, :]
    elif arr.ndim == 3:
        data = arr[min(level, arr.shape[0] - 1), :, :]
    elif arr.ndim == 2:
        data = arr[:, :]
    else:
        raise ValueError(f"{variable} shape {arr.shape} is not supported for 2D display.")

    desc = getattr(var, "description", variable)
    units = getattr(var, "units", "")
    return np.asarray(data, dtype=float), str(desc), str(units), variable


def _can_display_variable(ds: Any, name: str, lat_shape: tuple[int, int]) -> bool:
    if name in SKIP_NAMES:
        return False
    var = ds.variables[name]
    dims = var.dimensions
    has_y = "south_north" in dims or "south_north_stag" in dims
    has_x = "west_east" in dims or "west_east_stag" in dims
    if not has_y or not has_x:
        return False
    try:
        data, _, _, _ = _field_from_var(ds, name)
    except Exception:
        return False
    return data.shape == lat_shape and np.isfinite(data).any()


def _robust_range(data: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(data, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(valid, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _render_overlay(data: np.ndarray, image_cls: Any, colormaps: Any, cmap_name: str = "turbo") -> Any:
    arr = np.asarray(data, dtype=float)
    valid = np.isfinite(arr)
    lo, hi = _robust_range(arr)
    normalized = np.clip((arr - lo) / (hi - lo), 0, 1)
    rgba = colormaps[cmap_name](normalized, bytes=True)
    rgba[..., 3] = np.where(valid, 185, 0).astype(np.uint8)
    return image_cls.fromarray(np.flipud(rgba), mode="RGBA")


def _resample_grid(data: np.ndarray, source_dx: float, source_dy: float, target_meters: float) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        return arr
    if source_dx <= 0 or source_dy <= 0 or target_meters <= 0:
        return arr

    height, width = arr.shape
    new_width = max(2, int(round((width - 1) * source_dx / target_meters)) + 1)
    new_height = max(2, int(round((height - 1) * source_dy / target_meters)) + 1)
    if new_width == width and new_height == height:
        return arr

    x_old = np.arange(width, dtype=float)
    y_old = np.arange(height, dtype=float)
    x_new = np.linspace(0, width - 1, new_width)
    y_new = np.linspace(0, height - 1, new_height)

    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0)
    weights = finite.astype(float)

    interp_values = np.vstack([np.interp(x_new, x_old, row) for row in filled])
    interp_weights = np.vstack([np.interp(x_new, x_old, row) for row in weights])
    interp_values = np.vstack([np.interp(y_new, y_old, interp_values[:, idx]) for idx in range(new_width)]).T
    interp_weights = np.vstack([np.interp(y_new, y_old, interp_weights[:, idx]) for idx in range(new_width)]).T

    with np.errstate(invalid="ignore", divide="ignore"):
        result = interp_values / interp_weights
    result[interp_weights <= 0] = np.nan
    return result


def _target_resolution_keys(dx: float, dy: float) -> list[str]:
    if dx and dy and (dx < 1000 or dy < 1000):
        return []
    return list(TARGET_RESOLUTIONS.keys())


def _default_resolution_key(products: dict[str, dict[str, Any]]) -> str:
    if DEFAULT_RESOLUTION_KEY in products:
        return DEFAULT_RESOLUTION_KEY
    return next(iter(products), DEFAULT_RESOLUTION_KEY)


def _domain_from_file(source_file: Path, ds: Any) -> str:
    name = source_file.name.lower()
    if "wrfout_d01" in name:
        return "d01"
    if "wrfout_d02" in name:
        return "d02"
    grid_id = getattr(ds, "GRID_ID", "")
    return f"d{int(grid_id):02d}" if str(grid_id).isdigit() else "unknown"


def _stats(data: np.ndarray) -> dict[str, float | None]:
    valid = np.asarray(data, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(np.nanmin(valid)),
        "max": float(np.nanmax(valid)),
        "mean": float(np.nanmean(valid)),
    }


def _safe_info_text(value: Any) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").replace("|", "/").strip()


def _variable_information(ds: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, var in ds.variables.items():
        desc = _safe_info_text(getattr(var, "description", ""))
        units = _safe_info_text(getattr(var, "units", ""))
        dims = ", ".join(str(item) for item in getattr(var, "dimensions", ()))
        english_label, zh_desc, en_desc = VARIABLE_INFORMATION.get(
            name,
            (
                desc or name,
                f"WRF model output variable {name}. Units: {units or 'unknown'}. Dimensions: {dims or 'unknown'}.",
                desc or f"WRF model output variable {name}.",
            ),
        )
        rows.append(
            {
                "name": _safe_info_text(name),
                "english_label": _safe_info_text(english_label),
                "chinese_description": _safe_info_text(zh_desc),
                "english_description": _safe_info_text(en_desc),
                "units": units,
                "dimensions": dims,
            }
        )
    return rows

def _array_has_finite_values(value: Any) -> bool:
    try:
        arr = np.asarray(value)
        return arr.size > 0 and np.isfinite(arr.astype(float)).any()
    except Exception:
        return False


def _validate_wrf_entry(source_file: Path, ds: Any) -> None:
    if "wrfout" not in source_file.name.lower():
        raise _wrf_error("entry_validate", f"source file is not a wrfout file: {source_file.name}")
    if "Times" not in ds.variables:
        raise _wrf_error("entry_validate", "Times variable is missing")
    try:
        time_label = _time_label(ds)
    except Exception as exc:
        raise _wrf_error("entry_validate", "Times variable cannot be decoded", exc) from exc
    if not time_label:
        raise _wrf_error("entry_validate", "Times variable is empty")
    for coord_name in ("XLAT", "XLONG"):
        if coord_name not in ds.variables:
            raise _wrf_error("entry_validate", f"{coord_name} coordinate is missing")
        if not _array_has_finite_values(ds.variables[coord_name][:]):
            raise _wrf_error("entry_validate", f"{coord_name} coordinate has no valid finite values")

    available = [name for name in PRODUCT_VARIABLES if name in ds.variables]
    if not available:
        raise _wrf_error("entry_validate", f"none of the configured WRF product variables exists: {', '.join(PRODUCT_VARIABLES)}")
    core_available = [name for name in ("T2", "U10", "V10", "PSFC") if name in ds.variables]
    if not core_available:
        raise _wrf_error("entry_validate", "none of the core weather variables exists: T2, U10, V10, PSFC")
    for name in available:
        try:
            if not _array_has_finite_values(ds.variables[name][:]):
                raise _wrf_error("entry_validate", f"key variable {name} has no valid finite values")
        except WrfAdapterError:
            raise
        except Exception as exc:
            raise _wrf_error("entry_validate", f"key variable {name} cannot be read", exc) from exc


def _path_exists_for_meta(value: Any, *, product_root: Path | None = None, final_dir: Path | None = None) -> bool:
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        return False
    candidates: list[Path] = []
    if text.startswith("/data/") and product_root is not None:
        candidates.append(product_root / text.removeprefix("/data/"))
    elif text.startswith("data/") and product_root is not None:
        candidates.append(product_root / text.removeprefix("data/"))
    else:
        raw = Path(text)
        if raw.is_absolute():
            candidates.append(raw)
        if final_dir is not None:
            candidates.append(final_dir / raw.name)
        candidates.append(_resolve_path(text))
    return any(path.exists() and path.is_file() for path in candidates)


def _validate_wrf_meta(meta: dict[str, Any], *, meta_file: Path | None = None, product_root: Path | None = None, final_dir: Path | None = None) -> None:
    if not isinstance(meta, dict):
        raise _wrf_error("metadata_validate", "meta is not a JSON object")
    required = ["dataset_id", "data_type", "source_file", "meta_file", "webp_files", "resolution_products", "variables", "times", "bbox", "weather_info"]
    missing = [key for key in required if key not in meta or meta.get(key) in (None, "", [], {})]
    if missing:
        raise _wrf_error("metadata_validate", f"required meta fields are missing or empty: {', '.join(missing)}")
    if str(meta.get("data_type") or "").upper() != "WRF":
        raise _wrf_error("metadata_validate", f"unexpected data_type={meta.get('data_type')!r}")
    if meta_file is not None and (not meta_file.is_file()):
        raise _wrf_error("metadata_validate", f"meta file was not written: {meta_file}")
    bbox = meta.get("bbox")
    if not isinstance(bbox, dict):
        raise _wrf_error("metadata_validate", "bbox must be an object with west/south/east/north")
    coords = [bbox.get(key) for key in ("west", "south", "east", "north")]
    if not all(Number_is_finite(value) for value in coords):
        raise _wrf_error("metadata_validate", f"bbox contains invalid coordinates: {bbox}")
    west, south, east, north = map(float, coords)
    if west >= east or south >= north:
        raise _wrf_error("metadata_validate", f"bbox is not a valid extent: {bbox}")
    times = meta.get("times")
    if not isinstance(times, list) or not any(str(item).strip() for item in times):
        raise _wrf_error("metadata_validate", "times is empty")
    variables = meta.get("variables")
    if not isinstance(variables, list) or not variables:
        raise _wrf_error("metadata_validate", "variables is empty")
    for item in variables:
        if not isinstance(item, dict) or not item.get("name"):
            raise _wrf_error("metadata_validate", f"variable item misses required name: {item}")
    webp_files = meta.get("webp_files")
    if not isinstance(webp_files, list) or not webp_files:
        raise _wrf_error("metadata_validate", "webp_files is empty")
    missing_webps = [
        str(path)
        for path in webp_files
        if not _path_exists_for_meta(path, product_root=product_root, final_dir=final_dir)
    ]
    if missing_webps:
        raise _wrf_error("metadata_validate", f"webp_files contain missing files: {missing_webps[:5]}")
    products = meta.get("resolution_products")
    if not isinstance(products, dict) or not products:
        raise _wrf_error("metadata_validate", "resolution_products is empty")
    for key, product in products.items():
        if not isinstance(product, dict):
            raise _wrf_error("metadata_validate", f"resolution product {key} is not an object")
        product_files = product.get("webp_files")
        product_variables = product.get("variables")
        if not isinstance(product_files, list) or not product_files:
            raise _wrf_error("metadata_validate", f"resolution product {key} has no WebP files")
        if not isinstance(product_variables, list) or not product_variables:
            raise _wrf_error("metadata_validate", f"resolution product {key} has no variables")
        product_missing = [
            str(path)
            for path in product_files
            if not _path_exists_for_meta(path, product_root=product_root, final_dir=final_dir)
        ]
        if product_missing:
            raise _wrf_error("metadata_validate", f"resolution product {key} references missing WebP files: {product_missing[:5]}")


def Number_is_finite(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_before_db_write(
    *,
    meta: dict[str, Any],
    meta_file: str | Path,
    final_dir: str | Path,
    product_root: str | Path,
    result: dict[str, Any],
    assets: list[dict[str, Any]],
) -> None:
    """Validate published WRF meta, WebP paths, result fields, and DB asset rows before success commit."""
    product_root_path = Path(product_root).resolve()
    final_dir_path = Path(final_dir).resolve()
    meta_path = Path(meta_file).resolve()
    try:
        final_dir_path.relative_to(product_root_path)
        meta_path.relative_to(final_dir_path)
    except ValueError as exc:
        raise _wrf_error("db_prewrite_validate", "published WRF paths escaped product/final directory") from exc
    _validate_wrf_meta(meta, meta_file=meta_path, product_root=product_root_path, final_dir=final_dir_path)
    result_required = ["data_type", "meta_path", "default_webp_url", "webp_count", "adapter_name", "adapter_version", "meta_schema_version"]
    missing_result = [key for key in result_required if result.get(key) in (None, "", [])]
    if missing_result:
        raise _wrf_error("db_prewrite_validate", f"success result fields are missing: {', '.join(missing_result)}")
    if str(result.get("data_type") or "").upper() != "WRF":
        raise _wrf_error("db_prewrite_validate", f"unexpected success result data_type={result.get('data_type')!r}")
    if int(result.get("webp_count") or 0) <= 0:
        raise _wrf_error("db_prewrite_validate", "webp_count must be greater than zero")
    if not _path_exists_for_meta(result.get("default_webp_url"), product_root=product_root_path, final_dir=final_dir_path):
        raise _wrf_error("db_prewrite_validate", f"default_webp_url is missing on disk: {result.get('default_webp_url')}")
    if not isinstance(assets, list) or not assets:
        raise _wrf_error("db_prewrite_validate", "asset catalog is empty")
    required_asset_fields = ["asset_uuid", "file_uuid", "element_key", "frame_index", "resolution_key", "webp_url", "asset_status"]
    for index, asset in enumerate(assets):
        missing_asset = [key for key in required_asset_fields if asset.get(key) in (None, "")]
        if missing_asset:
            raise _wrf_error("db_prewrite_validate", f"asset[{index}] missing fields: {', '.join(missing_asset)}")
        if not _path_exists_for_meta(asset.get("webp_url"), product_root=product_root_path, final_dir=final_dir_path):
            raise _wrf_error("db_prewrite_validate", f"asset[{index}] WebP is missing on disk: {asset.get('webp_url')}")


def _cached_meta_if_ready(source_file: Path) -> dict[str, Any] | None:
    meta_file = source_file.with_name(f"{source_file.name}.meta.json")
    if not meta_file.exists():
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    webp_files = meta.get("webp_files")
    if not isinstance(webp_files, list) or not webp_files:
        return None
    products = meta.get("resolution_products")
    if not isinstance(products, dict) or not products:
        return None
    if any(not _resolve_path(str(item)).exists() for item in webp_files):
        return None
    for product in products.values():
        product_files = product.get("webp_files") if isinstance(product, dict) else None
        if not isinstance(product_files, list) or not product_files:
            return None
        if any(not _resolve_path(str(item)).exists() for item in product_files):
            return None
    return meta


def _canonical_uploaded_source(source_file: Path) -> Path:
    match = re.match(r"^(wrfout_d\d{2}_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})_\d+$", source_file.name)
    if not match:
        return source_file

    canonical = source_file.with_name(match.group(1))
    if not canonical.exists():
        return source_file

    try:
        source_file.unlink()
    except OSError:
        pass
    return canonical


def process_file(file_path: str, data_type: str = "WRF") -> dict:
    Dataset, Image, colormaps = _load_runtime()

    source_file = _canonical_uploaded_source(Path(file_path).resolve())
    cached_meta = _cached_meta_if_ready(source_file)
    if cached_meta is not None:
        return cached_meta

    meta_file = source_file.with_name(f"{source_file.name}.meta.json")
    webp_dir = source_file.parent / f"{source_file.name}.webps"
    try:
        if webp_dir.exists():
            shutil.rmtree(webp_dir)
        if meta_file.exists():
            meta_file.unlink()
        webp_dir.mkdir(parents=True, exist_ok=True)

        try:
            ds_context = Dataset(source_file)
        except Exception as exc:
            raise _wrf_error("file_read", f"failed to open wrfout file {source_file.name}", exc) from exc

        with ds_context as ds:
            try:
                _validate_wrf_entry(source_file, ds)
                lat, lon = _lat_lon(ds)
                variable_information = _variable_information(ds)
                bbox = {
                    "west": float(np.nanmin(lon)),
                    "south": float(np.nanmin(lat)),
                    "east": float(np.nanmax(lon)),
                    "north": float(np.nanmax(lat)),
                }
                time_label = _time_label(ds)
                domain = _domain_from_file(source_file, ds)
                dx = float(getattr(ds, "DX", 0) or 0)
                dy = float(getattr(ds, "DY", 0) or 0)
                grid = f"{lat.shape[1]} x {lat.shape[0]}"
                resolution = f"{dx / 1000:g} km" if dx else "unknown"
            except WrfAdapterError:
                raise
            except Exception as exc:
                raise _wrf_error("file_read", "failed to read WRF coordinates, Times, or global attributes", exc) from exc

            try:
                display_variables = [
                    name for name in PRODUCT_VARIABLES
                    if name in ds.variables and _can_display_variable(ds, name, lat.shape)
                ]
                if not display_variables:
                    display_variables = [
                        name for name in ds.variables
                        if _can_display_variable(ds, name, lat.shape)
                    ][:8]
                if not display_variables:
                    raise _wrf_error("variable_compute", "no displayable WRF variable contains finite gridded values")
            except WrfAdapterError:
                raise
            except Exception as exc:
                raise _wrf_error("variable_compute", "failed while selecting display variables", exc) from exc

            target_keys = _target_resolution_keys(dx, dy)
            if not target_keys:
                raise _wrf_error(
                    "product_generate",
                    f"source resolution is finer than 1km or invalid; skip rendering. dx={dx}, dy={dy}",
                )
            resolution_products: dict[str, dict[str, Any]] = {
                key: {
                    "label": key,
                    "resolution": key.replace("km", " km"),
                    "target_meters": int(TARGET_RESOLUTIONS[key]),
                    "webp_files": [],
                    "variables": [],
                    "grid": "",
                    "is_resampled": TARGET_RESOLUTIONS[key] != dx or TARGET_RESOLUTIONS[key] != dy,
                }
                for key in target_keys
            }
            source_variables: list[dict[str, Any]] = []
            primary_stats = {"min": None, "max": None, "mean": None}
            primary_unit = ""
            primary_element = "WRF variable"

            for name in display_variables:
                try:
                    data, desc, units, var_id = _field_from_var(ds, name)
                    label, business_desc = VARIABLE_LABELS.get(name, (desc or name, desc or name))
                    stat = _stats(data)
                    if stat["min"] is None or stat["max"] is None:
                        raise _wrf_error("variable_compute", f"variable {name} contains no finite values")
                    source_variables.append(
                        {
                            "name": name,
                            "label": label,
                            "description": business_desc,
                            "units": units,
                            "shape": list(data.shape),
                            "source_shape": list(data.shape),
                            "source_resolution": resolution,
                            **stat,
                        }
                    )
                except WrfAdapterError:
                    raise
                except Exception as exc:
                    raise _wrf_error("variable_compute", f"failed to compute variable {name}", exc) from exc

                for res_key, target_meters in TARGET_RESOLUTIONS.items():
                    if res_key not in resolution_products:
                        continue
                    try:
                        product = resolution_products[res_key]
                        output_data = _resample_grid(data, dx, dy, target_meters)
                        output_stat = _stats(output_data)
                        if output_stat["min"] is None or output_stat["max"] is None:
                            raise _wrf_error("product_generate", f"{name} {res_key} resample produced no finite values")
                        output_grid = f"{output_data.shape[1]} x {output_data.shape[0]}"
                        product["grid"] = output_grid
                        product["variables"].append(
                            {
                                "name": name,
                                "label": label,
                                "description": business_desc,
                                "units": units,
                                "shape": list(output_data.shape),
                                "source_shape": list(data.shape),
                                "source_resolution": resolution,
                                "grid": output_grid,
                                "resolution": product["resolution"],
                                **output_stat,
                            }
                        )

                        image = _render_overlay(output_data, Image, colormaps)
                        webp_path = webp_dir / res_key / f"{time_label.replace(':', '_')}_{var_id}.webp"
                        webp_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(webp_path, format="WEBP", lossless=True, quality=90, method=6)
                        if not webp_path.is_file() or webp_path.stat().st_size <= 0:
                            raise _wrf_error("product_generate", f"WebP was not written or is empty: {webp_path.name}")
                        product["webp_files"].append(_to_relative(webp_path))
                    except WrfAdapterError:
                        raise
                    except Exception as exc:
                        raise _wrf_error("product_generate", f"failed to generate {res_key} WebP for {name}", exc) from exc

                if name == display_variables[0]:
                    primary_stats = stat
                    primary_unit = units
            primary_element = "WRF variable"

            default_resolution = _default_resolution_key(resolution_products)
            default_product = resolution_products.get(default_resolution, {})
            webp_files = list(default_product.get("webp_files", []))
            variables = list(default_product.get("variables") or source_variables)
            if variables:
                first_variable = variables[0]
                primary_stats = {
                    "min": first_variable.get("min"),
                    "max": first_variable.get("max"),
                    "mean": first_variable.get("mean"),
                }
                primary_unit = first_variable.get("units", primary_unit)
            primary_element = "WRF variable"

            weather_info = {
                "source": "WRF",
                "product": "WRF-Chem model layer",
                "element": primary_element,
                "time": time_label.replace("_", " "),
                "level": "surface / near-surface or level 0",
                "range": (
                    f"{bbox['west']:.3f}E-{bbox['east']:.3f}E, "
                    f"{bbox['south']:.3f}N-{bbox['north']:.3f}N"
                ),
                "resolution": resolution,
                "displayResolution": default_resolution,
                "sourceResolution": resolution,
                "grid": grid,
                "validGrid": f"{lat.size}",
                "coverage": domain,
                "missing": "NaN/FillValue",
                "unit": primary_unit,
                "variables": str(len(display_variables)),
                "steps": "1",
                "status": "parsed",
                "quality": "transparent WebP overlay generated",
                "max": primary_stats["max"],
                "min": primary_stats["min"],
                "mean": primary_stats["mean"],
                "alert": "none",
                "update": datetime.now(timezone.utc).isoformat(),
                "bars": [0, 0, 0, 0, 0],
                "trend": [],
            }

            meta = {
                "dataset_id": build_dataset_id(source_file),
                "data_type": data_type,
                "file_name": source_file.name,
                "file_format": "NC",
                "source_file": source_file.as_posix(),
                "meta_file": meta_file.as_posix(),
                "webp_files": webp_files,
                "default_resolution": default_resolution,
                "source_resolution": resolution,
                "resolution_products": resolution_products,
                "variables": variables,
                "variable_information": variable_information,
                "times": [time_label],
                "levels": ["surface_or_level_0"],
                "bbox": bbox,
                "weather_info": weather_info,
                "extra": {
                    "status": "parsed",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "domain": domain,
                    "dx": dx,
                    "dy": dy,
                    "default_resolution": default_resolution,
                    "target_resolutions": target_keys,
                    "webp_dir": _to_relative(webp_dir),
                    "note": "WRF adapter parsed NetCDF and generated transparent WebP overlays for the frontend map.",
                },
            }

        _validate_wrf_meta(meta, meta_file=None)
        write_meta(meta_file, meta)
        _validate_wrf_meta(meta, meta_file=meta_file)
        return meta
    except Exception:
        shutil.rmtree(webp_dir, ignore_errors=True)
        if meta_file.exists():
            try:
                meta_file.unlink()
            except OSError:
                pass
        raise


def process_files(file_paths: list[str], data_type: str = "WRF") -> dict:
    paths = sorted((Path(item).resolve() for item in file_paths), key=lambda item: item.name)
    metas = [process_file(str(path), data_type=data_type) for path in paths]
    if not metas:
        raise ValueError("No WRF files were provided.")
    if len(metas) == 1:
        return metas[0]

    first = metas[0]
    all_times: list[str] = []
    all_webp_files: list[str] = []
    all_resolution_products: dict[str, dict[str, Any]] = {}
    source_files: list[str] = []
    bboxes = []

    for meta in metas:
        source_files.append(meta.get("source_file", ""))
        all_times.extend(str(item) for item in meta.get("times", []))
        all_webp_files.extend(str(item) for item in meta.get("webp_files", []))
        for key, product in (meta.get("resolution_products") or {}).items():
            target = all_resolution_products.setdefault(
                key,
                {
                    "label": product.get("label", key),
                    "resolution": product.get("resolution", key),
                    "target_meters": product.get("target_meters"),
                    "webp_files": [],
                    "variables": product.get("variables", []),
                    "grid": product.get("grid", ""),
                    "is_resampled": product.get("is_resampled", True),
                },
            )
            target["webp_files"].extend(str(item) for item in product.get("webp_files", []))
            if not target.get("variables") and product.get("variables"):
                target["variables"] = product.get("variables", [])
            if not target.get("grid") and product.get("grid"):
                target["grid"] = product.get("grid")
        if isinstance(meta.get("bbox"), dict):
            bboxes.append(meta["bbox"])

    bbox = first.get("bbox", {})
    if bboxes:
        bbox = {
            "west": min(float(item["west"]) for item in bboxes),
            "south": min(float(item["south"]) for item in bboxes),
            "east": max(float(item["east"]) for item in bboxes),
            "north": max(float(item["north"]) for item in bboxes),
        }

    all_times = sorted(set(all_times))
    weather_info = dict(first.get("weather_info", {}))
    default_resolution = first.get("default_resolution") or _default_resolution_key(all_resolution_products)
    default_product = all_resolution_products.get(default_resolution, {})
    if default_product.get("webp_files"):
        all_webp_files = list(default_product["webp_files"])
    weather_info.update(
        {
            "time": f"{all_times[0].replace('_', ' ')} - {all_times[-1].replace('_', ' ')}",
            "steps": str(len(all_times)),
            "displayResolution": default_resolution,
            "status": "parsed",
            "update": datetime.now(timezone.utc).isoformat(),
        }
    )

    batch_id = f"{paths[0].parent.name}_{all_times[0]}_{all_times[-1]}".replace(":", "_")
    meta_file = paths[0].parent / f"{batch_id}.folder.meta.json"
    combined = {
        "dataset_id": batch_id,
        "data_type": data_type,
        "file_format": "NC",
        "source_file": source_files[0],
        "source_files": source_files,
        "meta_file": meta_file.as_posix(),
        "webp_files": all_webp_files,
        "default_resolution": default_resolution,
        "source_resolution": first.get("source_resolution"),
        "resolution_products": all_resolution_products,
        "variables": default_product.get("variables") or first.get("variables", []),
        "variable_information": first.get("variable_information", []),
        "times": all_times,
        "levels": first.get("levels", ["surface_or_level_0"]),
        "bbox": bbox,
        "weather_info": weather_info,
        "extra": {
            **first.get("extra", {}),
            "status": "parsed",
            "file_count": len(metas),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    try:
        _validate_wrf_meta(combined, meta_file=None)
        write_meta(meta_file, combined)
        _validate_wrf_meta(combined, meta_file=meta_file)
    except Exception:
        if meta_file.exists():
            try:
                meta_file.unlink()
            except OSError:
                pass
        raise
    return combined

