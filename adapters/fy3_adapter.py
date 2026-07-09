from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from adapters import diff_methods

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "FY3"
GLOBAL_EXTENT = [-180.0, -90.0, 180.0, 90.0]
SCIENCE_RE = re.compile(r"FY3(?P<sat>[A-Z])_MERSI_GBAL_L1_(?P<date>\d{8})_(?P<time>\d{4})_1000M_MS\.HDF$", re.I)
GEO_RE = re.compile(r"FY3(?P<sat>[A-Z])_MERSI_GBAL_L1_(?P<date>\d{8})_(?P<time>\d{4})_GEO1K_MS\.HDF$", re.I)
DEFAULT_TARGET_RESOLUTION = 0.25
MAX_INTERPOLATION_POINTS = 300_000
MAX_DIFF_GRID_CELLS = int(os.environ.get("FY3_MAX_DIFF_GRID_CELLS", "50000000"))
DISPLAY_IMAGE_FORMAT = "WEBP"

ALL_BANDS = list(range(1, 26))
CORE_BANDS = ALL_BANDS
BAND_DATASETS = {
    **{band: ("Data/EV_250_Aggr.1KM_RefSB", band - 1, "reflective") for band in range(1, 5)},
    **{band: ("Data/EV_1KM_RefSB", band - 5, "reflective") for band in range(5, 20)},
    **{band: ("Data/EV_1KM_Emissive", band - 20, "emissive") for band in range(20, 24)},
    **{band: ("Data/EV_250_Aggr.1KM_Emissive", band - 24, "emissive") for band in range(24, 26)},
}
BAND_META = {
    1: ("蓝光可见光", "%", 0, 100),
    2: ("绿光可见光", "%", 0, 100),
    3: ("红光可见光", "%", 0, 100),
    4: ("近红外", "%", 0, 100),
    5: ("短波红外", "%", 0, 100),
    6: ("近红外", "%", 0, 100),
    7: ("近红外", "%", 0, 100),
    8: ("近红外", "%", 0, 100),
    9: ("近红外", "%", 0, 100),
    10: ("近红外", "%", 0, 100),
    11: ("短波红外", "%", 0, 100),
    12: ("短波红外", "%", 0, 100),
    13: ("短波红外", "%", 0, 100),
    14: ("短波红外", "%", 0, 100),
    15: ("短波红外", "%", 0, 100),
    16: ("短波红外", "%", 0, 100),
    17: ("短波红外", "%", 0, 100),
    18: ("短波红外", "%", 0, 100),
    19: ("短波红外", "%", 0, 100),
    20: ("红外窗口", "mW/(m2 cm-1 sr)", 0, 120),
    21: ("红外窗口", "mW/(m2 cm-1 sr)", 0, 120),
    22: ("红外探测", "mW/(m2 cm-1 sr)", 0, 120),
    23: ("红外探测", "mW/(m2 cm-1 sr)", 0, 120),
    24: ("红外辅助", "mW/(m2 cm-1 sr)", 0, 300),
    25: ("红外辅助", "mW/(m2 cm-1 sr)", 0, 300),
}


@dataclass(frozen=True)
class FY3FilePair:
    science_path: Path
    geo_path: Path
    scene_id: str
    satellite: str


def is_fy3_filename(filename: str) -> bool:
    name = Path(filename).name
    return bool(SCIENCE_RE.match(name) or GEO_RE.match(name))


def _match_scene(path: Path) -> tuple[str, str, str] | None:
    match = SCIENCE_RE.match(path.name) or GEO_RE.match(path.name)
    if not match:
        return None
    date = match.group("date")
    time = match.group("time")
    sat = f"FY-3{match.group('sat').upper()}"
    return f"{date}_{time}", sat, "geo" if GEO_RE.match(path.name) else "science"


def pair_fy3_files(paths: Iterable[str | Path]) -> FY3FilePair:
    scenes: dict[str, dict[str, Path | str]] = {}
    for item in paths:
        path = Path(item)
        parsed = _match_scene(path)
        if not parsed:
            continue
        scene_id, satellite, role = parsed
        scenes.setdefault(scene_id, {"satellite": satellite})[role] = path

    for scene_id in sorted(scenes):
        scene = scenes[scene_id]
        if scene.get("science") and scene.get("geo"):
            return FY3FilePair(
                science_path=Path(scene["science"]),
                geo_path=Path(scene["geo"]),
                scene_id=scene_id,
                satellite=str(scene["satellite"]),
            )
    raise ValueError("FY-3 解析需要同时提供 1000M_MS.HDF 与 GEO1K_MS.HDF 配对文件。")


def discover_fy3_pairs(source_dir: str | Path) -> list[FY3FilePair]:
    root = Path(source_dir).expanduser()
    if not root.exists():
        raise ValueError(f"FY-3 数据目录不存在：{root}")

    scenes: dict[str, dict[str, Path | str]] = {}
    for path in root.rglob("*.HDF"):
        parsed = _match_scene(path)
        if not parsed:
            continue
        scene_id, satellite, role = parsed
        scenes.setdefault(scene_id, {"satellite": satellite})[role] = path

    pairs: list[FY3FilePair] = []
    for scene_id in sorted(scenes):
        scene = scenes[scene_id]
        if not scene.get("science") or not scene.get("geo"):
            continue
        pairs.append(
            FY3FilePair(
                science_path=Path(scene["science"]),
                geo_path=Path(scene["geo"]),
                scene_id=scene_id,
                satellite=str(scene["satellite"]),
            )
        )
    return pairs


def select_upload_files(files: list[Any]) -> list[Any]:
    by_scene: dict[str, dict[str, Any]] = {}
    for item in files:
        parsed = _match_scene(Path(item.filename or ""))
        if not parsed:
            continue
        scene_id, _satellite, role = parsed
        by_scene.setdefault(scene_id, {})[role] = item
    for scene in by_scene.values():
        if scene.get("science") and scene.get("geo"):
            return [scene["science"], scene["geo"]]
    return files


def upload_target_dir(filename: str, target_dir: Path) -> Path:
    parsed = _match_scene(Path(filename))
    if not parsed:
        return target_dir
    scene_id, _satellite, _role = parsed
    date, time = scene_id.split("_")
    return target_dir / date / time / "raw"


def _scene_relative(path: Path, scene_dir: Path) -> str:
    try:
        return path.relative_to(scene_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _configured_extent() -> list[float] | None:
    raw = os.environ.get("FY3_EXTENT", "").strip()
    if not raw:
        return None
    try:
        values = [float(item.strip()) for item in raw.split(",")]
    except ValueError:
        return None
    if len(values) != 4:
        return None
    west, south, east, north = values
    if west >= east or south >= north:
        return None
    return values


def _configured_resolution() -> float:
    try:
        return max(0.01, float(os.environ.get("FY3_TARGET_RESOLUTION", DEFAULT_TARGET_RESOLUTION)))
    except ValueError:
        return DEFAULT_TARGET_RESOLUTION


def _format_degree_resolution(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):g}°"


def _dataset_scale(dataset: h5py.Dataset, index: int) -> tuple[float, float]:
    slope = dataset.attrs.get("Slope")
    intercept = dataset.attrs.get("Intercept")
    slope_value = float(np.asarray(slope).reshape(-1)[index]) if slope is not None else 1.0
    intercept_value = float(np.asarray(intercept).reshape(-1)[index]) if intercept is not None else 0.0
    return slope_value, intercept_value


def _valid_range(dataset: h5py.Dataset) -> tuple[float, float] | None:
    value = dataset.attrs.get("valid_range")
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size < 2:
        return None
    return float(arr[0]), float(arr[1])


def _fill_values(dataset: h5py.Dataset) -> set[float]:
    values = {65535.0, 65534.0, 65533.0}
    fill = dataset.attrs.get("FillValue")
    if fill is not None:
        values.update(float(item) for item in np.asarray(fill).reshape(-1))
    return values


def _calibrate_reflective(raw: np.ndarray, hdf: h5py.File, band: int, dataset: h5py.Dataset, index: int) -> np.ndarray:
    coeffs_ds = hdf.get("Calibration/VIS_Cal_Coeff")
    if coeffs_ds is not None and coeffs_ds.shape[0] >= band:
        k0, k1, k2 = [float(value) for value in coeffs_ds[band - 1, :3]]
        return k0 + k1 * raw + k2 * raw * raw
    slope, intercept = _dataset_scale(dataset, index)
    return intercept + slope * raw


def _calibrate_emissive(raw: np.ndarray, hdf: h5py.File, band: int, dataset: h5py.Dataset, index: int) -> np.ndarray:
    coeffs_ds = hdf.get("Calibration/IR_Cal_Coeff")
    coeff_index = band - 20
    if coeffs_ds is not None and 0 <= coeff_index < coeffs_ds.shape[0]:
        coeffs = np.asarray(coeffs_ds[coeff_index, :, 0], dtype=np.float32)
        k0, k1, k2, k3 = [float(value) for value in coeffs[:4]]
        return k0 + k1 * raw + k2 * raw * raw + k3 * raw * raw * raw
    slope, intercept = _dataset_scale(dataset, index)
    return intercept + slope * raw


def _read_calibrated_band(hdf: h5py.File, band: int) -> np.ndarray:
    dataset_name, index, kind = BAND_DATASETS[band]
    dataset = hdf[dataset_name]
    raw = np.asarray(dataset[index], dtype=np.float32)
    valid = np.isfinite(raw)
    if valid_range := _valid_range(dataset):
        low, high = valid_range
        valid &= (raw >= low) & (raw <= high)
    for fill in _fill_values(dataset):
        valid &= raw != fill
    raw = np.where(valid, raw, np.nan).astype(np.float32)
    if kind == "reflective":
        values = _calibrate_reflective(raw, hdf, band, dataset, index)
    else:
        values = _calibrate_emissive(raw, hdf, band, dataset, index)
    return np.asarray(values, dtype=np.float32)


def _scaled_geo(dataset: h5py.Dataset) -> np.ndarray:
    raw = np.asarray(dataset[:], dtype=np.float32)
    slope = dataset.attrs.get("Slope")
    intercept = dataset.attrs.get("Intercept")
    if slope is not None:
        raw = raw * float(np.asarray(slope).reshape(-1)[0])
    if intercept is not None:
        raw = raw + float(np.asarray(intercept).reshape(-1)[0])
    return raw


def _target_extent(lon: np.ndarray, lat: np.ndarray) -> list[float]:
    """返回目标网格的地理范围。优先 FY3_EXTENT 环境变量，否则从 swath 数据自动推算实际范围。"""
    configured = _configured_extent()
    if configured:
        return configured
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        return list(GLOBAL_EXTENT)
    lon_valid = lon[valid]
    lat_valid = lat[valid]
    margin = 0.5
    west = max(float(lon_valid.min()) - margin, -180.0)
    east = min(float(lon_valid.max()) + margin, 180.0)
    south = max(float(lat_valid.min()) - margin, -90.0)
    north = min(float(lat_valid.max()) + margin, 90.0)
    return [west, south, east, north]


def _build_grid(extent: list[float], resolution: float) -> dict[str, Any]:
    west, south, east, north = extent
    lon = np.arange(west, east + resolution * 0.5, resolution, dtype=np.float32)
    lat = np.arange(south, north + resolution * 0.5, resolution, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return {
        "extent": [float(west), float(south), float(east), float(north)],
        "resolution": float(resolution),
        "lon": lon2d,
        "lat": lat2d,
        "nx": int(lon.size),
        "ny": int(lat.size),
    }


def _sample_valid_points(lon: np.ndarray, lat: np.ndarray, data: np.ndarray, extent: list[float], sensor_zenith: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = extent
    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & np.isfinite(data)
        & (lon >= west)
        & (lon <= east)
        & (lat >= south)
        & (lat <= north)
    )
    if sensor_zenith is not None:
        valid &= np.isfinite(sensor_zenith) & (sensor_zenith < 65.0)
    y, x = np.nonzero(valid)
    if y.size > MAX_INTERPOLATION_POINTS:
        step = int(np.ceil(y.size / MAX_INTERPOLATION_POINTS))
        y = y[::step]
        x = x[::step]
    points = np.column_stack([lon[y, x], lat[y, x]]).astype(np.float32)
    values = data[y, x].astype(np.float32)
    return points, values


def _interpolate_to_grid(points: np.ndarray, values: np.ndarray, grid: dict[str, Any]) -> np.ndarray:
    if points.shape[0] < 3:
        return np.full((grid["ny"], grid["nx"]), np.nan, dtype=np.float32)
    target = np.column_stack([grid["lon"].ravel(), grid["lat"].ravel()])
    try:
        interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
        result = np.asarray(interpolator(target), dtype=np.float32)
    except Exception:
        result = np.full(target.shape[0], np.nan, dtype=np.float32)
    if np.isnan(result).any():
        nearest = NearestNDInterpolator(points, values)
        nearest_values = np.asarray(nearest(target), dtype=np.float32)
        result = np.where(np.isfinite(result), result, nearest_values)
    return result.reshape(grid["ny"], grid["nx"]).astype(np.float32)


def _rgba_from_data(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    alpha = np.isfinite(values)
    # 使用数据实际范围做归一化，避免硬编码 vmin/vmax 与校准数据不匹配导致黑图
    valid = values[alpha]
    if valid.size > 0:
        data_min = float(np.percentile(valid, 2))
        data_max = float(np.percentile(valid, 98))
    else:
        data_min, data_max = float(vmin), float(vmax)
    span = max(data_max - data_min, 1e-6)
    norm = np.clip((values - data_min) / span, 0.0, 1.0)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    gray = (norm * 255).astype(np.uint8)
    rgba = np.zeros((*gray.shape, 4), dtype=np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(alpha, 220, 0).astype(np.uint8)
    return rgba


def _write_webp(data: np.ndarray, output_path: Path, vmin: float, vmax: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgba_from_data(data, vmin, vmax), mode="RGBA").save(output_path, format=DISPLAY_IMAGE_FORMAT, lossless=True)


def _target_cell_count(extent: list[float], resolution: float) -> int:
    west, south, east, north = [float(item) for item in extent]
    nx = int(np.floor((east - west) / resolution + 0.5)) + 1
    ny = int(np.floor((north - south) / resolution + 0.5)) + 1
    return max(nx, 0) * max(ny, 0)


def _base_resolution_asset(variable: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": "original",
        "label": "原始",
        "png": variable.get("png"),
        "float32": variable.get("float32"),
        "grid": {"nx": grid["nx"], "ny": grid["ny"], "resolution": grid["resolution"]},
        "extent": grid["extent"],
        "resolution": grid["resolution"],
        "spatial_resolution": _format_degree_resolution(grid["resolution"]),
    }


def _write_diff_resolution_assets(
    scene_dir: Path,
    band_name: str,
    data: np.ndarray,
    grid: dict[str, Any],
    vmin: float,
    vmax: float,
) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for target_resolution in diff_methods.get_target_resolutions(float(grid["resolution"])):
        if _target_cell_count(grid["extent"], target_resolution) > MAX_DIFF_GRID_CELLS:
            continue
        key = diff_methods.resolution_key_for_target(target_resolution)
        resampled, target_grid = diff_methods.resample_to_resolution(data, grid, target_resolution)
        latlon_dir = scene_dir / "diff" / key / "latlon"
        float32_dir = scene_dir / "diff" / key / "float32"
        png_path = latlon_dir / f"{band_name}.webp"
        float32_path = float32_dir / f"{band_name}.float32"
        float32_path.parent.mkdir(parents=True, exist_ok=True)
        resampled.astype(np.float32).tofile(float32_path)
        _write_webp(resampled, png_path, vmin, vmax)
        assets[key] = {
            "key": key,
            "label": diff_methods.resolution_label_for_key(key),
            "png": _scene_relative(png_path, scene_dir),
            "float32": _scene_relative(float32_path, scene_dir),
            "grid": {"nx": target_grid["nx"], "ny": target_grid["ny"], "resolution": target_grid["resolution"]},
            "extent": target_grid["extent"],
            "resolution": target_grid["resolution"],
            "spatial_resolution": _format_degree_resolution(target_grid["resolution"]),
        }
    return assets


def _quality(data: np.ndarray) -> dict[str, Any]:
    ratio = float(np.isfinite(data).sum() / data.size) if data.size else 0.0
    warnings = []
    if ratio < 0.01:
        warnings.append("有效像素不足 1%")
    return {"valid_pixel_ratio": ratio, "warnings": warnings}


def _scene_dirs(pair: FY3FilePair) -> tuple[Path, Path, Path, Path]:
    date, time = pair.scene_id.split("_")
    scene_dir = DATA_DIR / date / time
    return scene_dir, scene_dir / "latlon", scene_dir / "float32", scene_dir / "meta"


def _band_names(bands: list[int] | None) -> list[str]:
    return [f"B{band:02d}" for band in (bands or CORE_BANDS)]


def _read_scene_meta(pair: FY3FilePair) -> dict[str, Any] | None:
    _scene_dir, _latlon_dir, _float32_dir, meta_dir = _scene_dirs(pair)
    meta_path = meta_dir / "scene.meta.json"
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cached_scene_matches(pair: FY3FilePair, target_resolution: float | None, bands: list[int] | None) -> dict[str, Any] | None:
    meta = _read_scene_meta(pair)
    if not meta:
        return None
    expected_resolution = float(target_resolution or _configured_resolution())
    actual_resolution = float(meta.get("resolution") or 0)
    if abs(actual_resolution - expected_resolution) > 1e-9:
        return None
    expected_bands = set(_band_names(bands))
    loaded_bands = set(str(item) for item in meta.get("loaded_bands", []))
    if not expected_bands.issubset(loaded_bands):
        return None
    for band_name in expected_bands:
        if not (DATA_DIR / pair.scene_id.split("_")[0] / pair.scene_id.split("_")[1] / "latlon" / f"{band_name}.webp").exists():
            return None
    return meta


def process_files(
    paths: list[str],
    data_type: str = "FY3",
    target_resolution: float | None = None,
    bands: list[int] | None = None,
) -> dict[str, Any]:
    pair = pair_fy3_files(paths)
    resolution = target_resolution or _configured_resolution()
    scene_dir, latlon_dir, float32_dir, meta_dir = _scene_dirs(pair)
    latlon_dir.mkdir(parents=True, exist_ok=True)
    float32_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(pair.geo_path, "r") as geo_hdf:
        lat = _scaled_geo(geo_hdf["Geolocation/Latitude"])
        lon = _scaled_geo(geo_hdf["Geolocation/Longitude"])
        sensor_ds = geo_hdf.get("Geolocation/SensorZenith")
        sensor_zenith = _scaled_geo(sensor_ds) if sensor_ds is not None else None

    extent = _target_extent(lon, lat)
    grid = _build_grid(extent, resolution)
    variables: list[dict[str, Any]] = []
    qualities: list[float] = []
    selected_bands = bands or CORE_BANDS
    with h5py.File(pair.science_path, "r") as science_hdf:
        for band in selected_bands:
            data = _read_calibrated_band(science_hdf, band)
            points, values = _sample_valid_points(lon, lat, data, extent, sensor_zenith)
            gridded = _interpolate_to_grid(points, values, grid)
            band_name = f"B{band:02d}"
            float32_path = float32_dir / f"{band_name}.float32"
            png_path = latlon_dir / f"{band_name}.webp"
            gridded.astype(np.float32).tofile(float32_path)
            name_cn, unit, vmin, vmax = BAND_META[band]
            _write_webp(gridded, png_path, vmin, vmax)
            band_quality = _quality(gridded)
            qualities.append(band_quality["valid_pixel_ratio"])
            variable = {
                "name": band_name,
                "name_cn": name_cn,
                "long_name": f"FY-3 MERSI {band_name}",
                "product_type": "variable",
                "unit": unit,
                "display_unit": unit,
                "shape": [grid["ny"], grid["nx"]],
                "dims": ["lat", "lon"],
                "stats": _stats(gridded),
                "vmin": vmin,
                "vmax": vmax,
                "legend_ticks": [f"{vmin:g}", f"{(vmin + vmax) / 2:g}", f"{vmax:g}"],
                "float32": _scene_relative(float32_path, scene_dir),
                "png": _scene_relative(png_path, scene_dir),
                "quality": band_quality,
            }
            resolution_assets = {"original": _base_resolution_asset(variable, grid)}
            resolution_assets.update(_write_diff_resolution_assets(scene_dir, band_name, gridded, grid, vmin, vmax))
            variable["resolution_assets"] = resolution_assets
            variables.append(variable)

    observed = datetime.strptime(pair.scene_id, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
    resolution_assets = variables[0].get("resolution_assets") if variables else {}
    resolution_options = diff_methods.build_resolution_options(grid["resolution"], resolution_assets)
    meta = {
        "business_type": "FY3",
        "data_type": data_type,
        "scene_id": pair.scene_id,
        "satellite": pair.satellite,
        "sensor": "MERSI",
        "observation_time": observed.isoformat().replace("+00:00", "Z"),
        "projection": "EPSG:4326",
        "grid_type": "latlon",
        "extent": grid["extent"],
        "resolution": grid["resolution"],
        "spatial_resolution": _format_degree_resolution(grid["resolution"]),
        "temporal_resolution": "5分钟",
        "data_format": "HDF5",
        "file_format": "HDF5",
        "resolutions": {item["key"]: item["resolution"] for item in resolution_options},
        "resolution_options": resolution_options,
        "grid": {"nx": grid["nx"], "ny": grid["ny"]},
        "variables": variables,
        "composites": [],
        "loaded_bands": [item["name"] for item in variables],
        "quality": {"valid_pixel_ratio": float(np.mean(qualities)) if qualities else 0.0, "warnings": []},
        "source_raw_dir": _scene_relative(pair.science_path.parent, scene_dir),
        "source_files": [_scene_relative(pair.science_path, scene_dir), _scene_relative(pair.geo_path, scene_dir)],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (meta_dir / "scene.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def process_pair(
    pair: FY3FilePair,
    data_type: str = "FY3",
    target_resolution: float | None = None,
    bands: list[int] | None = None,
) -> dict[str, Any]:
    return process_files(
        [pair.science_path.as_posix(), pair.geo_path.as_posix()],
        data_type=data_type,
        target_resolution=target_resolution,
        bands=bands,
    )


def process_directory(
    source_dir: str | Path,
    target_resolution: float | None = None,
    bands: list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    pairs = discover_fy3_pairs(source_dir)
    results: list[dict[str, Any]] = []
    failed = 0
    cached = 0
    for pair in pairs:
        try:
            if not force and (cached_meta := _cached_scene_matches(pair, target_resolution, bands)):
                cached += 1
                results.append(
                    {
                        "scene_id": pair.scene_id,
                        "status": "cached",
                        "bands": cached_meta.get("loaded_bands", []),
                        "extent": cached_meta.get("extent"),
                        "resolution": cached_meta.get("resolution"),
                        "quality": cached_meta.get("quality"),
                    }
                )
                continue
            meta = process_pair(pair, target_resolution=target_resolution, bands=bands)
            results.append(
                {
                    "scene_id": pair.scene_id,
                    "status": "ok",
                    "bands": meta.get("loaded_bands", []),
                    "extent": meta.get("extent"),
                    "resolution": meta.get("resolution"),
                    "quality": meta.get("quality"),
                }
            )
        except Exception as exc:
            failed += 1
            results.append({"scene_id": pair.scene_id, "status": "error", "error": str(exc)})
    return {
        "source_dir": Path(source_dir).expanduser().as_posix(),
        "pair_count": len(pairs),
        "processed": len([item for item in results if item["status"] == "ok"]),
        "cached": cached,
        "failed": failed,
        "results": results,
    }


def process_file(path: str, data_type: str = "FY3") -> dict[str, Any]:
    source = Path(path)
    parsed = _match_scene(source)
    if not parsed:
        raise ValueError("不支持的 FY-3 文件名。")
    scene_id, _satellite, _role = parsed
    candidates = list(source.parent.glob(f"*{scene_id.replace('_', '_')}*.HDF"))
    if len(candidates) < 2:
        candidates.extend(source.parent.parent.glob(f"*/raw/*{scene_id.replace('_', '_')}*.HDF"))
    return process_files([str(item) for item in candidates] + [str(source)], data_type=data_type)


def _stats(data: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(data, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
    }
