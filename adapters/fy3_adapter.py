from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import h5py
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator

from adapters.base import (
    build_resolution_options,
    get_target_resolutions,
    resample_to_resolution,
    resolution_key_for_target,
    resolution_label_for_key,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "FY3"
DEFAULT_BUSINESS_EXTENT = [73.0, 15.0, 135.0, 55.0]
SCIENCE_RE = re.compile(r"FY3(?P<sat>[A-Z])_MERSI_GBAL_L1_(?P<date>\d{8})_(?P<time>\d{4})_1000M_MS\.HDF$", re.I)
GEO_RE = re.compile(r"FY3(?P<sat>[A-Z])_MERSI_GBAL_L1_(?P<date>\d{8})_(?P<time>\d{4})_GEO1K_MS\.HDF$", re.I)
DEFAULT_TARGET_RESOLUTION = 0.25
MAX_INTERPOLATION_POINTS = 300_000
MAX_DIFF_GRID_CELLS = int(os.environ.get("FY3_MAX_DIFF_GRID_CELLS", "50000000"))
DISPLAY_IMAGE_FORMAT = "WEBP"

ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(progress_callback: ProgressCallback | None, **event: Any) -> None:
    if not progress_callback:
        return
    try:
        progress_callback(event)
    except Exception:
        pass


def _scene_meta_status(scene_dir: Path) -> tuple[bool, str | None, dict[str, Any] | None]:
    meta_path = scene_dir / "meta" / "scene.meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None, None
    return True, str(meta.get("status") or "parsed"), meta.get("quality")


def _is_displayable_quality(quality: dict[str, Any] | None, status: str | None = None) -> bool:
    if status == "no_coverage":
        return False
    try:
        return float((quality or {}).get("valid_pixel_ratio", 0)) >= 0.01
    except (TypeError, ValueError):
        return False

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


def _scene_key(scene_id: str, satellite: str) -> str:
    """Return the collision-safe identity while keeping the legacy scene_id public."""
    return f"{satellite.replace('-', '')}_{scene_id}"


def _published_scene_status(
    root: Path,
    scene_id: str,
    satellite: str,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    date, time = scene_id.split("_", 1)
    candidates = [root / date / time / "meta" / "scene.meta.json"]
    assets_root = root / "assets"
    if assets_root.is_dir():
        candidates.extend(assets_root.glob("*/*/*/meta/scene.meta.json"))
    for meta_path in candidates:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(meta.get("scene_id") or "") != scene_id:
            continue
        if str(meta.get("satellite") or "").upper() != satellite.upper():
            continue
        return True, str(meta.get("status") or "parsed"), meta.get("quality")
    return False, None, None


def pair_fy3_files(paths: Iterable[str | Path]) -> FY3FilePair:
    scenes: dict[tuple[str, str], dict[str, Path | str]] = {}
    for item in paths:
        path = Path(item)
        parsed = _match_scene(path)
        if not parsed:
            continue
        scene_id, satellite, role = parsed
        scenes.setdefault((satellite, scene_id), {"satellite": satellite})[role] = path

    for satellite, scene_id in sorted(scenes):
        scene = scenes[(satellite, scene_id)]
        if scene.get("science") and scene.get("geo"):
            return FY3FilePair(
                science_path=Path(scene["science"]),
                geo_path=Path(scene["geo"]),
                scene_id=scene_id,
                satellite=str(scene["satellite"]),
            )
    raise ValueError("FY-3 解析需要同时提供 1000M_MS.HDF 与 GEO1K_MS.HDF 配对文件。")


def validate_scene_inputs(pair: FY3FilePair, bands: list[int] | None = None) -> None:
    """Fail before creating products when a FY-3 pair is incomplete or empty."""
    selected_bands = bands or CORE_BANDS
    try:
        with h5py.File(pair.geo_path, "r") as geo_hdf:
            if "Geolocation/Latitude" not in geo_hdf or "Geolocation/Longitude" not in geo_hdf:
                raise ValueError("FY-3 GEO 文件缺少经纬度变量")
            lat = _scaled_geo(geo_hdf["Geolocation/Latitude"])
            lon = _scaled_geo(geo_hdf["Geolocation/Longitude"])
            if lat.size == 0 or lon.size == 0 or lat.shape != lon.shape or not np.isfinite(lat).any() or not np.isfinite(lon).any():
                raise ValueError("FY-3 GEO 经纬度为空、形状不一致或不含有效值")
        with h5py.File(pair.science_path, "r") as science_hdf:
            for band in selected_bands:
                values = _read_calibrated_band(science_hdf, band)
                if values.size == 0 or not np.isfinite(values).any():
                    raise ValueError(f"FY-3 通道 B{band:02d} 为空或不含有效值")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"FY-3 场景入口校验失败：{type(exc).__name__}: {exc}") from exc


def discover_fy3_pairs(source_dir: str | Path) -> list[FY3FilePair]:
    root = Path(source_dir).expanduser()
    if not root.exists():
        raise ValueError(f"FY-3 数据目录不存在：{root}")

    scenes: dict[tuple[str, str], dict[str, Path | str]] = {}
    for path in _fy3_raw_files(root):
        parsed = _match_scene(path)
        if not parsed:
            continue
        scene_id, satellite, role = parsed
        scenes.setdefault((satellite, scene_id), {"satellite": satellite})[role] = path

    pairs: list[FY3FilePair] = []
    for satellite, scene_id in sorted(scenes):
        scene = scenes[(satellite, scene_id)]
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


def _fy3_raw_files(source_dir: str | Path) -> list[Path]:
    """返回 canonical 根 raw 文件，并兼容读取旧的 日期/时次/raw 目录。"""
    root = Path(source_dir).expanduser()
    if root.is_file():
        return [root] if is_fy3_filename(root.name) else []

    search_dirs: list[Path] = []
    if root.name == "raw":
        search_dirs.append(root)
    else:
        search_dirs.extend([root / "raw", root])
        search_dirs.extend(sorted(root.glob("*/*/raw")))

    files_by_name: dict[str, Path] = {}
    seen_dirs: set[Path] = set()
    for directory in search_dirs:
        try:
            directory_key = directory.resolve()
        except OSError:
            directory_key = directory
        if directory_key in seen_dirs or not directory.is_dir():
            continue
        seen_dirs.add(directory_key)
        for path in sorted(directory.glob("*.HDF")):
            if is_fy3_filename(path.name):
                files_by_name.setdefault(path.name, path)
    return sorted(files_by_name.values(), key=lambda item: item.name)


def scan_raw_scenes(source_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    root = Path(source_dir).expanduser()
    if not root.exists():
        return []

    scenes: dict[tuple[str, str], dict[str, Any]] = {}
    for path in _fy3_raw_files(root):
        parsed = _match_scene(path)
        if not parsed:
            continue
        scene_id, satellite, role = parsed
        identity = (satellite, scene_id)
        scene = scenes.setdefault(
            identity,
            {
                "business_type": "FY3",
                "scene_id": scene_id,
                "scene_key": _scene_key(scene_id, satellite),
                "satellite": satellite,
                "date": scene_id.split("_")[0],
                "time": scene_id.split("_")[1],
                "raw_dir": path.parent.as_posix(),
                "files": [],
                "roles": set(),
            },
        )
        scene["files"].append(path.name)
        scene["roles"].add(role)
        if role == "science":
            scene["science_file"] = path.name
        elif role == "geo":
            scene["geo_file"] = path.name

    result: list[dict[str, Any]] = []
    for satellite, scene_id in sorted(scenes):
        scene = scenes[(satellite, scene_id)]
        roles = set(scene.pop("roles"))
        missing = [role for role in ("science", "geo") if role not in roles]
        parsed, meta_status, quality = _published_scene_status(root, scene_id, satellite)
        status = meta_status if parsed else ("ready_to_parse" if not missing else "raw_incomplete")
        result.append(
            {
                **scene,
                "file_count": len(scene["files"]),
                "complete": not missing,
                "missing": missing,
                "parsed": parsed,
                "status": status,
                "quality": quality,
                "displayable": parsed and _is_displayable_quality(quality, meta_status),
            }
        )
    return result


def select_upload_files(files: list[Any]) -> list[Any]:
    by_scene: dict[tuple[str, str], dict[str, Any]] = {}
    for item in files:
        parsed = _match_scene(Path(item.filename or ""))
        if not parsed:
            continue
        scene_id, satellite, role = parsed
        by_scene.setdefault((satellite, scene_id), {})[role] = item
    for scene in by_scene.values():
        if scene.get("science") and scene.get("geo"):
            return [scene["science"], scene["geo"]]
    return files


def upload_target_dir(filename: str, target_dir: Path) -> Path:
    return Path(target_dir) / "raw" if _match_scene(Path(filename)) else Path(target_dir)


def _scene_relative(path: Path, scene_dir: Path) -> str:
    try:
        return path.relative_to(scene_dir).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), start=scene_dir.resolve())).as_posix()


def _configured_extent() -> list[float]:
    raw = os.environ.get("FY3_EXTENT", "").strip()
    if not raw:
        return list(DEFAULT_BUSINESS_EXTENT)
    try:
        values = [float(item.strip()) for item in raw.split(",")]
    except ValueError:
        return list(DEFAULT_BUSINESS_EXTENT)
    if len(values) != 4:
        return list(DEFAULT_BUSINESS_EXTENT)
    west, south, east, north = values
    if west >= east or south >= north:
        return list(DEFAULT_BUSINESS_EXTENT)
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


def _target_extent(_lon: np.ndarray, _lat: np.ndarray) -> list[float]:
    """返回固定业务网格范围，避免极轨条带跨日期变更线时扩展成全球网格。"""
    return _configured_extent()


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
        "webp": variable.get("webp"),
        "image": variable.get("webp"),
        "grid": {"nx": grid["nx"], "ny": grid["ny"], "resolution": grid["resolution"]},
        "extent": grid["extent"],
        "resolution": grid["resolution"],
        "spatial_resolution": _format_degree_resolution(grid["resolution"]),
        "derived": False,
        "method": "source_grid",
        "source_resolution": grid["resolution"],
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
    if not np.isfinite(data).any():
        return assets
    for target_resolution in get_target_resolutions(float(grid["resolution"])):
        if _target_cell_count(grid["extent"], target_resolution) > MAX_DIFF_GRID_CELLS:
            continue
        key = resolution_key_for_target(target_resolution)
        resampled, target_grid = resample_to_resolution(data, grid, target_resolution)
        latlon_dir = scene_dir / "diff" / key / "latlon"
        webp_path = latlon_dir / f"{band_name}.webp"
        _write_webp(resampled, webp_path, vmin, vmax)
        assets[key] = {
            "key": key,
            "label": resolution_label_for_key(key),
            "webp": _scene_relative(webp_path, scene_dir),
            "image": _scene_relative(webp_path, scene_dir),
            "grid": {"nx": target_grid["nx"], "ny": target_grid["ny"], "resolution": target_grid["resolution"]},
            "extent": target_grid["extent"],
            "resolution": target_grid["resolution"],
            "spatial_resolution": _format_degree_resolution(target_grid["resolution"]),
            "derived": True,
            "method": "bilinear",
            "source_resolution": grid["resolution"],
        }
    return assets


def _quality(data: np.ndarray) -> dict[str, Any]:
    ratio = float(np.isfinite(data).sum() / data.size) if data.size else 0.0
    warnings = []
    if ratio < 0.01:
        warnings.append("有效像素不足 1%")
    return {"valid_pixel_ratio": ratio, "warnings": warnings}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    part_path = path.with_name(f"{path.name}.part")
    part_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    part_path.replace(path)


def _scene_dirs(
    pair: FY3FilePair,
    output_root: str | Path = DATA_DIR,
) -> tuple[Path, Path, Path]:
    date, time = pair.scene_id.split("_")
    scene_dir = Path(output_root) / date / time
    return scene_dir, scene_dir / "latlon", scene_dir / "meta"


def _band_names(bands: list[int] | None) -> list[str]:
    return [f"B{band:02d}" for band in (bands or CORE_BANDS)]


def _read_scene_meta(
    pair: FY3FilePair,
    output_root: str | Path = DATA_DIR,
) -> dict[str, Any] | None:
    _scene_dir, _latlon_dir, meta_dir = _scene_dirs(pair, output_root)
    meta_path = meta_dir / "scene.meta.json"
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cached_scene_matches(
    pair: FY3FilePair,
    target_resolution: float | None,
    bands: list[int] | None,
    output_root: str | Path = DATA_DIR,
) -> dict[str, Any] | None:
    meta = _read_scene_meta(pair, output_root)
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
        scene_dir, _latlon_dir, _meta_dir = _scene_dirs(pair, output_root)
        if not (scene_dir / "latlon" / f"{band_name}.webp").exists():
            return None
    return meta


def process_files(
    paths: list[str],
    data_type: str = "FY3",
    output_root: str | Path = DATA_DIR,
    target_resolution: float | None = None,
    bands: list[int] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    pair = pair_fy3_files(paths)
    validate_scene_inputs(pair, bands)
    resolution = target_resolution or _configured_resolution()
    scene_dir, latlon_dir, meta_dir = _scene_dirs(pair, output_root)
    latlon_dir.mkdir(parents=True, exist_ok=True)
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
        for band_index, band in enumerate(selected_bands, start=1):
            band_name = f"B{band:02d}"
            _emit_progress(
                progress_callback,
                stage="processing_band",
                scene_id=pair.scene_id,
                band=band_name,
                band_index=band_index,
                band_total=len(selected_bands),
            )
            data = _read_calibrated_band(science_hdf, band)
            points, values = _sample_valid_points(lon, lat, data, extent, sensor_zenith)
            gridded = _interpolate_to_grid(points, values, grid)
            webp_path = latlon_dir / f"{band_name}.webp"
            name_cn, unit, vmin, vmax = BAND_META[band]
            _write_webp(gridded, webp_path, vmin, vmax)
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
                "webp": _scene_relative(webp_path, scene_dir),
                "image": _scene_relative(webp_path, scene_dir),
                "quality": band_quality,
            }
            resolution_assets = {"original": _base_resolution_asset(variable, grid)}
            resolution_assets.update(_write_diff_resolution_assets(scene_dir, band_name, gridded, grid, vmin, vmax))
            variable["resolution_assets"] = resolution_assets
            variables.append(variable)
            _emit_progress(
                progress_callback,
                stage="band_completed",
                scene_id=pair.scene_id,
                band=band_name,
                band_index=band_index,
                band_total=len(selected_bands),
            )

    observed = datetime.strptime(pair.scene_id, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
    resolution_assets = variables[0].get("resolution_assets") if variables else {}
    resolution_options = build_resolution_options(grid["resolution"], resolution_assets)
    valid_pixel_ratio = float(np.mean(qualities)) if qualities else 0.0
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
        "temporal_resolution": "轨道过境 / 不定",
        "data_format": "HDF5",
        "file_format": "HDF5",
        "resolutions": {item["key"]: item["resolution"] for item in resolution_options},
        "resolution_options": resolution_options,
        "grid": {"nx": grid["nx"], "ny": grid["ny"]},
        "variables": variables,
        "composites": [],
        "loaded_bands": [item["name"] for item in variables],
        "status": "parsed" if valid_pixel_ratio >= 0.01 else "no_coverage",
        "quality": {
            "valid_pixel_ratio": valid_pixel_ratio,
            "warnings": [] if valid_pixel_ratio >= 0.01 else ["轨迹未覆盖当前固定业务区域"],
        },
        "source_raw_dir": _scene_relative(pair.science_path.parent, scene_dir),
        "source_files": [_scene_relative(pair.science_path, scene_dir), _scene_relative(pair.geo_path, scene_dir)],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(meta_dir / "scene.meta.json", meta)
    return meta


def process_pair(
    pair: FY3FilePair,
    data_type: str = "FY3",
    target_resolution: float | None = None,
    bands: list[int] | None = None,
    progress_callback: ProgressCallback | None = None,
    output_root: str | Path = DATA_DIR,
) -> dict[str, Any]:
    return process_files(
        [pair.science_path.as_posix(), pair.geo_path.as_posix()],
        data_type=data_type,
        output_root=output_root,
        target_resolution=target_resolution,
        bands=bands,
        progress_callback=progress_callback,
    )


def process_directory(
    source_dir: str | Path,
    target_resolution: float | None = None,
    bands: list[int] | None = None,
    force: bool = False,
    scene_ids: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    pairs = discover_fy3_pairs(source_dir)
    requested_scene_ids = None if scene_ids is None else {str(item) for item in scene_ids if str(item)}
    if requested_scene_ids is not None:
        pairs = [pair for pair in pairs if pair.scene_id in requested_scene_ids]
    results: list[dict[str, Any]] = []
    failed = 0
    cached = 0
    pair_total = len(pairs)
    for pair_index, pair in enumerate(pairs, start=1):
        _emit_progress(
            progress_callback,
            stage="processing_scene",
            scene_id=pair.scene_id,
            scene_index=pair_index,
            scene_total=pair_total,
        )
        try:
            if not force and (cached_meta := _cached_scene_matches(pair, target_resolution, bands)):
                cached += 1
                quality = cached_meta.get("quality")
                meta_status = str(cached_meta.get("status") or "parsed")
                item = {
                    "scene_id": pair.scene_id,
                    "status": "cached",
                    "meta_status": meta_status,
                    "displayable": _is_displayable_quality(quality, meta_status),
                    "bands": cached_meta.get("loaded_bands", []),
                    "extent": cached_meta.get("extent"),
                    "resolution": cached_meta.get("resolution"),
                    "quality": quality,
                }
                results.append(item)
                _emit_progress(
                    progress_callback,
                    stage="scene_completed",
                    scene_id=pair.scene_id,
                    scene_index=pair_index,
                    scene_total=pair_total,
                    result=item,
                )
                continue
            def pair_progress(event: dict[str, Any]) -> None:
                _emit_progress(
                    progress_callback,
                    **event,
                    scene_index=pair_index,
                    scene_total=pair_total,
                )

            meta = process_pair(
                pair,
                target_resolution=target_resolution,
                bands=bands,
                progress_callback=pair_progress,
            )
            quality = meta.get("quality")
            meta_status = str(meta.get("status") or "parsed")
            item = {
                "scene_id": pair.scene_id,
                "status": "ok",
                "meta_status": meta_status,
                "displayable": _is_displayable_quality(quality, meta_status),
                "bands": meta.get("loaded_bands", []),
                "extent": meta.get("extent"),
                "resolution": meta.get("resolution"),
                "quality": quality,
            }
            results.append(item)
            _emit_progress(
                progress_callback,
                stage="scene_completed",
                scene_id=pair.scene_id,
                scene_index=pair_index,
                scene_total=pair_total,
                result=item,
            )
        except Exception as exc:
            failed += 1
            item = {"scene_id": pair.scene_id, "status": "error", "error": str(exc), "displayable": False}
            results.append(item)
            _emit_progress(
                progress_callback,
                stage="scene_failed",
                scene_id=pair.scene_id,
                scene_index=pair_index,
                scene_total=pair_total,
                error=str(exc),
            )
    return {
        "source_dir": Path(source_dir).expanduser().as_posix(),
        "pair_count": len(pairs),
        "processed": len([item for item in results if item["status"] == "ok"]),
        "cached": cached,
        "failed": failed,
        "results": results,
    }


def update_from_raw(
    source_dir: str | Path = DATA_DIR,
    target_resolution: float | None = None,
    bands: list[int] | None = None,
    force: bool = False,
    scene_ids: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    scenes = scan_raw_scenes(source_dir)
    requested_scene_ids = None if scene_ids is None else {str(item) for item in scene_ids if str(item)}
    if requested_scene_ids is not None:
        scenes = [scene for scene in scenes if scene["scene_id"] in requested_scene_ids]
    incomplete = [scene for scene in scenes if not scene["complete"]]
    ready_ids = [scene["scene_id"] for scene in scenes if scene["complete"]]
    result = process_directory(
        source_dir,
        target_resolution=target_resolution,
        bands=bands,
        force=force,
        scene_ids=ready_ids if requested_scene_ids is not None else None,
        progress_callback=progress_callback,
    )
    return {
        **result,
        "scene_count": len(scenes),
        "ready_count": len([scene for scene in scenes if scene["complete"]]),
        "incomplete_count": len(incomplete),
        "incomplete": incomplete,
    }


def process_file(path: str, data_type: str = "FY3") -> dict[str, Any]:
    source = Path(path)
    parsed = _match_scene(source)
    if not parsed:
        raise ValueError("不支持的 FY-3 文件名。")
    scene_id, satellite, _role = parsed
    candidates = [
        item
        for item in source.parent.glob("*.HDF")
        if (matched := _match_scene(item)) and matched[0] == scene_id and matched[1] == satellite
    ]
    if len(candidates) < 2:
        candidates.extend(
            item
            for item in _fy3_raw_files(DATA_DIR)
            if (matched := _match_scene(item)) and matched[0] == scene_id and matched[1] == satellite
        )
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
