import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


SUPPORTED_TARGET_RESOLUTIONS = [0.03, 0.01]
RESOLUTION_LABELS = {
    0.03: ("3km", "显示插值 3km"),
    0.01: ("1km", "显示插值 1km"),
}


def get_target_resolutions(src_resolution: float) -> list[float]:
    source = float(src_resolution)
    if source <= 0.01:
        return []
    return [resolution for resolution in SUPPORTED_TARGET_RESOLUTIONS if resolution < source]


def resolution_key_for_target(target_resolution: float) -> str:
    value = round(float(target_resolution), 6)
    if value in RESOLUTION_LABELS:
        return RESOLUTION_LABELS[value][0]
    return f"{value:g}deg"


def resolution_label_for_key(key: str) -> str:
    for option_key, label in RESOLUTION_LABELS.values():
        if option_key == key:
            return label
    return key


def resolution_for_key(key: str) -> float | None:
    for resolution, (option_key, _label) in RESOLUTION_LABELS.items():
        if option_key == key:
            return resolution
    return None


def build_resolution_options(src_resolution: float, assets: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    options = [{"key": "original", "label": "原始", "resolution": float(src_resolution)}]
    available_assets = assets or {}
    for target in get_target_resolutions(src_resolution):
        key = resolution_key_for_target(target)
        if key not in available_assets:
            continue
        options.append({"key": key, "label": resolution_label_for_key(key), "resolution": target})
    return options


def resample_to_resolution(
    data: np.ndarray,
    src_grid: dict[str, Any],
    target_resolution: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(data, dtype=np.float32)
    west, south, east, north = [float(item) for item in src_grid["extent"]]
    resolution = float(target_resolution)
    lon = np.arange(west, east + resolution * 0.5, resolution, dtype=np.float32)
    lat = np.arange(south, north + resolution * 0.5, resolution, dtype=np.float32)

    src_ny, src_nx = values.shape[:2]
    src_lon = np.linspace(west, east, src_nx, dtype=np.float32)
    src_lat = np.linspace(south, north, src_ny, dtype=np.float32)
    target_lon, target_lat = np.meshgrid(lon, lat)

    x = (target_lon - src_lon[0]) / max(src_lon[-1] - src_lon[0], 1e-6) * (src_nx - 1)
    y = (target_lat - src_lat[0]) / max(src_lat[-1] - src_lat[0], 1e-6) * (src_ny - 1)
    sampled = map_coordinates(values, [y.ravel(), x.ravel()], order=1, mode="nearest").reshape(lat.size, lon.size)
    grid = {
        "extent": [west, south, east, north],
        "resolution": resolution,
        "nx": int(lon.size),
        "ny": int(lat.size),
        "projection": src_grid.get("projection", "EPSG:4326"),
        "grid_type": src_grid.get("grid_type", "latlon"),
    }
    return sampled.astype(np.float32), grid


def build_dataset_id(source_file: Path) -> str:
    return source_file.name.replace(".", "_")


def write_meta(meta_file: Path, meta: dict[str, Any]) -> None:
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    with meta_file.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def process_basic_file(
    file_path: str,
    data_type: str,
    file_format: str,
    weather_info: dict[str, Any],
) -> dict[str, Any]:
    source_file = Path(file_path).resolve()
    meta_file = source_file.with_name(f"{source_file.name}.meta.json")

    meta = {
        "dataset_id": build_dataset_id(source_file),
        "data_type": data_type,
        "file_format": file_format,
        "source_file": source_file.as_posix(),
        "meta_file": meta_file.as_posix(),
        "png_files": [],
        "variables": [],
        "times": [],
        "levels": [],
        "bbox": None,
        "weather_info": weather_info,
        "extra": {
            "status": "placeholder",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "请在对应 adapter 中补充真实解析逻辑和 PNG 生成逻辑。",
        },
    }

    write_meta(meta_file, meta)
    return meta
