from __future__ import annotations

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

VARIABLE_LABELS = {
    "PM2_5_DRY": ("PM2.5 干质量浓度", "近地面细颗粒物浓度，用于空气质量预报展示。"),
    "PM10": ("PM10 颗粒物浓度", "可吸入颗粒物浓度，适合与 PM2.5 联合展示污染过程。"),
    "AOD2D_OUT": ("气溶胶光学厚度", "柱积分气溶胶光学厚度，反映大气浑浊程度。"),
    "T2": ("2 米气温", "近地面 2 米气温，可用于天气背景场展示。"),
    "U10": ("10 米东西风", "10 米高度东西向风速，正值表示偏西风分量。"),
    "V10": ("10 米南北风", "10 米高度南北向风速，正值表示偏南风分量。"),
    "PSFC": ("地面气压", "模式地面气压场，可辅助判断天气系统。"),
    "PBLH": ("边界层高度", "行星边界层高度，对污染扩散能力判断很关键。"),
    "RAINC": ("累积对流降水", "对流降水累积量，用于识别强对流降水贡献。"),
    "RAINNC": ("累积非对流降水", "非对流降水累积量，常用于连续性降水展示。"),
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


def _load_runtime():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import colormaps
        from netCDF4 import Dataset
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "WRF adapter 需要安装 netCDF4、numpy、matplotlib、Pillow 后才能解析 wrfout 并生成 PNG。"
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


def process_file(file_path: str, data_type: str = "WRF") -> dict:
    Dataset, Image, colormaps = _load_runtime()

    source_file = Path(file_path).resolve()
    meta_file = source_file.with_name(f"{source_file.name}.meta.json")
    png_dir = source_file.parent / f"{source_file.name}.pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    with Dataset(source_file) as ds:
        lat, lon = _lat_lon(ds)
        bbox = {
            "west": float(np.nanmin(lon)),
            "south": float(np.nanmin(lat)),
            "east": float(np.nanmax(lon)),
            "north": float(np.nanmax(lat)),
        }
        time_label = _time_label(ds) or source_file.name
        domain = _domain_from_file(source_file, ds)
        dx = float(getattr(ds, "DX", 0) or 0)
        dy = float(getattr(ds, "DY", 0) or 0)
        grid = f"{lat.shape[1]} × {lat.shape[0]}"
        resolution = f"{dx / 1000:g} km" if dx else "未知"

        display_variables = [
            name for name in PRODUCT_VARIABLES
            if name in ds.variables and _can_display_variable(ds, name, lat.shape)
        ]
        if not display_variables:
            display_variables = [
                name for name in ds.variables
                if _can_display_variable(ds, name, lat.shape)
            ][:8]

        variables: list[dict[str, Any]] = []
        png_files: list[str] = []
        primary_stats = {"min": None, "max": None, "mean": None}
        primary_unit = ""
        primary_element = "WRF 变量"

        for name in display_variables:
            data, desc, units, var_id = _field_from_var(ds, name)
            label, business_desc = VARIABLE_LABELS.get(name, (desc or name, desc or name))
            stat = _stats(data)
            variables.append(
                {
                    "name": name,
                    "label": label,
                    "description": business_desc,
                    "units": units,
                    "shape": list(data.shape),
                    **stat,
                }
            )

            image = _render_overlay(data, Image, colormaps)
            png_path = png_dir / f"{time_label.replace(':', '_')}_{var_id}.png"
            image.save(png_path)
            png_files.append(png_path.as_posix())

            if name == display_variables[0]:
                primary_stats = stat
                primary_unit = units
                primary_element = label

        weather_info = {
            "source": "WRF",
            "product": "WRF-Chem 模式产品图层",
            "element": primary_element,
            "time": time_label.replace("_", " "),
            "level": "地面/近地面或第 0 层",
            "range": (
                f"{bbox['west']:.3f}°E-{bbox['east']:.3f}°E, "
                f"{bbox['south']:.3f}°N-{bbox['north']:.3f}°N"
            ),
            "resolution": resolution,
            "grid": grid,
            "validGrid": f"{lat.size}",
            "coverage": domain,
            "missing": "NaN/FillValue",
            "unit": primary_unit,
            "variables": str(len(display_variables)),
            "steps": "1",
            "status": "已解析",
            "quality": "已生成透明 PNG overlay",
            "max": primary_stats["max"],
            "min": primary_stats["min"],
            "mean": primary_stats["mean"],
            "alert": "无",
            "update": datetime.now(timezone.utc).isoformat(),
            "bars": [0, 0, 0, 0, 0],
            "trend": [],
        }

        meta = {
            "dataset_id": build_dataset_id(source_file),
            "data_type": data_type,
            "file_format": "NC",
            "source_file": source_file.as_posix(),
            "meta_file": meta_file.as_posix(),
            "png_files": png_files,
            "variables": variables,
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
                "png_dir": png_dir.as_posix(),
                "note": "WRF adapter 已完成 NetCDF 解析，并生成前端地图叠加用透明 PNG。",
            },
        }

    write_meta(meta_file, meta)
    return meta


def process_files(file_paths: list[str], data_type: str = "WRF") -> dict:
    paths = sorted((Path(item).resolve() for item in file_paths), key=lambda item: item.name)
    metas = [process_file(str(path), data_type=data_type) for path in paths]
    if not metas:
        raise ValueError("No WRF files were provided.")
    if len(metas) == 1:
        return metas[0]

    first = metas[0]
    all_times: list[str] = []
    all_png_files: list[str] = []
    source_files: list[str] = []
    bboxes = []

    for meta in metas:
        source_files.append(meta.get("source_file", ""))
        all_times.extend(str(item) for item in meta.get("times", []))
        all_png_files.extend(str(item) for item in meta.get("png_files", []))
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

    time_pairs = sorted(zip(all_times, metas), key=lambda item: item[0])
    all_times = [item[0] for item in time_pairs]
    weather_info = dict(first.get("weather_info", {}))
    weather_info.update(
        {
            "time": f"{all_times[0].replace('_', ' ')} - {all_times[-1].replace('_', ' ')}",
            "steps": str(len(all_times)),
            "status": "parsed_folder",
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
        "png_files": all_png_files,
        "variables": first.get("variables", []),
        "times": all_times,
        "levels": first.get("levels", ["surface_or_level_0"]),
        "bbox": bbox,
        "weather_info": weather_info,
        "extra": {
            **first.get("extra", {}),
            "status": "parsed_folder",
            "file_count": len(metas),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_meta(meta_file, combined)
    return combined
