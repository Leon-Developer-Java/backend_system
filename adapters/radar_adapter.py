from __future__ import annotations

import math
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import xarray as xr
from PIL import Image, ImageDraw

from adapters.base import build_dataset_id, write_meta


DEFAULT_RENDER_VAR = "observation.base_ref_cor_log"
FALLBACK_RENDER_VAR = "observation.prdt_crf_raw_log"

RADAR_PRODUCT_INFO = {
    "observation.base_ref_cor_log": ("DBZH", "反射率", "dBZ"),
    "observation.base_vel_raw_lin": ("VRAD", "径向速度", "m/s"),
    "observation.base_zdr_cor_log": ("ZDR", "差分反射率", "dB"),
    "observation.base_rhv_flt_lin": ("RHV", "相关系数", ""),
    "observation.base_kdp_lsf_x_lin": ("KDP", "差分传播相移率", "deg/km"),
    "observation.prdt_hcl_flt_lin": ("HCL", "水凝物分类", ""),
    "observation.prdt_mlt_hgt_pol": ("MLT", "融化层高度", "m"),
    "observation.prdt_hcl_srf_lin": ("HCL_SRF", "地面水凝物分类", ""),
    "observation.prdt_ccl_raw_lin": ("CCL", "云分类", ""),
    "observation.prdt_crf_raw_log": ("CRF", "组合反射率", "dBZ"),
    "observation.prdt_etp_raw_lin": ("ETP", "回波顶高", "km"),
    "observation.prdt_qpr_mix_lin": ("QPR", "定量降水估测", "mm/h"),
}

RADAR_THRESHOLDS = np.array(
    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    dtype=np.float32,
)
RADAR_COLORS = np.array(
    [
        (4, 233, 231, 160),
        (1, 159, 244, 175),
        (3, 0, 244, 190),
        (2, 253, 2, 205),
        (1, 197, 1, 215),
        (0, 142, 0, 225),
        (253, 248, 2, 230),
        (229, 188, 0, 235),
        (253, 149, 0, 240),
        (253, 0, 0, 242),
        (212, 0, 0, 245),
        (188, 0, 0, 248),
        (248, 0, 253, 248),
        (152, 84, 198, 250),
        (253, 253, 253, 255),
    ],
    dtype=np.uint8,
)

VELOCITY_THRESHOLDS = np.array(
    [-30, -20, -10, -5, -1, 0, 1, 5, 10, 20, 30],
    dtype=np.float32,
)
VELOCITY_COLORS = np.array(
    [
        (49, 54, 149, 235),
        (69, 117, 180, 235),
        (116, 173, 209, 230),
        (171, 217, 233, 225),
        (224, 243, 248, 210),
        (245, 245, 245, 180),
        (254, 224, 144, 220),
        (253, 174, 97, 230),
        (244, 109, 67, 235),
        (215, 48, 39, 240),
        (165, 0, 38, 245),
    ],
    dtype=np.uint8,
)

RADAR_SELECTABLE_PRODUCTS = [
    "observation.base_ref_cor_log",
    "observation.base_vel_raw_lin",
]


def process_file(file_path: str, data_type: str = "Radar") -> dict[str, Any]:
    source_file = Path(file_path).resolve()
    if source_file.suffix.lower() != ".nc":
        raise ValueError("当前雷达适配器已实现 CAP_FMT NetCDF 解析；CINRAD/bz2 基数据后续再接入 wradlib。")

    started = time.perf_counter()
    meta_file = source_file.with_name(f"{source_file.name}.meta.json")
    png_file = source_file.with_name(f"{source_file.stem}.png")

    try:
        with xr.open_dataset(source_file, decode_times=False) as dataset:
            render_name = _choose_render_variable(dataset)
            render_data, render_mode = _make_render_data(dataset[render_name])
            lat_values, lon_values = _lat_lon_for_var(dataset, dataset[render_name])
            extent = _extent(dataset, lat_values, lon_values)
            stations = _stations(dataset)

            _write_radar_png(
                png_file=png_file,
                values=render_data,
                extent=extent,
                stations=stations,
                variable_name=render_name,
            )

            render_stats = _stats(render_data)
            variables = _variables(dataset, source_file)
            levels = _levels_for_var(dataset, dataset[render_name])
            observation_time = _observation_time(dataset, source_file)
            resolution_lon = _resolution(lon_values)
            resolution_lat = _resolution(lat_values)
            valid_grid = int(np.isfinite(render_data).sum())
            total_grid = int(render_data.size)
            coverage = valid_grid / total_grid if total_grid else 0.0
            product_code, product_name, unit = _product_info(render_name)
            generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            render_description = (
                "vertical maximum value from height layers"
                if render_mode == "vertical_max"
                else "single layer product render"
            )
            format_specific = {
                "radar_name": str(dataset.attrs.get("information.name", "")),
                "radar_type": str(dataset.attrs.get("information.type.radar", "")),
                "draw_type": str(dataset.attrs.get("information.type.draw", "")),
                "institution": str(dataset.attrs.get("information.institution", "")),
                "producer": str(dataset.attrs.get("information.producer", "")),
                "version": str(dataset.attrs.get("information.version", "")),
                "stations": stations,
                "render": {
                    "variable": render_name,
                    "code": product_code,
                    "mode": render_mode,
                    "description": render_description,
                },
            }
            radar_extra = {
                "radar_name": format_specific["radar_name"],
                "radar_type": format_specific["radar_type"],
                "institution": format_specific["institution"],
                "stations": stations,
                "render": {
                    "variable": product_code,
                    "raw_name": render_name,
                    "mode": render_mode,
                    "description": render_description,
                },
            }

            weather_info = {
                "file": source_file.name,
                "source": "Radar",
                "product": "CAP_FMT 组合雷达 NetCDF",
                "element": f"{product_name} {product_code}",
                "time": observation_time["display"],
                "level": _level_text(levels, render_mode),
                "range": _range_text(extent),
                "resolution": f"{resolution_lon:.4f}° × {resolution_lat:.4f}°",
                "grid": f"{render_data.shape[1]} × {render_data.shape[0]}",
                "valid_grid": valid_grid,
                "validGrid": str(valid_grid),
                "coverage": f"{coverage:.1%}",
                "missing": "NaN",
                "unit": unit,
                "variable_count": len(variables),
                "step_count": 1,
                "vars": str(len(variables)),
                "variables": str(len(variables)),
                "steps": "1",
                "status": "解析完成",
                "quality": _quality_text(coverage),
                "max": render_stats["max"],
                "min": render_stats["min"],
                "mean": render_stats["mean"],
                "alert": _alert_text(render_stats["max"]),
                "updated_at": generated_at,
                "update": generated_at,
                "bars": _histogram_bars(render_data),
                "bars_labels": ["0-10", "10-20", "20-30", "30-40", ">=40"],
                "trend": [],
                "trend_times": [],
                "extent": extent,
                "png": png_file.as_posix(),
                "meta_file": meta_file.as_posix(),
            }

            meta: dict[str, Any] = {
                "schema_version": "1.0",
                "file": source_file.name,
                "element": weather_info["element"],
                "time": weather_info["time"],
                "level": weather_info["level"],
                "range": weather_info["range"],
                "grid": weather_info["grid"],
                "missing": weather_info["missing"],
                "unit": weather_info["unit"],
                "vars": weather_info["vars"],
                "steps": weather_info["steps"],
                "extent": extent,
                "dataset_id": build_dataset_id(source_file),
                "data_type": data_type,
                "file_format": "RADAR_NC_CAP_FMT",
                "source_file": source_file.as_posix(),
                "meta_file": meta_file.as_posix(),
                "png_files": [png_file.as_posix()],
                "default_png": png_file.as_posix(),
                "default_variable": product_code,
                "variables": variables,
                "composites": [],
                "times": [observation_time["iso"]],
                "levels": levels,
                "bbox": extent,
                "weather_info": weather_info,
                "file_detail": {
                    "name": source_file.name,
                    "path": source_file.as_posix(),
                    "format": "RADAR_NC_CAP_FMT",
                    "size_bytes": source_file.stat().st_size,
                    "mtime": datetime.fromtimestamp(source_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "parsed_at": generated_at,
                    "parse_duration_s": round(time.perf_counter() - started, 3),
                    "parse_status": "success",
                },
                "time_detail": {
                    "start": observation_time["iso"],
                    "end": observation_time["iso"],
                    "count": 1,
                    "step_seconds": _attr_number(dataset, "information.time.scan"),
                    "reference_time": None,
                },
                "spatial": {
                    "lon_min": extent[0],
                    "lon_max": extent[2],
                    "lat_min": extent[1],
                    "lat_max": extent[3],
                    "resolution_lon": resolution_lon,
                    "resolution_lat": resolution_lat,
                    "nx": int(render_data.shape[1]),
                    "ny": int(render_data.shape[0]),
                    "levels": levels,
                    "level_type": "height_m",
                    "projection": "latlon",
                },
                "format_specific": {
                    "radar_name": str(dataset.attrs.get("information.name", "")),
                    "radar_type": str(dataset.attrs.get("information.type.radar", "")),
                    "draw_type": str(dataset.attrs.get("information.type.draw", "")),
                    "institution": str(dataset.attrs.get("information.institution", "")),
                    "producer": str(dataset.attrs.get("information.producer", "")),
                    "version": str(dataset.attrs.get("information.version", "")),
                    "stations": stations,
                    "render": {
                        "variable": render_name,
                        "mode": render_mode,
                        "description": "30 个高度层取垂直最大值合成 PNG" if render_mode == "vertical_max" else "单层产品直接渲染",
                    },
                },
                "extra": {
                    "status": "parsed",
                    "generated_at": generated_at,
                    "cma": {},
                    "era5": {},
                    "gfs": {},
                    "himawari": {},
                    "radar": radar_extra,
                    "wrf": {},
                    "note": "当前实现面向 Z_RADR CAP_FMT NetCDF case。原始极坐标 CINRAD/bz2 后续接入 wradlib。",
                },
            }
    except OSError as exc:
        raise ValueError(f"雷达文件读取失败：{exc}") from exc

    write_meta(meta_file, meta)
    return meta


def process_files(file_paths: list[str], data_type: str = "Radar") -> dict[str, Any]:
    paths = [Path(item).resolve() for item in file_paths]
    if not paths:
        raise ValueError("No radar files were provided.")

    metas = [process_file(str(path), data_type=data_type) for path in paths]
    if len(metas) == 1:
        return metas[0]

    frames = [_frame_from_meta(meta, index) for index, meta in enumerate(metas)]
    frames.sort(key=lambda item: item.get("time") or item.get("file") or "")
    for index, frame in enumerate(frames):
        frame["index"] = index

    first = metas[0]
    times = [str(frame["time"]) for frame in frames if frame.get("time")]
    source_files = [str(frame["source_file"]) for frame in frames if frame.get("source_file")]
    png_files = [str(frame["png"]) for frame in frames if frame.get("png")]
    bbox = _merge_bboxes([frame.get("extent") for frame in frames])
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    weather_info = dict(first.get("weather_info", {}))
    if times:
        weather_info["time"] = f"{times[0]} - {times[-1]}"
    weather_info.update(
        {
            "file": f"{len(frames)} radar files",
            "step_count": len(frames),
            "steps": str(len(frames)),
            "status": "parsed_series",
            "updated_at": generated_at,
            "update": generated_at,
            "times": times,
        }
    )

    batch_id = f"{paths[0].parent.name}_{times[0]}_{times[-1]}" if times else f"{paths[0].parent.name}_radar_series"
    batch_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", batch_id).strip("._")
    meta_file = paths[0].parent / f"{batch_id}.series.meta.json"
    radar_extra = dict(first.get("extra", {}).get("radar", {}))
    radar_extra.update({"file_count": len(frames), "frame_count": len(frames)})

    combined = {
        "schema_version": "1.0",
        "file": weather_info["file"],
        "element": weather_info.get("element"),
        "time": weather_info.get("time"),
        "level": weather_info.get("level"),
        "range": weather_info.get("range"),
        "grid": weather_info.get("grid"),
        "missing": weather_info.get("missing"),
        "unit": weather_info.get("unit"),
        "vars": weather_info.get("vars"),
        "steps": weather_info.get("steps"),
        "extent": bbox,
        "dataset_id": batch_id,
        "data_type": data_type,
        "file_format": "RADAR_NC_CAP_FMT",
        "source_file": source_files[0] if source_files else "",
        "source_files": source_files,
        "meta_file": meta_file.as_posix(),
        "png_files": png_files,
        "default_png": png_files[0] if png_files else None,
        "default_variable": first.get("default_variable"),
        "variables": first.get("variables", []),
        "composites": [],
        "times": times,
        "levels": first.get("levels", []),
        "bbox": bbox,
        "frames": frames,
        "weather_info": weather_info,
        "file_detail": {
            "name": weather_info["file"],
            "path": paths[0].parent.as_posix(),
            "format": "RADAR_NC_CAP_FMT",
            "parsed_at": generated_at,
            "parse_status": "success",
            "file_count": len(frames),
        },
        "time_detail": {
            "start": times[0] if times else None,
            "end": times[-1] if times else None,
            "count": len(frames),
            "step_seconds": first.get("time_detail", {}).get("step_seconds"),
            "reference_time": None,
        },
        "spatial": first.get("spatial", {}),
        "format_specific": first.get("format_specific", {}),
        "extra": {
            **first.get("extra", {}),
            "status": "parsed_series",
            "generated_at": generated_at,
            "radar": radar_extra,
        },
    }
    write_meta(meta_file, combined)
    return combined


def build_render_catalog(source_file: str | Path, cache_dir: str | Path | None = None) -> dict[str, Any]:
    source_path = Path(source_file).resolve()
    if cache_dir is None:
        cache_path = source_path.parent / "_radar_renders" / source_path.stem
    else:
        cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(source_path, decode_times=False) as dataset:
        stations = _stations(dataset)
        products = []

        for variable_name in RADAR_SELECTABLE_PRODUCTS:
            if variable_name not in dataset.data_vars:
                continue

            data_array = dataset[variable_name]
            lat_values, lon_values = _lat_lon_for_var(dataset, data_array)
            extent = _extent(dataset, lat_values, lon_values)
            levels = _levels_for_var(dataset, data_array)
            product_code, product_name, unit = _product_info(variable_name)
            values = np.asarray(data_array.values, dtype=np.float32)
            values = np.where(np.isfinite(values), values, np.nan)

            product_levels = []
            for level_spec in _render_level_specs(values, levels):
                png_file = cache_path / f"{source_path.stem}.{_safe_name(variable_name)}.{level_spec['key']}.png"
                if _needs_render(png_file, source_path):
                    _write_radar_png(
                        png_file=png_file,
                        values=level_spec["values"],
                        extent=extent,
                        stations=stations,
                        variable_name=variable_name,
                    )

                product_levels.append(
                    {
                        "key": level_spec["key"],
                        "label": level_spec["label"],
                        "mode": level_spec["mode"],
                        "level": level_spec["level"],
                        "png": png_file.as_posix(),
                        "extent": extent,
                        "stats": _stats(level_spec["values"]),
                    }
                )

            products.append(
                {
                    "key": variable_name,
                    "code": product_code,
                    "name": product_name,
                    "label": f"{product_name} {product_code}",
                    "unit": unit,
                    "extent": extent,
                    "levels": product_levels,
                }
            )

    return {
        "source_file": source_path.as_posix(),
        "cache_dir": cache_path.as_posix(),
        "products": products,
    }


def build_grid_catalog(source_file: str | Path) -> dict[str, Any]:
    source_path = Path(source_file).resolve()

    with xr.open_dataset(source_path, decode_times=False) as dataset:
        observation_time = _observation_time(dataset, source_path)
        products = []

        for variable_name in RADAR_SELECTABLE_PRODUCTS:
            if variable_name not in dataset.data_vars:
                continue

            data_array = dataset[variable_name]
            lat_values, lon_values = _lat_lon_for_var(dataset, data_array)
            extent = _extent(dataset, lat_values, lon_values)
            levels = _levels_for_var(dataset, data_array)
            product_code, product_name, unit = _product_info(variable_name)

            products.append(
                {
                    "key": variable_name,
                    "code": product_code,
                    "name": product_name,
                    "label": f"{product_name} {product_code}",
                    "unit": unit,
                    "extent": extent,
                    "grid": _grid_shape(data_array),
                    "levels": _grid_level_options(data_array, levels),
                    "missing": -9999.0,
                }
            )

    return {
        "source_file": source_path.as_posix(),
        "file": source_path.name,
        "time": observation_time,
        "products": products,
    }


def read_grid_values(
    source_file: str | Path,
    variable_name: str,
    level_key: str,
    missing_value: float = -9999.0,
) -> dict[str, Any]:
    source_path = Path(source_file).resolve()

    with xr.open_dataset(source_path, decode_times=False) as dataset:
        if variable_name not in dataset.data_vars:
            raise ValueError(f"雷达变量不存在：{variable_name}")

        data_array = dataset[variable_name]
        lat_values, lon_values = _lat_lon_for_var(dataset, data_array)
        extent = _extent(dataset, lat_values, lon_values)
        levels = _levels_for_var(dataset, data_array)
        values = np.asarray(data_array.values, dtype=np.float32)
        values = np.where(np.isfinite(values), values, np.nan)
        grid_values, level_label, mode, level_value = _select_grid_level(values, levels, level_key)
        stats = _stats(grid_values)
        product_code, product_name, unit = _product_info(variable_name)

    clean_values = np.where(np.isfinite(grid_values), grid_values, missing_value).astype("<f4", copy=False)

    return {
        "file": source_path.name,
        "product": {
            "key": variable_name,
            "code": product_code,
            "name": product_name,
            "label": f"{product_name} {product_code}",
            "unit": unit,
        },
        "level": {
            "key": level_key,
            "label": level_label,
            "mode": mode,
            "level": level_value,
        },
        "extent": extent,
        "grid": {"nx": int(clean_values.shape[1]), "ny": int(clean_values.shape[0])},
        "stats": stats,
        "missing": missing_value,
        "dtype": "float32",
        "byte_order": "little",
        "bytes": clean_values.tobytes(order="C"),
    }


def _choose_render_variable(dataset: xr.Dataset) -> str:
    if DEFAULT_RENDER_VAR in dataset.data_vars:
        return DEFAULT_RENDER_VAR
    if FALLBACK_RENDER_VAR in dataset.data_vars:
        return FALLBACK_RENDER_VAR

    for name, data_array in dataset.data_vars.items():
        if name.startswith("observation.") and np.issubdtype(data_array.dtype, np.number):
            if 2 <= data_array.ndim <= 3:
                return name
    raise ValueError("未找到可渲染的雷达观测变量。")


def _make_render_data(data_array: xr.DataArray) -> tuple[np.ndarray, str]:
    values = np.asarray(data_array.values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, np.nan)

    if values.ndim == 3 and values.shape[0] > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(values, axis=0), "vertical_max"
    if values.ndim == 3:
        return values[0], "single_layer"
    if values.ndim == 2:
        return values, "single_layer"
    raise ValueError(f"变量 {data_array.name} 的维度 {values.shape} 暂不支持渲染。")


def _write_radar_png(
    png_file: Path,
    values: np.ndarray,
    extent: list[float],
    stations: list[dict[str, Any]],
    variable_name: str = DEFAULT_RENDER_VAR,
) -> None:
    png_file.parent.mkdir(parents=True, exist_ok=True)
    image_values = np.flipud(values)
    rgba = _colorize_values(image_values, variable_name)

    image = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(image)
    west, south, east, north = extent
    width, height = image.size
    for station in stations:
        lon = station.get("longitude")
        lat = station.get("latitude")
        if lon is None or lat is None or east == west or north == south:
            continue
        x = int(round((lon - west) / (east - west) * (width - 1)))
        y = int(round((north - lat) / (north - south) * (height - 1)))
        if 0 <= x < width and 0 <= y < height:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255, 240), outline=(20, 20, 20, 240), width=1)

    image.save(png_file)


def _colorize_values(values: np.ndarray, variable_name: str) -> np.ndarray:
    rgba = np.zeros((values.shape[0], values.shape[1], 4), dtype=np.uint8)
    product_code, _, _ = _product_info(variable_name)

    if product_code == "VRAD":
        valid = np.isfinite(values)
        color_index = np.digitize(values[valid], VELOCITY_THRESHOLDS, right=False) - 1
        color_index = np.clip(color_index, 0, len(VELOCITY_COLORS) - 1)
        rgba[valid] = VELOCITY_COLORS[color_index]
        return rgba

    valid = np.isfinite(values) & (values >= RADAR_THRESHOLDS[0])
    color_index = np.digitize(values[valid], RADAR_THRESHOLDS, right=False) - 1
    color_index = np.clip(color_index, 0, len(RADAR_COLORS) - 1)
    rgba[valid] = RADAR_COLORS[color_index]
    return rgba


def _render_level_specs(values: np.ndarray, levels: list[float]) -> list[dict[str, Any]]:
    if values.ndim == 3 and values.shape[0] > 1:
        specs: list[dict[str, Any]] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            specs.append(
                {
                    "key": "max",
                    "label": "垂直最大值",
                    "mode": "vertical_max",
                    "level": None,
                    "values": np.nanmax(values, axis=0),
                }
            )

        for index, level in enumerate(levels):
            specs.append(
                {
                    "key": f"level-{index}",
                    "label": f"{level:g} m",
                    "mode": "single_level",
                    "level": level,
                    "values": values[index],
                }
            )
        return specs

    if values.ndim == 3:
        level = levels[0] if levels else None
        return [
            {
                "key": "level-0",
                "label": f"{level:g} m" if level is not None else "单层",
                "mode": "single_level",
                "level": level,
                "values": values[0],
            }
        ]

    if values.ndim == 2:
        return [
            {
                "key": "surface",
                "label": "单层",
                "mode": "single_level",
                "level": None,
                "values": values,
            }
        ]

    raise ValueError(f"变量维度 {values.shape} 暂不支持渲染。")


def _grid_shape(data_array: xr.DataArray) -> dict[str, int]:
    if data_array.ndim == 3:
        return {"nx": int(data_array.shape[2]), "ny": int(data_array.shape[1])}
    if data_array.ndim == 2:
        return {"nx": int(data_array.shape[1]), "ny": int(data_array.shape[0])}
    raise ValueError(f"变量 {data_array.name} 的维度 {data_array.shape} 暂不支持渲染。")


def _grid_level_options(data_array: xr.DataArray, levels: list[float]) -> list[dict[str, Any]]:
    if data_array.ndim == 3 and data_array.shape[0] > 1:
        options = [
            {
                "key": "max",
                "label": "垂直最大值",
                "mode": "vertical_max",
                "level": None,
            }
        ]
        options.extend(
            {
                "key": f"level-{index}",
                "label": f"{level:g} m",
                "mode": "single_level",
                "level": level,
            }
            for index, level in enumerate(levels)
        )
        return options

    if data_array.ndim == 3:
        level = levels[0] if levels else None
        return [
            {
                "key": "level-0",
                "label": f"{level:g} m" if level is not None else "单层",
                "mode": "single_level",
                "level": level,
            }
        ]

    if data_array.ndim == 2:
        return [
            {
                "key": "surface",
                "label": "单层",
                "mode": "single_level",
                "level": None,
            }
        ]

    raise ValueError(f"变量 {data_array.name} 的维度 {data_array.shape} 暂不支持渲染。")


def _select_grid_level(
    values: np.ndarray,
    levels: list[float],
    level_key: str,
) -> tuple[np.ndarray, str, str, float | None]:
    if values.ndim == 3 and values.shape[0] > 1 and level_key == "max":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(values, axis=0), "垂直最大值", "vertical_max", None

    if values.ndim == 3:
        if level_key == "surface":
            index = 0
        else:
            match = re.fullmatch(r"level-(\d+)", level_key)
            if not match:
                raise ValueError(f"不支持的高度层：{level_key}")
            index = int(match.group(1))
        if index < 0 or index >= values.shape[0]:
            raise ValueError(f"高度层索引越界：{level_key}")
        level = levels[index] if index < len(levels) else None
        label = f"{level:g} m" if level is not None else f"第 {index + 1} 层"
        return values[index], label, "single_level", level

    if values.ndim == 2 and level_key in {"surface", "level-0", "max"}:
        return values, "单层", "single_level", None

    raise ValueError(f"变量维度 {values.shape} 不支持高度层 {level_key}。")


def _needs_render(png_file: Path, source_file: Path) -> bool:
    return not png_file.exists() or png_file.stat().st_mtime < source_file.stat().st_mtime


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "radar"


def _frame_from_meta(meta: dict[str, Any], index: int) -> dict[str, Any]:
    source_file = str(meta.get("source_file") or meta.get("file_detail", {}).get("path") or "")
    weather_info = dict(meta.get("weather_info", {}))
    times = meta.get("times") if isinstance(meta.get("times"), list) else []
    time_value = str(times[0]) if times else str(weather_info.get("time") or "")
    png = meta.get("default_png")
    if not png and isinstance(meta.get("png_files"), list) and meta["png_files"]:
        png = meta["png_files"][0]
    if not png:
        png = weather_info.get("png")

    return {
        "index": index,
        "file": Path(source_file).name if source_file else str(meta.get("file") or ""),
        "source_file": source_file,
        "meta_file": meta.get("meta_file"),
        "time": time_value,
        "time_label": weather_info.get("time") or time_value,
        "extent": meta.get("bbox") or meta.get("extent") or weather_info.get("extent"),
        "png": png,
        "default_png": png,
        "weather_info": weather_info,
    }


def _merge_bboxes(extents: list[Any]) -> list[float] | None:
    values: list[list[float]] = []
    for extent in extents:
        if isinstance(extent, dict):
            candidate = [extent.get("west"), extent.get("south"), extent.get("east"), extent.get("north")]
        else:
            candidate = extent
        if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
            continue
        try:
            nums = [float(item) for item in candidate]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(item) for item in nums) and nums[0] < nums[2] and nums[1] < nums[3]:
            values.append(nums)
    if not values:
        return None
    return [
        _round_float(min(item[0] for item in values)),
        _round_float(min(item[1] for item in values)),
        _round_float(max(item[2] for item in values)),
        _round_float(max(item[3] for item in values)),
    ]


def _grid_url(source_file: Path, variable_name: str, level_key: str = "max") -> str:
    query = urlencode(
        {
            "file": source_file.name,
            "product": variable_name,
            "level": level_key,
        }
    )
    return f"/api/radar/grid?{query}"


def _lat_lon_for_var(dataset: xr.Dataset, data_array: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    dims = list(data_array.dims)
    if len(dims) >= 3:
        lat_dim = dims[-2]
        lon_dim = dims[-1]
    elif len(dims) == 2:
        lat_dim = dims[0]
        lon_dim = dims[1]
    else:
        raise ValueError(f"变量 {data_array.name} 缺少二维经纬网格。")

    lat_name = lat_dim.replace("axis.", "coordinate.")
    lon_name = lon_dim.replace("axis.", "coordinate.")
    if lat_name not in dataset or lon_name not in dataset:
        raise ValueError(f"变量 {data_array.name} 缺少坐标变量 {lat_name}/{lon_name}。")

    return np.asarray(dataset[lat_name].values, dtype=np.float64), np.asarray(dataset[lon_name].values, dtype=np.float64)


def _levels_for_var(dataset: xr.Dataset, data_array: xr.DataArray) -> list[float]:
    if data_array.ndim < 3:
        return []
    level_name = data_array.dims[0].replace("axis.", "coordinate.")
    if level_name not in dataset:
        return []
    return [_round_float(item) for item in np.asarray(dataset[level_name].values, dtype=np.float64).tolist()]


def _extent(dataset: xr.Dataset, lat_values: np.ndarray, lon_values: np.ndarray) -> list[float]:
    west = _attr_number(dataset, "information.longitude.west")
    east = _attr_number(dataset, "information.longitude.east")
    south = _attr_number(dataset, "information.latitude.south")
    north = _attr_number(dataset, "information.latitude.north")

    if None in {west, east, south, north}:
        west = float(np.nanmin(lon_values))
        east = float(np.nanmax(lon_values))
        south = float(np.nanmin(lat_values))
        north = float(np.nanmax(lat_values))

    return [_round_float(west), _round_float(south), _round_float(east), _round_float(north)]


def _stations(dataset: xr.Dataset) -> list[dict[str, Any]]:
    names_key = "coordinate.radarlist.1.1"
    lon_key = "coordinate.radarlist.longitude"
    lat_key = "coordinate.radarlist.latitude"
    height_key = "coordinate.radarlist.height"
    if not all(key in dataset for key in [names_key, lon_key, lat_key, height_key]):
        return []

    stations = []
    for name, lon, lat, height in zip(
        dataset[names_key].values,
        dataset[lon_key].values,
        dataset[lat_key].values,
        dataset[height_key].values,
    ):
        stations.append(
            {
                "name": str(name),
                "longitude": _round_float(lon),
                "latitude": _round_float(lat),
                "height_m": _round_float(height),
            }
        )
    return stations


def _variables(dataset: xr.Dataset, source_file: Path) -> list[dict[str, Any]]:
    variables = []
    for name, data_array in dataset.data_vars.items():
        if not name.startswith("observation."):
            continue
        product_code, product_name, unit = _product_info(name)
        stat = _stats(np.asarray(data_array.values, dtype=np.float32))
        variables.append(
            {
                "name": product_code,
                "short_name": product_code,
                "raw_name": name,
                "long_name": product_name,
                "name_cn": product_name,
                "unit": unit,
                "display_unit": None,
                "units": unit,
                "units_original": unit,
                "dims": list(data_array.dims),
                "shape": [int(item) for item in data_array.shape],
                "level": None,
                "missing": "NaN",
                "description": None,
                "wavelength": None,
                "float32": _grid_url(source_file, name),
                "netcdf": source_file.as_posix(),
                "png": None,
                "category": "雷达产品",
                "definition": "",
                "applications": ["强对流监测", "短临预报", "降水估测"],
                "stats": stat,
            }
        )
    return variables


def _product_info(name: str) -> tuple[str, str, str]:
    return RADAR_PRODUCT_INFO.get(name, (name.split(".")[-1], name.split(".")[-1], ""))


def _observation_time(dataset: xr.Dataset, source_file: Path) -> dict[str, str]:
    timestamp = _attr_number(dataset, "information.time.start")
    if timestamp is not None:
        dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    else:
        match = re.search(r"_(\d{14})_", source_file.name)
        if not match:
            dt = datetime.fromtimestamp(source_file.stat().st_mtime, tz=timezone.utc)
        else:
            dt = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

    return {
        "iso": dt.isoformat().replace("+00:00", "Z"),
        "display": dt.strftime("%Y-%m-%d %H:%M"),
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    numeric = np.asarray(values, dtype=np.float64)
    valid = numeric[np.isfinite(numeric)]
    if valid.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": _round_float(np.nanmin(valid)),
        "max": _round_float(np.nanmax(valid)),
        "mean": _round_float(np.nanmean(valid)),
        "std": _round_float(np.nanstd(valid)),
    }


def _histogram_bars(values: np.ndarray) -> list[int]:
    valid = values[np.isfinite(values) & (values >= 0)]
    if valid.size == 0:
        return [0, 0, 0, 0, 0]
    counts, _ = np.histogram(valid, bins=[0, 10, 20, 30, 40, math.inf])
    return [int(item) for item in counts.tolist()]


def _resolution(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    diffs = np.diff(values.astype(np.float64))
    diffs = np.abs(diffs[np.isfinite(diffs)])
    if diffs.size == 0:
        return 0.0
    return _round_float(float(np.median(diffs)))


def _range_text(extent: list[float]) -> str:
    west, south, east, north = extent
    return f"{west:g}°E-{east:g}°E, {south:g}°N-{north:g}°N"


def _level_text(levels: list[float], render_mode: str) -> str:
    if not levels:
        return "单层产品"
    if render_mode == "vertical_max" and len(levels) > 1:
        return f"{levels[0]:g}-{levels[-1]:g} m 高度层最大值合成（{len(levels)} 层）"
    return f"{levels[0]:g} m"


def _quality_text(coverage: float) -> str:
    if coverage >= 0.2:
        return "有效覆盖较高"
    if coverage >= 0.05:
        return "有效覆盖正常"
    return "有效覆盖较低"


def _alert_text(max_value: float | None) -> str:
    if max_value is None:
        return "无"
    if max_value >= 45:
        return "存在强回波"
    if max_value >= 35:
        return "存在中等回波"
    return "无"


def _attr_number(dataset: xr.Dataset, key: str) -> float | None:
    value = dataset.attrs.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_float(value: Any) -> float:
    return round(float(value), 6)


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "NaN"
    return f"{value:.2f}"
