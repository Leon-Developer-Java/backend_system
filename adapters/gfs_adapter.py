from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cfgrib
import matplotlib
import numpy as np
import xarray as xr
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEMA_VERSION = "2.0"
WEBP_QUALITY = 82
WEBP_METHOD = 2

# 只展示常用近地面业务变量；后续扩展时在此加入变量名即可。
ALLOWED_VARS = {
    "t2m", "2t",
    "d2m", "2d",
    "tp", "apcp",
    "sp", "msl", "prmsl",
    "u10", "v10", "10u", "10v", "ugrd", "vgrd",
}


def _source_name(data_type: str | None) -> str:
    return "ECMWF" if str(data_type or "").strip().upper() in {"EC", "ECMWF", "IFS"} else "GFS"


def _product_name(source: str) -> str:
    return "ECMWF 数值预报产品" if source == "ECMWF" else "GFS 数值预报产品"


def _open_grib_groups(file_path: str) -> list[xr.Dataset]:
    """按 cfgrib 分组打开 GRIB/GRIB2。"""
    try:
        return list(cfgrib.open_datasets(file_path, backend_kwargs={"indexpath": ""}))
    except Exception:
        return [
            xr.open_dataset(
                file_path,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )
        ]


def _find_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = next((name for name in ("latitude", "lat", "y") if name in ds.coords or name in ds.dims), None)
    lon_name = next((name for name in ("longitude", "lon", "x") if name in ds.coords or name in ds.dims), None)
    if not lat_name or not lon_name:
        raise ValueError("无法识别经纬度坐标。")
    return lat_name, lon_name


def _coord_values(ds: xr.Dataset, name: str) -> list[str]:
    if name not in ds.coords:
        return []
    values = np.asarray(ds.coords[name].values).reshape(-1)
    return [str(value).replace(".000000000", "") for value in values]


def _time_labels(ds: xr.Dataset, n: int) -> list[str]:
    valid_times = _coord_values(ds, "valid_time")
    base_times = _coord_values(ds, "time")
    steps = _coord_values(ds, "step")

    if len(valid_times) == n:
        return valid_times
    if len(steps) == n:
        base = base_times[0] if base_times else "time"
        return [f"{base} + {step}" for step in steps]
    if n == 1:
        return [valid_times[0] if valid_times else (base_times[0] if base_times else "step000")]
    return [f"step{i:03d}" for i in range(n)]


def _step_hours(ds: xr.Dataset, n: int) -> list[int]:
    if "step" not in ds.coords:
        return list(range(n))

    values = np.asarray(ds.coords["step"].values).reshape(-1)
    if values.size != n:
        return list(range(n))

    result: list[int] = []
    for i, value in enumerate(values):
        try:
            result.append(int(round(float(value / np.timedelta64(1, "h")))))
            continue
        except Exception:
            pass

        text = str(value)
        match = re.search(r"(?:(\d+)\s+days?\s+)?(\d{1,2}):(\d{2}):(\d{2})", text)
        if match:
            result.append(int(match.group(1) or 0) * 24 + int(match.group(2)))
            continue

        match = re.search(r"(\d+)", text)
        result.append(int(match.group(1)) if match else i)

    return result


def _cycle_time(ds: xr.Dataset) -> str:
    values = _coord_values(ds, "time")
    return values[0] if values else "待解析"


def _normalize_longitude(arr: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    if lon.size >= 2 and np.nanmin(lon) >= 0 and np.nanmax(lon) > 180:
        normalized = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(normalized)
        return arr[:, :, order], normalized[order]
    return arr, lon


def _extract_array_lat_lon(ds: xr.Dataset, var_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_name, lon_name = _find_lat_lon_names(ds)
    da = ds[var_name]

    for dim in list(da.dims):
        if dim not in {lat_name, lon_name, "time", "valid_time", "step"}:
            da = da.isel({dim: 0})

    arr = np.squeeze(np.asarray(da.values, dtype=float))
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim > 3:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    elif arr.ndim != 3:
        raise ValueError(f"变量 {var_name} 无法渲染，shape={arr.shape}")

    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)

    # 输出统一为北到南、经度递增。
    if lat.size >= 2 and lat[0] < lat[-1]:
        lat = lat[::-1]
        arr = arr[:, ::-1, :]
    if lon.size >= 2 and lon[0] > lon[-1]:
        lon = lon[::-1]
        arr = arr[:, :, ::-1]

    arr, lon = _normalize_longitude(arr, lon)
    return arr, lat, lon


def _infer_var_type(var_name: str, units: str, long_name: str) -> str:
    text = f"{var_name} {units} {long_name}".lower()
    if any(key in text for key in ("t2m", "2t", "d2m", "2d", "temperature", "dewpoint")):
        return "temperature"
    if any(key in text for key in ("tp", "apcp", "precip", "rain")):
        return "precipitation"
    if any(key in text for key in ("pressure", "prmsl", "msl", "sp")):
        return "pressure"
    if any(key in text for key in ("wind", "u10", "v10", "10u", "10v", "ugrd", "vgrd")):
        return "wind"
    return "generic"


def _convert_values(var_name: str, units: str, long_name: str, values: np.ndarray) -> tuple[np.ndarray, str, str, str]:
    arr = np.asarray(values, dtype=float)
    lower = str(units or "").lower()
    var_type = _infer_var_type(var_name, units, long_name)

    if var_type == "temperature" and lower in {"k", "kelvin"}:
        return arr - 273.15, "°C", "K → °C", var_type
    if var_type == "pressure" and lower in {"pa", "pascal", "pascals"}:
        return arr / 100.0, "hPa", "Pa → hPa", var_type
    if var_type == "precipitation" and lower in {"m", "meter", "metre"}:
        return arr * 1000.0, "mm", "m → mm", var_type
    if var_type == "precipitation" and "kg" in lower and "m" in lower:
        return arr, "mm", f"{units} → mm", var_type
    return arr, units or "未知", "未转换", var_type


def _business_label(var_name: str, long_name: str) -> str:
    mapping = {
        "t2m": "2米气温", "2t": "2米气温",
        "d2m": "2米露点温度", "2d": "2米露点温度",
        "tp": "累积降水", "apcp": "累积降水",
        "sp": "地面气压", "msl": "海平面气压", "prmsl": "海平面气压",
        "u10": "10米U风", "10u": "10米U风", "ugrd": "U风分量",
        "v10": "10米V风", "10v": "10米V风", "vgrd": "V风分量",
    }
    return mapping.get(var_name.lower(), long_name or var_name)


def _product_category(var_type: str) -> str:
    return {
        "temperature": "温度产品",
        "precipitation": "降水产品",
        "pressure": "气压产品",
        "wind": "风场产品",
    }.get(var_type, "数值预报产品")


def _priority(var_name: str) -> int:
    return {
        "t2m": 1000, "2t": 1000,
        "d2m": 900, "2d": 900,
        "tp": 850, "apcp": 850,
        "sp": 800, "msl": 790, "prmsl": 790,
        "u10": 700, "10u": 700, "ugrd": 690,
        "v10": 680, "10v": 680, "vgrd": 670,
    }.get(var_name.lower(), 100)


def _frame_stats(values: np.ndarray) -> dict[str, Any]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"min": None, "mean": None, "max": None, "p05": None, "p50": None, "p95": None, "missing_ratio": 1.0}
    return {
        "min": round(float(valid.min()), 3),
        "mean": round(float(valid.mean()), 3),
        "max": round(float(valid.max()), 3),
        "p05": round(float(np.percentile(valid, 5)), 3),
        "p50": round(float(np.percentile(valid, 50)), 3),
        "p95": round(float(np.percentile(valid, 95)), 3),
        "missing_ratio": round(float(1.0 - valid.size / values.size), 6),
    }


def _global_stats(arr3d: np.ndarray) -> dict[str, Any]:
    valid = arr3d[np.isfinite(arr3d)]
    if valid.size == 0:
        return {"min": None, "mean": None, "max": None, "p05": None, "p95": None, "missing_ratio": 1.0}
    return {
        "min": round(float(valid.min()), 3),
        "mean": round(float(valid.mean()), 3),
        "max": round(float(valid.max()), 3),
        "p05": round(float(np.percentile(valid, 5)), 3),
        "p95": round(float(np.percentile(valid, 95)), 3),
        "missing_ratio": round(float(1.0 - valid.size / arr3d.size), 6),
    }


def _color_range(var_name: str, var_type: str, stats: dict[str, Any]) -> dict[str, Any]:
    name = var_name.lower()
    if name in {"tp", "apcp"}:
        ref = max(float(stats.get("p95") or 0), float(stats.get("max") or 0))
        levels = [1, 2, 5, 10, 25, 50, 100, 150]
        vmax = next((value for value in levels if ref <= value), 150)
        return {"min": 0.0, "max": float(vmax), "mode": "precip_dynamic"}
    if name in {"sp", "msl", "prmsl"}:
        mean = float(stats.get("mean") or 1013.0)
        return {"min": round(mean - 18, 3), "max": round(mean + 18, 3), "mode": "pressure_centered"}
    p05 = stats.get("p05")
    p95 = stats.get("p95")
    if p05 is not None and p95 is not None and float(p95) > float(p05):
        return {"min": float(p05), "max": float(p95), "mode": "p05_p95"}
    return {"min": stats.get("min"), "max": stats.get("max"), "mode": "min_max"}


def _legend_ticks(color_range: dict[str, Any]) -> list[str]:
    vmin = color_range.get("min")
    vmax = color_range.get("max")
    if vmin is None or vmax is None or float(vmax) <= float(vmin):
        return ["低", "较低", "中", "较高", "高"]
    values = np.linspace(float(vmin), float(vmax), 5)
    return [f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}" for value in values]


def _gradient(var_type: str) -> str:
    if var_type == "precipitation":
        return "linear-gradient(to right, #f8fafc, #93c5fd, #22c55e, #facc15, #ef4444)"
    if var_type == "pressure":
        return "linear-gradient(to right, #7c3aed, #2563eb, #22c55e, #facc15, #ef4444)"
    if var_type == "wind":
        return "linear-gradient(to right, #e0f2fe, #38bdf8, #2563eb, #7c3aed, #ef4444)"
    return "linear-gradient(to right, #1e40af, #0ea5e9, #22c55e, #facc15, #ef4444)"


def _to_data_url(path: Path) -> str:
    normalized = str(path).replace("\\", "/")
    marker = "/data/"
    index = normalized.lower().find(marker)
    return normalized[index:] if index >= 0 else f"/data/{path.name}"


def _save_one_webp(values2d: np.ndarray, output_path: Path, var_type: str, vmin: float | None, vmax: float | None) -> None:
    arr = np.asarray(values2d, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        arr = np.zeros_like(arr)
        valid = np.ones_like(arr, dtype=bool)

    if vmin is None:
        vmin = float(arr[valid].min())
    if vmax is None:
        vmax = float(arr[valid].max())
    if abs(vmax - vmin) < 1e-12:
        normalized = np.zeros_like(arr, dtype=float)
    else:
        normalized = np.clip((arr - vmin) / (vmax - vmin), 0, 1)

    cmap = plt.get_cmap("viridis" if var_type == "pressure" else "turbo")
    rgba = cmap(normalized)
    rgba[..., 3] = np.where(valid, 0.72, 0.0)
    rgba8 = np.clip(rgba * 255, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba8, mode="RGBA").save(
        output_path,
        format="WEBP",
        quality=WEBP_QUALITY,
        method=WEBP_METHOD,
    )


def _save_webp_frames(
    source_path: Path,
    var_name: str,
    arr3d: np.ndarray,
    var_type: str,
    color_range: dict[str, Any],
    times: list[str],
    forecast_hours: list[int],
) -> tuple[str, list[dict[str, Any]]]:
    safe_key = re.sub(r"[^0-9A-Za-z_-]", "_", var_name)
    frames: list[dict[str, Any]] = []
    first_path: Path | None = None

    for index, frame in enumerate(arr3d):
        output_path = source_path.with_name(f"{source_path.name}_{safe_key}_step{index:03d}.webp")
        _save_one_webp(
            frame,
            output_path,
            var_type=var_type,
            vmin=color_range.get("min"),
            vmax=color_range.get("max"),
        )
        if first_path is None:
            first_path = output_path
        frames.append({
            "index": index,
            "forecast_hour": forecast_hours[index] if index < len(forecast_hours) else index,
            "forecast_label": f"F{(forecast_hours[index] if index < len(forecast_hours) else index):03d}",
            "valid_time": times[index] if index < len(times) else f"step{index:03d}",
            "url": _to_data_url(output_path),
            "stats": _frame_stats(np.asarray(frame, dtype=float)),
        })

    if first_path is None:
        raise ValueError(f"变量 {var_name} 没有生成 WebP。")

    compat_path = source_path.with_name(f"{source_path.name}_{safe_key}.webp")
    if first_path.resolve() != compat_path.resolve():
        shutil.copyfile(first_path, compat_path)

    return _to_data_url(compat_path), frames


def _build_layers(source_path: Path, groups: list[xr.Dataset]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    records: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()

    for group_index, ds in enumerate(groups):
        for var_name in ds.data_vars:
            if var_name not in ALLOWED_VARS or var_name in seen:
                continue

            try:
                da = ds[var_name]
                attrs = da.attrs
                long_name = attrs.get("long_name", var_name)
                units = attrs.get("units", "")
                short_name = attrs.get("GRIB_shortName", var_name)
                type_of_level = attrs.get("GRIB_typeOfLevel", attrs.get("typeOfLevel", "surface"))
                step_type = attrs.get("GRIB_stepType", attrs.get("stepType", "unknown"))

                raw, lat, lon = _extract_array_lat_lon(ds, var_name)
                values, display_unit, conversion, var_type = _convert_values(var_name, units, long_name, raw)

                times = _time_labels(ds, values.shape[0])
                forecast_hours = _step_hours(ds, values.shape[0])
                stats = _global_stats(values)
                color_range = _color_range(var_name, var_type, stats)
                image_url, frames = _save_webp_frames(
                    source_path,
                    var_name,
                    values,
                    var_type,
                    color_range,
                    times,
                    forecast_hours,
                )

                lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
                lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
                resolution = "待解析"
                if lat.size >= 2 and lon.size >= 2:
                    resolution = f"{abs(float(lat[1] - lat[0])):.2f}° × {abs(float(lon[1] - lon[0])):.2f}°"

                label = _business_label(var_name, long_name)
                layer = {
                    "key": var_name,
                    "label": label,
                    "element": label,
                    "elementCode": var_name,
                    "elementEnglish": long_name,
                    "shortName": short_name,
                    "productCategory": _product_category(var_type),
                    "varType": var_type,
                    "rawUnit": units,
                    "unit": display_unit,
                    "displayUnit": display_unit,
                    "conversion": conversion,
                    "groupIndex": group_index,
                    "level": f"{type_of_level}, stepType={step_type}",
                    "typeOfLevel": type_of_level,
                    "stepType": step_type,
                    "issue_time": _cycle_time(ds),
                    "time": times[0] if len(times) == 1 else f"{times[0]} 至 {times[-1]}",
                    "times": times,
                    "forecast_hours": forecast_hours,
                    "forecast_labels": [f"F{hour:03d}" for hour in forecast_hours],
                    "extent": [round(lon_min, 4), round(lat_min, 4), round(lon_max, 4), round(lat_max, 4)],
                    "bbox": {
                        "south": round(lat_min, 4),
                        "north": round(lat_max, 4),
                        "west": round(lon_min, 4),
                        "east": round(lon_max, 4),
                    },
                    "range": f"纬度 {lat_min:.4f} ~ {lat_max:.4f}，经度 {lon_min:.4f} ~ {lon_max:.4f}",
                    "resolution": resolution,
                    "grid": {"nx": int(lon.size), "ny": int(lat.size), "text": f"{lat.size} × {lon.size}"},
                    "gridText": f"{lat.size} × {lon.size}",
                    "coverage": f"{(1.0 - float(stats['missing_ratio'])) * 100:.2f}%",
                    "missingText": f"{float(stats['missing_ratio']) * 100:.2f}%",
                    "quality": "正常" if float(stats["missing_ratio"]) <= 0.01 else "存在缺测",
                    "min": stats.get("min"),
                    "mean": stats.get("mean"),
                    "max": stats.get("max"),
                    "color_range": color_range,
                    "legend_ticks": _legend_ticks(color_range),
                    "gradient": _gradient(var_type),
                    "image_format": "webp",
                    "image_url": image_url,
                    "image_urls": [frame["url"] for frame in frames],
                    "frames": frames,
                    "render_mode": "webp",
                }

                records.append((_priority(var_name), var_name, layer))
                seen.add(var_name)
            except Exception as exc:
                print(f"[WARN] Skip variable {var_name}: {exc}")

    if not records:
        raise ValueError("GRIB 文件中没有可渲染的常用变量。")

    records.sort(key=lambda item: item[0], reverse=True)
    layers = {name: layer for _, name, layer in records}
    options = [
        {
            "key": name,
            "label": layer["label"],
            "unit": layer["unit"],
            "varType": layer["varType"],
            "productCategory": layer["productCategory"],
        }
        for _, name, layer in records
    ]
    return options, layers, options[0]["key"]


def process_file(file_path: str, data_type: str = "GFS") -> dict[str, Any]:
    """解析 GRIB/GRIB2，仅生成 WebP 时序图和一个紧凑 meta.json。"""
    source_path = Path(file_path).resolve()
    if not source_path.exists():
        raise ValueError(f"文件不存在：{source_path}")

    source = _source_name(data_type)
    groups = _open_grib_groups(str(source_path))
    variable_options, variable_layers, default_variable = _build_layers(source_path, groups)
    default_layer = variable_layers[default_variable]

    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": source_path.name.replace(".", "_"),
        "source": source,
        "business_type": source,
        "data_type": source,
        "product": _product_name(source),
        "file_name": source_path.name,
        "file_format": "GRIB2" if source_path.suffix.lower() == ".grib2" else "GRIB",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_time": default_layer.get("issue_time"),
        "extent": default_layer.get("extent"),
        "bbox": default_layer.get("bbox"),
        "grid": default_layer.get("grid"),
        "resolution": default_layer.get("resolution"),
        "default_variable": default_variable,
        "variable_options": variable_options,
        "variable_layers": variable_layers,
        "weather_info": {
            "source": source,
            "product": _product_name(source),
            "element": default_layer.get("element"),
            "time": default_layer.get("time"),
            "level": default_layer.get("level"),
            "range": default_layer.get("range"),
            "resolution": default_layer.get("resolution"),
            "grid": default_layer.get("gridText"),
            "unit": default_layer.get("unit"),
            "missing": default_layer.get("missingText"),
            "status": "解析成功",
            "quality": default_layer.get("quality"),
            "min": default_layer.get("min"),
            "mean": default_layer.get("mean"),
            "max": default_layer.get("max"),
            "mainVariable": default_variable,
            "image_format": "webp",
            "image_url": default_layer.get("image_url"),
            "image_urls": default_layer.get("image_urls", []),
            "render_mode": "webp",
        },
        "image_format": "webp",
        "image_url": default_layer.get("image_url"),
        "image_urls": default_layer.get("image_urls", []),
        "render_mode": "webp",
    }

    meta_path = source_path.with_name(f"{source_path.name}.meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
