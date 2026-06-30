from __future__ import annotations

import json
import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import cfgrib
import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adapters.base import process_basic_file


MISSING_VALUE = -9999.0


# ============================================================
# GFS / ECMWF GRIB adapter
# 功能：
# 1. 支持 GRIB / GRIB2
# 2. 支持多变量解析：t2m / d2m / sp / tp 等
# 3. 每个变量生成独立 PNG 序列
# 4. 返回 variable_options + variable_layers，供前端变量下拉切换
# 5. 保留 png_url / png_urls 等旧字段，兼容旧前端
# ============================================================


def _open_grib_groups(file_path: str) -> list[xr.Dataset]:
    """cfgrib 会把不同 level/typeOfLevel/stepType 拆成多个 dataset。"""
    try:
        return list(
            cfgrib.open_datasets(
                file_path,
                backend_kwargs={"indexpath": ""},
            )
        )
    except Exception:
        ds = xr.open_dataset(
            file_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
        return [ds]


def _find_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = None
    lon_name = None

    for name in ["latitude", "lat", "y"]:
        if name in ds.coords or name in ds.dims:
            lat_name = name
            break

    for name in ["longitude", "lon", "x"]:
        if name in ds.coords or name in ds.dims:
            lon_name = name
            break

    if not lat_name or not lon_name:
        raise ValueError("无法识别经纬度坐标。")

    return lat_name, lon_name


def _fmt_time(v: Any) -> str:
    try:
        return str(v).replace(".000000000", "")
    except Exception:
        return str(v)


def _coord_values(ds: xr.Dataset, name: str) -> list[str]:
    if name not in ds.coords:
        return []

    arr = np.asarray(ds.coords[name].values).reshape(-1)
    if arr.size == 0:
        return []

    return [_fmt_time(x) for x in arr]


def _time_labels(ds: xr.Dataset, n: int) -> list[str]:
    valid_times = _coord_values(ds, "valid_time")
    steps = _coord_values(ds, "step")
    base_times = _coord_values(ds, "time")

    if len(valid_times) == n:
        return valid_times

    if len(steps) == n:
        base = base_times[0] if base_times else "time"
        return [f"{base} + {s}" for s in steps]

    if n == 1:
        if valid_times:
            return [valid_times[0]]
        if base_times:
            return [base_times[0]]
        return ["step000"]

    return [f"step{i:03d}" for i in range(n)]


def _summarize_time(labels: list[str]) -> str:
    if not labels:
        return "待解析"
    if len(labels) == 1:
        return labels[0]
    return f"{labels[0]} 至 {labels[-1]}"


def _summarize_steps(labels: list[str]) -> str:
    if not labels:
        return "待解析"
    return str(len(labels))


def _infer_var_type(var_name: str, units: str, long_name: str) -> str:
    text = f"{var_name} {units} {long_name}".lower()

    if any(k in text for k in ["t2m", "2t", "temperature", "tmp", "d2m", "dewpoint"]):
        return "temperature"

    if any(k in text for k in ["tp", "apcp", "precip", "rain", "total precipitation"]):
        return "precipitation"

    if any(k in text for k in ["pressure", "prmsl", "msl", "sp", "pa"]):
        return "pressure"

    if any(k in text for k in ["wind", "u10", "v10", "ugrd", "vgrd"]):
        return "wind"

    return "generic"


def _convert_values(
    var_name: str,
    units: str,
    long_name: str,
    values: np.ndarray,
) -> tuple[np.ndarray, str, str, str]:
    arr = np.asarray(values, dtype=float)
    units_lower = (units or "").lower()
    var_type = _infer_var_type(var_name, units, long_name)

    if var_type == "temperature" and units_lower in ["k", "kelvin"]:
        return arr - 273.15, "°C", "K → °C", var_type

    if var_type == "pressure" and units_lower in ["pa", "pascal", "pascals"]:
        return arr / 100.0, "hPa", "Pa → hPa", var_type

    if var_type == "precipitation" and units_lower in ["m", "meter", "metre"]:
        return arr * 1000.0, "mm", "m → mm", var_type

    if var_type == "precipitation" and "kg" in units_lower and "m" in units_lower:
        return arr, "mm", f"{units} → mm", var_type

    return arr, units or "未知", "未转换", var_type


def _extract_array_lat_lon(ds: xr.Dataset, var_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_name, lon_name = _find_lat_lon_names(ds)

    da = ds[var_name]
    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)

    keep_dims = {lat_name, lon_name, "time", "valid_time", "step"}

    for dim in list(da.dims):
        if dim not in keep_dims:
            da = da.isel({dim: 0})

    arr = np.asarray(da.values, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        pass
    elif arr.ndim > 3:
        h = arr.shape[-2]
        w = arr.shape[-1]
        arr = arr.reshape(-1, h, w)
    else:
        raise ValueError(f"变量 {var_name} 维度不适合渲染：shape={arr.shape}")

    # latitude 从南到北时，翻转成北到南，便于 PNG 和 extent 对应。
    if lat.size >= 2 and lat[0] < lat[-1]:
        lat = lat[::-1]
        arr = arr[:, ::-1, :]

    # longitude 从大到小时，翻转。
    if lon.size >= 2 and lon[0] > lon[-1]:
        lon = lon[::-1]
        arr = arr[:, :, ::-1]

    return arr, lat, lon


def _stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    total = int(arr.size)
    mask = np.isfinite(arr)
    valid_count = int(mask.sum())
    missing_count = total - valid_count
    missing_ratio = missing_count / total if total else 1.0

    if valid_count == 0:
        return {
            "max": None,
            "min": None,
            "mean": None,
            "valid_count": 0,
            "total_count": total,
            "missing_count": missing_count,
            "missing_ratio": missing_ratio,
        }

    valid = arr[mask]

    return {
        "max": round(float(np.nanmax(valid)), 3),
        "min": round(float(np.nanmin(valid)), 3),
        "mean": round(float(np.nanmean(valid)), 3),
        "valid_count": valid_count,
        "total_count": total,
        "missing_count": missing_count,
        "missing_ratio": missing_ratio,
    }


def _make_bars(values: np.ndarray, bins: int = 5) -> list[int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return [0] * bins

    hist, _ = np.histogram(arr, bins=bins)
    if hist.max() == 0:
        return [0] * bins

    return [int(round(x)) for x in hist / hist.max() * 100]


def _make_trend(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 3:
        series = np.nanmean(arr, axis=(1, 2))
    else:
        series = np.asarray([np.nanmean(arr)])

    return [round(float(x), 3) for x in series]


def _quality_text(missing_ratio: float) -> str:
    if missing_ratio <= 0.01:
        return "正常"
    if missing_ratio <= 0.10:
        return "少量缺测"
    if missing_ratio <= 0.30:
        return "部分缺测"
    return "缺测较多"


def _alert_text(var_type: str, max_value: Any) -> str:
    if max_value is None:
        return "无"

    try:
        max_v = float(max_value)
    except Exception:
        return "无"

    if var_type == "temperature" and max_v >= 35:
        return "高温风险"

    if var_type == "precipitation" and max_v >= 50:
        return "强降水风险"

    if var_type == "wind" and max_v >= 17:
        return "大风风险"

    return "无"


def _collect_variable_names(groups: list[xr.Dataset]) -> str:
    items = []

    for ds in groups:
        for var in ds.data_vars:
            da = ds[var]
            long_name = da.attrs.get("long_name", "")
            units = da.attrs.get("units", "")

            if long_name or units:
                items.append(f"{var}({long_name}, {units})")
            else:
                items.append(var)

    text = "; ".join(items)
    if len(text) > 500:
        text = text[:500] + "..."

    return text or "待解析"


def _var_priority(var_name: str, long_name: str = "") -> int:
    name = var_name.lower()
    text = f"{name} {long_name}".lower()

    if name in {"t2m", "2t", "tmp"}:
        return 1000
    if name in {"d2m", "2d"}:
        return 900
    if name in {"tp", "apcp"}:
        return 850
    if name in {"sp", "msl", "prmsl"}:
        return 800
    if name in {"u10", "v10", "ugrd", "vgrd"}:
        return 700
    if "temperature" in text:
        return 600
    if "precip" in text:
        return 500
    if "pressure" in text:
        return 400
    return 100




def _product_category(var_name: str) -> str:
    mapping = {
        "t2m": "温度产品",
        "d2m": "湿度产品",
        "tp": "降水产品",
        "sp": "气压产品",
    }
    return mapping.get(var_name.lower(), "数值预报产品")


def _business_label(var_name: str, long_name: str) -> str:
    mapping = {
        "t2m": "2米气温",
        "d2m": "2米露点温度",
        "tp": "累积降水",
        "sp": "地面气压",
    }
    return mapping.get(var_name.lower(), long_name or var_name)


def _fixed_color_range(var_name: str, var_type: str, stats: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """
    业务气象系统必须固定色标范围，避免时间动画逐帧自适应导致颜色不可比较。
    返回：vmin, vmax, mode
    """
    name = var_name.lower()
    if name == "t2m":
        return -50.0, 50.0, "fixed"
    if name == "d2m":
        return -50.0, 40.0, "fixed"
    if name == "tp":
        return 0.0, 100.0, "fixed"
    if name == "sp":
        return 850.0, 1100.0, "fixed"

    if stats.get("min") is not None and stats.get("max") is not None:
        return float(stats["min"]), float(stats["max"]), "auto"
    return None, None, "auto"


def _step_hours(ds: xr.Dataset, n: int) -> list[int | None]:
    if "step" not in ds.coords:
        return [None] * n

    vals = np.asarray(ds.coords["step"].values).reshape(-1)
    if vals.size != n:
        return [None] * n

    out: list[int | None] = []
    for v in vals:
        try:
            # np.timedelta64
            h = int(round(float(v / np.timedelta64(1, "h"))))
            out.append(h)
            continue
        except Exception:
            pass

        text = str(v)
        # 兼容 "0 days 06:00:00"
        m = re.search(r"(?:(\d+)\s+days?\s+)?(\d{1,2}):(\d{2}):(\d{2})", text)
        if m:
            days = int(m.group(1) or 0)
            hours = int(m.group(2))
            out.append(days * 24 + hours)
            continue

        m = re.search(r"(\d+)", text)
        out.append(int(m.group(1)) if m else None)

    return out


def _forecast_labels(ds: xr.Dataset, n: int) -> list[str]:
    hours = _step_hours(ds, n)
    labels: list[str] = []
    for i, h in enumerate(hours):
        labels.append(f"F{h:03d}" if h is not None else f"F{i:03d}")
    return labels


def _cycle_time(ds: xr.Dataset) -> str:
    vals = _coord_values(ds, "time")
    return vals[0] if vals else "待解析"

def _display_label(var_name: str, long_name: str) -> str:
    mapping = {
        "t2m": "2m temperature",
        "2t": "2m temperature",
        "d2m": "2m dewpoint",
        "2d": "2m dewpoint",
        "sp": "surface pressure",
        "msl": "mean sea-level pressure",
        "prmsl": "mean sea-level pressure",
        "tp": "total precipitation",
        "apcp": "total precipitation",
        "u10": "10m U wind",
        "v10": "10m V wind",
    }
    return mapping.get(var_name.lower(), long_name or var_name)


def _gradient_for_var(var_type: str) -> str:
    if var_type == "precipitation":
        return "linear-gradient(to right, #f8fafc, #93c5fd, #22c55e, #facc15, #ef4444)"
    if var_type == "pressure":
        return "linear-gradient(to right, #7c3aed, #2563eb, #22c55e, #facc15, #ef4444)"
    if var_type == "wind":
        return "linear-gradient(to right, #e0f2fe, #38bdf8, #2563eb, #7c3aed, #ef4444)"
    return "linear-gradient(to right, #1e40af, #0ea5e9, #22c55e, #facc15, #ef4444)"


def _legend_ticks(vmin: Any, vmax: Any) -> list[str]:
    if vmin is None or vmax is None:
        return ["低", "较低", "中", "较高", "高"]

    try:
        a = float(vmin)
        b = float(vmax)
    except Exception:
        return ["低", "较低", "中", "较高", "高"]

    if abs(a - b) < 1e-12:
        return [f"{a:.0f}"] * 5

    vals = np.linspace(a, b, 5)
    return [f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}" for v in vals]


def _save_one_png(
    values2d: np.ndarray,
    output_path: Path,
    var_type: str = "generic",
    vmin: float | None = None,
    vmax: float | None = None,
) -> str:
    arr = np.asarray(values2d, dtype=float)
    valid = np.isfinite(arr)

    if not valid.any():
        arr = np.zeros_like(arr, dtype=float)
        valid = np.isfinite(arr)

    if vmin is None:
        vmin = float(np.nanmin(arr[valid]))
    if vmax is None:
        vmax = float(np.nanmax(arr[valid]))

    if abs(vmax - vmin) < 1e-12:
        norm = np.zeros_like(arr, dtype=float)
    else:
        norm = (arr - vmin) / (vmax - vmin)

    norm = np.clip(norm, 0, 1)

    if var_type == "pressure":
        cmap = plt.get_cmap("viridis")
    else:
        cmap = plt.get_cmap("turbo")

    rgba = cmap(norm)
    rgba[..., 3] = np.where(valid, 0.72, 0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, rgba)

    return str(output_path).replace("\\", "/")


def _to_png_url(png_path: Path) -> str:
    parts = list(png_path.parts)

    if "data" in parts:
        idx = parts.index("data")
        rel_parts = parts[idx + 1:]
        return "/data/" + "/".join(rel_parts).replace("\\", "/")

    return f"/data/GFS/{png_path.name}"


def _save_png_sequence_for_var(
    path: Path,
    var_key: str,
    arr3d: np.ndarray,
    var_type: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[str, str, list[str], list[str]]:
    png_files: list[str] = []
    png_urls: list[str] = []

    safe_key = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in var_key)

    for i in range(arr3d.shape[0]):
        step_path = path.with_name(f"{path.name}_{safe_key}_step{i:03d}.png")
        png_file = _save_one_png(arr3d[i], step_path, var_type=var_type, vmin=vmin, vmax=vmax)
        png_files.append(png_file)
        png_urls.append(_to_png_url(step_path))

    if not png_files:
        raise ValueError(f"变量 {var_key} 没有生成任何 PNG。")

    compat_path = path.with_name(f"{path.name}_{safe_key}.png")
    first_path = Path(png_files[0])
    if first_path.resolve() != compat_path.resolve():
        shutil.copyfile(first_path, compat_path)

    return (
        str(compat_path).replace("\\", "/"),
        _to_png_url(compat_path),
        png_files,
        png_urls,
    )



def _step_stats(values2d: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values2d, dtype=float)
    total = int(arr.size)
    mask = np.isfinite(arr)
    valid_count = int(mask.sum())

    if valid_count == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "valid_count": 0,
            "total_count": total,
            "missing_ratio": 1.0,
        }

    valid = arr[mask]
    return {
        "min": round(float(np.nanmin(valid)), 3),
        "max": round(float(np.nanmax(valid)), 3),
        "mean": round(float(np.nanmean(valid)), 3),
        "valid_count": valid_count,
        "total_count": total,
        "missing_ratio": round(float(1.0 - valid_count / total), 6) if total else 1.0,
    }


def _save_grid_sequence_for_var(
    path: Path,
    var_key: str,
    arr3d: np.ndarray,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    保存 float32 数值矩阵。
    PNG 负责显示，float32 负责前端鼠标点查真实值。
    """
    grid_files: list[str] = []
    grid_urls: list[str] = []
    step_stats: list[dict[str, Any]] = []

    safe_key = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in var_key)

    for i in range(arr3d.shape[0]):
        values2d = np.asarray(arr3d[i], dtype=float)
        step_stats.append(_step_stats(values2d))

        out = np.where(np.isfinite(values2d), values2d, MISSING_VALUE).astype("<f4")
        grid_path = path.with_name(f"{path.name}_{safe_key}_step{i:03d}.float32")
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        out.tofile(grid_path)

        grid_files.append(str(grid_path).replace("\\", "/"))
        grid_urls.append(_to_png_url(grid_path))

    return grid_files, grid_urls, step_stats

def _build_variable_layers(path: Path, groups: list[xr.Dataset]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    raw_layers: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()

    # 为了让前端变量下拉保持干净，默认只展示 GFS 的 4 个核心业务变量。
    # 后续想扩展风场/高空层，只需要把对应变量名加入这里。
    allowed_vars = {"t2m", "d2m", "tp", "sp"}

    for group_index, ds in enumerate(groups):
        for var_name in ds.data_vars:
            if var_name not in allowed_vars:
                continue

            if var_name in seen:
                continue

            try:
                da = ds[var_name]
                attrs = da.attrs

                long_name = attrs.get("long_name", var_name)
                units = attrs.get("units", "")
                short_name = attrs.get("GRIB_shortName", var_name)
                type_of_level = attrs.get("GRIB_typeOfLevel", attrs.get("typeOfLevel", "surface"))
                step_type = attrs.get("GRIB_stepType", attrs.get("stepType", "unknown"))

                raw_arr, lat, lon = _extract_array_lat_lon(ds, var_name)
                converted_arr, display_unit, conversion, var_type = _convert_values(
                    var_name,
                    units,
                    long_name,
                    raw_arr,
                )

                times = _time_labels(ds, converted_arr.shape[0])
                s = _stats(converted_arr)

                lat_min = round(float(np.nanmin(lat)), 4)
                lat_max = round(float(np.nanmax(lat)), 4)
                lon_min = round(float(np.nanmin(lon)), 4)
                lon_max = round(float(np.nanmax(lon)), 4)

                extent = [lon_min, lat_min, lon_max, lat_max]
                bbox = {
                    "south": lat_min,
                    "north": lat_max,
                    "west": lon_min,
                    "east": lon_max,
                }

                vmin, vmax, color_range_mode = _fixed_color_range(var_name, var_type, s)
                forecast_labels = _forecast_labels(ds, converted_arr.shape[0])
                cycle_time = _cycle_time(ds)

                png_file, png_url, png_files, png_urls = _save_png_sequence_for_var(
                    path=path,
                    var_key=var_name,
                    arr3d=converted_arr,
                    var_type=var_type,
                    vmin=vmin,
                    vmax=vmax,
                )

                grid_files, grid_urls, step_stats = _save_grid_sequence_for_var(
                    path=path,
                    var_key=var_name,
                    arr3d=converted_arr,
                )

                label = _business_label(var_name, long_name)
                english_label = _display_label(var_name, long_name)
                product_category = _product_category(var_name)

                layer = {
                    "key": var_name,
                    "label": label,
                    "englishLabel": english_label,
                    "productCategory": product_category,
                    "productType": product_category,
                    "element": f"{label}（{var_name} / {long_name}）",
                    "long_name": long_name,
                    "shortName": short_name,
                    "rawUnit": units,
                    "unit": display_unit,
                    "displayUnit": display_unit,
                    "conversion": conversion,
                    "varType": var_type,
                    "groupIndex": group_index,
                    "level": f"{type_of_level}, stepType={step_type}",
                    "time": _summarize_time(times),
                    "times": times,
                    "valid_times": times,
                    "forecast_labels": forecast_labels,
                    "cycle_time": cycle_time,
                    "issue_time": cycle_time,
                    "steps": _summarize_steps(times),
                    "extent": extent,
                    "bbox": bbox,
                    "range": f"纬度 {lat_min} ~ {lat_max}，经度 {lon_min} ~ {lon_max}",
                    "resolution": "待解析" if lat.size < 2 or lon.size < 2 else f"{abs(float(lat[1] - lat[0])):.2f}° × {abs(float(lon[1] - lon[0])):.2f}°",
                    "grid": {
                        "nx": int(lon.size),
                        "ny": int(lat.size),
                        "text": f"{lat.size} × {lon.size}",
                    },
                    "gridText": f"{lat.size} × {lon.size}",
                    "validGrid": f"{s['valid_count']} / {s['total_count']}",
                    "coverage": f"{(1 - s['missing_ratio']) * 100:.2f}%",
                    "missing": MISSING_VALUE,
                    "missingText": f"{s['missing_ratio'] * 100:.2f}%",
                    "quality": _quality_text(float(s["missing_ratio"])),
                    "max": s["max"],
                    "min": s["min"],
                    "mean": s["mean"],
                    "alert": _alert_text(var_type, s["max"]),
                    "bars": _make_bars(converted_arr),
                    "trend": _make_trend(converted_arr),
                    "gradient": _gradient_for_var(var_type),
                    "legend_ticks": _legend_ticks(vmin, vmax),
                    "color_range": {"min": vmin, "max": vmax, "mode": color_range_mode},
                    "png": png_file,
                    "png_url": png_url,
                    "png_files": png_files,
                    "png_urls": png_urls,
                    "grid_files": grid_files,
                    "grid_urls": grid_urls,
                    "step_stats": step_stats,
                }

                priority = _var_priority(var_name, long_name)
                raw_layers.append((priority, var_name, layer))
                seen.add(var_name)

            except Exception as e:
                print(f"[WARN] Skip variable {var_name}: {e}")
                continue

    if not raw_layers:
        raise ValueError("GRIB 文件中没有可渲染变量。")

    raw_layers.sort(key=lambda x: x[0], reverse=True)

    variable_options: list[dict[str, Any]] = []
    variable_layers: dict[str, Any] = {}

    for _, var_name, layer in raw_layers:
        variable_options.append({
            "key": var_name,
            "label": layer["label"],
            "element": layer["element"],
            "productCategory": layer.get("productCategory"),
            "unit": layer["unit"],
            "varType": layer["varType"],
            "min": layer["min"],
            "max": layer["max"],
            "legend_ticks": layer["legend_ticks"],
            "gradient": layer["gradient"],
        })
        variable_layers[var_name] = layer

    default_var = variable_options[0]["key"]
    return variable_options, variable_layers, default_var


def _build_weather_info_from_layer(
    path: Path,
    groups: list[xr.Dataset],
    layer: dict[str, Any],
    variable_options: list[dict[str, Any]],
    variable_layers: dict[str, Any],
    default_variable: str,
) -> dict[str, Any]:
    return {
        "source": "GFS",
        "product": "GFS/ECMWF 数值预报产品",
        "productCategory": layer.get("productCategory", "数值预报产品"),
        "issueTime": layer.get("issue_time", "待解析"),
        "cycleTime": layer.get("cycle_time", "待解析"),
        "element": layer.get("element", "GRIB 变量"),
        "time": layer.get("time", "待解析"),
        "level": layer.get("level", "待解析"),
        "range": layer.get("range", "待解析"),
        "resolution": layer.get("resolution", "待解析"),
        "grid": layer.get("gridText", "待解析"),
        "validGrid": layer.get("validGrid", "待解析"),
        "coverage": layer.get("coverage", "待解析"),
        "missing": layer.get("missingText", "待解析"),
        "unit": layer.get("unit", "待解析"),
        "variables": _collect_variable_names(groups),
        "steps": layer.get("steps", "待解析"),
        "status": "解析成功",
        "quality": layer.get("quality", "待解析"),
        "max": layer.get("max"),
        "min": layer.get("min"),
        "mean": layer.get("mean"),
        "alert": layer.get("alert", "无"),
        "update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bars": layer.get("bars", []),
        "trend": layer.get("trend", []),
        "mainVariable": default_variable,
        "mainVariableName": layer.get("long_name"),
        "shortName": layer.get("shortName"),
        "rawUnit": layer.get("rawUnit"),
        "displayUnit": layer.get("displayUnit"),
        "conversion": layer.get("conversion"),
        "varType": layer.get("varType"),
        "groupIndex": layer.get("groupIndex"),
        "extent": layer.get("extent", []),
        "bbox": layer.get("bbox"),
        "png": layer.get("png"),
        "png_url": layer.get("png_url"),
        "png_files": layer.get("png_files", []),
        "png_urls": layer.get("png_urls", []),
        "grid_files": layer.get("grid_files", []),
        "grid_urls": layer.get("grid_urls", []),
        "gridShape": layer.get("grid", {}),
        "step_stats": layer.get("step_stats", []),
        "times": layer.get("times", []),
        "forecast_labels": layer.get("forecast_labels", []),
        "valid_times": layer.get("valid_times", []),
        "fileSizeMB": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else None,
        "variable_options": variable_options,
        "variable_layers": variable_layers,
        "default_variable": default_variable,
    }


def _build_panel_meta(
    path: Path,
    weather_info: dict[str, Any],
    extent: list[float],
    png_url: str,
    png_urls: list[str],
    times: list[str],
) -> dict[str, Any]:
    return {
        "file": path.name,
        "element": weather_info.get("element", "待解析"),
        "time": weather_info.get("time", "待解析"),
        "level": weather_info.get("level", "待解析"),
        "range": weather_info.get("range", "待解析"),
        "grid": weather_info.get("grid", "待解析"),
        "missing": weather_info.get("missing", "待解析"),
        "unit": weather_info.get("unit", "待解析"),
        "vars": weather_info.get("variables", "待解析"),
        "steps": weather_info.get("steps", "待解析"),
        "extent": extent,
        "png_url": png_url,
        "png_urls": png_urls,
        "grid_urls": weather_info.get("grid_urls", []),
        "gridShape": weather_info.get("gridShape", {}),
        "step_stats": weather_info.get("step_stats", []),
        "times": times,
        "forecast_labels": weather_info.get("forecast_labels", []),
        "issueTime": weather_info.get("issueTime", "待解析"),
        "cycleTime": weather_info.get("cycleTime", "待解析"),
        "productCategory": weather_info.get("productCategory", "待解析"),
        "status": weather_info.get("status", "待解析"),
        "quality": weather_info.get("quality", "待解析"),
        "max": weather_info.get("max"),
        "min": weather_info.get("min"),
        "mean": weather_info.get("mean"),
        "alert": weather_info.get("alert"),
        "variable_options": weather_info.get("variable_options", []),
        "variable_layers": weather_info.get("variable_layers", {}),
        "default_variable": weather_info.get("default_variable"),
    }


def _write_meta_again(result: dict[str, Any]) -> None:
    meta_file = result.get("meta_file")
    if not meta_file:
        return

    try:
        meta_path = Path(meta_file)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def process_file(file_path: str, data_type: str = "GFS") -> dict[str, Any]:
    path = Path(file_path)
    file_format = "GRIB2" if path.suffix.lower() == ".grib2" else "GRIB"

    weather_info: dict[str, Any] = {
        "source": "GFS",
        "product": "GFS/ECMWF 数值预报产品",
        "element": "GRIB 变量",
        "time": "待解析",
        "level": "待解析",
        "range": "待解析",
        "resolution": "待解析",
        "grid": "待解析",
        "validGrid": "待解析",
        "coverage": "待解析",
        "missing": "待解析",
        "unit": "待解析",
        "variables": "待解析",
        "steps": "待解析",
        "status": "已接收",
        "quality": "待解析",
        "max": None,
        "min": None,
        "mean": None,
        "alert": "无",
        "update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bars": [0, 0, 0, 0, 0],
        "trend": [],
        "variable_options": [],
        "variable_layers": {},
        "default_variable": None,
    }

    extent: list[float] = []
    bbox: dict[str, float] | None = None
    png_file: str | None = None
    png_url: str | None = None
    png_files: list[str] = []
    png_urls: list[str] = []
    times: list[str] = []
    variable_options: list[dict[str, Any]] = []
    variable_layers: dict[str, Any] = {}
    default_variable: str | None = None

    try:
        groups = _open_grib_groups(str(path))

        variable_options, variable_layers, default_variable = _build_variable_layers(path, groups)
        main_layer = variable_layers[default_variable]

        extent = main_layer.get("extent", [])
        bbox = main_layer.get("bbox")
        png_file = main_layer.get("png")
        png_url = main_layer.get("png_url")
        png_files = main_layer.get("png_files", [])
        png_urls = main_layer.get("png_urls", [])
        times = main_layer.get("times", [])

        weather_info = _build_weather_info_from_layer(
            path=path,
            groups=groups,
            layer=main_layer,
            variable_options=variable_options,
            variable_layers=variable_layers,
            default_variable=default_variable,
        )

    except Exception as exc:
        weather_info.update({
            "status": "解析失败",
            "quality": "异常",
            "alert": "解析失败",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    result = process_basic_file(
        str(path),
        data_type=data_type,
        file_format=file_format,
        weather_info=weather_info,
    )

    if not isinstance(result, dict):
        return weather_info

    panel_meta = _build_panel_meta(
        path=path,
        weather_info=weather_info,
        extent=extent,
        png_url=png_url or "",
        png_urls=png_urls,
        times=times,
    )

    result["file_name"] = path.name
    result["directory"] = str(path.parent).replace("\\", "/") + "/"
    result["business_type"] = data_type
    result["data_type"] = data_type

    result["weather_info"] = weather_info
    result["meta"] = panel_meta

    result["bbox"] = bbox
    result["extent"] = extent
    result["times"] = times

    # 兼容旧前端：默认变量仍然暴露为 png_url/png_urls。
    result["png"] = png_file
    result["png_url"] = png_url
    result["png_files"] = png_files
    result["png_urls"] = png_urls

    # 新前端：多变量图层。
    result["variable_options"] = variable_options
    result["variable_layers"] = variable_layers
    result["default_variable"] = default_variable

    result["extra"] = {
        "parser": "adapters.gfs_adapter.process_file",
        "main_variable": weather_info.get("mainVariable"),
        "main_variable_name": weather_info.get("mainVariableName"),
        "raw_unit": weather_info.get("rawUnit"),
        "display_unit": weather_info.get("displayUnit"),
        "conversion": weather_info.get("conversion"),
        "extent": extent,
        "bbox": bbox,
        "png_url": png_url,
        "png_urls": png_urls,
        "grid_urls": weather_info.get("grid_urls", []),
        "gridShape": weather_info.get("gridShape", {}),
        "step_stats": weather_info.get("step_stats", []),
        "times": times,
        "variable_options": variable_options,
        "variable_layers": variable_layers,
        "default_variable": default_variable,
    }

    result.update(weather_info)

    _write_meta_again(result)
    return result
