from __future__ import annotations

import json
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


def _open_grib_groups(file_path: str) -> list[xr.Dataset]:
    """
    优先使用 cfgrib.open_datasets，避免 tp / instant / accum 混合时被跳过。
    """
    try:
        groups = cfgrib.open_datasets(
            file_path,
            backend_kwargs={"indexpath": ""}
        )
        return list(groups)
    except Exception:
        ds = xr.open_dataset(
            file_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""}
        )
        return [ds]


def _find_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "x"]

    lat_name = None
    lon_name = None

    for name in lat_candidates:
        if name in ds.coords or name in ds.dims:
            lat_name = name
            break

    for name in lon_candidates:
        if name in ds.coords or name in ds.dims:
            lon_name = name
            break

    if lat_name is None or lon_name is None:
        raise ValueError("无法识别经纬度坐标。")

    return lat_name, lon_name


def _format_time_value(value: Any) -> str:
    try:
        if isinstance(value, np.datetime64):
            return str(value).replace(".000000000", "")
        return str(value).replace(".000000000", "")
    except Exception:
        return str(value)


def _summarize_time(ds: xr.Dataset) -> str:
    """
    优先使用 valid_time，其次 time。
    """
    for name in ["valid_time", "time"]:
        if name in ds.coords:
            arr = np.asarray(ds.coords[name].values).reshape(-1)
            if arr.size == 0:
                continue

            first = _format_time_value(arr[0])
            last = _format_time_value(arr[-1])

            if first == last:
                return first

            return f"{first} 至 {last}"

    return "待解析"


def _summarize_steps(ds: xr.Dataset) -> str:
    if "step" not in ds.coords:
        return "待解析"

    arr = np.asarray(ds.coords["step"].values).reshape(-1)

    if arr.size == 0:
        return "待解析"

    first = str(arr[0])
    last = str(arr[-1])

    if arr.size == 1:
        return first

    return f"{first} 至 {last}，共 {arr.size} 个时效"


def _get_resolution(lat: np.ndarray, lon: np.ndarray) -> str:
    if lat.size >= 2:
        dlat = abs(float(lat[1]) - float(lat[0]))
    else:
        dlat = 0.0

    if lon.size >= 2:
        dlon = abs(float(lon[1]) - float(lon[0]))
    else:
        dlon = 0.0

    if dlat > 0 and dlon > 0:
        return f"{dlat:.2f}° × {dlon:.2f}°"

    return "待解析"


def _infer_var_type(var_name: str, units: str, long_name: str) -> str:
    text = f"{var_name} {units} {long_name}".lower()

    if any(k in text for k in ["t2m", "2t", "temperature", "tmp", "d2m", "dewpoint"]):
        return "temperature"

    if any(k in text for k in ["tp", "precip", "rain", "apcp", "total precipitation"]):
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
    """
    返回：转换后数组、展示单位、转换说明、变量类型
    """
    arr = np.asarray(values, dtype=float)
    units_lower = (units or "").lower()
    var_type = _infer_var_type(var_name, units, long_name)

    # 温度：K → ℃
    if var_type == "temperature" and units_lower in ["k", "kelvin"]:
        return arr - 273.15, "°C", "K → °C", var_type

    # 气压：Pa → hPa
    if var_type == "pressure" and units_lower in ["pa", "pascal", "pascals"]:
        return arr / 100.0, "hPa", "Pa → hPa", var_type

    # 降水：m → mm
    if var_type == "precipitation" and units_lower in ["m", "meter", "metre"]:
        return arr * 1000.0, "mm", "m → mm", var_type

    # GFS 降水常见 kg m**-2，数值上近似 mm
    if var_type == "precipitation" and "kg" in units_lower and "m" in units_lower:
        return arr, "mm", f"{units} → mm", var_type

    if units:
        return arr, units, "未转换", var_type

    return arr, "未知", "未转换", var_type


def _choose_main_variable(groups: list[xr.Dataset]) -> tuple[int, xr.Dataset, str]:
    """
    选择主展示变量。
    优先级：
    1. t2m
    2. tp
    3. prmsl / msl / sp
    4. d2m
    5. u10 / v10
    6. 第一个变量
    """
    priority = [
        "t2m", "2t", "tmp",
        "tp", "apcp",
        "prmsl", "msl", "sp",
        "d2m", "2d", "dpt",
        "u10", "v10", "ugrd", "vgrd",
    ]

    for wanted in priority:
        for gi, ds in enumerate(groups):
            for var in ds.data_vars:
                if var.lower() == wanted.lower():
                    return gi, ds, var

    for gi, ds in enumerate(groups):
        data_vars = list(ds.data_vars)
        if data_vars:
            return gi, ds, data_vars[0]

    raise ValueError("GRIB 文件中没有可用变量。")


def _to_2d_or_3d_array(ds: xr.Dataset, var_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输出：
    - values: 2D 或 3D 数组
    - lat
    - lon

    如果变量有额外维度，例如 level / heightAboveGround，默认取第 0 层。
    """
    lat_name, lon_name = _find_lat_lon_names(ds)

    da = ds[var_name]

    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)

    keep_dims = {lat_name, lon_name, "time", "valid_time", "step"}

    for dim in list(da.dims):
        if dim not in keep_dims:
            da = da.isel({dim: 0})

    arr = np.asarray(da.values, dtype=float)

    # 去掉长度为 1 的维度
    arr = np.squeeze(arr)

    # 最终保证最后两维是 lat × lon
    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        pass
    elif arr.ndim > 3:
        h = arr.shape[-2]
        w = arr.shape[-1]
        arr = arr.reshape(-1, h, w)
    else:
        raise ValueError(f"变量 {var_name} 的维度不适合渲染：shape={arr.shape}")

    # 纬度统一为北到南
    if lat.size >= 2 and lat[0] < lat[-1]:
        lat = lat[::-1]
        if arr.ndim == 2:
            arr = arr[::-1, :]
        else:
            arr = arr[:, ::-1, :]

    # 经度统一为西到东
    if lon.size >= 2 and lon[0] > lon[-1]:
        lon = lon[::-1]
        if arr.ndim == 2:
            arr = arr[:, ::-1]
        else:
            arr = arr[:, :, ::-1]

    return arr, lat, lon


def _stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    total_count = int(arr.size)
    valid_mask = np.isfinite(arr)
    valid_count = int(valid_mask.sum())
    missing_count = total_count - valid_count
    missing_ratio = missing_count / total_count if total_count else 1.0

    if valid_count == 0:
        return {
            "max": None,
            "min": None,
            "mean": None,
            "valid_count": 0,
            "total_count": total_count,
            "missing_count": missing_count,
            "missing_ratio": missing_ratio,
        }

    valid = arr[valid_mask]

    return {
        "max": round(float(np.nanmax(valid)), 3),
        "min": round(float(np.nanmin(valid)), 3),
        "mean": round(float(np.nanmean(valid)), 3),
        "valid_count": valid_count,
        "total_count": total_count,
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

    scaled = hist / hist.max() * 100
    return [int(round(x)) for x in scaled]


def _make_trend(values: np.ndarray, target_len: int = 8) -> list[float]:
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 2:
        mean_value = float(np.nanmean(arr))
        return [round(mean_value, 3)] * target_len

    if arr.ndim == 3:
        series = np.nanmean(arr, axis=(1, 2))
    else:
        series = np.asarray([np.nanmean(arr)])

    series = np.asarray(series, dtype=float)

    if series.size == 0:
        return [0.0] * target_len

    if series.size == target_len:
        return [round(float(x), 3) for x in series]

    x_old = np.linspace(0, 1, series.size)
    x_new = np.linspace(0, 1, target_len)
    interp = np.interp(x_new, x_old, series)

    return [round(float(x), 3) for x in interp]


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


def _save_png(values: np.ndarray, output_path: Path) -> str:
    """
    生成前端 WebglLayer 可用的透明 PNG。
    注意：这里不是地理严格配准图，只是 MVP 阶段的格点场贴图。
    """
    arr = np.asarray(values, dtype=float)

    if arr.ndim == 3:
        arr2d = arr[0]
    elif arr.ndim == 2:
        arr2d = arr
    else:
        arr2d = np.squeeze(arr)
        if arr2d.ndim > 2:
            arr2d = arr2d[0]

    valid = np.isfinite(arr2d)

    if not valid.any():
        arr2d = np.zeros_like(arr2d, dtype=float)
        valid = np.isfinite(arr2d)

    vmin = float(np.nanmin(arr2d[valid]))
    vmax = float(np.nanmax(arr2d[valid]))

    if abs(vmax - vmin) < 1e-12:
        norm = np.zeros_like(arr2d, dtype=float)
    else:
        norm = (arr2d - vmin) / (vmax - vmin)

    norm = np.clip(norm, 0, 1)

    cmap = plt.get_cmap("turbo")
    rgba = cmap(norm)

    # 有效格点半透明，无效格点透明
    rgba[..., 3] = np.where(valid, 0.72, 0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, rgba)

    return str(output_path).replace("\\", "/")


def _write_meta_again(result: dict[str, Any]) -> None:
    """
    process_basic_file 可能先写了 meta.json。
    这里把补强后的 result 再写回一次，确保 meta.json 中有 png/png_files/weather_info。
    """
    meta_file = result.get("meta_file")

    if not meta_file:
        return

    try:
        meta_path = Path(meta_file)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass


def process_file(file_path: str, data_type: str = "GFS") -> dict[str, Any]:
    path = Path(file_path)

    weather_info: dict[str, Any] = {
        "source": "GFS",
        "product": "GFS 数值预报产品",
        "element": "GRIB2 变量",
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
        "max": "待解析",
        "min": "待解析",
        "mean": "待解析",
        "alert": "无",
        "update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bars": [0, 0, 0, 0, 0],
        "trend": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    file_format = "GRIB2" if path.suffix.lower() == ".grib2" else "GRIB"

    try:
        groups = _open_grib_groups(str(path))
        group_index, ds, main_var = _choose_main_variable(groups)

        da = ds[main_var]
        attrs = da.attrs

        long_name = attrs.get("long_name", main_var)
        units = attrs.get("units", "")
        short_name = attrs.get("GRIB_shortName", main_var)
        type_of_level = attrs.get("GRIB_typeOfLevel", attrs.get("typeOfLevel", "surface"))
        step_type = attrs.get("GRIB_stepType", attrs.get("stepType", "unknown"))

        raw_arr, lat, lon = _to_2d_or_3d_array(ds, main_var)
        converted_arr, display_unit, conversion, var_type = _convert_values(
            main_var,
            units,
            long_name,
            raw_arr,
        )

        s = _stats(converted_arr)

        lat_min = round(float(np.nanmin(lat)), 4)
        lat_max = round(float(np.nanmax(lat)), 4)
        lon_min = round(float(np.nanmin(lon)), 4)
        lon_max = round(float(np.nanmax(lon)), 4)

        png_path = path.with_name(path.name + ".png")
        png_file = _save_png(converted_arr, png_path)

        weather_info.update({
            "source": "GFS",
            "product": "GFS 数值预报产品",
            "element": f"{main_var} / {long_name}",
            "time": _summarize_time(ds),
            "level": f"{type_of_level}，stepType={step_type}",
            "range": f"纬度 {lat_min} ~ {lat_max}，经度 {lon_min} ~ {lon_max}",
            "resolution": _get_resolution(lat, lon),
            "grid": f"{lat.size} × {lon.size}",
            "validGrid": f"{s['valid_count']} / {s['total_count']}",
            "coverage": f"{(1 - s['missing_ratio']) * 100:.2f}%",
            "missing": f"{s['missing_ratio'] * 100:.2f}%",
            "unit": display_unit,
            "variables": _collect_variable_names(groups),
            "steps": _summarize_steps(ds),
            "status": "解析成功",
            "quality": _quality_text(float(s["missing_ratio"])),
            "max": s["max"],
            "min": s["min"],
            "mean": s["mean"],
            "alert": _alert_text(var_type, s["max"]),
            "update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "bars": _make_bars(converted_arr),
            "trend": _make_trend(converted_arr),

            "mainVariable": main_var,
            "mainVariableName": long_name,
            "shortName": short_name,
            "rawUnit": units,
            "displayUnit": display_unit,
            "conversion": conversion,
            "varType": var_type,
            "groupIndex": group_index,
            "latMin": lat_min,
            "latMax": lat_max,
            "lonMin": lon_min,
            "lonMax": lon_max,
            "fileSizeMB": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else None,
            "png": png_file,
            "png_file": png_file,
        })

    except Exception as exc:
        weather_info.update({
            "status": "解析失败",
            "quality": "异常",
            "alert": "解析失败",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    basic_result = process_basic_file(
        str(path),
        data_type=data_type,
        file_format=file_format,
        weather_info=weather_info,
    )

    if isinstance(basic_result, dict):
        basic_result["weather_info"] = weather_info
        basic_result.update(weather_info)

        variable_text = weather_info.get("variables", "")
        variable_items = []
        if isinstance(variable_text, str) and variable_text not in ["", "待解析"]:
            variable_items = [x.strip() for x in variable_text.split(";") if x.strip()]

        time_text = weather_info.get("time", "")
        times = []
        if isinstance(time_text, str) and time_text not in ["", "待解析"]:
            times = [time_text]

        level_text = weather_info.get("level", "")
        levels = []
        if isinstance(level_text, str) and level_text not in ["", "待解析"]:
            levels = [level_text]

        lat_min = weather_info.get("latMin")
        lat_max = weather_info.get("latMax")
        lon_min = weather_info.get("lonMin")
        lon_max = weather_info.get("lonMax")

        bbox = None
        if None not in [lat_min, lat_max, lon_min, lon_max]:
            bbox = {
                "south": lat_min,
                "north": lat_max,
                "west": lon_min,
                "east": lon_max,
            }

        png_file = weather_info.get("png_file") or weather_info.get("png")

        basic_result["variables"] = variable_items
        basic_result["times"] = times
        basic_result["levels"] = levels
        basic_result["bbox"] = bbox

        if png_file:
            basic_result["png_files"] = [png_file]
            basic_result["png"] = png_file

        basic_result["extra"] = {
            "status": "parsed",
            "parser": "adapters.gfs_adapter.process_file",
            "main_variable": weather_info.get("mainVariable"),
            "main_variable_name": weather_info.get("mainVariableName"),
            "var_type": weather_info.get("varType"),
            "raw_unit": weather_info.get("rawUnit"),
            "display_unit": weather_info.get("displayUnit"),
            "conversion": weather_info.get("conversion"),
            "group_index": weather_info.get("groupIndex"),
            "file_size_mb": weather_info.get("fileSizeMB"),
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "png": png_file,
        }

        _write_meta_again(basic_result)
        return basic_result

    weather_info["basic_result"] = basic_result
    return weather_info