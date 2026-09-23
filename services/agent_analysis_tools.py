"""Scoped data-analysis tools built on parsed adapter outputs.

The first batch is intentionally small:
- ERA5 time-series statistics
- WRF forecast summary
- Radar echo-change summary

Each tool starts from adapter-produced meta.json, then reads the referenced
source file only when a calculation is required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from services.agent_plotting import line_chart


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "backend_system" / "data"
DATA_PUBLIC_BASE = os.getenv("WEATHER_DATA_BASE", "http://127.0.0.1:8002").strip().rstrip("/")

DEFAULT_ERA5_META = DATA_ROOT / "ERA5" / "data_0.nc.meta.json"
DEFAULT_WRF_META = DATA_ROOT / "WRF" / "wrfout_d01_2025-07-16_08_00_00.meta.json"
RADAR_META_GLOB = "Radar/**/*.meta.json"

_LABELS = {
    "analyze_era5_timeseries": "分析 ERA5 时序变化中",
    "analyze_wrf_forecast": "分析 WRF 预报结果中",
    "analyze_radar_echo_change": "分析雷达回波变化中",
}


def label_for(name: str) -> str:
    return _LABELS.get(name, "")


def list_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_era5_timeseries",
                "description": "基于 ERA5 已解析 meta.json 和原始 NC 文件，分析指定变量的逐时次均值、最大值、最小值和变化趋势。适合 t2m/tp/u10/v10/sp/ssrd。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "description": "必填，ERA5 变量名，例如 t2m、tp、u10、v10、sp、ssrd"},
                        "meta_file": {"type": "string", "description": "可选，ERA5 meta.json 路径；不填使用当前样例 data_0.nc.meta.json"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_wrf_forecast",
                "description": "基于 WRF 已解析 meta.json 和 wrfout 文件，输出 T2/RAINNC/U10/V10 等变量的预报结果统计和已有图像引用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variables": {"type": "string", "description": "可选，逗号分隔变量名，例如 T2,RAINNC,U10,V10；不填默认 T2,RAINNC,U10,V10"},
                        "meta_file": {"type": "string", "description": "可选，WRF meta.json 路径；不填使用当前样例 wrfout meta"},
                        "resolution": {"type": "string", "description": "可选，已有渲染分辨率，例如 1km 或 3km"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_radar_echo_change",
                "description": "基于连续 Radar meta/source 文件，分析 DBZH 等变量的最大值、平均值和强回波格点变化趋势。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variable": {"type": "string", "description": "可选，雷达变量名，默认 DBZH"},
                        "meta_files": {"type": "string", "description": "可选，逗号分隔 Radar meta.json 路径；不填自动使用现有 Radar meta"},
                        "threshold": {"type": "string", "description": "可选，强回波阈值，默认 35"},
                    },
                },
            },
        },
    ]


def dispatch(name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    args = args if isinstance(args, dict) else {}
    if name == "analyze_era5_timeseries":
        variable = str(args.get("variable") or "").strip()
        if not variable:
            return _need_params(name, "ERA5 时序变化分析", ["variable"])
        return analyze_era5_timeseries(variable=variable, meta_file=str(args.get("meta_file") or ""))
    if name == "analyze_wrf_forecast":
        return analyze_wrf_forecast(
            variables=args.get("variables"),
            meta_file=str(args.get("meta_file") or ""),
            resolution=str(args.get("resolution") or ""),
        )
    if name == "analyze_radar_echo_change":
        return analyze_radar_echo_change(
            variable=str(args.get("variable") or "DBZH").strip() or "DBZH",
            meta_files=args.get("meta_files"),
            threshold=args.get("threshold"),
        )
    return {"status": "error", "summary": f"未知分析工具：{name}", "image_url": None}


def analyze_era5_timeseries(*, variable: str, meta_file: str = "") -> dict[str, Any]:
    meta_path = _resolve_meta_file(meta_file, DEFAULT_ERA5_META)
    meta = _read_json(meta_path)
    source = _source_from_meta(meta, meta_path)
    if not source.exists():
        return _error(f"ERA5 原始文件不存在：{source}", meta_path)

    with xr.open_dataset(source) as dataset:
        if variable not in dataset:
            return _error(f"ERA5 变量不存在：{variable}。可选变量：{', '.join(dataset.data_vars)}", meta_path)
        arr = dataset[variable]
        time_dim = _time_dim(arr)
        spatial_dims = [dim for dim in arr.dims if dim != time_dim]
        reduced_mean = arr.mean(dim=spatial_dims, skipna=True) if spatial_dims else arr
        reduced_max = arr.max(dim=spatial_dims, skipna=True) if spatial_dims else arr
        reduced_min = arr.min(dim=spatial_dims, skipna=True) if spatial_dims else arr

        times = _time_values(dataset, arr, time_dim, meta)
        means = _to_float_list(reduced_mean.values)
        maxs = _to_float_list(reduced_max.values)
        mins = _to_float_list(reduced_min.values)

    means = [_convert_value(variable, value) for value in means]
    maxs = [_convert_value(variable, value) for value in maxs]
    mins = [_convert_value(variable, value) for value in mins]
    unit = _display_unit(variable, _variable_unit(meta, variable))
    records = [
        {"time": times[index] if index < len(times) else str(index), "mean": means[index], "max": maxs[index], "min": mins[index]}
        for index in range(min(len(means), len(maxs), len(mins)))
    ]
    if not records:
        return _error(f"ERA5 变量 {variable} 没有可统计数据。", meta_path)

    peak = max(records, key=lambda item: _safe_number(item["mean"]))
    low = min(records, key=lambda item: _safe_number(item["mean"]))
    trend = _trend([item["mean"] for item in records])
    chart_url = line_chart(
        f"ERA5 {variable} 时序变化",
        {f"{variable} mean": [float(item["mean"]) for item in records]},
        [item["time"] for item in records],
    )
    summary = (
        f"ERA5 {variable} 时序分析完成，共 {len(records)} 个时次。"
        f"均值最高出现在 {peak['time']}，约 {peak['mean']:.3f}{unit}；"
        f"均值最低出现在 {low['time']}，约 {low['mean']:.3f}{unit}。"
        f"整体趋势：{trend}。"
    )
    return _ok(summary, image_url=chart_url, records=records, sources=_sources(meta_path, source), warnings=_source_warnings(meta, source))


def analyze_wrf_forecast(*, variables: Any = None, meta_file: str = "", resolution: str = "") -> dict[str, Any]:
    meta_path = _resolve_meta_file(meta_file, DEFAULT_WRF_META)
    meta = _read_json(meta_path)
    source = _source_from_meta(meta, meta_path)
    if not source.exists():
        return _error(f"WRF 原始文件不存在：{source}", meta_path)

    selected = _split_values(variables) or ["T2", "RAINNC", "U10", "V10"]
    records = []
    with xr.open_dataset(source) as dataset:
        for name in selected:
            if name not in dataset:
                records.append({"variable": name, "status": "missing"})
                continue
            values = np.asarray(dataset[name].values, dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                records.append({"variable": name, "status": "empty"})
                continue
            stats = {
                "variable": name,
                "status": "ok",
                "min": _convert_value(name, float(np.nanmin(finite))),
                "max": _convert_value(name, float(np.nanmax(finite))),
                "mean": _convert_value(name, float(np.nanmean(finite))),
                "unit": _display_unit(name, _variable_unit(meta, name)),
            }
            records.append(stats)

    ok_records = [item for item in records if item.get("status") == "ok"]
    images = _wrf_images(meta, [item["variable"] for item in ok_records], resolution)
    if not ok_records:
        return _error("WRF 未找到可统计变量。", meta_path)

    lines = [
        f"{item['variable']}：均值 {item['mean']:.3f}{item['unit']}，范围 {item['min']:.3f} - {item['max']:.3f}{item['unit']}"
        for item in ok_records
    ]
    summary = (
        f"WRF 预报结果分析完成，时次 {', '.join(str(t) for t in (meta.get('times') or [])[:3]) or '未标明'}。"
        + "；".join(lines)
        + "。"
    )
    return _ok(summary, image_url=images[0] if images else None, records=records, image_urls=images, sources=_sources(meta_path, source), warnings=_source_warnings(meta, source))


def analyze_radar_echo_change(*, variable: str = "DBZH", meta_files: Any = None, threshold: Any = None) -> dict[str, Any]:
    threshold_value = _float_value(threshold, 35.0)
    paths = [_resolve_meta_file(item, None) for item in _split_values(meta_files)]
    if not paths:
        paths = [
            path
            for path in sorted(DATA_ROOT.glob(RADAR_META_GLOB))
            if ".adapter_staging" not in path.parts
        ][:8]
    if not paths:
        return _error(f"Radar 数据目录下没有可分析的 meta.json：{DATA_ROOT / 'Radar'}", None)
    records = []
    sources = []
    images = []
    for meta_path in paths:
        meta = _read_json(meta_path)
        source = _source_from_meta(meta, meta_path)
        if not source.exists():
            records.append({"meta_file": str(meta_path), "status": "missing_source"})
            continue
        with xr.open_dataset(source, decode_times=False) as dataset:
            data_name = _dataset_variable_name(dataset, meta, variable)
            if not data_name:
                records.append({"meta_file": str(meta_path), "status": "missing_variable", "time": _meta_time(meta)})
                continue
            values = np.asarray(dataset[data_name].values, dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                records.append({"meta_file": str(meta_path), "status": "empty", "time": _meta_time(meta)})
                continue
            strong_count = int(np.sum(finite >= threshold_value))
            records.append(
                {
                    "time": _meta_time(meta),
                    "variable": variable,
                    "source_variable": data_name,
                    "status": "ok",
                    "max": float(np.nanmax(finite)),
                    "mean": float(np.nanmean(finite)),
                    "strong_count": strong_count,
                    "threshold": threshold_value,
                    "source_file": str(source),
                    "meta_file": str(meta_path),
                }
            )
            sources.append({"meta_file": str(meta_path), "source_file": str(source)})
            image = _first_image(meta)
            if image:
                images.append(_public_data_url(image))

    ok_records = [item for item in records if item.get("status") == "ok"]
    if not ok_records:
        return _error(f"Radar 未找到可统计的 {variable} 数据。", paths[0] if paths else None)

    trend = _trend([item["max"] for item in ok_records])
    peak = max(ok_records, key=lambda item: _safe_number(item["max"]))
    chart_url = line_chart(
        f"Radar {variable} 回波变化",
        {f"{variable} max": [float(item["max"]) for item in ok_records], "strong count": [float(item["strong_count"]) for item in ok_records]},
        [item["time"] for item in ok_records],
    )
    summary = (
        f"Radar {variable} 回波变化分析完成，共 {len(ok_records)} 个有效时次。"
        f"最大回波出现在 {peak['time']}，约 {peak['max']:.3f}；"
        f"按阈值 {threshold_value:g} 统计，整体趋势：{trend}。"
    )
    return _ok(summary, image_url=chart_url, records=records, image_urls=images[:8], sources=sources, warnings=[])


def _ok(summary: str, *, image_url: str | None = None, records: list[dict] | None = None, image_urls: list[str] | None = None, sources: Any = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": "ok",
        "summary": summary,
        "image_url": image_url,
        "records": records or [],
        "image_urls": image_urls or ([image_url] if image_url else []),
        "sources": sources or [],
        "warnings": warnings or [],
    }


def _error(message: str, meta_path: Path | None = None) -> dict[str, Any]:
    data = {"meta_file": str(meta_path)} if meta_path else {}
    return {"status": "error", "summary": message, "image_url": None, "data": data}


def _need_params(name: str, title: str, missing: list[str]) -> dict[str, Any]:
    return {
        "status": "need_params",
        "model": name,
        "model_name": title,
        "missing": [{"name": item, "label": item, "type": "text", "required": True} for item in missing],
        "summary": f"调用 {title} 还需要补充参数：{'、'.join(missing)}。",
        "image_url": None,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_meta_file(value: str | Path | None, default: Path | None) -> Path:
    raw = str(value or "").strip()
    path = Path(raw) if raw else default
    if path is None:
        raise ValueError("meta_file is required")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists() or not path.name.endswith(".meta.json"):
        raise ValueError(f"meta file not found: {path}")
    try:
        path.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"meta file outside data root: {path}") from exc
    return path


def _source_from_meta(meta: dict[str, Any], meta_path: Path) -> Path:
    source = str(meta.get("source_file") or "").strip()
    candidates = []
    if source:
        candidates.append(Path(source))
    if meta_path.name.endswith(".meta.json"):
        candidates.append(meta_path.with_name(meta_path.name[: -len(".meta.json")]))
    if meta.get("file"):
        candidates.append(meta_path.parent / str(meta.get("file")))
    for item in candidates:
        path = item if item.is_absolute() else PROJECT_ROOT / item
        if path.exists():
            return path.resolve()
    return (candidates[-1] if candidates else meta_path).resolve()


def _time_dim(arr: xr.DataArray) -> str | None:
    for dim in arr.dims:
        low = dim.lower()
        if "time" in low or low in {"step", "valid_time"}:
            return dim
    return arr.dims[0] if arr.dims else None


def _time_values(dataset: xr.Dataset, arr: xr.DataArray, time_dim: str | None, meta: dict[str, Any]) -> list[str]:
    meta_times = [str(item) for item in (meta.get("times") or []) if item not in (None, "")]
    if time_dim and time_dim in dataset.coords:
        values = dataset.coords[time_dim].values
        return [str(np.datetime_as_string(item, unit="m") if np.issubdtype(values.dtype, np.datetime64) else item) for item in values]
    if time_dim and time_dim in arr.coords:
        values = arr.coords[time_dim].values
        return [str(item) for item in np.atleast_1d(values)]
    return meta_times or [str(index) for index in range(arr.sizes.get(time_dim, 1) if time_dim else 1)]


def _to_float_list(values: Any) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(item) for item in array if np.isfinite(item)]


def _convert_value(variable: str, value: float) -> float:
    key = variable.lower()
    if key in {"t2m", "t2", "d2m"} and value > 150:
        return value - 273.15
    if key in {"tp"}:
        return value * 1000.0
    if key in {"sp", "psfc"} and value > 2000:
        return value / 100.0
    return value


def _display_unit(variable: str, raw_unit: str = "") -> str:
    key = variable.lower()
    if key in {"t2m", "t2", "d2m"}:
        return "C"
    if key in {"tp", "rainc", "rainnc"}:
        return "mm"
    if key in {"sp", "psfc"}:
        return "hPa"
    return raw_unit or ""


def _variable_unit(meta: dict[str, Any], variable: str) -> str:
    for item in meta.get("variables") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("short_name") or item.get("raw_name") or "")
        if name.lower() == variable.lower():
            return str(item.get("unit") or item.get("units") or item.get("display_unit") or "")
    return ""


def _trend(values: list[Any]) -> str:
    nums = [float(item) for item in values if item is not None and np.isfinite(float(item))]
    if len(nums) < 2:
        return "时次不足，暂不判断"
    delta = nums[-1] - nums[0]
    span = max(nums) - min(nums)
    if span == 0 or abs(delta) < max(abs(span) * 0.15, 1e-6):
        return "整体较平稳"
    return "整体上升" if delta > 0 else "整体下降"


def _safe_number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _split_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _wrf_images(meta: dict[str, Any], variables: list[str], resolution: str = "") -> list[str]:
    product = None
    products = meta.get("resolution_products") if isinstance(meta.get("resolution_products"), dict) else {}
    if products:
        key = resolution or meta.get("default_resolution") or next(iter(products))
        product = products.get(key) or next(iter(products.values()))
    variables_meta = product.get("variables") if isinstance(product, dict) else meta.get("variables")
    webps = product.get("webp_files") if isinstance(product, dict) else meta.get("webp_files")
    if not isinstance(variables_meta, list) or not isinstance(webps, list):
        return [_public_data_url(str(item)) for item in (meta.get("webp_files") or [])[:4]]
    images = []
    webps_by_index = [str(item) for item in webps]
    for index, item in enumerate(variables_meta):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name in variables:
            webp = item.get("webp") or item.get("webp_url")
            if not webp and index < len(webps_by_index):
                webp = webps_by_index[index]
            if webp:
                images.append(_public_data_url(str(webp)))
    if images:
        return images
    return [_public_data_url(str(item)) for item in webps[:4]]


def _dataset_variable_name(dataset: xr.Dataset, meta: dict[str, Any], requested: str) -> str:
    if requested in dataset:
        return requested
    wanted = requested.lower()
    for item in meta.get("variables") or []:
        if not isinstance(item, dict):
            continue
        aliases = [
            str(item.get("name") or ""),
            str(item.get("short_name") or ""),
            str(item.get("raw_name") or ""),
            str(item.get("code") or ""),
        ]
        if any(alias.lower() == wanted for alias in aliases):
            raw_name = str(item.get("raw_name") or item.get("name") or "")
            if raw_name in dataset:
                return raw_name
    for name in dataset.data_vars:
        if name.lower() == wanted or name.lower().endswith("." + wanted):
            return str(name)
    return ""


def _meta_time(meta: dict[str, Any]) -> str:
    times = meta.get("times") if isinstance(meta.get("times"), list) else []
    return str(times[0]) if times else str(meta.get("time") or "")


def _first_image(meta: dict[str, Any]) -> str:
    for key in ("default_webp", "default_png", "png", "webp"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("webp_files", "png_files"):
        values = meta.get(key)
        if isinstance(values, list) and values:
            return str(values[0])
    return ""


def _public_data_url(value: str) -> str:
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if text.startswith("/data/"):
        return f"{DATA_PUBLIC_BASE}{text}"
    if text.startswith("data/"):
        return f"{DATA_PUBLIC_BASE}/{text}"
    return text


def _sources(meta_path: Path, source: Path) -> list[dict[str, str]]:
    return [{"meta_file": str(meta_path), "source_file": str(source)}]


def _source_warnings(meta: dict[str, Any], source: Path) -> list[str]:
    raw = str(meta.get("source_file") or "")
    if raw and Path(raw) != source:
        return [f"meta.source_file 指向旧路径，已使用当前项目内文件：{source}"]
    return []
