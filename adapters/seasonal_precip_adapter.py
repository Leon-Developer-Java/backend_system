"""Adapter for uploaded seasonal precipitation forecast products.

The adapter deliberately accepts both NetCDF and GRIB/GRIB2.  Product names
and provider metadata are read from file attributes when available; the
display contract is kept compatible with the existing WebGL/GribLayer code.
"""

from __future__ import annotations

import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import xarray as xr
from PIL import Image

matplotlib.use("Agg")

from adapters import gfs_adapter


ADAPTER_VERSION = "0.1.0"
SCHEMA_VERSION = "2.0"
DATA_TYPE = "SEASONAL_PRECIP"
WEBP_QUALITY = 82
WEBP_METHOD = 2

_PRECIP_HINTS = (
    "precip", "rain", "rainfall", "prate", "apcp", "tp", "total_precip",
    "降水", "降雨",
)
_ANOMALY_HINTS = ("anomaly", "anom", "departure", "距平", "偏差", "percentile")
_PROBABILITY_HINTS = (
    "prob", "probability", "below", "normal", "above", "tercile", "quantile",
    "概率", "低于", "正常", "高于",
)
_LAT_NAMES = ("latitude", "lat", "y")
_LON_NAMES = ("longitude", "lon", "x")
_TIME_NAMES = ("time", "valid_time", "forecast_time", "lead_time", "lead", "month", "season", "target", "step", "forecastmonth")
_CATEGORY_NAMES = ("category", "categories", "tercile", "class", "probability_category", "quantile")
_PRECIP_TOKEN_RE = re.compile(r"(?:^|[_\W])(pr|ppt|tp|apcp|precip|rain)(?:$|[_\W])", re.IGNORECASE)


def _open_groups(source: Path) -> list[xr.Dataset]:
    if source.suffix.lower() in {".grib", ".grib2", ".grb", ".grb2"}:
        return gfs_adapter._open_grib_groups(str(source))
    # scipy uses Python file I/O and can open classic NetCDF in Windows paths
    # containing Chinese characters. Decode the provider's calendar months below.
    with source.open("rb") as handle:
        signature = handle.read(4)
    if signature[:3] == b"CDF":
        ds = xr.open_dataset(source, engine="scipy", decode_times=False)
        for name in ds.coords:
            units = str(ds[name].attrs.get("units", ""))
            match = re.match(r"months since (\d{4})-(\d{2})", units)
            if not match:
                continue
            origin = int(match[1]) * 12 + int(match[2]) - 1
            dates = []
            for value in np.asarray(ds[name].values).reshape(-1):
                year, month = divmod(origin + int(round(float(value))), 12)
                dates.append(np.datetime64(f"{year:04d}-{month + 1:02d}-01", 'ns'))
            ds = ds.assign_coords({name: (ds[name].dims, np.asarray(dates).reshape(ds[name].shape))})
        return [xr.decode_cf(ds)]
    return [xr.open_dataset(source)]


def _coord_name(ds: xr.Dataset, candidates: tuple[str, ...], axis: str) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    for name in ds.coords:
        attrs = ds[name].attrs
        if str(attrs.get("axis", "")).upper() == axis:
            return name
        if str(attrs.get("standard_name", "")).lower() in {
            "latitude" if axis == "Y" else "longitude"
        }:
            return name
    return None


def _find_lat_lon(ds: xr.Dataset) -> tuple[str, str]:
    lat = _coord_name(ds, _LAT_NAMES, "Y")
    lon = _coord_name(ds, _LON_NAMES, "X")
    if not lat or not lon:
        raise ValueError("无法识别季节降水产品的经纬度坐标。")
    return lat, lon


def _category_dimension(da: xr.DataArray, lat_name: str, lon_name: str) -> str | None:
    for dim in da.dims:
        if dim not in {lat_name, lon_name} and dim.lower() in _CATEGORY_NAMES:
            return dim
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(_text(item) for item in value[:4])
    return str(value).strip()


def _value_labels(ds: xr.Dataset, dim: str, count: int) -> list[str]:
    if dim not in ds.coords:
        return [f"step{i:03d}" for i in range(count)]
    values = np.asarray(ds[dim].values).reshape(-1)
    labels = [_text(value) for value in values[:count]]
    return labels + [f"step{i:03d}" for i in range(len(labels), count)]


def _frame_labels(ds: xr.Dataset, leading_dims: list[str], frame_count: int) -> list[str]:
    if "valid_time" in ds.coords:
        valid = np.asarray(ds.valid_time.values).reshape(-1)
        if valid.size == frame_count and np.issubdtype(valid.dtype, np.datetime64):
            return [str(item) for item in valid]
    time_dim = next((name for name in leading_dims if name.lower() in _TIME_NAMES), None)
    if time_dim:
        values = _value_labels(ds, time_dim, int(ds.sizes.get(time_dim, frame_count)))
        if len(leading_dims) == 1:
            return values[:frame_count]

    labels: list[str] = []
    dimensions = [dim for dim in leading_dims if dim in ds.sizes]
    ranges = [range(int(ds.sizes[dim])) for dim in dimensions]
    for index, combo in enumerate(itertools.product(*ranges)):
        parts = []
        for dim, item in zip(dimensions, combo):
            values = _value_labels(ds, dim, int(ds.sizes[dim]))
            parts.append(f"{dim}={values[item] if item < len(values) else item}")
        labels.append(" / ".join(parts) or f"step{index:03d}")
        if len(labels) >= frame_count:
            break
    return labels or [f"step{i:03d}" for i in range(frame_count)]


def _forecast_hours(ds: xr.Dataset, leading_dims: list[str], frame_count: int) -> list[int]:
    for dim in leading_dims:
        if dim.lower() not in {"lead", "lead_time", "step", "forecast_hour", "forecast_hours"}:
            continue
        if dim not in ds.coords:
            continue
        result: list[int] = []
        for index, value in enumerate(np.asarray(ds[dim].values).reshape(-1)[:frame_count]):
            try:
                result.append(int(round(float(value / np.timedelta64(1, "h")))))
                continue
            except Exception:
                pass
            match = re.search(r"(\d+)", _text(value))
            result.append(int(match.group(1)) if match else index)
        if len(result) == frame_count:
            return result
    return list(range(frame_count))


def _extract_frames(da: xr.DataArray, ds: xr.Dataset, lat_name: str, lon_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    if lat_name not in da.dims or lon_name not in da.dims:
        raise ValueError(f"变量 {da.name} 不是经纬度网格。")
    leading_dims = [dim for dim in da.dims if dim not in {lat_name, lon_name}]
    ordered = da.transpose(*leading_dims, lat_name, lon_name)
    values = np.asarray(ordered.values, dtype=float)
    lat = np.asarray(ds[lat_name].values, dtype=float).reshape(-1)
    lon = np.asarray(ds[lon_name].values, dtype=float).reshape(-1)
    if values.ndim == 2:
        values = values[None, :, :]
    else:
        values = values.reshape((-1, values.shape[-2], values.shape[-1]))
    if values.shape[-2:] != (lat.size, lon.size):
        raise ValueError(f"变量 {da.name} 网格形状与经纬度不一致。")
    if lat.size >= 2 and lat[0] < lat[-1]:
        lat = lat[::-1]
        values = values[:, ::-1, :]
    if lon.size >= 2 and lon[0] > lon[-1]:
        lon = lon[::-1]
        values = values[:, :, ::-1]
    if lon.size >= 2 and np.nanmin(lon) >= 0 and np.nanmax(lon) > 180:
        normalized = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(normalized)
        lon = normalized[order]
        values = values[:, :, order]
    values = np.where(np.isfinite(values), values, np.nan)
    labels = _frame_labels(ds, leading_dims, values.shape[0])
    hours = _forecast_hours(ds, leading_dims, values.shape[0])
    return values, lat, lon, labels, hours


def _classify(name: str, attrs: dict[str, Any]) -> str | None:
    text = " ".join([name, _text(attrs.get("long_name")), _text(attrs.get("standard_name")), _text(attrs.get("description"))]).lower()
    if any(key in text for key in _PROBABILITY_HINTS):
        return "probability"
    if any(key in text for key in _ANOMALY_HINTS):
        return "anomaly"
    if any(key in text for key in _PRECIP_HINTS) or _PRECIP_TOKEN_RE.search(text):
        return "precipitation"
    return None


def _label(name: str, attrs: dict[str, Any], kind: str) -> str:
    if name in {"prob_above", "prob_norm", "prob_below"}:
        return {"prob_above": "降水偏多概率", "prob_norm": "降水正常概率", "prob_below": "降水偏少概率"}[name]
    explicit = _text(attrs.get("long_name") or attrs.get("description"))
    if explicit:
        return explicit
    units = _text(attrs.get("units")).lower().replace(" ", "")
    if kind == "precipitation" and ("/s" in units or "s-1" in units):
        return f"降水率 · {name}"
    if kind == "probability":
        return f"降水概率 · {name}"
    if kind == "anomaly":
        return f"降水距平 · {name}"
    return f"累计降水 · {name}"


def _convert(values: np.ndarray, units: str, kind: str, name: str) -> tuple[np.ndarray, str, str]:
    raw_units = _text(units)
    lower = raw_units.lower().replace(" ", "").replace("**", "")
    result = np.asarray(values, dtype=float)
    if kind == "probability":
        finite = result[np.isfinite(result)]
        if "%" not in raw_units and "percent" not in lower and finite.size and float(np.nanmax(finite)) <= 1.0:
            result = result * 100.0
        return result, "%", f"{raw_units or 'fraction'} → %"
    if lower in {"m", "meter", "metre", "mwater"}:
        result = result * 1000.0
        return result, "mm", f"{raw_units} → mm"
    if kind in {"precipitation", "anomaly"} and ("s-1" in lower or "/s" in lower):
        if lower.startswith("kg") or lower.startswith("mm"):
            return result * 86400.0, "mm/day", f"{raw_units} → mm/day"
        if lower.startswith("m"):
            return result * 86400000.0, "mm/day", f"{raw_units} → mm/day"
    if "kg" in lower and "m-2" in lower and "s-1" not in lower and "/s" not in lower:
        return result, "mm", f"{raw_units} → mm"
    if kind == "anomaly" and ("percent" in lower or "pct" in name.lower() or "%" in raw_units):
        return result, "%", "保持百分比距平"
    return result, raw_units or "mm", "未转换"


def _stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"min": None, "mean": None, "max": None, "p05": None, "p95": None, "missing_ratio": 1.0}
    return {
        "min": round(float(finite.min()), 3),
        "mean": round(float(finite.mean()), 3),
        "max": round(float(finite.max()), 3),
        "p05": round(float(np.percentile(finite, 5)), 3),
        "p95": round(float(np.percentile(finite, 95)), 3),
        "missing_ratio": round(float(1.0 - finite.size / values.size), 6),
    }


def _range(kind: str, values: np.ndarray) -> dict[str, Any]:
    stats = _stats(values)
    if kind == "probability":
        return {"min": 0.0, "max": 100.0, "mode": "probability"}
    if kind == "anomaly":
        extent = max(abs(float(stats.get("p05") or 0)), abs(float(stats.get("p95") or 0)), 1.0)
        return {"min": -extent, "max": extent, "mode": "anomaly"}
    vmax = max(float(stats.get("p95") or 0), float(stats.get("max") or 0), 1.0)
    return {"min": 0.0, "max": vmax, "mode": "precipitation"}


def _gradient(kind: str) -> str:
    if kind == "probability":
        return "linear-gradient(to right, #eff6ff, #93c5fd, #2563eb, #7c3aed, #be123c)"
    if kind == "anomaly":
        return "linear-gradient(to right, #2563eb, #93c5fd, #f8fafc, #facc15, #dc2626)"
    return "linear-gradient(to right, #f8fafc, #93c5fd, #22c55e, #facc15, #ef4444)"


def _infer_source_metadata(source: Path, attrs: dict[str, Any]) -> tuple[str, str, str]:
    name = source.name.lower()
    product = _text(attrs.get("title") or attrs.get("product") or attrs.get("model"))
    provider = _text(attrs.get("institution") or attrs.get("provider") or attrs.get("source"))
    issue_time = _text(attrs.get("issue_time") or attrs.get("forecast_reference_time") or attrs.get("init_time"))
    cpc = re.fullmatch(r"(.+)\.prate\.(\d{6})\.ENSMEAN\.(anom|fcst)\.nc", source.name)
    probability = re.fullmatch(r"prate\.(\d{6})\.prob(?:\.adj)?\.(mon|seas)\.nc", source.name)
    cds = re.fullmatch(r"CDS_(ecmwf|ukmo|meteo_france)_s(\d+)_(\d{6})_(precipitation|anomaly)\.grib", source.name)
    if cpc or probability:
        model = cpc[1] if cpc else "NMME 降水概率"
        if model == "NMME": model = "NMME 多模式集合平均"
        if probability:
            model += " · " + ("三个月" if probability[2] == 'seas' else '月度')
            if '.adj.' in source.name: model += ' · 校正'
        product = product or f"NOAA {model}"
        provider = provider or "NOAA/CPC"
        issue_time = issue_time or datetime.strptime(cpc[2] if cpc else probability[1], "%Y%m").replace(tzinfo=timezone.utc).isoformat()
        return product, provider, issue_time
    if cds:
        centre = {'ecmwf': 'ECMWF', 'ukmo': '英国气象局', 'meteo_france': '法国气象局'}[cds[1]]
        return product or f"{centre} · 系统 {cds[2]}", provider or "Copernicus CDS", issue_time or datetime.strptime(cds[3], '%Y%m').replace(tzinfo=timezone.utc).isoformat()
    if "cancm4" in name:
        product = product or "NOAA NMME CanCM4"
        provider = provider or "NOAA/NCEI"
    elif "cfsv2" in name or "nmme" in name:
        product = product or "NOAA NMME CFSv2"
        provider = provider or "NOAA/NCEI"
    if not issue_time:
        match = re.search(r"\.(\d{10}|\d{8})_", source.name)
        if match:
            try:
                value = match.group(1)
                issue_time = datetime.strptime(value, "%Y%m%d" if len(value) == 8 else "%Y%m%d%H").replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass
    return product or source.stem, provider, issue_time


def _save_frame(values: np.ndarray, path: Path, color_range: dict[str, Any], kind: str) -> None:
    from matplotlib import pyplot as plt

    finite = np.isfinite(values)
    if not finite.any():
        values = np.zeros_like(values)
        finite = np.ones_like(values, dtype=bool)
    vmin, vmax = float(color_range["min"]), float(color_range["max"])
    normalized = np.zeros_like(values, dtype=float) if vmax <= vmin else np.clip((values - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap("RdBu_r" if kind == "anomaly" else "turbo")
    rgba = cmap(normalized)
    rgba[..., 3] = np.where(finite, 0.72, 0.0)
    Image.fromarray(np.clip(rgba * 255, 0, 255).astype(np.uint8), mode="RGBA").save(path, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)


def _public_url(path: Path) -> str:
    text = str(path).replace("\\", "/")
    marker = "/data/"
    index = text.lower().find(marker)
    return text[index:] if index >= 0 else f"/data/{path.name}"


def _build_variable(source: Path, ds: xr.Dataset, name: str, da: xr.DataArray, lat_name: str, lon_name: str, kind: str, index: int) -> dict[str, Any]:
    attrs = dict(da.attrs)
    raw, lat, lon, times, hours = _extract_frames(da, ds, lat_name, lon_name)
    values, unit, conversion = _convert(raw, _text(attrs.get("units")), kind, name)
    period_months = 3 if re.fullmatch(r"prate\.\d{6}\.prob(?:\.adj)?\.seas\.nc", source.name) else 1
    labels = times
    if period_months == 3:
        labels = [f"{value[:7]} 至 {str(np.datetime64(value[:7], 'M') + 2)}" for value in times]
    stats = _stats(values)
    color_range = _range(kind, values)
    safe = re.sub(r"[^0-9A-Za-z_-]", "_", name)
    frames = []
    for frame_index, frame in enumerate(values):
        path = source.with_name(f"{source.name}_{safe}_step{frame_index:03d}.webp")
        _save_frame(frame, path, color_range, kind)
        frame_stats = _stats(frame)
        frames.append({
            "index": frame_index,
            "forecast_hour": hours[frame_index] if frame_index < len(hours) else frame_index,
            "forecast_label": labels[frame_index] if frame_index < len(labels) else f"step{frame_index:03d}",
            "valid_time": times[frame_index] if frame_index < len(times) else f"step{frame_index:03d}",
            "url": _public_url(path),
            "stats": frame_stats,
        })
    extent = [round(float(np.nanmin(lon)), 4), round(float(np.nanmin(lat)), 4), round(float(np.nanmax(lon)), 4), round(float(np.nanmax(lat)), 4)]
    label = _label(name, attrs, kind)
    product_category = {"precipitation": "累计降水", "anomaly": "降水距平", "probability": "降水概率"}[kind]
    if kind == 'precipitation' and unit == 'mm/day':
        product_category = '降水率'
    return {
        "key": name,
        "label": label,
        "element": label,
        "elementCode": name,
        "elementEnglish": _text(attrs.get("long_name") or attrs.get("standard_name") or name),
        "productCategory": product_category,
        "varType": kind,
        "rawUnit": _text(attrs.get("units")),
        "unit": unit,
        "displayUnit": unit,
        "conversion": conversion,
        "times": times,
        "forecast_hours": hours,
        "forecast_labels": labels,
        "period_months": period_months,
        "extent": extent,
        "bbox": {"west": extent[0], "south": extent[1], "east": extent[2], "north": extent[3]},
        "range": f"纬度 {extent[1]:.4f} ~ {extent[3]:.4f}，经度 {extent[0]:.4f} ~ {extent[2]:.4f}",
        "resolution": f"{abs(float(lat[1] - lat[0])):.3f}° × {abs(float(lon[1] - lon[0])):.3f}°" if len(lat) > 1 and len(lon) > 1 else "待解析",
        "grid": {"nx": int(len(lon)), "ny": int(len(lat)), "text": f"{len(lat)} × {len(lon)}"},
        "min": stats.get("min"), "mean": stats.get("mean"), "max": stats.get("max"),
        "missingText": f"{float(stats.get('missing_ratio') or 0) * 100:.2f}%",
        "quality": "正常" if float(stats.get("missing_ratio") or 0) <= 0.01 else "存在缺测",
        "color_range": color_range,
        "legend_ticks": [f"{value:.0f}" for value in np.linspace(float(color_range["min"]), float(color_range["max"]), 5)],
        "gradient": _gradient(kind),
        "image_format": "webp",
        "image_url": frames[0]["url"],
        "image_urls": [frame["url"] for frame in frames],
        "frames": frames,
        "render_mode": "webp",
        "source_index": index,
    }


def process_file(file_path: str, data_type: str = DATA_TYPE, **_: Any) -> dict[str, Any]:
    source = Path(file_path).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"季节降水产品不存在或为空：{source}")
    groups = _open_groups(source)
    records: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for group_index, ds in enumerate(groups):
        if re.fullmatch(r"prate\.\d{6}\.prob(?:\.adj)?\.(mon|seas)\.nc", source.name) and all(name in ds for name in ('prob_above', 'prob_norm', 'prob_below')):
            # CPC labels these fields 'percent', but its three categories sum
            # to 1. Detect the actual encoding before converting to percent.
            total = (ds.prob_above + ds.prob_norm + ds.prob_below).values
            finite = total[np.isfinite(total)]
            if finite.size and np.allclose(finite, 1.0, atol=1e-5):
                for name in ('prob_above', 'prob_norm', 'prob_below'):
                    ds[name].attrs['units'] = '1'
        lat_name, lon_name = _find_lat_lon(ds)
        for name in ds.data_vars:
            if name in seen:
                continue
            da = ds[name]
            kind = _classify(str(name), dict(da.attrs))
            if not kind:
                continue
            category_dim = _category_dimension(da, lat_name, lon_name)
            category_items = [(None, da)]
            if category_dim:
                labels = _value_labels(ds, category_dim, int(ds.sizes.get(category_dim, 0)))
                category_items = [
                    (labels[index] if index < len(labels) else str(index), da.isel({category_dim: index}))
                    for index in range(int(ds.sizes.get(category_dim, 0)))
                ]
            for category_label, category_da in category_items:
                layer_name = str(name)
                layer_kind = "probability" if category_label and kind == "precipitation" else kind
                if category_label:
                    category_key = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(category_label)).strip("_").lower() or "category"
                    layer_name = f"{name}_{category_key}"
                try:
                    layer = _build_variable(source, ds, layer_name, category_da, lat_name, lon_name, layer_kind, group_index)
                except Exception as exc:
                    print(f"[WARN] Skip seasonal variable {layer_name}: {exc}")
                    continue
                if category_label:
                    layer["label"] = f"{layer['label']} · {category_label}"
                    layer["element"] = layer["label"]
                priority = {"precipitation": 300, "anomaly": 200, "probability": 100}[layer_kind]
                records.append((priority, layer_name, layer))
                seen.add(str(name))
    if not records:
        raise ValueError("文件中没有识别到季节降水、距平或概率变量。")
    records.sort(key=lambda item: (-item[0], item[1]))
    layers = {name: layer for _, name, layer in records}
    options = [{"key": name, "label": layer["label"], "unit": layer["unit"], "varType": layer["varType"], "productCategory": layer["productCategory"]} for _, name, layer in records]
    default = records[0][1]
    layer = layers[default]
    attrs = {}
    for ds in groups:
        attrs.update({str(key): value for key, value in ds.attrs.items()})
    product, provider, issue_time = _infer_source_metadata(source, attrs)
    meta_path = source.with_name(f"{source.name}.meta.json")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": source.name.replace(".", "_"),
        "source": DATA_TYPE,
        "business_type": DATA_TYPE,
        "data_type": DATA_TYPE,
        "product": product,
        "provider": provider,
        "file_name": source.name,
        "file_format": source.suffix.lower().lstrip(".").upper(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "issue_time": issue_time,
        "run_time": issue_time,
        "extent": layer["extent"],
        "bbox": layer["bbox"],
        "grid": layer["grid"],
        "resolution": layer["resolution"],
        "default_variable": default,
        "variable_options": options,
        "variable_layers": layers,
        "times": layer["times"],
        "image_format": "webp",
        "image_url": layer["image_url"],
        "image_urls": layer["image_urls"],
        "render_mode": "webp",
        "weather_info": {
            "source": DATA_TYPE,
            "product": product,
            "provider": provider,
            "element": layer["label"],
            "time": layer["times"][0] if layer["times"] else "",
            "range": layer["range"],
            "unit": layer["unit"],
            "status": "解析成功",
            "quality": layer["quality"],
            "mainVariable": default,
            "image_format": "webp",
            "image_url": layer["image_url"],
            "image_urls": layer["image_urls"],
            "render_mode": "webp",
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for dataset in groups:
        dataset.close()
    return meta
