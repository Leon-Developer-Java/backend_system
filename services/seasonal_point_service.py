"""Authorized original-grid seasonal point series, never decoded from colors."""
from collections import OrderedDict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from DB.config import create_database_engine
from DB.repository import get_active_user, get_display_resource
from adapters import seasonal_precip_adapter as adapter

ENGINE = create_database_engine()
LOCK = threading.Lock()
CACHE = OrderedDict()


class PointSeriesError(ValueError):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def timestamp(value):
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date).astimezone(timezone.utc)
    except (TypeError, ValueError):
        raise PointSeriesError("有效时次无法解析，未替换为其他月份。") from None


def _period(meta, layer):
    value = layer.get("period_months")
    if value in (1, 3):
        return value
    name = str(meta.get("file_name") or "")
    if re.fullmatch(r"prate\.\d{6}\.prob(?:\.adj)?\.seas\.nc", name):
        return 3
    if re.fullmatch(r"(?:CFSv2|CanESM5|GEM5\.2_NEMO|NASA_GEOS5v2|NCAR_CCSM4|NCAR_CESM1|NMME)\.prate\.\d{6}\.ENSMEAN\.(?:fcst|anom)\.nc", name):
        return 1
    return None


def _data_array(ds, variable, lat, lon):
    if variable in ds.data_vars and not adapter._category_dimension(ds[variable], lat, lon):
        return ds[variable]
    matches = []
    for name in ds.data_vars:
        da = ds[name]
        dim = adapter._category_dimension(da, lat, lon)
        if not dim:
            continue
        for index, label in enumerate(adapter._value_labels(ds, dim, ds.sizes[dim])):
            suffix = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", label).strip("_").lower() or "category"
            if variable == f"{name}_{suffix}":
                matches.append(da.isel({dim:index}))
    if len(matches) != 1:
        raise PointSeriesError("原始文件中的变量缺失或不唯一。")
    return matches[0]


def sample_series(source, meta, variable, times, longitude, latitude, bbox=None):
    """Reuse the map adapter's orientation, time and unit conversions exactly."""
    layer = (meta.get("variable_layers") or {}).get(variable)
    if not isinstance(layer, dict):
        raise PointSeriesError("指定变量不存在。")
    if not 1 <= len(times) <= 36 or len(set(map(timestamp, times))) != len(times):
        raise PointSeriesError("请选择1至36个不同的真实有效时次。", 400)
    targets = list(map(timestamp, times))
    frames = layer.get("frames") or []
    indices = []
    for target in targets:
        match = [i for i,f in enumerate(frames) if timestamp(f.get("valid_time")) == target]
        if len(match) != 1:
            raise PointSeriesError("指定月份缺失或存在多个成员，未自动替换。")
        indices.append(match[0])
    groups = adapter._open_groups(Path(source))
    found = []
    try:
        for ds in groups:
            lat_name, lon_name = adapter._find_lat_lon(ds)
            try:
                da = _data_array(ds, variable, lat_name, lon_name)
            except PointSeriesError:
                continue
            if da.size > 16_000_000:
                raise PointSeriesError("该变量规模较大，暂不提供即时格点查询。")
            # CPC's legacy 'percent' fields sometimes hold fractions; use the
            # same three-category encoding check as process_file.
            if re.fullmatch(r"prate\.\d{6}\.prob(?:\.adj)?\.(?:mon|seas)\.nc", str(meta.get("file_name") or "")) and all(k in ds for k in ("prob_above","prob_norm","prob_below")):
                total = (ds.prob_above + ds.prob_norm + ds.prob_below).values
                finite = total[np.isfinite(total)]
                if finite.size and np.allclose(finite, 1, atol=1e-5):
                    da = da.copy(deep=False)
                    da.attrs = {**da.attrs, "units":"1"}
            raw, lat, lon, labels, _ = adapter._extract_frames(da, ds, lat_name, lon_name)
            if any(i >= len(labels) or timestamp(labels[i]) != target for i,target in zip(indices,targets)):
                raise PointSeriesError("原始数据时次与地图元数据不一致，请重新解析文件。")
            values, unit, _ = adapter._convert(raw, str(da.attrs.get("units") or ""), layer.get("varType"), variable)
            if str(layer.get("unit") or layer.get("displayUnit") or unit) != unit:
                raise PointSeriesError("原始值单位与地图单位不一致，请重新解析。")
            if lat.ndim != 1 or lon.ndim != 1 or not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
                raise PointSeriesError("暂不支持该坐标网格。")
            dx = float(np.median(np.abs(np.diff(lon)))) if len(lon)>1 else None
            dy = float(np.median(np.abs(np.diff(lat)))) if len(lat)>1 else None
            if bbox is not None:
                w,s,e,n=bbox
                if not (lon.min()<=w<e<=lon.max() and lat.min()<=s<n<=lat.max()):
                    raise PointSeriesError('数据不覆盖指定范围。',400)
                xs=np.flatnonzero((lon>=w)&(lon<=e));ys=np.flatnonzero((lat>=s)&(lat<=n))
                if len(xs)*len(ys)>5000 or len(xs)*len(ys)*len(indices)>250000:
                    raise PointSeriesError('区域格点过多，请缩小范围后查看原始格点图。',400)
                region_values=values[np.ix_(indices,ys,xs)]
                found.append({'status':'succeeded' if len(xs) and len(ys) else 'no_data','method':'original_region_grid',
                    'bbox':list(bbox),'longitudes':lon[xs].tolist(),'latitudes':lat[ys].tolist(),
                    'grid_spacing':{'longitude':dx,'latitude':dy},'unit':unit,'variable_key':variable,'result_type':layer.get('varType'),
                    'series':[{'valid_time':target.isoformat(),'values':[[float(v) if np.isfinite(v) else None for v in row] for row in region_values[j]]} for j,target in enumerate(targets)]})
                continue
            if not (float(lat.min()) <= latitude <= float(lat.max()) and float(lon.min()) <= longitude <= float(lon.max())):
                raise PointSeriesError("请求位置不在原始网格覆盖范围内。")
            iy, ix = int(np.argmin(abs(lat-latitude))), int(np.argmin(abs(lon-longitude)))
            def samples(y,x):
                return [{"valid_time": t.isoformat(), "value": float(values[i,y,x]) if np.isfinite(values[i,y,x]) else None,
                         "forecast_label": str(frames[i].get("forecast_label") or labels[i])[:160]} for i,t in zip(indices,targets)]
            neighbors = []
            for y in range(max(0,iy-1),min(len(lat),iy+2)):
                for x in range(max(0,ix-1),min(len(lon),ix+2)):
                    neighbors.append({"longitude":float(lon[x]), "latitude":float(lat[y]), "series":samples(y,x)})
            found.append({"status":"succeeded", "method":"nearest_grid_point", "interpolation":"none",
                "requested_point":{"longitude":longitude,"latitude":latitude},
                "sample_point":{"longitude":float(lon[ix]),"latitude":float(lat[iy])},
                "grid_spacing":{"longitude":dx,"latitude":dy}, "unit":unit, "variable_key":variable,
                "result_type":layer.get("varType"), "period_months":_period(meta,layer), "series":samples(iy,ix), "neighbors":neighbors,
                "scope":"single_original_grid_point", "regional_average":False,
                "note":"最近原始格点值，非城市面积平均；保留原单位和有效时间，不从图像颜色估值，不自动累计。"})
        if len(found) != 1:
            raise PointSeriesError("变量缺失或跨多个GRIB分组不唯一，未返回猜测数值。")
        return found[0]
    finally:
        for ds in groups:
            ds.close()


def authorized_series(user_token, file_uuid, variable, times, longitude, latitude, bbox=None):
    """Authorize every request before reading cached or original data."""
    if bbox is not None:
        if len(bbox)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in bbox) or not (-180<=bbox[0]<bbox[2]<=180 and -90<=bbox[1]<bbox[3]<=90):
            raise PointSeriesError('经纬度范围无效。',400)
    user = get_active_user(ENGINE, str(user_token.get("sub") or ""))
    if not user or int(user.get("role") or 0)<1 or user.get("token_version") != user_token.get("token_version"):
        raise PointSeriesError("登录状态已失效，请重新登录。",401)
    resource = get_display_resource(ENGINE,user["uuid"],int(user["role"]),file_uuid)
    if not resource:
        raise PointSeriesError("数据不存在或无权访问。",404)
    if str(resource.get("data_type") or "").upper() not in {"SEASONAL_PRECIP","SEASONALPRECIP","SEASONAL_PRECIPITATION"}:
        raise PointSeriesError("该文件不是季节降水产品。",400)
    raw_root=Path(os.getenv("RAW_STORAGE_ROOT",str(ROOT/'storage'/'raw'))).resolve()
    product_root=Path(os.getenv("PRODUCT_DATA_ROOT",str(ROOT/'backend'/'data'))).resolve()
    source=(raw_root/str(resource.get("source_path") or "")).resolve()
    meta_path=(product_root/str(resource.get("meta_path") or "").lstrip('/\\')).resolve()
    if not source.is_relative_to(raw_root) or not meta_path.is_relative_to(product_root):
        raise PointSeriesError("数据路径不在已配置目录内。")
    if not source.is_file() or not meta_path.is_file():
        raise PointSeriesError("原始数据或元数据缺失，无法提供格点值。")
    if source.stat().st_size > 512*1024*1024:
        raise PointSeriesError("原始文件过大，暂不提供即时格点查询。")
    signature=(file_uuid,str(source),source.stat().st_mtime_ns,source.stat().st_size,meta_path.stat().st_mtime_ns,variable,tuple(times),longitude,latitude,tuple(bbox) if bbox is not None else None)
    with LOCK:
        if signature not in CACHE:
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            CACHE[signature]={**sample_series(source,meta,variable,times,longitude,latitude,bbox=bbox),"file_uuid":file_uuid}
            while len(CACHE)>32:
                CACHE.popitem(last=False)
        CACHE.move_to_end(signature)
        return CACHE[signature]
