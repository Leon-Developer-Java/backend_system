"""GridRad-Severe V4.2 reader using the existing radar display contract.

Sparse indexing follows the previously verified GridRad local reader. Floating
point source values and missingness are retained; only display assets are images.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import warnings

import netCDF4
import numpy as np

from adapters.base import write_meta

FIELDS = {
    "Reflectivity": ("observation.base_ref_cor_log", "Z_H"),
    "DifferentialReflectivity": ("observation.base_zdr_cor_log", "Z_DR"),
    "DifferentialPhase": ("observation.base_kdp_lsf_x_lin", "K_DP"),
    "CorrelationCoefficient": ("observation.base_rhv_flt_lin", "r_HV"),
}


def is_gridrad(path):
    try:
        with netCDF4.Dataset(path) as nc:
            return "index" in nc.variables and "Altitude" in nc.variables and "Reflectivity" in nc.variables
    except (OSError, RuntimeError):
        return False  # Preserve the existing h5netcdf CAP_FMT reader's fallback.


def read_fields(path, options):
    with netCDF4.Dataset(path) as nc:
        lon, lat, alt = (np.asarray(nc[k][:], dtype=np.float64) for k in ("Longitude", "Latitude", "Altitude"))
        if str(getattr(nc["Altitude"], "units", "")).strip() != "km":
            raise ValueError("GridRad高度单位须为km。")
        lon = np.where(lon > 180, lon - 360, lon)
        if any(c.ndim != 1 or not c.size or not np.all(np.isfinite(c)) or np.any(np.diff(c) <= 0) for c in (lon, lat, alt)):
            raise ValueError("GridRad坐标无效。")
        index = np.asarray(nc["index"][:], dtype=np.int64)
        shape = (len(alt), len(lat), len(lon))
        if index.ndim != 1 or np.any(index < 0) or np.any(index >= np.prod(shape)) or np.unique(index).size != index.size:
            raise ValueError("GridRad稀疏索引越界或重复。")
        timestamp = datetime.strptime(str(nc.Analysis_time), "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        variables = options.get("variables") or [k for k in FIELDS if k in nc.variables]
        if not variables or any(v not in FIELDS or v not in nc.variables for v in variables):
            raise ValueError("GridRad源文件未包含所选要素。")
        if "DifferentialPhase" in variables and "km" not in str(nc["DifferentialPhase"].units):
            raise ValueError("DifferentialPhase不能确认为KDP。")
        area = options.get("area")
        x, y = np.arange(len(lon)), np.arange(len(lat))
        if area is not None:
            n, w, s, e = area
            # Coordinates are grid centers; permit up to half a native cell at boundaries.
            dx, dy = abs(float(np.median(np.diff(lon)))), abs(float(np.median(np.diff(lat))))
            if w < lon[0] - dx / 2 or e > lon[-1] + dx / 2 or s < lat[0] - dy / 2 or n > lat[-1] + dy / 2:
                raise ValueError("所选区域超出该事件网格，不自动拼接或填充。")
            x, y = np.flatnonzero((lon >= w) & (lon <= e)), np.flatnonzero((lat >= s) & (lat <= n))
        if len(x) < 2 or len(y) < 2:
            raise ValueError("所选区域没有至少2×2个原始格点。")
        heights = options.get("heights_km") or alt.tolist()
        z = []
        for height in heights:
            matches = np.flatnonzero(np.isclose(alt, height, rtol=0, atol=1e-6))
            if len(matches) != 1:
                raise ValueError(f"源文件没有海拔{height:g}km高度层，不自动垂直插值。")
            z.append(int(matches[0]))
        fields = {}
        qc = options.get("quality_control", "raw")
        official = None
        qc_hash = None
        if qc == "official_v4_2":
            code = Path(__file__).with_name("gridrad_official_v4_2.py")
            spec = importlib.util.spec_from_file_location("gridrad_official_v4_2", code)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            official = module.remove_clutter(module.filter(module.read_file(str(path))))
            qc_hash = hashlib.sha256(code.read_bytes()).hexdigest()
        elif qc != "raw":
            raise ValueError("不支持的GridRad质控模式。")
        for name in variables:
            if nc[name].shape != index.shape:
                raise ValueError("GridRad要素长度与索引不一致。")
            if official is None:
                dense = np.full(shape, np.nan, dtype=np.float32)
                dense.reshape(-1)[index] = np.ma.filled(nc[name][:], np.nan)
            else:
                dense = np.asarray(official[FIELDS[name][1]]["values"], dtype=np.float32)
            fields[name] = dense[np.ix_(z, y, x)].copy()
    return {"time": timestamp, "lon": lon[x], "lat": lat[y], "alt_m": alt[z] * 1000,
            "fields": fields, "qc": qc, "qc_hash": qc_hash}


def process_file(path, data_type="Radar", options=None, output_dir=None):
    from adapters import radar_adapter as radar
    path = Path(path).resolve()
    data = read_fields(path, options or {})
    root = Path(output_dir).resolve() if output_dir else path.parent / "_gridrad_products" / path.stem
    root.mkdir(parents=True, exist_ok=True)
    def public(path):
        product_root = (options or {}).get("_product_root")
        return "/data/" + path.resolve().relative_to(Path(product_root).resolve()).as_posix() if product_root else radar.public_data_path(path)
    lon, lat = data["lon"], data["lat"]
    dx, dy = float(np.median(np.diff(lon))), float(np.median(np.diff(lat)))
    extent = [float(lon[0] - dx / 2), float(lat[0] - dy / 2), float(lon[-1] + dx / 2), float(lat[-1] + dy / 2)]
    products, variables, layers = [], [], {}
    for name, values in data["fields"].items():
        mapped, _ = FIELDS[name]
        info = radar._product_meta(mapped)
        specs = [{"key": f"level-{i}", "label": f"{h:g} m 海拔", "mode": "single_level", "level": float(h), "values": values[i]} for i, h in enumerate(data["alt_m"])]
        if name == "Reflectivity":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                specs.insert(0, {"key": "max", "label": "所选高度组合反射率", "mode": "vertical_max", "level": None, "values": np.nanmax(values, axis=0)})
        rendered = []
        for spec in specs:
            image = root / f"{name}.{spec['key']}.webp"
            # Shared renderer expects north at the top of each image.
            radar._write_radar_webp(image, spec["values"], mapped)
            rendered.append({k: v for k, v in spec.items() if k != "values"} | {
                "webp_url": public(image), "webp": public(image),
                "extent": extent, "stats": radar._stats(spec["values"]), "legend": info["legend"]})
            key = info["code"] + "_" + spec["key"]
            layers[key] = {"raw_name": name, "productCode": info["code"], "label": info["label"], "unit": info["unit"],
                           "level_type": "height_m_msl", "level": spec["level"] if spec["level"] is not None else "selected_column_max",
                           "times": [data["time"]], "webp_urls": [public(image)],
                           "width": len(lon), "height": len(lat), "extent": extent}
        products.append({**info, "key": name, "raw_name": name, "extent": extent, "levels": rendered})
        variables.append({"name": info["code"], "raw_name": name, "name_cn": info["name_cn"], "long_name": info["name_en"],
                          "unit": info["unit"], "shape": list(values.shape), "stats": radar._stats(values)})
    default = products[0]["levels"][0]["webp_url"]
    meta_path = root / (path.stem + ".meta.json")
    generated = datetime.now(timezone.utc).isoformat()
    meta = {"schema_version": "1.0", "dataset_id": path.stem, "data_type": data_type,
            "file_format": "GRIDRAD_SEVERE_V4_2", "file_name": path.name, "file": path.name,
            "source_file": str(path), "meta_file": str(meta_path), "default_variable": products[0]["code"],
            "default_webp": default, "webp_files": [r["webp_url"] for p in products for r in p["levels"]],
            "variables": variables, "variable_layers": layers, "display_products": products,
            "product_catalog": [{**p, "selectable": True} for p in products],
            "times": [data["time"]], "levels": data["alt_m"].tolist(), "extent": extent, "bbox": extent,
            "time_detail": {"start": data["time"], "end": data["time"], "count": 1, "step_seconds": 300},
            "spatial": {"nx": len(lon), "ny": len(lat), "level_type": "height_m_msl", "projection": "EPSG:4326", "resolution_lon": dx, "resolution_lat": dy},
            "format_specific": {"radar_name": "GridRad-Severe", "radar_type": "NEXRAD merged grid", "institution": "NSF NCAR GDEX",
                                "version": "4.2", "quality_control": data["qc"], "qc_script_sha256": data["qc_hash"]},
            "weather_info": {"type": "Radar", "file": path.name, "element": products[0]["name_cn"], "time": data["time"],
                             "unit": products[0]["unit"], "level": "海拔高度", "extent": extent, "steps": "1", "step_count": 1,
                             "vars": str(len(products)), "temporal_resolution": "5分钟", "status": "解析完成", "updated_at": generated,
                             "webp_url": default},
            "file_detail": {"path": str(path), "format": "GRIDRAD_SEVERE_V4_2", "parse_status": "success"},
            "extra": {"radar": {"source": "GridRad-Severe", "height_reference": "MSL", "quality_control": data["qc"]}}}
    write_meta(meta_path, meta)
    return meta


def process_files(paths, data_type="Radar", options=None, output_dir=None):
    if not paths:
        raise ValueError("没有GridRad源文件。")
    root = Path(output_dir).resolve() if output_dir else Path(paths[0]).parent / "_gridrad_series"
    metas = [process_file(path, data_type, options, root / f"frame{i:04d}") for i, path in enumerate(paths)]
    metas.sort(key=lambda m: m["times"][0])
    if len({m["times"][0] for m in metas}) != len(metas):
        raise ValueError("GridRad序列存在重复时间，不自动覆盖。")
    geometry = [(m["bbox"], m["levels"], m["spatial"]["nx"], m["spatial"]["ny"]) for m in metas]
    if any(g != geometry[0] for g in geometry[1:]):
        raise ValueError("事件网格或高度层发生变化，请分组处理，不自动重采样。")
    meta = dict(metas[0])
    layers = {}
    for key, layer in meta["variable_layers"].items():
        layers[key] = {**layer, "times": [m["times"][0] for m in metas],
                       "webp_urls": [m["variable_layers"][key]["webp_urls"][0] for m in metas]}
    meta.update(dataset_id=root.name, file_name=f"GridRad_{len(metas)}_frames", source_files=[m["source_file"] for m in metas],
                frames=[{"index": i, "time": m["times"][0], "source_file": m["source_file"], "meta_file": m["meta_file"],
                         "webp_url": m["default_webp"], "extent": m["extent"]} for i, m in enumerate(metas)],
                times=[m["times"][0] for m in metas], variable_layers=layers,
                webp_files=[p for m in metas for p in m["webp_files"]], meta_file=str(root / "gridrad.series.meta.json"))
    meta["weather_info"] = {**meta["weather_info"], "step_count": len(metas), "steps": str(len(metas)), "times": meta["times"]}
    meta["time_detail"] = {"start": meta["times"][0], "end": meta["times"][-1], "count": len(metas), "step_seconds": (options or {}).get("sampling_minutes", 5) * 60}
    write_meta(Path(meta["meta_file"]), meta)
    return meta
