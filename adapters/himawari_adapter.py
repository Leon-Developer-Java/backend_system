from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from PIL import Image
from scipy.ndimage import map_coordinates

if __package__ in {None, ""}:
    sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())

from adapters.base import process_basic_file


HSD_FILENAME_RE = re.compile(
    r"HS_H(?P<sat>\d{2})_(?P<date>\d{8})_(?P<time>\d{4})_"
    r"(?P<band>B\d{2})_(?P<region>[A-Z0-9]+)_(?P<resolution>R\d{2})_"
    r"S(?P<segment>\d{2})(?P<total>\d{2})\.DAT(?:\.bz2)?$",
    re.IGNORECASE,
)

CHINA_EXTENT = [73, 18, 136, 54]
LATLON_RESOLUTION = 0.04
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "Himawari"


def _band(
    key: str,
    name_zh: str,
    plain_name: str,
    category: str,
    wavelength: str,
    unit: str,
    description: str,
    uses: list[str],
    cautions: list[str],
    vmin: float,
    vmax: float,
) -> dict[str, Any]:
    return {
        "key": key,
        "name_zh": name_zh,
        "plain_name": plain_name,
        "category": category,
        "wavelength": wavelength,
        "unit": unit,
        "display_unit": "%" if unit == "%" else "degC",
        "description": description,
        "uses": uses,
        "cautions": cautions,
        "cmap": "gray",
        "vmin": vmin,
        "vmax": vmax,
    }


BAND_CATALOG: dict[str, dict[str, Any]] = {
    "B01": _band("B01", "蓝光反射率", "白天云和地表颜色辅助通道", "可见光反射率", "0.47um", "%", "反映云、海洋和地表对蓝光的反射强弱，数值越大通常越亮。", ["识别白天云区", "辅助真彩色合成"], ["夜间没有太阳照射，该通道不可用于夜间云图。"], 0, 120),
    "B02": _band("B02", "绿光反射率", "白天自然颜色辅助通道", "可见光反射率", "0.51um", "%", "反映目标对绿光的反射强弱，是真彩色合成的重要组成部分。", ["真彩色云图", "白天云系观察"], ["只代表反射光强，不是温度或降水强度。"], 0, 120),
    "B03": _band("B03", "红光反射率", "白天高分辨率云图通道", "可见光反射率", "0.64um", "%", "红光反射率能清楚显示白天云的纹理和边界。", ["白天云系监测", "真彩色合成"], ["夜间不可用，强反射不等于云顶更高。"], 0, 120),
    "B04": _band("B04", "近红外反射率", "植被和云相态辅助通道", "近红外反射率", "0.86um", "%", "对植被、水体和部分云相态差异比较敏感。", ["区分陆地植被和水体", "辅助自然色合成"], ["该通道仍依赖太阳光，夜间不适合使用。"], 0, 120),
    "B05": _band("B05", "雪冰云粒子反射率", "雪冰和云粒子大小辅助通道", "近红外反射率", "1.6um", "%", "对雪冰、云粒子大小和低云特征较敏感。", ["识别雪冰覆盖", "辅助云相态分析"], ["不是直接的雪深或云水含量，只是反射率观测。"], 0, 120),
    "B06": _band("B06", "短波近红外反射率", "云粒子和地表反射辅助通道", "近红外反射率", "2.3um", "%", "对云粒子和地表类型差异有响应，常用于组合产品。", ["自然色增强", "云和地表区分"], ["白天使用效果更可靠，不能单独解释为天气强弱。"], 0, 120),
    "B07": _band("B07", "短波红外亮温", "夜间低云雾和热点辅助通道", "红外亮温", "3.9um", "K", "夜间常用于低云雾和热点识别，白天会受太阳反射影响。", ["夜间低云雾识别", "火点辅助判断"], ["白天会混入太阳反射，解释时要区分昼夜。"], 200, 330),
    "B08": _band("B08", "高层水汽亮温", "高空水汽分布通道", "水汽亮温", "6.2um", "K", "主要反映对流层中高层水汽和云的辐射特征。", ["分析高空水汽", "识别干侵入"], ["它不是地面湿度，主要代表较高层大气信息。"], 200, 300),
    "B09": _band("B09", "中高层水汽亮温", "中高空水汽结构通道", "水汽亮温", "6.9um", "K", "用于观察中高层水汽和云系结构。", ["中高层水汽分析", "气团 RGB"], ["不能直接代表近地面湿度。"], 200, 300),
    "B10": _band("B10", "中层水汽亮温", "中层大气水汽通道", "水汽亮温", "7.3um", "K", "辅助判断中层干湿和云系发展环境。", ["中层水汽分析", "对流发展环境判断"], ["亮温受温度和水汽共同影响。"], 200, 310),
    "B11": _band("B11", "云相态红外亮温", "云相态和沙尘辅助通道", "红外亮温", "8.6um", "K", "常和窗口通道组合，用于区分云、沙尘和云相态差异。", ["沙尘 RGB", "云相态辅助分析"], ["通常要和其他红外通道组合使用。"], 200, 320),
    "B12": _band("B12", "臭氧吸收带亮温", "气团和高空动力辅助通道", "红外亮温", "9.6um", "K", "受臭氧吸收影响，常用于气团 RGB。", ["气团 RGB", "高空动力分析"], ["不应单独解释为臭氧浓度定量产品。"], 200, 320),
    "B13": _band("B13", "红外窗口亮温", "云顶温度观测通道", "红外亮温", "10.4um", "K", "温度越低，通常表示云顶越高或越冷。", ["观察深对流", "识别台风云系"], ["它不是地面气温，而是卫星看到的辐射亮温。"], 200, 320),
    "B14": _band("B14", "长波红外窗口亮温", "标准红外云图通道", "红外亮温", "11.2um", "K", "适合全天候观察云顶温度和大范围云系。", ["全天候云图", "云顶温度分析"], ["晴空下可能反映地表或海表辐射。"], 200, 320),
    "B15": _band("B15", "分裂窗红外亮温", "水汽和沙尘差异辅助通道", "红外亮温", "12.4um", "K", "常与 10.4um 或 11.2um 通道做差。", ["沙尘 RGB", "薄云识别"], ["亮温差产品比单独通道更有解释价值。"], 200, 320),
    "B16": _band("B16", "二氧化碳吸收带亮温", "高云和云顶高度辅助通道", "红外亮温", "13.3um", "K", "对高层云和云顶高度相关信息较敏感。", ["高云识别", "云顶高度辅助判断"], ["不是二氧化碳浓度产品。"], 200, 310),
}

COMPOSITE_CATALOG: dict[str, dict[str, Any]] = {
    "true_color": {"key": "true_color", "name_zh": "真彩色云图", "plain_name": "接近人眼看到的白天卫星图", "source_bands": ["B03", "B02", "B01"], "description": "用于直观看云、陆地、水体和大范围天气系统。", "cautions": ["夜间没有可见光，真彩色产品不可用或效果很差。"]},
    "natural_color": {"key": "natural_color", "name_zh": "自然色云图", "plain_name": "增强陆地、水体、植被和云差异的白天图像", "source_bands": ["B05", "B04", "B03"], "description": "比真彩色更强调地表和云的差异。", "cautions": ["这是组合增强图，不是单一物理变量。"]},
    "air_mass": {"key": "air_mass", "name_zh": "气团 RGB", "plain_name": "观察干侵入、高空动力和气团差异的增强图", "source_bands": ["B08", "B10", "B12", "B13"], "description": "通过水汽和臭氧吸收带差异突出不同气团和高空动力结构。", "cautions": ["颜色代表通道组合差异，不是气团名称的直接分类。"]},
    "dust": {"key": "dust", "name_zh": "沙尘 RGB", "plain_name": "辅助识别沙尘、薄云和低层特征的增强图", "source_bands": ["B11", "B13", "B14", "B15"], "description": "通过分裂窗和红外通道差异增强沙尘及薄云信号。", "cautions": ["沙尘识别需要结合地面观测和天气背景确认。"]},
    "night_microphysics": {"key": "night_microphysics", "name_zh": "夜间微物理 RGB", "plain_name": "夜间低云、雾和云相态辅助识别图", "source_bands": ["B07", "B13", "B15"], "description": "利用短波红外和长波红外差异，在夜间辅助识别低云雾。", "cautions": ["白天 B07 会受太阳反射影响，夜间解释更稳定。"]},
    "water_vapor_enhanced": {"key": "water_vapor_enhanced", "name_zh": "水汽增强图", "plain_name": "突出中高层水汽和干湿结构的增强图", "source_bands": ["B08", "B09", "B10"], "description": "将多个水汽通道组合，突出中高层大气干湿结构。", "cautions": ["它不是地面湿度图，主要反映中高层大气。"]},
}


def parse_hsd_filename(filename: str) -> dict[str, Any] | None:
    match = HSD_FILENAME_RE.match(Path(filename).name)
    if not match:
        return None
    groups = match.groupdict()
    return {
        "satellite": f"Himawari-{int(groups['sat'])}",
        "date": groups["date"],
        "time": groups["time"],
        "band": groups["band"].upper(),
        "region": groups["region"].upper(),
        "resolution": groups["resolution"].upper(),
        "segment": int(groups["segment"]),
        "total_segments": int(groups["total"]),
    }


def is_hsd_filename(filename: str) -> bool:
    return parse_hsd_filename(Path(filename).name) is not None


def upload_target_dir(filename: str, base_dir: str | Path) -> Path:
    hsd_info = parse_hsd_filename(Path(filename).name)
    if not hsd_info:
        return Path(base_dir)
    return Path(base_dir) / hsd_info["date"] / hsd_info["time"] / "raw"


def select_upload_files(files: list[Any]) -> list[Any]:
    hsd_files = [item for item in files if getattr(item, "filename", None) and is_hsd_filename(item.filename)]
    return hsd_files or files


def scan_hsd_scenes(input_root: str | Path, min_files: int = 10) -> list[dict[str, Any]]:
    root = Path(input_root)
    scenes: list[dict[str, Any]] = []
    for raw_dir in sorted(root.glob("*/*/raw")):
        files = sorted(raw_dir.glob("HS_H*.DAT.bz2"))
        infos = [info for item in files if (info := parse_hsd_filename(item.name))]
        if len(infos) < min_files:
            continue
        scenes.append({"date": infos[0]["date"], "time": infos[0]["time"], "raw_dir": raw_dir, "file_count": len(files), "bands": sorted({item["band"] for item in infos})})
    return scenes


def build_latlon_grid(extent: list[float] | None = None, resolution: float = LATLON_RESOLUTION) -> dict[str, Any]:
    west, south, east, north = extent or CHINA_EXTENT
    nx = int(round((east - west) / resolution)) + 1
    ny = int(round((north - south) / resolution)) + 1
    return {"projection": "EPSG:4326", "grid_type": "regular_latlon", "extent": [west, south, east, north], "resolution": resolution, "nx": nx, "ny": ny}


def _normalize_for_png(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(values)
    norm = np.zeros(values.shape, dtype=np.float32)
    if vmax > vmin:
        norm[valid] = np.clip((values[valid] - vmin) / (vmax - vmin), 0, 1)
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    gray = (norm * 255).astype(np.uint8)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    rgba[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return rgba


def _render_png(data: np.ndarray, png_path: Path, catalog: dict[str, Any]) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = _normalize_for_png(data, float(catalog.get("vmin", np.nanmin(data))), float(catalog.get("vmax", np.nanmax(data))))
    Image.fromarray(rgba).save(png_path)


def _latlon_coords(grid: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    west, south, east, north = grid["extent"]
    lon = np.linspace(west, east, grid["nx"], dtype=np.float32)
    lat = np.linspace(north, south, grid["ny"], dtype=np.float32)
    return lat, lon


def _write_netcdf(data: np.ndarray, nc_path: Path, band: str, catalog: dict[str, Any], grid: dict[str, Any]) -> None:
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    lat, lon = _latlon_coords(grid)
    dataset = xr.Dataset(
        data_vars={band: (("lat", "lon"), data, {"long_name": catalog["name_zh"], "plain_name": catalog["plain_name"], "units": catalog["unit"], "description": catalog["description"]})},
        coords={"lat": lat, "lon": lon},
        attrs={"projection": grid["projection"], "grid_type": grid["grid_type"], "extent": ",".join(str(item) for item in grid["extent"]), "resolution": grid["resolution"]},
    )
    dataset.to_netcdf(nc_path)
    dataset.close()


def write_latlon_variable(output_dir: str | Path, band: str, data: np.ndarray, grid: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    latlon_dir = output_dir / "latlon"
    latlon_dir.mkdir(parents=True, exist_ok=True)
    band = band.upper()
    catalog = BAND_CATALOG[band]
    values = np.asarray(data, dtype=np.float32)
    png_path = latlon_dir / f"{band}.png"
    float32_path = latlon_dir / f"{band}.float32"
    nc_path = latlon_dir / f"{band}.nc"
    _render_png(values, png_path, catalog)
    values.tofile(float32_path)
    _write_netcdf(values, nc_path, band, catalog, grid)
    finite = values[np.isfinite(values)]
    stats = {"min": float(np.nanmin(finite)) if finite.size else None, "max": float(np.nanmax(finite)) if finite.size else None, "mean": float(np.nanmean(finite)) if finite.size else None}
    public_meta = {key: value for key, value in catalog.items() if key not in {"cmap", "vmin", "vmax"}}
    return {**public_meta, "grid": {"nx": grid["nx"], "ny": grid["ny"]}, "extent": grid["extent"], "png": png_path.as_posix(), "float32": float32_path.as_posix(), "netcdf": nc_path.as_posix(), "stats": stats}


def write_scene_metadata(scene_dir: str | Path, date: str, time: str, satellite: str, raw_dir: str | Path, raw_file_count: int, grid: dict[str, Any], variables: list[dict[str, Any]], composites: list[dict[str, Any]]) -> dict[str, Any]:
    scene_dir = Path(scene_dir)
    meta_dir = scene_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "scene_id": f"{date}_{time}",
        "satellite": satellite,
        "observation_time": f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}:00Z",
        "projection": grid["projection"],
        "grid_type": grid["grid_type"],
        "extent": grid["extent"],
        "resolution": grid["resolution"],
        "grid": {"nx": grid["nx"], "ny": grid["ny"]},
        "variables": variables,
        "composites": composites,
        "loaded_bands": [item["key"] for item in variables],
        "source_raw_dir": Path(raw_dir).as_posix(),
        "raw_file_count": raw_file_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with (meta_dir / "scene.meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    return meta


def _read_reusable_scene_metadata(scene_dir: Path) -> dict[str, Any] | None:
    meta_path = scene_dir / "meta" / "scene.meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)
    products = list(meta.get("variables", [])) + list(meta.get("composites", []))
    if not products:
        return None
    for item in products:
        png = item.get("png")
        if png and not Path(png).exists():
            return None
    return meta


def _find_raw_dir(input_root: str | Path, date: str | None, time: str | None) -> Path:
    root = Path(input_root)
    if root.is_file():
        root = root.parent
    if list(root.glob("HS_H*.DAT*")):
        return root
    if date and time:
        raw_dir = root / date / time / "raw"
        if raw_dir.exists():
            return raw_dir
    scenes = scan_hsd_scenes(root, min_files=1)
    if scenes:
        return scenes[0]["raw_dir"]
    raise FileNotFoundError(f"未找到 Himawari HSD raw 目录: {root}")


def _scene_info_from_files(files: list[Path]) -> dict[str, Any]:
    infos = [info for item in files if (info := parse_hsd_filename(item.name))]
    if not infos:
        raise ValueError("未找到可识别的 Himawari HSD 文件名。")
    return {"satellite": infos[0]["satellite"], "date": infos[0]["date"], "time": infos[0]["time"], "bands": sorted({item["band"] for item in infos})}


def _resample_dataset_to_latlon(dataset: Any, grid: dict[str, Any]) -> np.ndarray:
    from pyproj import Transformer

    area = dataset.attrs.get("area") or dataset.area
    source_height, source_width = dataset.shape[:2]
    x0, y0, x1, y1 = area.area_extent
    target_lat, target_lon = _latlon_coords(grid)
    lon2d, lat2d = np.meshgrid(target_lon, target_lat)
    transformer = Transformer.from_crs("EPSG:4326", area.crs, always_xy=True)
    xs, ys = transformer.transform(lon2d, lat2d)
    cols = (xs - x0) / (x1 - x0) * (source_width - 1)
    rows = (y1 - ys) / (y1 - y0) * (source_height - 1)
    valid = np.isfinite(rows) & np.isfinite(cols) & (rows >= 0) & (rows <= source_height - 1) & (cols >= 0) & (cols <= source_width - 1)
    if not np.any(valid):
        return np.full((grid["ny"], grid["nx"]), np.nan, dtype=np.float32)
    row_min = max(0, int(np.floor(np.nanmin(rows[valid]))) - 2)
    row_max = min(source_height - 1, int(np.ceil(np.nanmax(rows[valid]))) + 2)
    col_min = max(0, int(np.floor(np.nanmin(cols[valid]))) - 2)
    col_max = min(source_width - 1, int(np.ceil(np.nanmax(cols[valid]))) + 2)
    source = np.asarray(dataset.values[row_min : row_max + 1, col_min : col_max + 1], dtype=np.float32)
    sampled = np.full((grid["ny"], grid["nx"]), np.nan, dtype=np.float32)
    coords = np.vstack([(rows[valid] - row_min).ravel(), (cols[valid] - col_min).ravel()])
    sampled[valid] = map_coordinates(source, coords, order=1, mode="constant", cval=np.nan).astype(np.float32)
    return sampled


def _available_ahi_bands(scene: Any, requested: list[str] | None) -> list[str]:
    available = {str(item).upper() for item in scene.available_dataset_names()}
    candidates = [item.upper() for item in requested] if requested else list(BAND_CATALOG)
    return [band for band in candidates if band in available]


def _normalize_channel(values: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.zeros(data.shape, dtype=np.float32)
    if low is None or high is None:
        low, high = np.nanpercentile(valid, [2, 98])
    if high <= low:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - low) / (high - low), 0, 1).astype(np.float32)


def _save_rgb_png(rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.all(np.isfinite(rgb), axis=-1), 255, 0).astype(np.uint8)
    Image.fromarray(rgba).save(output_path)


def _rgb_from_composite(key: str, arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    if key == "true_color" and all(item in arrays for item in ("B03", "B02", "B01")):
        return np.stack([_normalize_channel(arrays["B03"]), _normalize_channel(arrays["B02"]), _normalize_channel(arrays["B01"])], axis=-1)
    if key == "natural_color" and all(item in arrays for item in ("B05", "B04", "B03")):
        return np.stack([_normalize_channel(arrays["B05"]), _normalize_channel(arrays["B04"]), _normalize_channel(arrays["B03"])], axis=-1)
    if key == "air_mass" and all(item in arrays for item in ("B08", "B10", "B12", "B13")):
        return np.stack([_normalize_channel(arrays["B08"] - arrays["B10"], -25, 0), _normalize_channel(arrays["B12"] - arrays["B13"], -40, 5), _normalize_channel(arrays["B08"], 210, 260)], axis=-1)
    if key == "dust" and all(item in arrays for item in ("B11", "B13", "B14", "B15")):
        return np.stack([_normalize_channel(arrays["B15"] - arrays["B13"], -4, 2), _normalize_channel(arrays["B14"] - arrays["B11"], 0, 15), _normalize_channel(arrays["B13"], 260, 320)], axis=-1)
    if key == "night_microphysics" and all(item in arrays for item in ("B07", "B13", "B15")):
        return np.stack([_normalize_channel(arrays["B15"] - arrays["B13"], -6, 2), _normalize_channel(arrays["B13"] - arrays["B07"], -4, 8), _normalize_channel(arrays["B13"], 240, 300)], axis=-1)
    if key == "water_vapor_enhanced" and all(item in arrays for item in ("B08", "B09", "B10")):
        return np.stack([_normalize_channel(arrays["B08"], 210, 260), _normalize_channel(arrays["B09"], 215, 270), _normalize_channel(arrays["B10"], 220, 280)], axis=-1)
    return None


def write_composites(scene_dir: str | Path, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    scene_dir = Path(scene_dir)
    output = []
    for key, catalog in COMPOSITE_CATALOG.items():
        rgb = _rgb_from_composite(key, arrays)
        if rgb is None:
            continue
        png_path = scene_dir / "composites" / f"{key}.png"
        _save_rgb_png(rgb, png_path)
        output.append({**catalog, "png": png_path.as_posix()})
    return output


def process_scene(input_root: str | Path, output_root: str | Path = DATA_DIR, date: str | None = None, time: str | None = None, bands: list[str] | None = None, extent: list[float] | None = None, resolution: float = LATLON_RESOLUTION, composites: bool = True) -> dict[str, Any]:
    raw_dir = _find_raw_dir(input_root, date, time)
    files = sorted(raw_dir.glob("HS_H*.DAT*"))
    if not files:
        raise FileNotFoundError(f"未找到 Himawari HSD 文件: {raw_dir}")
    scene_info = _scene_info_from_files(files)
    date = date or scene_info["date"]
    time = time or scene_info["time"]
    grid = build_latlon_grid(extent=extent, resolution=resolution)
    scene_dir = Path(output_root) / date / time
    if bands is None and extent is None and resolution == LATLON_RESOLUTION and composites:
        if meta := _read_reusable_scene_metadata(scene_dir):
            return meta

    from satpy import Scene

    filenames = [item.as_posix() for item in files]
    probe_scene = Scene(reader="ahi_hsd", filenames=filenames)
    load_bands = _available_ahi_bands(probe_scene, bands)
    if not load_bands:
        raise ValueError("HSD 场景中没有可解析的 AHI B01-B16 通道。")
    variables: list[dict[str, Any]] = []
    resampled_arrays: dict[str, np.ndarray] = {}
    for band in load_bands:
        scene = Scene(reader="ahi_hsd", filenames=filenames)
        scene.load([band])
        values = _resample_dataset_to_latlon(scene[band], grid)
        variables.append(write_latlon_variable(scene_dir, band, values, grid))
        if composites:
            resampled_arrays[band] = values
        del scene, values
    composite_meta = write_composites(scene_dir, resampled_arrays) if composites else []
    return write_scene_metadata(scene_dir, date, time, scene_info["satellite"], raw_dir, len(files), grid, variables, composite_meta)


def process_file(file_path: str, data_type: str = "Himawari") -> dict:
    try:
        return process_scene(Path(file_path))
    except Exception:
        weather_info = {"source": "Himawari", "product": "葵花卫星产品", "element": "卫星通道", "time": "解析失败", "level": "卫星观测", "range": "待解析", "resolution": "待解析", "grid": "待解析", "validGrid": "待解析", "coverage": "待解析", "missing": "待解析", "unit": "待解析", "variables": "待解析", "steps": "待解析", "status": "已接收但未形成完整 HSD 场景", "quality": "待解析", "max": "待解析", "min": "待解析", "mean": "待解析", "alert": "请上传完整 HSD raw 场景目录或分段集合。", "update": "待解析", "bars": [0, 0, 0, 0, 0], "trend": [0, 0, 0, 0, 0, 0, 0, 0]}
        return process_basic_file(file_path, data_type=data_type, file_format="HSD", weather_info=weather_info)


def main() -> int:
    parser = argparse.ArgumentParser(description="解析 Himawari HSD 为等经纬网格产品")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", default=DATA_DIR.as_posix())
    parser.add_argument("--date")
    parser.add_argument("--time")
    parser.add_argument("--bands", help="逗号分隔，例如 B01,B02,B13；默认解析所有可用通道")
    parser.add_argument("--extent", default=",".join(str(item) for item in CHINA_EXTENT))
    parser.add_argument("--resolution", type=float, default=LATLON_RESOLUTION)
    parser.add_argument("--no-composites", action="store_true")
    args = parser.parse_args()
    bands = [item.strip().upper() for item in args.bands.split(",") if item.strip()] if args.bands else None
    extent = [float(item.strip()) for item in args.extent.split(",")]
    meta = process_scene(args.input_root, args.output_root, args.date, args.time, bands, extent, args.resolution, not args.no_composites)
    print(json.dumps({"scene_id": meta["scene_id"], "variables": meta["loaded_bands"], "composites": [item["key"] for item in meta["composites"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
