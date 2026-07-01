from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ftplib
import json
import os
import posixpath
import re
import shutil
import sys
from threading import Lock
import time as time_module
from datetime import datetime, timedelta, timezone
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
FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_ROOT = "/jma/hsd"
FTP_PORT = 21
FTP_TIMEOUT = 60
HIMAWARI_TARGET_BANDS = ["B13", "B03", "B02", "B01"]
HIMAWARI_QUICK_BANDS = HIMAWARI_TARGET_BANDS
HIMAWARI_FULL_BANDS = HIMAWARI_TARGET_BANDS
TRUE_COLOR_BANDS = ["B03", "B02", "B01"]
SLOW_MIN_REMAINING_MINUTES = 120
B13_FAST_BANDS = ["B13"]
VISIBLE_COLOR_BANDS = ["B03", "B02", "B01"]
VISIBLE_LOCAL_START_HOUR = 6
VISIBLE_LOCAL_END_HOUR = 18
VISIBLE_LOCAL_UTC_OFFSET_HOURS = 8
PARTIAL_MAX_AGE_HOURS = 6
DEFAULT_WINDOW_HOURS = 24
DEFAULT_LATEST_DELAY_MINUTES = 60


def _emit_progress(progress_callback: Any, **event: Any) -> None:
    if not progress_callback:
        return
    try:
        progress_callback(event)
    except Exception:
        pass


def _ordered_unique_bands(bands: list[str] | None) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for band in bands or []:
        key = str(band).upper()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _band_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"B(\d{2})", str(value).upper())
    return (int(match.group(1)) if match else 999, str(value))


def _scene_datetime(date: str, scene_time: str) -> datetime | None:
    try:
        return datetime.strptime(f"{date}{scene_time}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _scene_beijing_datetime(date: str, scene_time: str) -> datetime | None:
    scene_dt = _scene_datetime(date, scene_time)
    if not scene_dt:
        return None
    return scene_dt + timedelta(hours=VISIBLE_LOCAL_UTC_OFFSET_HOURS)


def _is_visible_light_slot(date: str, scene_time: str) -> bool:
    local_dt = _scene_beijing_datetime(date, scene_time)
    if not local_dt:
        return False
    return VISIBLE_LOCAL_START_HOUR <= local_dt.hour < VISIBLE_LOCAL_END_HOUR


def _visible_light_bands_for_slot(date: str, scene_time: str) -> list[str]:
    return list(VISIBLE_COLOR_BANDS) if _is_visible_light_slot(date, scene_time) else []


def _product_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("key") or "").strip()


def _merge_keyed_items(existing: list[dict[str, Any]], incoming: list[dict[str, Any]], sort_bands: bool = False) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in existing + incoming:
        key = _product_name(item)
        if not key:
            continue
        merged[key] = item
    values = list(merged.values())
    if sort_bands:
        return sorted(values, key=lambda item: _band_sort_key(_product_name(item)))
    return values


def _stats_template(stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = stats or {}
    return {
        "min": stats.get("min"),
        "max": stats.get("max"),
        "mean": stats.get("mean"),
        "std": stats.get("std"),
    }


def _grid_shape(item: dict[str, Any], grid: dict[str, Any]) -> list[int]:
    shape = item.get("shape")
    if isinstance(shape, list) and shape:
        return shape
    item_grid = item.get("grid") if isinstance(item.get("grid"), dict) else grid
    ny = item_grid.get("ny") if isinstance(item_grid, dict) else None
    nx = item_grid.get("nx") if isinstance(item_grid, dict) else None
    return [int(ny), int(nx)] if ny and nx else []


def _normalize_himawari_variable(item: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    name = _product_name(item)
    return {
        "name": name,
        "long_name": item.get("long_name") or item.get("plain_name"),
        "short_name": item.get("short_name") or name or None,
        "raw_name": item.get("raw_name"),
        "name_cn": item.get("name_cn") or item.get("name_zh"),
        "unit": item.get("unit"),
        "display_unit": item.get("display_unit"),
        "shape": _grid_shape(item, grid),
        "dims": item.get("dims") or ["lat", "lon"],
        "level": item.get("level"),
        "missing": item.get("missing"),
        "stats": _stats_template(item.get("stats")),
        "category": item.get("category"),
        "description": item.get("description"),
        "wavelength": item.get("wavelength"),
        "float32": item.get("float32"),
        "netcdf": item.get("netcdf"),
        "png": item.get("png"),
    }


def _normalize_himawari_composite(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _product_name(item),
        "name_cn": item.get("name_cn") or item.get("name_zh"),
        "description": item.get("description") or item.get("long_name") or item.get("plain_name"),
        "source_bands": item.get("source_bands") or [],
        "float32": item.get("float32"),
        "netcdf": item.get("netcdf"),
        "png": item.get("png"),
    }


def _format_extent_label(extent: list[Any]) -> str:
    if not isinstance(extent, list) or len(extent) != 4:
        return ""
    west, south, east, north = extent
    return f"{west}°E-{east}°E, {south}°N-{north}°N"


def _default_himawari_variable(variables: list[dict[str, Any]]) -> str | None:
    names = [_product_name(item) for item in variables if _product_name(item)]
    if "B13" in names:
        return "B13"
    return names[0] if names else None


def _default_himawari_png(variables: list[dict[str, Any]], composites: list[dict[str, Any]], default_variable: str | None) -> str | None:
    for item in variables:
        if _product_name(item) == default_variable and item.get("png"):
            return item["png"]
    for item in variables:
        if _product_name(item) == "B13" and item.get("png"):
            return item["png"]
    for item in composites:
        if _product_name(item) == "true_color" and item.get("png"):
            return item["png"]
    for item in variables + composites:
        if item.get("png"):
            return item["png"]
    return None


def _weather_info(meta: dict[str, Any], variables: list[dict[str, Any]], default_variable: str | None, generated_at: str | None) -> dict[str, Any]:
    existing = meta.get("weather_info") if isinstance(meta.get("weather_info"), dict) else {}
    default_item = next((item for item in variables if _product_name(item) == default_variable), variables[0] if variables else {})
    stats = default_item.get("stats") or {}
    extent = meta.get("extent") or meta.get("bbox") or CHINA_EXTENT
    grid = meta.get("grid") if isinstance(meta.get("grid"), dict) else {}
    return {
        "source": existing.get("source") or "Himawari",
        "product": existing.get("product") or "葵花静止卫星 HSD 等经纬度网格产品",
        "element": existing.get("element") or default_item.get("name_cn") or default_item.get("long_name") or default_variable or "卫星通道",
        "time": existing.get("time") or meta.get("observation_time") or "",
        "level": existing.get("level") or "卫星观测",
        "range": existing.get("range") or _format_extent_label(extent),
        "resolution": existing.get("resolution") or (f"{meta.get('resolution')}°" if meta.get("resolution") is not None else ""),
        "grid": existing.get("grid") or (f"{grid.get('nx')} × {grid.get('ny')}" if grid.get("nx") and grid.get("ny") else ""),
        "valid_grid": existing.get("valid_grid"),
        "coverage": existing.get("coverage"),
        "missing": existing.get("missing"),
        "unit": existing.get("unit") or default_item.get("display_unit") or default_item.get("unit"),
        "variable_count": existing.get("variable_count") or len(variables),
        "step_count": existing.get("step_count") or 1,
        "status": existing.get("status") or "解析完成",
        "quality": existing.get("quality") or "等经纬度网格已生成",
        "max": existing.get("max") if existing.get("max") is not None else stats.get("max"),
        "min": existing.get("min") if existing.get("min") is not None else stats.get("min"),
        "mean": existing.get("mean") if existing.get("mean") is not None else stats.get("mean"),
        "alert": existing.get("alert"),
        "updated_at": existing.get("updated_at") or generated_at,
        "bars": existing.get("bars") or [],
        "bars_labels": existing.get("bars_labels") or [],
        "trend": existing.get("trend") or [],
        "trend_times": existing.get("trend_times") or [],
    }


def normalize_himawari_meta(meta: dict[str, Any], meta_path: str | Path | None = None) -> dict[str, Any]:
    meta = dict(meta)
    scene_id = meta.get("scene_id") or ""
    observation_time = meta.get("observation_time") or (meta.get("times") or [""])[0]
    extent = list(meta.get("extent") or meta.get("bbox") or CHINA_EXTENT)
    grid = meta.get("grid") if isinstance(meta.get("grid"), dict) else {}
    if not grid:
        grid = build_latlon_grid(extent, float(meta.get("resolution") or LATLON_RESOLUTION))
    grid = {"nx": int(grid.get("nx")), "ny": int(grid.get("ny"))} if grid.get("nx") and grid.get("ny") else grid
    generated_at = meta.get("generated_at") or meta.get("extra", {}).get("generated_at") or datetime.now(timezone.utc).isoformat()
    variables = [_normalize_himawari_variable(item, grid) for item in meta.get("variables", []) if _product_name(item)]
    composites = [_normalize_himawari_composite(item) for item in meta.get("composites", []) if _product_name(item)]
    loaded_bands = sorted({_product_name(item) for item in variables if _product_name(item).upper().startswith("B")}, key=_band_sort_key)
    default_variable = meta.get("default_variable") or _default_himawari_variable(variables)
    png_files = [item["png"] for item in variables + composites if item.get("png")]
    default_png = meta.get("default_png") or _default_himawari_png(variables, composites, default_variable)
    source_raw_dir = meta.get("source_raw_dir") or meta.get("source_file") or ""
    raw_file_count = int(meta.get("raw_file_count") or meta.get("extra", {}).get("himawari", {}).get("raw_file_count") or 0)
    retention_managed = bool(meta.get("retention_managed") or meta.get("extra", {}).get("himawari", {}).get("retention_managed"))
    meta_file = Path(meta_path).as_posix() if meta_path else meta.get("meta_file", "")
    projection = meta.get("projection") or meta.get("extra", {}).get("himawari", {}).get("projection") or "EPSG:4326"
    grid_type = meta.get("grid_type") or meta.get("extra", {}).get("himawari", {}).get("grid_type") or "regular_latlon"
    resolution = meta.get("resolution")
    normalized = {
        "schema_version": "1.0",
        "dataset_id": meta.get("dataset_id") or f"{scene_id}_himawari_hsd",
        "data_type": "Himawari",
        "file_format": "HSD",
        "source_file": source_raw_dir,
        "meta_file": meta_file,
        "png_files": png_files,
        "default_png": default_png,
        "default_variable": default_variable,
        "times": [observation_time] if observation_time else [],
        "levels": [],
        "bbox": extent,
        "scene_id": scene_id,
        "satellite": meta.get("satellite") or meta.get("extra", {}).get("himawari", {}).get("satellite") or "Himawari-9",
        "observation_time": observation_time,
        "projection": projection,
        "grid_type": grid_type,
        "extent": extent,
        "resolution": resolution,
        "grid": grid,
        "variables": variables,
        "composites": composites,
        "weather_info": {},
        "extra": {
            "status": meta.get("extra", {}).get("status") or "parsed",
            "generated_at": generated_at,
            "note": meta.get("extra", {}).get("note"),
            "cma": meta.get("extra", {}).get("cma", {}),
            "era5": meta.get("extra", {}).get("era5", {}),
            "gfs": meta.get("extra", {}).get("gfs", {}),
            "himawari": {
                "satellite": meta.get("satellite") or meta.get("extra", {}).get("himawari", {}).get("satellite") or "Himawari-9",
                "projection": projection,
                "grid_type": grid_type,
                "loaded_bands": loaded_bands,
                "raw_file_count": raw_file_count,
                "source_raw_dir": source_raw_dir,
                "retention_managed": retention_managed,
            },
            "radar": meta.get("extra", {}).get("radar", {}),
            "wrf": meta.get("extra", {}).get("wrf", {}),
        },
        "loaded_bands": loaded_bands,
        "source_raw_dir": source_raw_dir,
        "raw_file_count": raw_file_count,
        "retention_managed": retention_managed,
        "generated_at": generated_at,
    }
    normalized["weather_info"] = _weather_info(normalized, variables, default_variable, generated_at)
    return normalized


def _merge_scene_metadata(
    scene_dir: Path,
    meta: dict[str, Any],
    variables: list[dict[str, Any]],
    composites: list[dict[str, Any]],
    raw_file_count: int,
    retention_managed: bool,
) -> dict[str, Any]:
    meta_path = scene_dir / "meta" / "scene.meta.json"
    if not meta_path.exists():
        return normalize_himawari_meta(meta, meta_path)
    try:
        with meta_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    except (OSError, json.JSONDecodeError):
        return normalize_himawari_meta(meta, meta_path)

    existing = normalize_himawari_meta(existing, meta_path)
    incoming = normalize_himawari_meta(meta, meta_path)
    merged_variables = _merge_keyed_items(existing.get("variables", []), variables, sort_bands=True)
    merged_composites = _merge_keyed_items(existing.get("composites", []), composites)
    loaded_bands = sorted({_product_name(item) for item in merged_variables if _product_name(item)}, key=_band_sort_key)
    incoming.update(
        {
            "variables": merged_variables,
            "composites": merged_composites,
            "loaded_bands": loaded_bands,
            "raw_file_count": int(existing.get("raw_file_count") or 0) + raw_file_count,
            "retention_managed": bool(existing.get("retention_managed")) or retention_managed,
        }
    )
    incoming["extra"]["himawari"]["raw_file_count"] = incoming["raw_file_count"]
    incoming["extra"]["himawari"]["retention_managed"] = incoming["retention_managed"]
    incoming["extra"]["himawari"]["loaded_bands"] = loaded_bands
    return normalize_himawari_meta(incoming, meta_path)


def _validate_hsd_date_time(date: str, time: str) -> tuple[str, str]:
    if not re.fullmatch(r"\d{8}", date or ""):
        raise ValueError("Himawari 日期必须是 YYYYMMDD，例如 20260616。")
    if not re.fullmatch(r"\d{4}", time or ""):
        raise ValueError("Himawari 时次必须是 HHMM，例如 0000。")
    return date, time


def build_himawari_remote_dir(date: str, time: str, root: str = FTP_ROOT) -> str:
    return build_himawari_remote_dirs(date, time, root)[0]


def build_himawari_remote_dirs(date: str, time: str, root: str = FTP_ROOT) -> list[str]:
    date, time = _validate_hsd_date_time(date, time)
    values = {
        "date": date,
        "time": time,
        "yyyy": date[:4],
        "yyyymm": date[:6],
        "dd": date[6:8],
        "hh": time[:2],
    }
    root = (root or FTP_ROOT).rstrip("/")
    dirs: list[str] = []
    if "{" in root:
        dirs.append(root.format(**values))
        base = root.split("{", 1)[0].rstrip("/")
    else:
        base = root
    dirs.extend(
        [
            posixpath.join(base, values["yyyymm"], values["dd"], values["hh"]),
            posixpath.join(base, values["yyyymm"], values["dd"], time),
            posixpath.join(base, date, time),
        ]
    )
    unique_dirs = []
    for item in dirs:
        if item not in unique_dirs:
            unique_dirs.append(item)
    return unique_dirs


def latest_himawari_slot(now: datetime | None = None, delay_minutes: int = DEFAULT_LATEST_DELAY_MINUTES) -> tuple[str, str]:
    base = now or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    target = base.astimezone(timezone.utc) - timedelta(minutes=delay_minutes)
    minute = target.minute - (target.minute % 10)
    target = target.replace(minute=minute, second=0, microsecond=0)
    return target.strftime("%Y%m%d"), target.strftime("%H%M")


def himawari_slot_window(
    now: datetime | None = None,
    hours: int = DEFAULT_WINDOW_HOURS,
    delay_minutes: int = DEFAULT_LATEST_DELAY_MINUTES,
    interval_minutes: int = 10,
) -> list[tuple[str, str]]:
    if hours <= 0:
        raise ValueError("Himawari 自动下载保留窗口必须大于 0 小时。")
    if interval_minutes <= 0:
        raise ValueError("Himawari 自动下载间隔必须大于 0 分钟。")

    latest_date, latest_time = latest_himawari_slot(now=now, delay_minutes=delay_minutes)
    latest = datetime.strptime(f"{latest_date}{latest_time}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    earliest = latest - timedelta(hours=hours)
    if earliest > latest:
        earliest = latest
    slots = []
    current = earliest
    while current <= latest:
        slots.append((current.strftime("%Y%m%d"), current.strftime("%H%M")))
        current += timedelta(minutes=interval_minutes)
    return slots


def _matching_remote_hsd_files(
    names: list[str],
    date: str,
    scene_time: str,
    bands: list[str] | None = None,
    region: str = "FLDK",
) -> list[str]:
    requested = {item.upper() for item in bands} if bands else None
    requested_region = region.upper() if region else None
    matched = []
    for name in names:
        filename = Path(name).name
        info = parse_hsd_filename(filename)
        if not info:
            continue
        if info["date"] != date or info["time"] != scene_time:
            continue
        if requested and info["band"] not in requested:
            continue
        if requested_region and info["region"] != requested_region:
            continue
        matched.append(name)
    return sorted(matched, key=lambda item: Path(item).name)


def _close_ftp(ftp: Any) -> None:
    try:
        ftp.quit()
    except Exception:
        close = getattr(ftp, "close", None)
        if close:
            close()


def _ftp_cwd_first_existing(ftp: Any, remote_dirs: list[str]) -> str:
    errors: list[str] = []
    for remote_dir in remote_dirs:
        try:
            ftp.cwd(remote_dir)
            return remote_dir
        except ftplib.error_perm as exc:
            errors.append(f"{remote_dir}: {exc}")
    raise ftplib.error_perm("Himawari 远端目录不存在，已尝试: " + " | ".join(errors))


def _remote_retr_path(remote_dir: str, remote_name: str) -> str:
    if remote_name.startswith("/"):
        return remote_name
    return posixpath.join(remote_dir, Path(remote_name).name)


def _download_himawari_file(
    remote_dir: str,
    remote_name: str,
    target: Path,
    host: str,
    user: str,
    password: str,
    timeout: int,
    ftp_factory: Any,
) -> str:
    ftp = ftp_factory()
    part = target.with_name(f"{target.name}.part")
    try:
        ftp.connect(host, FTP_PORT, timeout=timeout)
        ftp.login(user, password)
        target.parent.mkdir(parents=True, exist_ok=True)
        resume_at = part.stat().st_size if part.exists() else 0
        mode = "ab" if resume_at > 0 else "wb"
        with part.open(mode) as file:
            ftp.retrbinary(f"RETR {_remote_retr_path(remote_dir, remote_name)}", file.write, rest=resume_at or None)
        part.replace(target)
        return target.as_posix()
    except Exception:
        raise
    finally:
        _close_ftp(ftp)


def download_himawari_hsd_scene(
    date: str,
    time: str,
    output_root: str | Path = DATA_DIR,
    bands: list[str] | None = None,
    overwrite: bool = False,
    parse_after_download: bool = False,
    delete_raw_after_parse: bool = True,
    retention_hours: int = DEFAULT_WINDOW_HOURS,
    ftp_factory: Any = ftplib.FTP,
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
    remote_root: str | None = None,
    timeout: int = FTP_TIMEOUT,
    phase: str | None = None,
    progress_callback: Any = None,
    file_workers: int = 1,
    latest_delay_minutes: int = 0,
) -> dict[str, Any]:
    date, time = _validate_hsd_date_time(date, time)
    env = os.environ
    host = (host or env.get("HIMAWARI_FTP_HOST", FTP_HOST)).strip()
    user = (user or env.get("HIMAWARI_FTP_USER") or "").strip()
    password = (password or env.get("HIMAWARI_FTP_PASSWORD") or "").strip()
    remote_root = (remote_root or env.get("HIMAWARI_FTP_ROOT", FTP_ROOT)).strip()
    if not user or not password:
        raise ValueError("请设置 HIMAWARI_FTP_USER 和 HIMAWARI_FTP_PASSWORD 后再下载 Himawari HSD。")
    remote_dirs = build_himawari_remote_dirs(date, time, remote_root)
    raw_dir = Path(output_root) / date / time / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    scene_id = f"{date}_{time}"

    ftp = ftp_factory()
    downloaded: list[str] = []
    skipped: list[str] = []
    try:
        _emit_progress(progress_callback, stage="connecting", phase=phase, scene_id=scene_id, detail="连接 Himawari FTP")
        ftp.connect(host, FTP_PORT, timeout=timeout)
        ftp.login(user, password)
        _emit_progress(progress_callback, stage="listing", phase=phase, scene_id=scene_id, detail="读取远端小时目录")
        remote_dir = _ftp_cwd_first_existing(ftp, remote_dirs)
        remote_files = _matching_remote_hsd_files(list(ftp.nlst()), date, time, bands)
        download_targets: list[tuple[int, str, str, Path]] = []
        for index, remote_name in enumerate(remote_files, start=1):
            raw_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(remote_name).name
            target = raw_dir / filename
            if target.exists() and target.stat().st_size > 0 and not overwrite:
                skipped.append(target.as_posix())
                continue
            download_targets.append((index, remote_name, filename, target))

        if file_workers > 1 and len(download_targets) > 1:
            done_count = 0
            with ThreadPoolExecutor(max_workers=max(1, min(int(file_workers), len(download_targets)))) as executor:
                futures = {}
                for index, remote_name, filename, target in download_targets:
                    _emit_progress(
                        progress_callback,
                        stage="downloading",
                        phase=phase,
                        scene_id=scene_id,
                        file=filename,
                        band=(parse_hsd_filename(filename) or {}).get("band"),
                        detail="并发下载 HSD 分段",
                        queue_done=done_count,
                        queue_total=len(remote_files),
                    )
                    futures[
                        executor.submit(
                            _download_himawari_file,
                            remote_dir,
                            remote_name,
                            target,
                            host,
                            user,
                            password,
                            timeout,
                            ftp_factory,
                        )
                    ] = filename
                for future in as_completed(futures):
                    downloaded.append(future.result())
                    done_count += 1
                    filename = futures[future]
                    _emit_progress(
                        progress_callback,
                        stage="downloading",
                        phase=phase,
                        scene_id=scene_id,
                        file=filename,
                        band=(parse_hsd_filename(filename) or {}).get("band"),
                        detail="HSD 分段下载完成",
                        queue_done=len(skipped) + done_count,
                        queue_total=len(remote_files),
                    )
        else:
            for index, remote_name, filename, target in download_targets:
                _emit_progress(
                    progress_callback,
                    stage="downloading",
                    phase=phase,
                    scene_id=scene_id,
                    file=filename,
                    band=(parse_hsd_filename(filename) or {}).get("band"),
                    detail="下载 HSD 分段",
                    queue_done=index - 1,
                    queue_total=len(remote_files),
                )
                part = target.with_name(f"{target.name}.part")
                try:
                    resume_at = part.stat().st_size if part.exists() else 0
                    mode = "ab" if resume_at > 0 else "wb"
                    with part.open(mode) as file:
                        ftp.retrbinary(f"RETR {remote_name}", file.write, rest=resume_at or None)
                except Exception:
                    raise
                part.replace(target)
                downloaded.append(target.as_posix())
        _emit_progress(progress_callback, stage="downloaded", phase=phase, scene_id=scene_id, detail="HSD 分段下载完成", queue_done=len(remote_files), queue_total=len(remote_files))
    except Exception:
        if parse_after_download and raw_dir.exists():
            _emit_progress(progress_callback, stage="failed", phase=phase, scene_id=scene_id, detail="下载失败，保留 raw 供下一轮续传")
        raise
    finally:
        _close_ftp(ftp)

    result: dict[str, Any] = {
        "date": date,
        "time": time,
        "phase": phase,
        "remote_dir": remote_dir,
        "tried_remote_dirs": remote_dirs,
        "raw_dir": raw_dir.as_posix(),
        "downloaded": downloaded,
        "skipped": skipped,
        "downloaded_count": len(downloaded),
        "skipped_count": len(skipped),
        "file_count": len(downloaded) + len(skipped),
    }
    if parse_after_download:
        if not _raw_scene_has_complete_bands(raw_dir, bands=bands):
            result["parse_skipped"] = "HSD raw 分段未完整，等待下一轮自动补齐后再解析。"
            _emit_progress(progress_callback, stage="downloaded", phase=phase, scene_id=scene_id, detail="raw 分段不完整，保留给下一轮续传")
            return result
        result["parsed"] = process_downloaded_himawari_scene(
            raw_dir,
            output_root,
            delete_raw=delete_raw_after_parse,
            retention_hours=retention_hours,
            latest_delay_minutes=latest_delay_minutes,
            progress_callback=progress_callback,
            phase=phase,
        )
    return result


def run_himawari_download_loop(
    date: str | None,
    scene_time: str | None,
    interval_minutes: int,
    iterations: int | None = None,
    latest: bool = False,
    latest_delay_minutes: int = DEFAULT_LATEST_DELAY_MINUTES,
    **download_kwargs: Any,
) -> list[dict[str, Any]]:
    results = []
    count = 0
    while True:
        if latest:
            run_date, run_time = latest_himawari_slot(delay_minutes=latest_delay_minutes)
        else:
            if not date or not scene_time:
                raise ValueError("非 latest 模式必须提供 date 和 time。")
            run_date, run_time = date, scene_time
        results.append(download_himawari_hsd_scene(run_date, run_time, **download_kwargs))
        count += 1
        if iterations is not None and count >= iterations:
            return results
        time_module.sleep(interval_minutes * 60)


def _parse_observation_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _scene_datetime_from_path(scene_dir: Path) -> datetime | None:
    try:
        date = scene_dir.parent.name
        scene_time = scene_dir.name
        return datetime.strptime(f"{date}{scene_time}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _scene_observation_time(scene_dir: Path) -> datetime | None:
    meta_path = scene_dir / "meta" / "scene.meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as file:
                meta = json.load(file)
            if parsed := _parse_observation_time(meta.get("observation_time")):
                return parsed
        except (OSError, json.JSONDecodeError):
            pass
    return _scene_datetime_from_path(scene_dir)


def _is_retention_managed_scene(scene_dir: Path) -> bool:
    meta_path = scene_dir / "meta" / "scene.meta.json"
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as file:
            meta = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(meta.get("retention_managed"))


def cleanup_himawari_retention(
    output_root: str | Path = DATA_DIR,
    retention_hours: int = DEFAULT_WINDOW_HOURS,
    now: datetime | None = None,
    delay_minutes: int = 0,
) -> list[str]:
    root = Path(output_root)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if delay_minutes > 0:
        latest_date, latest_time = latest_himawari_slot(now=current, delay_minutes=delay_minutes)
        reference = datetime.strptime(f"{latest_date}{latest_time}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    else:
        reference = current.astimezone(timezone.utc)
    cutoff = reference - timedelta(hours=retention_hours)
    removed: list[str] = []
    for scene_dir in sorted(root.glob("*/*")):
        if not scene_dir.is_dir() or scene_dir.name == "raw":
            continue
        if not _is_retention_managed_scene(scene_dir):
            continue
        observation_time = _scene_observation_time(scene_dir)
        if not observation_time or observation_time >= cutoff:
            continue
        shutil.rmtree(scene_dir)
        removed.append(scene_dir.as_posix())
        date_dir = scene_dir.parent
        try:
            if date_dir.is_dir() and not any(date_dir.iterdir()):
                date_dir.rmdir()
        except OSError:
            pass
    return removed


def process_downloaded_himawari_scene(
    raw_dir: str | Path,
    output_root: str | Path = DATA_DIR,
    delete_raw: bool = True,
    retention_hours: int = DEFAULT_WINDOW_HOURS,
    now: datetime | None = None,
    latest_delay_minutes: int = 0,
    **process_kwargs: Any,
) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    progress_callback = process_kwargs.pop("progress_callback", None)
    phase = process_kwargs.pop("phase", None)
    date_name = raw_path.parent.parent.name
    time_name = raw_path.parent.name
    scene_id = f"{date_name}_{time_name}" if re.fullmatch(r"\d{8}", date_name) and re.fullmatch(r"\d{4}", time_name) else raw_path.parent.name
    _emit_progress(progress_callback, stage="parsing", phase=phase, scene_id=scene_id, detail="解析 HSD raw")
    try:
        meta = process_scene(raw_path, output_root=output_root, progress_callback=progress_callback, phase=phase, retention_managed=True, **process_kwargs)
    except Exception as exc:
        if delete_raw and raw_path.exists():
            _emit_progress(progress_callback, stage="cleanup_raw", phase=phase, scene_id=scene_id, detail="解析失败，删除 raw")
            shutil.rmtree(raw_path, ignore_errors=True)
            _remove_empty_scene_dirs(raw_path.parent)
        _emit_progress(progress_callback, stage="failed", phase=phase, scene_id=scene_id, detail=str(exc))
        raise
    if delete_raw and raw_path.exists():
        _emit_progress(progress_callback, stage="cleanup_raw", phase=phase, scene_id=meta.get("scene_id"), detail="解析成功，删除 raw")
        shutil.rmtree(raw_path)
        try:
            if raw_path.parent.is_dir() and not any(raw_path.parent.iterdir()):
                raw_path.parent.rmdir()
        except OSError:
            pass
    cleanup_himawari_retention(output_root, retention_hours=retention_hours, now=now, delay_minutes=latest_delay_minutes)
    _emit_progress(progress_callback, stage="parsed", phase=phase, scene_id=meta.get("scene_id"), detail="PNG/meta 生成完成")
    return meta


def cleanup_partial_himawari_downloads(
    output_root: str | Path = DATA_DIR,
    max_age_hours: int = PARTIAL_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> list[str]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current.timestamp() - max(0, max_age_hours) * 3600
    removed: list[str] = []
    for path in sorted(Path(output_root).glob("*/*/raw/*.part")):
        try:
            if max_age_hours > 0 and path.stat().st_mtime >= cutoff:
                continue
        except OSError:
            continue
        try:
            path.unlink()
            removed.append(path.as_posix())
        except OSError:
            pass
    return removed


def _remove_empty_scene_dirs(scene_dir: Path) -> None:
    try:
        if scene_dir.is_dir() and not any(scene_dir.iterdir()):
            date_dir = scene_dir.parent
            scene_dir.rmdir()
            if date_dir.is_dir() and not any(date_dir.iterdir()):
                date_dir.rmdir()
    except OSError:
        pass


def cleanup_himawari_raw_dirs(
    output_root: str | Path = DATA_DIR,
    max_age_hours: int = PARTIAL_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> list[str]:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current.timestamp() - max(0, max_age_hours) * 3600
    removed: list[str] = []
    for raw_dir in sorted(Path(output_root).glob("*/*/raw")):
        if not raw_dir.is_dir():
            continue
        newest_mtime = max((path.stat().st_mtime for path in raw_dir.glob("*") if path.exists()), default=0)
        if max_age_hours > 0 and newest_mtime >= cutoff:
            continue
        try:
            shutil.rmtree(raw_dir)
            removed.append(raw_dir.as_posix())
            _remove_empty_scene_dirs(raw_dir.parent)
        except OSError:
            pass
    return removed


def is_himawari_scene_complete(scene_dir: str | Path) -> bool:
    try:
        return _read_reusable_scene_metadata(Path(scene_dir)) is not None
    except (OSError, json.JSONDecodeError):
        return False


def _scene_has_bands(scene_dir: str | Path, bands: list[str]) -> bool:
    try:
        meta = _read_reusable_scene_metadata(Path(scene_dir))
    except (OSError, json.JSONDecodeError):
        return False
    if not meta:
        return False
    loaded = {str(item).upper() for item in meta.get("loaded_bands", [])}
    loaded.update(str(item).upper() for item in meta.get("extra", {}).get("himawari", {}).get("loaded_bands", []))
    loaded.update(_product_name(item).upper() for item in meta.get("variables", []) if _product_name(item))
    return set(bands).issubset(loaded)


def _target_himawari_jobs(
    date: str,
    scene_time: str,
    root: Path,
    target_bands: list[str],
    phase: str = "download",
) -> list[tuple[str, str, str, list[str]]]:
    scene_dir = root / date / scene_time
    target_bands = _ordered_unique_bands(target_bands)
    missing_bands = [band for band in target_bands if not _scene_has_bands(scene_dir, [band])]
    if not missing_bands:
        return []
    return [(date, scene_time, phase, missing_bands)]


def _build_himawari_priority_jobs(
    scene_slots: list[tuple[str, str]],
    root: Path,
    target_bands: list[str],
) -> list[tuple[str, str, str, list[str]]]:
    requested = set(_ordered_unique_bands(target_bands))
    scene_order = list(reversed(scene_slots))
    jobs: list[tuple[str, str, str, list[str]]] = []

    if "B13" in requested:
        for date, scene_time in scene_order:
            jobs.extend(_target_himawari_jobs(date, scene_time, root, B13_FAST_BANDS, phase="quick_b13"))

    visible_targets = [band for band in VISIBLE_COLOR_BANDS if band in requested]
    if visible_targets:
        for date, scene_time in scene_order:
            visible_bands = [band for band in _visible_light_bands_for_slot(date, scene_time) if band in visible_targets]
            if visible_bands:
                jobs.extend(_target_himawari_jobs(date, scene_time, root, visible_bands, phase="visible_color"))

    fallback_targets = [band for band in target_bands if band not in set(B13_FAST_BANDS + VISIBLE_COLOR_BANDS)]
    if fallback_targets:
        for date, scene_time in scene_order:
            jobs.extend(_target_himawari_jobs(date, scene_time, root, fallback_targets, phase="download"))

    return jobs


def _raw_scene_has_complete_bands(raw_dir: str | Path, bands: list[str] | None = None) -> bool:
    raw_path = Path(raw_dir)
    requested = {item.upper() for item in bands} if bands else set(BAND_CATALOG)
    segments: dict[str, set[int]] = {}
    totals: dict[str, int] = {}
    for file_path in raw_path.glob("HS_H*.DAT*"):
        if file_path.name.endswith(".part"):
            continue
        info = parse_hsd_filename(file_path.name)
        if not info or info["band"] not in requested:
            continue
        if info["region"] != "FLDK":
            continue
        segments.setdefault(info["band"], set()).add(info["segment"])
        totals[info["band"]] = info["total_segments"]

    if not requested:
        return False
    for band in requested:
        total = totals.get(band)
        if not total:
            return False
        if segments.get(band) != set(range(1, total + 1)):
            return False
    return True


def recover_himawari_scene_window(
    output_root: str | Path = DATA_DIR,
    slots: list[tuple[str, str]] | None = None,
    hours: int = DEFAULT_WINDOW_HOURS,
    delay_minutes: int = DEFAULT_LATEST_DELAY_MINUTES,
    interval_minutes: int = 10,
    retention_hours: int = DEFAULT_WINDOW_HOURS,
    bands: list[str] | None = None,
    quick_bands: list[str] | None = None,
    max_scenes_per_run: int | None = None,
    max_jobs_per_run: int | None = 0,
    max_workers: int = 1,
    queue: str = "download",
    slow_min_remaining_minutes: int = SLOW_MIN_REMAINING_MINUTES,
    now: datetime | None = None,
    downloader: Any = download_himawari_hsd_scene,
    processor: Any = process_downloaded_himawari_scene,
    raise_errors: bool = False,
    progress_callback: Any = None,
    **download_kwargs: Any,
) -> dict[str, Any]:
    root = Path(output_root)
    scene_slots = slots or himawari_slot_window(now=now, hours=hours, delay_minutes=delay_minutes, interval_minutes=interval_minutes)
    requested_bands = [item.upper() for item in (bands or quick_bands or HIMAWARI_TARGET_BANDS)]
    target_band_list = [item for item in _ordered_unique_bands(requested_bands) if item in HIMAWARI_TARGET_BANDS] or list(HIMAWARI_TARGET_BANDS)
    queue_key = str(queue or "download").lower()
    if queue_key in {"all", "fast", "quick", "slow", "full"}:
        queue_key = "download"
    if queue_key != "download":
        raise ValueError("Himawari queue 必须是 download。")
    job_limit = max_jobs_per_run
    if job_limit is None and max_scenes_per_run is not None:
        job_limit = max_scenes_per_run
    if job_limit is None:
        job_limit = 0
    retention_now = now or datetime.now(timezone.utc)
    if retention_now.tzinfo is None:
        retention_now = retention_now.replace(tzinfo=timezone.utc)
    result: dict[str, Any] = {
        "queue": queue_key,
        "slots": [f"{date}_{scene_time}" for date, scene_time in scene_slots],
        "skipped": [],
        "target_complete": [],
        "processed_raw": [],
        "downloaded": [],
        "errors": [],
        "stopped": None,
        "phase": None,
        "removed_part_files": cleanup_partial_himawari_downloads(root),
        "removed_expired": cleanup_himawari_retention(root, retention_hours=retention_hours, now=retention_now, delay_minutes=delay_minutes),
    }

    handled = 0
    result_lock = Lock()

    def handle_scene(date: str, scene_time: str, phase: str, wanted_bands: list[str]) -> dict[str, Any]:
        nonlocal handled
        scene_id = f"{date}_{scene_time}"
        scene_dir = root / date / scene_time
        raw_dir = scene_dir / "raw"
        try:
            _emit_progress(progress_callback, stage="checking", phase=phase, scene_id=scene_id, detail="检查本地结果和 raw 完整性")
            if _raw_scene_has_complete_bands(raw_dir, bands=wanted_bands):
                processor(
                    raw_dir,
                    root,
                    delete_raw=True,
                    retention_hours=retention_hours,
                    now=retention_now,
                    latest_delay_minutes=delay_minutes,
                    bands=wanted_bands,
                    progress_callback=progress_callback,
                    phase=phase,
                )
                return {"kind": "processed_raw", "scene_id": scene_id}
            downloader_result = downloader(
                date,
                scene_time,
                output_root=root,
                bands=wanted_bands,
                overwrite=False,
                parse_after_download=True,
                delete_raw_after_parse=True,
                retention_hours=retention_hours,
                latest_delay_minutes=delay_minutes,
                phase=phase,
                progress_callback=progress_callback,
                **download_kwargs,
            )
            return {"kind": "downloaded", "scene_id": scene_id, "result": downloader_result}
        except Exception as exc:
            if raise_errors:
                raise
            _emit_progress(progress_callback, stage="failed", phase=phase, scene_id=scene_id, detail=str(exc))
            return {
                "kind": "error",
                "scene_id": scene_id,
                "error": str(exc),
                "stopped": "FTP 登录失败，已停止本轮 Himawari 自动恢复。" if _is_ftp_auth_error(exc) else None,
            }

    def record_job(job_result: dict[str, Any]) -> None:
        nonlocal handled
        with result_lock:
            kind = job_result.get("kind")
            scene_id = job_result.get("scene_id")
            if kind == "processed_raw":
                result["processed_raw"].append(scene_id)
                handled += 1
            elif kind == "downloaded":
                result["downloaded"].append(scene_id)
                handled += 1
            elif kind == "error":
                result["errors"].append({"scene_id": scene_id, "error": job_result.get("error")})
                if job_result.get("stopped") and not result["stopped"]:
                    result["stopped"] = job_result["stopped"]

    for date, scene_time in scene_slots:
        scene_id = f"{date}_{scene_time}"
        scene_dir = root / date / scene_time
        if _scene_has_bands(scene_dir, target_band_list):
            result["target_complete"].append(scene_id)

    priority_jobs = _build_himawari_priority_jobs(scene_slots, root, target_band_list)

    if job_limit:
        priority_jobs = priority_jobs[:job_limit]

    if priority_jobs:
        result["phase"] = priority_jobs[0][2]
    worker_count = max(1, min(int(max_workers or 1), len(priority_jobs) or 1))
    if worker_count == 1:
        for date, scene_time, phase, wanted_bands in priority_jobs:
            record_job(handle_scene(date, scene_time, phase, wanted_bands))
            if result["stopped"]:
                break
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(handle_scene, date, scene_time, phase, wanted_bands)
                for date, scene_time, phase, wanted_bands in priority_jobs
            ]
            for future in as_completed(futures):
                record_job(future.result())
                if result["stopped"]:
                    break

    if result["downloaded"] or result["processed_raw"] or result["skipped"]:
        result["removed_expired"].extend(cleanup_himawari_retention(root, retention_hours=retention_hours, now=retention_now, delay_minutes=delay_minutes))
    _emit_progress(progress_callback, stage="idle", phase=result["phase"], detail="本轮 Himawari 自动任务结束")
    return result


def _is_ftp_auth_error(exc: Exception) -> bool:
    if not isinstance(exc, ftplib.error_perm):
        return False
    message = str(exc).lower()
    return "530" in message or "login" in message or "auth" in message


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


def write_latlon_variable(output_dir: str | Path, band: str, data: np.ndarray, grid: dict[str, Any], save_intermediates: bool = False) -> dict[str, Any]:
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
    if save_intermediates:
        values.tofile(float32_path)
        _write_netcdf(values, nc_path, band, catalog, grid)
    else:
        for path in (float32_path, nc_path):
            if path.exists():
                path.unlink()
    finite = values[np.isfinite(values)]
    stats = {"min": float(np.nanmin(finite)) if finite.size else None, "max": float(np.nanmax(finite)) if finite.size else None, "mean": float(np.nanmean(finite)) if finite.size else None, "std": None}
    return {
        "name": catalog["key"],
        "long_name": catalog["plain_name"],
        "short_name": catalog["key"],
        "raw_name": catalog["key"],
        "name_cn": catalog["name_zh"],
        "unit": catalog["unit"],
        "display_unit": catalog["display_unit"],
        "shape": [grid["ny"], grid["nx"]],
        "dims": ["lat", "lon"],
        "level": None,
        "missing": None,
        "stats": stats,
        "category": catalog["category"],
        "description": catalog["description"],
        "wavelength": catalog["wavelength"],
        "png": png_path.as_posix(),
        "float32": float32_path.as_posix() if save_intermediates else None,
        "netcdf": nc_path.as_posix() if save_intermediates else None,
    }


def write_scene_metadata(
    scene_dir: str | Path,
    date: str,
    time: str,
    satellite: str,
    raw_dir: str | Path,
    raw_file_count: int,
    grid: dict[str, Any],
    variables: list[dict[str, Any]],
    composites: list[dict[str, Any]],
    retention_managed: bool = False,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir)
    meta_dir = scene_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "scene.meta.json"
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
        "loaded_bands": [_product_name(item) for item in variables if _product_name(item)],
        "source_raw_dir": Path(raw_dir).as_posix(),
        "raw_file_count": raw_file_count,
        "retention_managed": retention_managed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta = _merge_scene_metadata(scene_dir, meta, variables, composites, raw_file_count, retention_managed)
    with meta_path.open("w", encoding="utf-8") as file:
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


def _normalize_reflectance_channel(values: np.ndarray, high: float = 100.0, gamma: float = 0.8) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    scaled = np.clip(data / high, 0, 1)
    if gamma and gamma != 1:
        scaled = np.power(scaled, gamma)
    return np.where(np.isfinite(scaled), scaled, 0).astype(np.float32)


def _save_rgb_png(rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.all(np.isfinite(rgb), axis=-1), 255, 0).astype(np.uint8)
    Image.fromarray(rgba).save(output_path)


def _rgb_from_composite(key: str, arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    if key == "true_color" and all(item in arrays for item in ("B03", "B02", "B01")):
        return np.stack([
            _normalize_reflectance_channel(arrays["B03"]),
            _normalize_reflectance_channel(arrays["B02"]),
            _normalize_reflectance_channel(arrays["B01"]),
        ], axis=-1)
    if key == "natural_color" and all(item in arrays for item in ("B05", "B04", "B03")):
        return np.stack([
            _normalize_reflectance_channel(arrays["B05"]),
            _normalize_reflectance_channel(arrays["B04"]),
            _normalize_reflectance_channel(arrays["B03"]),
        ], axis=-1)
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
        output.append(
            {
                "name": catalog["key"],
                "name_cn": catalog["name_zh"],
                "description": catalog["description"],
                "source_bands": catalog["source_bands"],
                "float32": None,
                "netcdf": None,
                "png": png_path.as_posix(),
            }
        )
    return output


def process_scene(
    input_root: str | Path,
    output_root: str | Path = DATA_DIR,
    date: str | None = None,
    time: str | None = None,
    bands: list[str] | None = None,
    extent: list[float] | None = None,
    resolution: float = LATLON_RESOLUTION,
    composites: bool = True,
    progress_callback: Any = None,
    phase: str | None = None,
    retention_managed: bool = False,
    save_intermediates: bool | None = None,
) -> dict[str, Any]:
    raw_dir = _find_raw_dir(input_root, date, time)
    files = sorted(raw_dir.glob("HS_H*.DAT*"))
    if not files:
        raise FileNotFoundError(f"未找到 Himawari HSD 文件: {raw_dir}")
    scene_info = _scene_info_from_files(files)
    date = date or scene_info["date"]
    time = time or scene_info["time"]
    scene_id = f"{date}_{time}"
    grid = build_latlon_grid(extent=extent, resolution=resolution)
    scene_dir = Path(output_root) / date / time
    if save_intermediates is None:
        save_intermediates = os.environ.get("HIMAWARI_OUTPUT_MODE", "display").strip().lower() == "debug"
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
    for index, band in enumerate(load_bands, start=1):
        _emit_progress(
            progress_callback,
            stage="processing_band",
            phase=phase,
            scene_id=scene_id,
            band=band,
            file=f"{band}.png",
            detail=f"解析并重采样 {band}",
            queue_done=index - 1,
            queue_total=len(load_bands),
        )
        scene = Scene(reader="ahi_hsd", filenames=filenames)
        scene.load([band])
        values = _resample_dataset_to_latlon(scene[band], grid)
        variables.append(write_latlon_variable(scene_dir, band, values, grid, save_intermediates=save_intermediates))
        if composites:
            resampled_arrays[band] = values
        del scene, values
    _emit_progress(progress_callback, stage="compositing", phase=phase, scene_id=scene_id, detail="生成 RGB 合成产品")
    composite_meta = write_composites(scene_dir, resampled_arrays) if composites else []
    _emit_progress(progress_callback, stage="writing_meta", phase=phase, scene_id=scene_id, detail="写入 scene.meta.json")
    return write_scene_metadata(scene_dir, date, time, scene_info["satellite"], raw_dir, len(files), grid, variables, composite_meta, retention_managed=retention_managed)


def process_file(file_path: str, data_type: str = "Himawari") -> dict:
    try:
        return process_scene(Path(file_path))
    except Exception:
        weather_info = {"source": "Himawari", "product": "葵花卫星产品", "element": "卫星通道", "time": "解析失败", "level": "卫星观测", "range": "待解析", "resolution": "待解析", "grid": "待解析", "validGrid": "待解析", "coverage": "待解析", "missing": "待解析", "unit": "待解析", "variables": "待解析", "steps": "待解析", "status": "已接收但未形成完整 HSD 场景", "quality": "待解析", "max": "待解析", "min": "待解析", "mean": "待解析", "alert": "请上传完整 HSD raw 场景目录或分段集合。", "update": "待解析", "bars": [0, 0, 0, 0, 0], "trend": [0, 0, 0, 0, 0, 0, 0, 0]}
        return process_basic_file(file_path, data_type=data_type, file_format="HSD", weather_info=weather_info)


def main() -> int:
    parser = argparse.ArgumentParser(description="解析 Himawari HSD 为等经纬网格产品")
    parser.add_argument("--input-root", help="本地 HSD raw 根目录；未使用 --download 时必填")
    parser.add_argument("--output-root", default=DATA_DIR.as_posix())
    parser.add_argument("--date")
    parser.add_argument("--time")
    parser.add_argument("--bands", help="逗号分隔，例如 B01,B02,B13；默认解析所有可用通道")
    parser.add_argument("--extent", default=",".join(str(item) for item in CHINA_EXTENT))
    parser.add_argument("--resolution", type=float, default=LATLON_RESOLUTION)
    parser.add_argument("--no-composites", action="store_true")
    parser.add_argument("--download", action="store_true", help="从 Himawari FTP 下载 HSD raw 文件")
    parser.add_argument("--latest", action="store_true", help="下载当前时间向前延迟后的最新 10 分钟时次")
    parser.add_argument("--latest-delay-minutes", type=int, default=DEFAULT_LATEST_DELAY_MINUTES)
    parser.add_argument("--interval-minutes", type=int, help="定期下载间隔；设置后持续循环，配合 --iterations 可限制次数")
    parser.add_argument("--iterations", type=int, help="定期下载循环次数；不设置则持续运行")
    parser.add_argument("--overwrite", action="store_true", help="重新下载并覆盖已有 HSD 文件")
    parser.add_argument("--parse-after-download", action="store_true", help="下载完成后立即调用现有解析流程")
    parser.add_argument("--keep-raw", action="store_true", help="下载后解析成功也保留 HSD raw 文件；默认解析成功后删除 raw")
    parser.add_argument("--retention-hours", type=int, default=DEFAULT_WINDOW_HOURS, help="解析结果窗口小时数，默认 24")
    parser.add_argument("--ftp-host", default=None, help="默认读取 HIMAWARI_FTP_HOST 或 ftp.ptree.jaxa.jp")
    parser.add_argument("--ftp-root", default=None, help="默认读取 HIMAWARI_FTP_ROOT 或 /jma/hsd，支持 {yyyymm}/{dd}/{time} 模板")
    args = parser.parse_args()
    bands = [item.strip().upper() for item in args.bands.split(",") if item.strip()] if args.bands else None

    if args.download:
        if not args.latest and (not args.date or not args.time):
            parser.error("--download 需要同时提供 --date 和 --time，或使用 --latest")
        download_kwargs = {
            "output_root": args.output_root,
            "bands": bands,
            "overwrite": args.overwrite,
            "parse_after_download": args.parse_after_download,
            "delete_raw_after_parse": not args.keep_raw,
            "retention_hours": args.retention_hours,
            "latest_delay_minutes": args.latest_delay_minutes,
            "host": args.ftp_host,
            "remote_root": args.ftp_root,
        }
        try:
            if args.interval_minutes:
                results = run_himawari_download_loop(
                    args.date,
                    args.time,
                    args.interval_minutes,
                    iterations=args.iterations,
                    latest=args.latest,
                    latest_delay_minutes=args.latest_delay_minutes,
                    **download_kwargs,
                )
                print(json.dumps({"runs": results}, ensure_ascii=False, indent=2))
                return 0
            date, scene_time = latest_himawari_slot(delay_minutes=args.latest_delay_minutes) if args.latest else (args.date, args.time)
            result = download_himawari_hsd_scene(date, scene_time, **download_kwargs)
        except ValueError as exc:
            parser.exit(2, f"{exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if not args.input_root:
        parser.error("--input-root is required unless --download is used")

    extent = [float(item.strip()) for item in args.extent.split(",")]
    meta = process_scene(args.input_root, args.output_root, args.date, args.time, bands, extent, args.resolution, not args.no_composites)
    print(json.dumps({"scene_id": meta["scene_id"], "variables": meta["loaded_bands"], "composites": [_product_name(item) for item in meta["composites"]]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
