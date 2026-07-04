# -*- coding: utf-8 -*-
"""
GFS / ECMWF 自动下载脚本

功能：
1. GFS：自动查找 NOMADS 最新可用 GFS cycle；
2. ECMWF：自动查找 ECMWF Open Data 最新可用 IFS cycle；
3. 下载 2米气温、2米露点、地面气压、累积降水四个变量；
4. 多 forecast hour 顺序合并成一个 GRIB2 文件；
5. GFS 保存到 backend_system/data/GFS/wait_process/；
6. ECMWF 保存到 backend_system/data/ECMWF/wait_process/；
7. 写入 download_manifest.json 和 latest_download.json；
8. 可选 --parse-after：下载后统一调用 adapters/gfs_adapter.py 生成 meta.json + PNG + float32。

说明：
- GFS 来源：NOMADS filter_gfs_0p25.pl；
- ECMWF 来源：ECMWF Open Data data.ecmwf.int/forecasts；
- ECMWF Open Data 通常为 3 小时间隔，不是 1 小时间隔；
  如果传 --lead-step 1，本脚本会自动转成 3 小时间隔；
- ECMWF Open Data 不需要 cdsapi 账号。
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NOMADS_GFS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
ECMWF_OPEN_DATA_ROOT = "https://data.ecmwf.int/forecasts"

# ECMWF Open Data index 中的 surface 参数。
# 2t: 2m temperature
# 2d: 2m dewpoint
# sp: surface pressure
# tp: total precipitation
ECMWF_SURFACE_PARAMS = {"2t", "2d", "sp", "tp"}

_SSL_CONTEXT = None


def set_ssl_context(insecure_ssl: bool = False) -> None:
    """
    Windows / 校园网 / 公司代理环境下，访问 ECMWF Open Data 可能出现
    CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain。

    默认仍使用系统证书校验。
    只有显式传入 --insecure-ssl 时，才跳过证书校验。
    """
    global _SSL_CONTEXT

    if insecure_ssl:
        _SSL_CONTEXT = ssl._create_unverified_context()
        print("[WARN] SSL certificate verification is disabled by --insecure-ssl.")
    else:
        _SSL_CONTEXT = None


def safe_urlopen(req, timeout: int):
    if _SSL_CONTEXT is not None:
        return urlopen(req, timeout=timeout, context=_SSL_CONTEXT)
    return urlopen(req, timeout=timeout)



def backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def cycle_candidates(max_back_hours: int = 72) -> list[datetime]:
    """
    GFS / ECMWF 通常都有 00/06/12/18 UTC 起报。
    从新到旧尝试，最新 cycle 若未发布会自动回退。
    """
    now = utc_now()
    base_date = now.date()
    cycles: list[datetime] = []

    for day_back in range(0, 5):
        d = base_date - timedelta(days=day_back)
        for cyc in [18, 12, 6, 0]:
            dt = datetime(d.year, d.month, d.day, cyc, tzinfo=timezone.utc)
            age = now - dt
            if timedelta(hours=0) <= age <= timedelta(hours=max_back_hours):
                cycles.append(dt)

    return sorted(set(cycles), reverse=True)


def is_grib_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 16:
        return False

    with path.open("rb") as f:
        return f.read(4) == b"GRIB"


def download_url(url: str, out_file: Path, retries: int = 3, timeout: int = 90) -> bool:
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            req = Request(
                url,
                headers={"User-Agent": "NUIST-SmartWeather-NWP-Downloader/1.0"},
            )

            with safe_urlopen(req, timeout=timeout) as resp:
                first = resp.read(4)

                if first != b"GRIB":
                    rest = resp.read(300)
                    msg = (first + rest).decode("utf-8", errors="ignore")
                    print(f"[WARN] Not GRIB response, attempt={attempt}: {msg[:160]}")
                    time.sleep(5)
                    continue

                with tmp_file.open("wb") as f:
                    f.write(first)
                    shutil.copyfileobj(resp, f)

            if is_grib_file(tmp_file):
                tmp_file.replace(out_file)
                return True

            print(f"[WARN] Downloaded file is not valid GRIB: {tmp_file}")

        except Exception as e:
            print(f"[WARN] download failed attempt={attempt}: {e}")
            time.sleep(5 * attempt)

    if tmp_file.exists():
        tmp_file.unlink(missing_ok=True)

    return False


def http_get_text(url: str, timeout: int = 90, retries: int = 3) -> str | None:
    for attempt in range(1, retries + 1):
        try:
            req = Request(
                url,
                headers={"User-Agent": "NUIST-SmartWeather-ECMWF-Downloader/1.0"},
            )
            with safe_urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            return data.decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] get text failed attempt={attempt}: {e}")
            time.sleep(4 * attempt)

    return None


def download_byte_range(url: str, start: int, end: int, timeout: int = 90, retries: int = 3) -> bytes | None:
    """
    从 ECMWF Open Data 大 GRIB2 文件中按 index 的 offset/length 下载指定字段。
    """
    headers = {
        "User-Agent": "NUIST-SmartWeather-ECMWF-Downloader/1.0",
        "Range": f"bytes={start}-{end}",
    }

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with safe_urlopen(req, timeout=timeout) as resp:
                data = resp.read()

            if data[:4] == b"GRIB":
                return data

            print(f"[WARN] byte-range response is not GRIB, attempt={attempt}, size={len(data)}")
            time.sleep(4 * attempt)

        except Exception as e:
            print(f"[WARN] byte-range download failed attempt={attempt}: {e}")
            time.sleep(4 * attempt)

    return None


def combine_grib_files(part_files: list[Path], combined_file: Path) -> None:
    """
    GRIB2 message 可以顺序拼接。
    合并后 adapter 可以一次解析多时次。
    """
    tmp_file = combined_file.with_suffix(combined_file.suffix + ".tmp")

    with tmp_file.open("wb") as fout:
        for p in part_files:
            with p.open("rb") as fin:
                shutil.copyfileobj(fin, fout)

    tmp_file.replace(combined_file)


def save_manifest(manifest_file: Path, record: dict) -> None:
    if manifest_file.exists():
        try:
            old = json.loads(manifest_file.read_text(encoding="utf-8"))
            if not isinstance(old, list):
                old = []
        except Exception:
            old = []
    else:
        old = []

    old.append(record)
    manifest_file.write_text(json.dumps(old[-100:], ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# GFS
# =============================================================================

def build_gfs_url(cycle_dt: datetime, forecast_hour: int) -> str:
    """
    构造 NOMADS GFS 0.25° filter URL。
    这里不传 subregion，默认下载全球数据。

    当前下载变量：
    - TMP: 2m temperature；
    - DPT: 2m dewpoint；
    - PRES: surface pressure；
    - APCP: accumulated precipitation。
    """
    ymd = cycle_dt.strftime("%Y%m%d")
    cyc = cycle_dt.strftime("%H")
    fh = f"{forecast_hour:03d}"

    params = {
        "dir": f"/gfs.{ymd}/{cyc}/atmos",
        "file": f"gfs.t{cyc}z.pgrb2.0p25.f{fh}",

        "lev_2_m_above_ground": "on",
        "lev_surface": "on",

        "var_TMP": "on",
        "var_DPT": "on",
        "var_PRES": "on",
        "var_APCP": "on",
    }

    return NOMADS_GFS_FILTER + "?" + urlencode(params)


def find_latest_available_gfs_cycle(args) -> datetime | None:
    probe_dir = Path(args.output_dir) / "_probe"
    ensure_dir(probe_dir)

    for cycle_dt in cycle_candidates(args.max_back_hours):
        ymd = cycle_dt.strftime("%Y%m%d")
        cyc = cycle_dt.strftime("%H")
        url = build_gfs_url(cycle_dt=cycle_dt, forecast_hour=0)
        probe_file = probe_dir / f"probe_gfs_{ymd}_{cyc}_f000.grib2"

        print(f"[CHECK][GFS] Try cycle {ymd} {cyc}Z")
        ok = download_url(url, probe_file, retries=1, timeout=args.timeout)
        if ok:
            print(f"[OK][GFS] Latest available cycle: {ymd} {cyc}Z")
            return cycle_dt

    return None


def run_gfs(args) -> int:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    latest_cycle = find_latest_available_gfs_cycle(args)
    if latest_cycle is None:
        print("[ERROR] No available GFS cycle found.")
        return 2

    ymd = latest_cycle.strftime("%Y%m%d")
    cyc = latest_cycle.strftime("%H")
    lead_hours = list(range(args.lead_start, args.lead_end + 1, args.lead_step))

    part_dir = output_dir / f"parts_gfs_{ymd}_{cyc}z"
    ensure_dir(part_dir)

    combined_name = f"gfs_realtime_{ymd}_{cyc}z_f{lead_hours[0]:03d}_f{lead_hours[-1]:03d}.grib2"
    combined_file = output_dir / combined_name

    if combined_file.exists() and combined_file.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] Combined file already exists: {combined_file}")
        if args.parse_after:
            parse_after_download(combined_file, "GFS")
        return 0

    downloaded: list[Path] = []
    failed: list[int] = []

    for fh in lead_hours:
        part_file = part_dir / f"gfs_{ymd}_{cyc}z_f{fh:03d}.grib2"
        url = build_gfs_url(cycle_dt=latest_cycle, forecast_hour=fh)

        print(f"[DOWNLOAD][GFS] f{fh:03d}")
        ok = download_url(url, part_file, retries=args.retries, timeout=args.timeout)

        if ok:
            downloaded.append(part_file)
            print(f"[OK][GFS] {part_file.name}  {part_file.stat().st_size / 1024:.1f} KB")
        else:
            failed.append(fh)
            print(f"[FAIL][GFS] f{fh:03d}")

    return finish_download(
        source="GFS",
        output_dir=output_dir,
        combined_file=combined_file,
        downloaded=downloaded,
        failed=failed,
        lead_hours=lead_hours,
        cycle_dt=latest_cycle,
        args=args,
        extent=[0.0, -90.0, 359.75, 90.0],
        source_name="GFS_NOMADS_GLOBAL",
    )


# =============================================================================
# ECMWF Open Data
# =============================================================================

def normalize_ecmwf_leads(args) -> list[int]:
    """
    ECMWF IFS Open Data 常规是 3 小时间隔。
    这里先按业务展示需求取 0~lead_end 的 3 小时间隔。
    """
    start = max(0, int(args.lead_start))
    end = max(start, int(args.lead_end))

    if args.lead_step < 3:
        print("[WARN][ECMWF] ECMWF Open Data 常用 3 小时间隔；已将 lead-step 自动调整为 3。")
        step = 3
    else:
        step = int(args.lead_step)

    if step % 3 != 0:
        print(f"[WARN][ECMWF] lead-step={step} 不是 3 的倍数；已调整为 3。")
        step = 3

    leads = [h for h in range(start, end + 1, step) if h % 3 == 0]

    if 0 not in leads and start == 0:
        leads.insert(0, 0)

    return leads


def ecmwf_max_step_for_cycle(cycle_dt: datetime) -> int:
    cyc = int(cycle_dt.strftime("%H"))
    return 240 if cyc in {0, 12} else 90


def build_ecmwf_file_url(cycle_dt: datetime, forecast_hour: int) -> str:
    """
    ECMWF Open Data IFS 0.25° oper/fc 文件 URL。
    """
    ymd = cycle_dt.strftime("%Y%m%d")
    cyc = cycle_dt.strftime("%H")
    base = f"{ymd}{cyc}0000"
    return f"{ECMWF_OPEN_DATA_ROOT}/{ymd}/{cyc}z/ifs/0p25/oper/{base}-{forecast_hour}h-oper-fc.grib2"


def build_ecmwf_index_url(cycle_dt: datetime, forecast_hour: int) -> str:
    return build_ecmwf_file_url(cycle_dt, forecast_hour).replace(".grib2", ".index")


def parse_ecmwf_index(index_text: str) -> list[dict]:
    rows: list[dict] = []

    for line in index_text.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        except Exception:
            continue

    return rows


def select_ecmwf_surface_records(rows: list[dict]) -> list[dict]:
    selected: list[dict] = []

    for item in rows:
        param = str(item.get("param", "")).lower()
        levtype = str(item.get("levtype", "")).lower()
        typ = str(item.get("type", "")).lower()
        stream = str(item.get("stream", "")).lower()

        if param not in ECMWF_SURFACE_PARAMS:
            continue

        if levtype and levtype != "sfc":
            continue

        if typ and typ != "fc":
            continue

        if stream and stream != "oper":
            continue

        if "_offset" not in item or "_length" not in item:
            continue

        selected.append(item)

    order = {"2t": 0, "2d": 1, "sp": 2, "tp": 3}
    selected.sort(key=lambda x: order.get(str(x.get("param", "")).lower(), 99))
    return selected


def download_ecmwf_step_subset(
    cycle_dt: datetime,
    forecast_hour: int,
    out_file: Path,
    retries: int = 3,
    timeout: int = 90,
) -> bool:
    """
    下载 ECMWF 单个 forecast step 的四个 surface 字段：
    1. 下载 .index；
    2. 找 param=2t/2d/sp/tp 的 offset/length；
    3. 用 HTTP Range 分别下载 GRIB message；
    4. 顺序拼成一个小 GRIB2。
    """
    grib_url = build_ecmwf_file_url(cycle_dt, forecast_hour)
    index_url = build_ecmwf_index_url(cycle_dt, forecast_hour)

    index_text = http_get_text(index_url, timeout=timeout, retries=retries)
    if not index_text:
        print(f"[WARN][ECMWF] index not available: {index_url}")
        return False

    rows = parse_ecmwf_index(index_text)
    records = select_ecmwf_surface_records(rows)

    if not records:
        print(f"[WARN][ECMWF] no target params in index: f{forecast_hour:03d}")
        return False

    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")

    try:
        with tmp_file.open("wb") as fout:
            for rec in records:
                offset = int(rec["_offset"])
                length = int(rec["_length"])
                start = offset
                end = offset + length - 1
                param = rec.get("param")

                print(f"[ECMWF] f{forecast_hour:03d} param={param} bytes={start}-{end}")
                data = download_byte_range(
                    grib_url,
                    start=start,
                    end=end,
                    timeout=timeout,
                    retries=retries,
                )

                if not data or data[:4] != b"GRIB":
                    raise RuntimeError(f"download field failed: f{forecast_hour:03d} param={param}")

                fout.write(data)

        if is_grib_file(tmp_file):
            tmp_file.replace(out_file)
            return True

        print(f"[WARN][ECMWF] invalid GRIB after byte-range: {tmp_file}")

    except Exception as e:
        print(f"[WARN][ECMWF] step subset failed f{forecast_hour:03d}: {e}")

    tmp_file.unlink(missing_ok=True)
    return False


def find_latest_available_ecmwf_cycle(args) -> datetime | None:
    """
    用 step=0 index 探测最新 ECMWF Open Data cycle。
    """
    for cycle_dt in cycle_candidates(args.max_back_hours):
        ymd = cycle_dt.strftime("%Y%m%d")
        cyc = cycle_dt.strftime("%H")
        index_url = build_ecmwf_index_url(cycle_dt, 0)

        print(f"[CHECK][ECMWF] Try cycle {ymd} {cyc}Z")
        index_text = http_get_text(index_url, timeout=args.timeout, retries=1)

        if not index_text:
            continue

        rows = parse_ecmwf_index(index_text)
        records = select_ecmwf_surface_records(rows)

        # step=0 可能没有 tp，但通常应该有 2t/2d/sp。命中 >=2 个就认为 cycle 可用。
        if len(records) >= 2:
            print(f"[OK][ECMWF] Latest available cycle: {ymd} {cyc}Z")
            return cycle_dt

        print(f"[WARN][ECMWF] index exists but target fields insufficient: {len(records)}")

    return None


def run_ecmwf(args) -> int:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    latest_cycle = find_latest_available_ecmwf_cycle(args)
    if latest_cycle is None:
        print("[ERROR] No available ECMWF cycle found.")
        return 2

    max_step = ecmwf_max_step_for_cycle(latest_cycle)
    requested_leads = normalize_ecmwf_leads(args)
    lead_hours = [h for h in requested_leads if h <= max_step]

    if not lead_hours:
        print(f"[ERROR][ECMWF] No valid lead hours. max_step={max_step}")
        return 2

    if requested_leads[-1] > max_step:
        print(f"[WARN][ECMWF] Current cycle max_step={max_step}h, truncated leads to f{lead_hours[-1]:03d}.")

    ymd = latest_cycle.strftime("%Y%m%d")
    cyc = latest_cycle.strftime("%H")

    part_dir = output_dir / f"parts_ecmwf_{ymd}_{cyc}z"
    ensure_dir(part_dir)

    combined_name = f"ecmwf_realtime_{ymd}_{cyc}z_f{lead_hours[0]:03d}_f{lead_hours[-1]:03d}.grib2"
    combined_file = output_dir / combined_name

    if combined_file.exists() and combined_file.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] Combined file already exists: {combined_file}")
        if args.parse_after:
            parse_after_download(combined_file, "ECMWF")
        return 0

    downloaded: list[Path] = []
    failed: list[int] = []

    for fh in lead_hours:
        part_file = part_dir / f"ecmwf_{ymd}_{cyc}z_f{fh:03d}.grib2"

        print(f"[DOWNLOAD][ECMWF] f{fh:03d}")
        ok = download_ecmwf_step_subset(
            latest_cycle,
            forecast_hour=fh,
            out_file=part_file,
            retries=args.retries,
            timeout=args.timeout,
        )

        if ok:
            downloaded.append(part_file)
            print(f"[OK][ECMWF] {part_file.name}  {part_file.stat().st_size / 1024:.1f} KB")
        else:
            failed.append(fh)
            print(f"[FAIL][ECMWF] f{fh:03d}")

    return finish_download(
        source="ECMWF",
        output_dir=output_dir,
        combined_file=combined_file,
        downloaded=downloaded,
        failed=failed,
        lead_hours=lead_hours,
        cycle_dt=latest_cycle,
        args=args,
        extent=[0.0, -90.0, 359.75, 90.0],
        source_name="ECMWF_OPEN_DATA_IFS_GLOBAL",
    )


# =============================================================================
# Parse / finish / clean
# =============================================================================

def parse_after_download(grib_file: Path, source: str) -> None:
    """
    统一使用同一个 adapter：
    adapters.gfs_adapter.process_file(file, data_type="GFS" 或 "ECMWF")
    """
    root = backend_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from adapters.gfs_adapter import process_file

    source = source.upper()

    print(f"[PARSE][{source}] Start parse: {grib_file}")
    result = process_file(str(grib_file), source)
    print(f"[PARSE][{source}] Done.")
    print(json.dumps({
        "status": result.get("status"),
        "file": result.get("file_name"),
        "source": result.get("source"),
        "data_type": result.get("data_type"),
        "default_variable": result.get("default_variable"),
        "variables": [v.get("key") for v in result.get("variable_options", [])],
        "n_layers": len(result.get("variable_layers") or {}),
        "n_png_default": len(result.get("png_urls") or []),
        "n_grid_default": len(result.get("grid_urls") or result.get("binary_urls") or []),
        "extent": result.get("extent"),
    }, ensure_ascii=False, indent=2))


def finish_download(
    source: str,
    output_dir: Path,
    combined_file: Path,
    downloaded: list[Path],
    failed: list[int],
    lead_hours: list[int],
    cycle_dt: datetime,
    args,
    extent: list[float],
    source_name: str,
) -> int:
    min_required = min(args.min_success, len(lead_hours))

    if len(downloaded) < min_required:
        print(f"[ERROR][{source}] Too few files downloaded: {len(downloaded)} < {min_required}")
        return 3

    print(f"[COMBINE][{source}] {len(downloaded)} files -> {combined_file.name}")
    combine_grib_files(downloaded, combined_file)

    ymd = cycle_dt.strftime("%Y%m%d")
    cyc = cycle_dt.strftime("%H")

    record = {
        "time_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cycle_utc": f"{ymd} {cyc}Z",
        "source": source_name,
        "data_type": source,
        "combined_file": str(combined_file).replace("\\", "/"),
        "forecast_hours": lead_hours,
        "downloaded_count": len(downloaded),
        "failed_hours": failed,
        "extent": extent,
        "status": "success" if not failed else "partial_success",
    }

    manifest_file = output_dir.parent / "download_manifest.json"
    save_manifest(manifest_file, record)

    latest_file = output_dir.parent / "latest_download.json"
    latest_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE][{source}] Download finished.")
    print(json.dumps(record, ensure_ascii=False, indent=2))

    if args.parse_after:
        parse_after_download(combined_file, source)

    cleanup_old_cycles(output_dir, source=source, keep_cycles=args.keep_cycles)

    return 0


def cleanup_old_cycles(output_dir: Path, source: str, keep_cycles: int = 3) -> None:
    if keep_cycles <= 0 or not output_dir.exists():
        return

    source_l = source.lower()

    if source_l == "ecmwf":
        main_glob = "ecmwf_realtime_*_f*.grib2"
        parts_prefix_from = "ecmwf_realtime_"
        parts_prefix_to = "parts_ecmwf_"
    else:
        main_glob = "gfs_realtime_*_f*.grib2"
        parts_prefix_from = "gfs_realtime_"
        parts_prefix_to = "parts_gfs_"

    prefixes = set()

    for p in output_dir.glob(main_glob):
        parts = p.name.split("_")
        if len(parts) >= 4:
            prefixes.add("_".join(parts[:4]))

    ordered = sorted(prefixes, reverse=True)
    keep = set(ordered[:keep_cycles])
    remove = [x for x in ordered if x not in keep]

    if not remove:
        print(f"[CLEAN][{source}] Nothing to clean. keep_cycles={keep_cycles}")
        return

    for prefix in remove:
        for p in output_dir.glob(prefix + "*"):
            try:
                if p.is_file():
                    p.unlink(missing_ok=True)
                    print(f"[CLEAN][{source}] remove file {p.name}")
            except Exception as e:
                print(f"[WARN][{source}] clean file failed {p}: {e}")

        parts_prefix = prefix.replace(parts_prefix_from, parts_prefix_to)
        for d in output_dir.glob(parts_prefix + "*"):
            try:
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"[CLEAN][{source}] remove dir {d.name}")
            except Exception as e:
                print(f"[WARN][{source}] clean dir failed {d}: {e}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto download latest global GFS or ECMWF forecast data.")

    parser.add_argument("--source", default="GFS", choices=["GFS", "ECMWF"])
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--lead-start", type=int, default=0)
    parser.add_argument("--lead-end", type=int, default=47)
    parser.add_argument("--lead-step", type=int, default=1)

    parser.add_argument("--max-back-hours", type=int, default=72)
    parser.add_argument("--min-success", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)

    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="跳过 HTTPS 证书校验。仅在 Windows/代理环境出现 CERTIFICATE_VERIFY_FAILED 时使用。",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parse-after", action="store_true")
    parser.add_argument("--keep-cycles", type=int, default=3, help="服务器上保留最近多少个 cycle，默认 3；设为 0 表示不清理。")

    return parser


def resolve_output_dir(args) -> None:
    if args.output_dir:
        return

    root = backend_root()

    if args.source.upper() == "ECMWF":
        args.output_dir = str(root / "data" / "ECMWF" / "wait_process")
    else:
        args.output_dir = str(root / "data" / "GFS" / "wait_process")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.source = args.source.upper()

    resolve_output_dir(args)
    set_ssl_context(args.insecure_ssl)

    print(json.dumps({
        "source": args.source,
        "output_dir": args.output_dir,
        "lead_start": args.lead_start,
        "lead_end": args.lead_end,
        "lead_step": args.lead_step,
        "parse_after": args.parse_after,
    }, ensure_ascii=False, indent=2))

    if args.source == "ECMWF":
        return run_ecmwf(args)

    return run_gfs(args)


if __name__ == "__main__":
    raise SystemExit(main())
