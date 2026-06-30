# -*- coding: utf-8 -*-
"""
GFS/ECMWF 自动下载脚本（全球 GFS 版）
作者：刘家鹤

功能：
1. 自动查找 NOMADS 上最新可用 GFS cycle；
2. 下载指定预报时效的全球 GFS 0.25° GRIB2 数据；
3. 将多个 forecast hour 顺序合并成一个 GRIB2 文件；
4. 保存到 backend_system/data/GFS/wait_process/；
5. 写入 download_manifest.json 和 latest_download.json；
6. 可选 --parse-after：下载后调用 adapters/gfs_adapter.py 生成 meta.json + PNG。

说明：
- 不加 --parse-after：只下载 GRIB2，适合 T1-GFS-DL 自动下载任务。
- 加 --parse-after：下载后生成 PNG，适合前端展示和变量选择测试。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NOMADS_GFS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"


def backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def cycle_candidates(max_back_hours: int = 48) -> list[datetime]:
    """
    GFS 通常 00/06/12/18 UTC 起报。
    从新到旧尝试，最新 cycle 若未发布会自动回退。
    """
    now = utc_now()
    base_date = now.date()
    cycles: list[datetime] = []

    for day_back in range(0, 4):
        d = base_date - timedelta(days=day_back)
        for cyc in [18, 12, 6, 0]:
            dt = datetime(d.year, d.month, d.day, cyc, tzinfo=timezone.utc)
            age = now - dt
            if timedelta(hours=0) <= age <= timedelta(hours=max_back_hours):
                cycles.append(dt)

    return sorted(set(cycles), reverse=True)


def build_gfs_url(cycle_dt: datetime, forecast_hour: int) -> str:
    """
    构造 NOMADS GFS 0.25° filter URL。
    这里不传 subregion，默认下载全球数据。

    当前下载变量：
    - TMP: 2m temperature / pressure-level temperature 等；
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
                headers={"User-Agent": "NUIST-SmartWeather-GFS-Downloader/1.0"},
            )

            with urlopen(req, timeout=timeout) as resp:
                first = resp.read(4)

                if first != b"GRIB":
                    rest = resp.read(300)
                    msg = (first + rest).decode("utf-8", errors="ignore")
                    print(f"[WARN] Not GRIB response, attempt={attempt}: {msg[:120]}")
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


def find_latest_available_cycle(args) -> datetime | None:
    probe_dir = Path(args.output_dir) / "_probe"
    ensure_dir(probe_dir)

    for cycle_dt in cycle_candidates(args.max_back_hours):
        ymd = cycle_dt.strftime("%Y%m%d")
        cyc = cycle_dt.strftime("%H")
        url = build_gfs_url(cycle_dt=cycle_dt, forecast_hour=0)
        probe_file = probe_dir / f"probe_gfs_{ymd}_{cyc}_f000.grib2"

        print(f"[CHECK] Try cycle {ymd} {cyc}Z")
        ok = download_url(url, probe_file, retries=1, timeout=args.timeout)
        if ok:
            print(f"[OK] Latest available cycle: {ymd} {cyc}Z")
            return cycle_dt

    return None


def combine_grib_files(part_files: list[Path], combined_file: Path) -> None:
    """
    GRIB2 message 可以顺序拼接。
    合并后 gfs_adapter.py 可以一次解析多时次。
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



def cleanup_old_cycles(output_dir: Path, keep_cycles: int = 3) -> None:
    """
    清理旧 GFS cycle，避免服务器磁盘持续膨胀。
    保留最近 keep_cycles 个 gfs_realtime_YYYYMMDD_HHz 前缀对应的 grib2/png/float32/meta/idx 和 parts 目录。
    """
    if keep_cycles <= 0 or not output_dir.exists():
        return

    prefixes = set()
    for p in output_dir.glob("gfs_realtime_*_f*.grib2"):
        # gfs_realtime_20260626_06z_f000_f047.grib2 -> gfs_realtime_20260626_06z
        parts = p.name.split("_")
        if len(parts) >= 4:
            prefixes.add("_".join(parts[:4]))

    ordered = sorted(prefixes, reverse=True)
    keep = set(ordered[:keep_cycles])
    remove = [x for x in ordered if x not in keep]

    if not remove:
        print(f"[CLEAN] Nothing to clean. keep_cycles={keep_cycles}")
        return

    for prefix in remove:
        for p in output_dir.glob(prefix + "*"):
            try:
                if p.is_file():
                    p.unlink(missing_ok=True)
                    print(f"[CLEAN] remove file {p.name}")
            except Exception as e:
                print(f"[WARN] clean file failed {p}: {e}")

        # 对应 parts_gfs_YYYYMMDD_HHz
        parts_prefix = prefix.replace("gfs_realtime_", "parts_gfs_")
        for d in output_dir.glob(parts_prefix + "*"):
            try:
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"[CLEAN] remove dir {d.name}")
            except Exception as e:
                print(f"[WARN] clean dir failed {d}: {e}")


def parse_after_download(grib_file: Path) -> None:
    root = backend_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from adapters.gfs_adapter import process_file

    print(f"[PARSE] Start parse: {grib_file}")
    result = process_file(str(grib_file), "GFS")
    print("[PARSE] Done.")
    print(json.dumps({
        "status": result.get("status"),
        "file": result.get("file_name"),
        "default_variable": result.get("default_variable"),
        "variables": [v.get("key") for v in result.get("variable_options", [])],
        "n_layers": len(result.get("variable_layers") or {}),
        "n_png_default": len(result.get("png_urls") or []),
        "extent": result.get("extent"),
    }, ensure_ascii=False, indent=2))


def run(args) -> int:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    latest_cycle = find_latest_available_cycle(args)
    if latest_cycle is None:
        print("[ERROR] No available GFS cycle found.")
        return 2

    ymd = latest_cycle.strftime("%Y%m%d")
    cyc = latest_cycle.strftime("%H")
    lead_hours = list(range(args.lead_start, args.lead_end + 1, args.lead_step))

    part_dir = output_dir / f"parts_gfs_{ymd}_{cyc}z"
    ensure_dir(part_dir)

    combined_name = f"gfs_realtime_{ymd}_{cyc}z_f{args.lead_start:03d}_f{args.lead_end:03d}.grib2"
    combined_file = output_dir / combined_name

    if combined_file.exists() and combined_file.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] Combined file already exists: {combined_file}")
        if args.parse_after:
            parse_after_download(combined_file)
        return 0

    downloaded: list[Path] = []
    failed: list[int] = []

    for fh in lead_hours:
        part_file = part_dir / f"gfs_{ymd}_{cyc}z_f{fh:03d}.grib2"
        url = build_gfs_url(cycle_dt=latest_cycle, forecast_hour=fh)

        print(f"[DOWNLOAD] f{fh:03d}")
        ok = download_url(url, part_file, retries=args.retries, timeout=args.timeout)

        if ok:
            downloaded.append(part_file)
            print(f"[OK] {part_file.name}  {part_file.stat().st_size / 1024:.1f} KB")
        else:
            failed.append(fh)
            print(f"[FAIL] f{fh:03d}")

    min_required = min(args.min_success, len(lead_hours))
    if len(downloaded) < min_required:
        print(f"[ERROR] Too few files downloaded: {len(downloaded)} < {min_required}")
        return 3

    print(f"[COMBINE] {len(downloaded)} files -> {combined_file.name}")
    combine_grib_files(downloaded, combined_file)

    record = {
        "time_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cycle_utc": f"{ymd} {cyc}Z",
        "source": "GFS_NOMADS_GLOBAL",
        "combined_file": str(combined_file).replace("\\", "/"),
        "forecast_hours": lead_hours,
        "downloaded_count": len(downloaded),
        "failed_hours": failed,
        "extent": [0.0, -90.0, 359.75, 90.0],
        "status": "success" if not failed else "partial_success",
    }

    manifest_file = output_dir.parent / "download_manifest.json"
    save_manifest(manifest_file, record)

    latest_file = output_dir.parent / "latest_download.json"
    latest_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] Download finished.")
    print(json.dumps(record, ensure_ascii=False, indent=2))

    if args.parse_after:
        parse_after_download(combined_file)

    cleanup_old_cycles(output_dir, keep_cycles=args.keep_cycles)

    return 0


def build_parser() -> argparse.ArgumentParser:
    root = backend_root()
    default_output = root / "data" / "GFS" / "wait_process"

    parser = argparse.ArgumentParser(description="Auto download latest global GFS data from NOMADS.")
    parser.add_argument("--source", default="GFS", choices=["GFS", "ECMWF"])
    parser.add_argument("--output-dir", default=str(default_output))

    parser.add_argument("--lead-start", type=int, default=0)
    parser.add_argument("--lead-end", type=int, default=47)
    parser.add_argument("--lead-step", type=int, default=1)

    parser.add_argument("--max-back-hours", type=int, default=48)
    parser.add_argument("--min-success", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parse-after", action="store_true")
    parser.add_argument("--keep-cycles", type=int, default=3, help="服务器上保留最近多少个 GFS cycle，默认 3；设为 0 表示不清理。")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.source == "ECMWF":
        print("[WARN] ECMWF 通常需要授权账号/API。当前脚本先实现 GFS，ECMWF 保留接口。")
        return 1

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
