"""Download ERA5 single-level data and optionally parse it for display.

Usage examples:
  python scripts/download_era5.py --date 2025-06-16 --parse
  python scripts/download_era5.py --start-date 2025-06-16 --end-date 2025-06-18 --area 54 73 18 135

The script uses the official CDS API client. Configure credentials with
%USERPROFILE%\\.cdsapirc or the CDSAPI_URL and CDSAPI_KEY environment variables.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


DEFAULT_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "surface_pressure",
]

DEFAULT_TIMES = [f"{hour:02d}:00" for hour in range(24)]
DEFAULT_AREA = [54.0, 73.0, 18.0, 135.0]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERA5_DATA_DIR = PROJECT_ROOT / "data" / "ERA5"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _client():
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency cdsapi. Install it in this backend environment with: "
            "python -m pip install cdsapi"
        ) from exc
    return cdsapi.Client()


def _download_day(client, current: date, args: argparse.Namespace) -> Path:
    ERA5_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else ERA5_DATA_DIR / f"era5_{current:%Y%m%d}.nc"
    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": args.variables,
        "year": f"{current.year:04d}",
        "month": f"{current.month:02d}",
        "day": f"{current.day:02d}",
        "time": args.times,
        "area": args.area,
    }
    client.retrieve("reanalysis-era5-single-levels", request, str(output))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ERA5 NetCDF files through the CDS API.")
    parser.add_argument("--date", help="Single date in YYYY-MM-DD format.")
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--area", nargs=4, type=float, default=DEFAULT_AREA, metavar=("N", "W", "S", "E"))
    parser.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    parser.add_argument("--times", nargs="+", default=DEFAULT_TIMES)
    parser.add_argument("--output", help="Output path. Use only with --date for a single file.")
    parser.add_argument("--parse", action="store_true", help="Run the ERA5 parser after each download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.date:
        start = end = _parse_date(args.date)
    else:
        if not args.start_date or not args.end_date:
            raise SystemExit("Use --date or both --start-date and --end-date.")
        start = _parse_date(args.start_date)
        end = _parse_date(args.end_date)
    if start > end:
        raise SystemExit("--start-date must be before or equal to --end-date.")
    if args.output and start != end:
        raise SystemExit("--output can only be used with a single --date.")

    client = _client()
    downloaded: list[Path] = []
    for current in _date_range(start, end):
        output = _download_day(client, current, args)
        downloaded.append(output)
        print(f"downloaded: {output}")
        if args.parse:
            from adapters.era5_adapter import process_file

            meta = process_file(str(output))
            print(f"parsed: {meta.get('meta_file')}")

    print(f"done: {len(downloaded)} file(s)")


if __name__ == "__main__":
    main()
