#!/usr/bin/env python3
"""Backfill the real NBM CONUS forecast Zarr for a date range.

Unlike the analysis sample (short + full-res + committed to GitHub), the forecast
store is large — (init_time, lead_day 1-7, y, x) for 5 vars means ~7x the data of
the analysis. It is therefore written to a gitignored local path by default and
intended for an external object store (HuggingFace / R2 / Source Coop), not the
git repo. This script just drives ``NbmForecastDataset.backfill``.

Examples:
    # Last 7 days (00z inits), full resolution
    python scripts/build_forecast.py --start 2026-06-04 --end 2026-06-10

    # Most recent init only (operational update)
    python scripts/build_forecast.py --operational
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from nbm_to_zarr.noaa.nbm.forecast import NbmForecastDataset

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/noaa-nbm-conus-forecast.zarr")  # gitignored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Start date YYYY-MM-DD (00z inits)")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--operational", action="store_true", help="Latest init only")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-std", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ds = NbmForecastDataset(include_std=args.include_std)

    if args.operational:
        ds.operational_update(args.output)
    else:
        if not (args.start and args.end):
            parser.error("provide --start and --end, or --operational")
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
        ds.backfill(args.output, start, end)

    logger.info("Forecast store written to %s", args.output)


if __name__ == "__main__":
    main()
