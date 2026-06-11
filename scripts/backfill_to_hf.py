#!/usr/bin/env python3
"""Backfill NBM data week-by-week, pushing each week to HuggingFace, then pruning.

Built for a disk-constrained, busy machine: process one ISO week at a time into a
standalone per-week Zarr, upload it to a HuggingFace dataset repo under
``YYYY-Www/``, then delete the local copy before the next week. Peak local
footprint is ~one week.

The HF repo is laid out as::

    <repo>/2026-W23/   (a complete zarr for that week)
    <repo>/2026-W22/
    ...

Each week is independently openable; concatenating the weeks along the time/
init_time dim reconstructs the full archive.

Data availability: NBM CONUS core on AWS starts ~2020-10-01. Earlier dates 404
and are skipped automatically (so an end date in the distant past just stops).

Direction: by default walks **backwards** from --end to --start (most recent
first), so the freshest data lands first and the archive deepens over time.

Auth: HF_TOKEN env var (write token).

Example (run in background, resumes if interrupted):
    HF_TOKEN=... python scripts/backfill_to_hf.py analysis \\
        --repo nakas/nbm-conus-analysis --start 2020-10-01 --end 2026-06-10
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from huggingface_hub import HfApi

from nbm_to_zarr.noaa.nbm.analysis import NbmAnalysisDataset
from nbm_to_zarr.noaa.nbm.forecast import NbmForecastDataset

logger = logging.getLogger(__name__)

SCRATCH = Path("data/_backfill_week.zarr")
# NBM CONUS core archive floor on AWS (verified: 2020-09 is 404, 2020-10 is 200).
DATA_FLOOR = datetime(2020, 10, 1)


def iso_weeks(start: datetime, end: datetime) -> list[tuple[str, datetime, datetime]]:
    """Return (label, monday, sunday) for each ISO week overlapping [start, end]."""
    weeks: list[tuple[str, datetime, datetime]] = []
    # Snap to the Monday on/before start.
    cur = start - timedelta(days=start.weekday())
    while cur <= end:
        wk_start = max(cur, start)
        wk_end = min(cur + timedelta(days=6), end)
        iso = cur.isocalendar()
        label = f"{iso.year}-W{iso.week:02d}"
        weeks.append((label, wk_start, wk_end))
        cur += timedelta(days=7)
    return weeks


def week_done(api: HfApi, repo: str, label: str, token: str | None) -> bool:
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset", token=token)
    except Exception:  # noqa: BLE001
        return False
    return any(f.startswith(f"{label}/") and f.endswith(".zarray") for f in files)


def build_week(kind: str, first: datetime, last: datetime, include_std: bool) -> bool:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    ds = (
        NbmAnalysisDataset(include_std=include_std)
        if kind == "analysis"
        else NbmForecastDataset(include_std=include_std)
    )
    ds.backfill(SCRATCH, first, last)
    return SCRATCH.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["analysis", "forecast"])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--include-std", action="store_true")
    parser.add_argument("--forward", action="store_true", help="Oldest first (default: newest first)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN")

    start = max(datetime.fromisoformat(args.start), DATA_FLOOR)
    end = datetime.fromisoformat(args.end)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True, token=token)

    weeks = iso_weeks(start, end)
    if not args.forward:
        weeks = list(reversed(weeks))
    logger.info("%s: %d weeks %s..%s -> %s (%s first)",
                args.kind, len(weeks), start.date(), end.date(), args.repo,
                "oldest" if args.forward else "newest")

    done = skipped = built = 0
    for label, first, last in weeks:
        if week_done(api, args.repo, label, token):
            logger.info("[%s] already on HF, skip", label)
            skipped += 1
            continue
        logger.info("[%s] building %s .. %s", label, first.date(), last.date())
        try:
            if not build_week(args.kind, first, last, args.include_std):
                logger.warning("[%s] no data (likely pre-archive), skip", label)
                skipped += 1
                continue
        except Exception:
            logger.exception("[%s] build failed, skip", label)
            skipped += 1
            continue

        try:
            api.upload_folder(
                repo_id=args.repo,
                repo_type="dataset",
                folder_path=str(SCRATCH),
                path_in_repo=label,
                commit_message=f"Add {args.kind} {label}",
                token=token,
            )
            built += 1
            done += 1
            logger.info("[%s] uploaded (%d done, %d skipped)", label, done, skipped)
        except Exception:
            logger.exception("[%s] upload failed, keeping local for retry", label)
            continue
        finally:
            shutil.rmtree(SCRATCH, ignore_errors=True)

    logger.info("Backfill finished: %d uploaded, %d skipped", built, skipped)


if __name__ == "__main__":
    main()
