#!/usr/bin/env python3
"""Upload an NBM Zarr store to a HuggingFace dataset repo.

HuggingFace dataset repos are free, public, and TB-scale — the right home for the
large forecast store and for an analysis archive that grows past GitHub's short
window. ``upload_large_folder`` handles the many small Zarr chunk files with
resume + dedup, so re-running after a backfill only pushes new/changed chunks.

``upload_large_folder`` uploads a folder to the *repo root*, so we use one repo
per store (the Zarr store IS the repo). Open it with
``xr.open_zarr("https://huggingface.co/datasets/<repo>/resolve/main")``.

Auth: set ``HF_TOKEN`` (a write token from https://huggingface.co/settings/tokens)
or run ``hf auth login`` first.

Examples:
    # Push the forecast store to its own dataset repo
    python scripts/upload_hf.py --repo andrewnakas/nbm-conus-forecast \\
        --store data/noaa-nbm-conus-forecast.zarr

    # Push the analysis archive
    python scripts/upload_hf.py --repo andrewnakas/nbm-conus-analysis \\
        --store data/noaa-nbm-conus-analysis.zarr

    # Verify a remote store opens
    python scripts/upload_hf.py --repo andrewnakas/nbm-conus-analysis --verify-only
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)


def hf_zarr_url(repo: str) -> str:
    """The ``https://`` URL xarray can open the Zarr store (repo root) from."""
    return f"https://huggingface.co/datasets/{repo}/resolve/main"


def ensure_repo(api: HfApi, repo: str, token: str | None) -> None:
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, token=token)


def upload_store(api: HfApi, repo: str, store: Path, token: str | None) -> None:
    if not store.exists():
        raise SystemExit(f"store not found: {store}")
    logger.info("Uploading %s -> datasets/%s (repo root)", store, repo)
    api.upload_large_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(store),
    )
    logger.info("Done. Open with:\n  xr.open_zarr('%s')", hf_zarr_url(repo))


def verify(repo: str) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    import xarray as xr

    url = hf_zarr_url(repo)
    ds = xr.open_zarr(url)
    logger.info("Opened %s", url)
    logger.info("  dims=%s vars=%s", dict(ds.sizes), list(ds.data_vars))
    time_dim = "time" if "time" in ds.coords else "init_time"
    logger.info(
        "  %s: %s .. %s",
        time_dim,
        str(ds[time_dim].values.min())[:10],
        str(ds[time_dim].values.max())[:10],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="HF dataset repo id (user/name)")
    parser.add_argument("--store", type=Path, help="Local Zarr store to upload")
    parser.add_argument("--verify-only", action="store_true", help="Just open the remote store")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.verify_only:
        verify(args.repo)
        return

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    ensure_repo(api, args.repo, token)
    if not args.store:
        parser.error("--store is required unless --verify-only")
    upload_store(api, args.repo, args.store, token)


if __name__ == "__main__":
    main()
