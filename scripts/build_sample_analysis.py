#!/usr/bin/env python3
"""Build a short, full-resolution, GitHub-hostable NBM analysis Zarr.

We host a *full-resolution* (native ~2.5 km, 2345 x 1597) NBM analysis on GitHub
itself, kept short — at most ~7 days. Zarr stores the grid as many small,
zstd-compressed chunk files (here 512 x 512), so each file stays well under
GitHub's 100 MB cap and the whole short window is a tractable repo size; no
external object store, no Git LFS, no egress bills. Real meteorological fields
(smooth gradients, precip mostly zero) compress far better than the worst case.

Two modes:

  --real    Fetch real NBM data for the last ``--days`` days at full resolution
            via the analysis reformatter (needs network + eccodes). This is the
            path the daily GitHub Actions job uses.

  (default) Generate a *synthetic* full-resolution analysis with the identical
            schema/coords/grid (no network or eccodes), so the repo always ships
            a working, openable store and tests run anywhere. Flagged synthetic
            in the attrs.

A rolling 7-day window keeps the committed store bounded: each run rewrites it
with the most recent ``--days`` days.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nbm_to_zarr.base.region_job import RegionJob
from nbm_to_zarr.noaa.nbm import grid
from nbm_to_zarr.noaa.nbm.analysis import NbmAnalysisDataset
from nbm_to_zarr.noaa.nbm.variables import variable_set

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/noaa-nbm-conus-analysis-sample.zarr")
MAX_DAYS = 7  # rolling window cap — keep the committed store short


def _base_coords(times: np.ndarray) -> dict[str, xr.DataArray]:
    """Full-resolution native grid coordinates (no coarsening)."""
    x, y = grid.grid_xy()
    lat, lon = grid.grid_latlon(x, y)
    return {
        "time": xr.DataArray(times, dims=("time",), attrs={"standard_name": "time"}),
        "y": xr.DataArray(y, dims=("y",), attrs={"units": "meters"}),
        "x": xr.DataArray(x, dims=("x",), attrs={"units": "meters"}),
        "latitude": xr.DataArray(lat, dims=("y", "x"), attrs={"units": "degrees_north"}),
        "longitude": xr.DataArray(lon, dims=("y", "x"), attrs={"units": "degrees_east"}),
        "spatial_ref": xr.DataArray(0, attrs=grid.spatial_ref_attrs()),
    }


def _attrs(synthetic: bool) -> dict[str, str]:
    tc = NbmAnalysisDataset().template_config
    attrs = tc.dataset_attributes.model_dump()
    extra = attrs.pop("extra", {})
    attrs.update(extra)
    attrs["id"] = "noaa-nbm-conus-analysis-sample"
    attrs["title"] = "NOAA NBM CONUS Analysis — short sample (full-res, GitHub-hosted)"
    attrs["resolution"] = "Native ~2.5 km Lambert conformal (2345 x 1597), no coarsening"
    attrs["window"] = f"Rolling last {MAX_DAYS} days (full resolution)"
    attrs["hosting"] = "Committed directly to this GitHub repo (free, no object store)"
    attrs["data_status"] = (
        "SYNTHETIC demo data — schema/coords/grid are real, values are generated. "
        "Run build_sample_analysis.py --real to publish real data."
        if synthetic
        else "Real NBM-derived analysis at full resolution."
    )
    return attrs


def _encoding(ds: xr.Dataset) -> dict[str, dict]:
    from numcodecs import Blosc

    comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    enc = {}
    for v in ds.data_vars:
        chunks = (1, min(512, ds.sizes["y"]), min(512, ds.sizes["x"]))
        enc[v] = {"compressor": comp, "chunks": chunks}
    return enc


def build_synthetic(days: int, start: datetime) -> xr.Dataset:
    """Full-resolution synthetic analysis with the real schema/grid."""
    times = pd.date_range(start=start, periods=days, freq="1D").values
    coords = _base_coords(times)
    ny, nx = coords["latitude"].shape
    lat2d = coords["latitude"].values
    rng = np.random.default_rng(42)

    data_vars: dict[str, xr.DataArray] = {}
    for v in variable_set(include_std=False):
        day_idx = np.arange(days)[:, None, None]
        seasonal = np.cos(2 * np.pi * (day_idx / 365.0))
        lat_field = lat2d[None, :, :]
        if v.name in ("tmean", "tmax", "tmin"):
            base = 20.0 - 0.6 * (lat_field - 25.0) + 6.0 * seasonal
            bump = {"tmax": 6.0, "tmin": -6.0}.get(v.name, 0.0)
            field = base + bump + rng.normal(0, 1.0, size=(days, ny, nx))
        elif v.name == "precip":
            field = np.clip(rng.gamma(0.6, 4.0, size=(days, ny, nx)) - 1.0, 0, None)
        elif v.name == "srad":
            field = np.clip(18.0 + 8.0 * seasonal + rng.normal(0, 1.0, size=(days, ny, nx)), 0, None)
        else:
            field = rng.normal(0, 1.0, size=(days, ny, nx))
        data_vars[v.name] = xr.DataArray(
            field.astype(np.float32),
            dims=("time", "y", "x"),
            attrs={"units": v.units, "long_name": v.long_name},
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs.update(_attrs(synthetic=True))
    return ds


def build_real(days: int, end: datetime) -> xr.Dataset:
    """Fetch real NBM analysis for the last ``days`` days at full resolution."""
    start = end - timedelta(days=days - 1)
    tmp = Path("data/_analysis_sample_tmp.zarr")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    NbmAnalysisDataset().backfill(tmp, start, end)
    ds = xr.open_zarr(tmp).compute()
    ds.attrs.update(_attrs(synthetic=False))
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Fetch real NBM data")
    parser.add_argument("--days", type=int, default=MAX_DAYS, help=f"Days (max {MAX_DAYS})")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: yesterday UTC)")
    parser.add_argument("--start", default="2025-01-01", help="Synthetic start date")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    days = min(args.days, MAX_DAYS)

    if args.real:
        from datetime import UTC

        end = (
            datetime.fromisoformat(args.end)
            if args.end
            else (datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        )
        ds = build_real(days, end)
    else:
        ds = build_synthetic(days, datetime.fromisoformat(args.start))

    # Bit-round mantissas (10 bits) before writing for much smaller chunks.
    for v in ds.data_vars:
        ds[v].values = RegionJob._apply_bit_rounding(ds[v].values.astype(np.float32), 10)

    if args.output.exists():
        import shutil

        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(args.output, mode="w", zarr_format=2, encoding=_encoding(ds))

    size_mb = _dir_size_mb(args.output)
    largest = _largest_file_mb(args.output)
    logger.info("Wrote %s (%.1f MB total, largest chunk %.2f MB)", args.output, size_mb, largest)
    logger.info("dims=%s vars=%s", dict(ds.sizes), list(ds.data_vars))
    if largest > 95:
        logger.warning("A chunk is %.1f MB — near GitHub's 100 MB/file cap.", largest)


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def _largest_file_mb(path: Path) -> float:
    files = [f.stat().st_size for f in path.rglob("*") if f.is_file()]
    return (max(files) / (1024 * 1024)) if files else 0.0


if __name__ == "__main__":
    main()
