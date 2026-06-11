#!/usr/bin/env python3
"""Build a short, GitHub-hostable NBM analysis Zarr.

The full-resolution NBM analysis grid (2345 x 1597, 5 vars) is ~75 MB/day
uncompressed — too large to keep many days of in a git repo. But a *spatially
coarsened, short* analysis dataset is a few MB and fits comfortably on GitHub
itself (no external object store, no LFS, no egress bills). This is exactly the
"host a really short analysis dataset on GitHub for free" path.

Two modes:

  --real    Fetch real NBM data for the requested days via the analysis
            reformatter, then coarsen spatially by --coarsen before writing.
            Requires network + eccodes. Use this to publish a real sample.

  (default) Generate a small *synthetic* analysis Zarr with the identical
            schema/coords. No network or eccodes needed — guarantees the repo
            always ships a working, openable demo store and lets CI/tests run
            anywhere. Clearly flagged as synthetic in the attrs.

Output goes to ``data/noaa-nbm-conus-analysis-sample.zarr`` by default, which the
GitHub Actions workflow commits straight into the repo.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nbm_to_zarr.noaa.nbm import grid
from nbm_to_zarr.noaa.nbm.analysis import NbmAnalysisDataset
from nbm_to_zarr.noaa.nbm.variables import variable_set

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("data/noaa-nbm-conus-analysis-sample.zarr")


def _coarsen_grid(coarsen: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return coarsened (x, y, lat2d, lon2d) by striding the native grid."""
    x, y = grid.grid_xy()
    xc = x[::coarsen]
    yc = y[::coarsen]
    lat, lon = grid.grid_latlon(xc, yc)
    return xc, yc, lat, lon


def _base_coords(times: np.ndarray, coarsen: int) -> dict[str, xr.DataArray]:
    xc, yc, lat, lon = _coarsen_grid(coarsen)
    return {
        "time": xr.DataArray(times, dims=("time",), attrs={"standard_name": "time"}),
        "y": xr.DataArray(yc, dims=("y",), attrs={"units": "meters"}),
        "x": xr.DataArray(xc, dims=("x",), attrs={"units": "meters"}),
        "latitude": xr.DataArray(lat, dims=("y", "x"), attrs={"units": "degrees_north"}),
        "longitude": xr.DataArray(lon, dims=("y", "x"), attrs={"units": "degrees_east"}),
        "spatial_ref": xr.DataArray(0, attrs=grid.spatial_ref_attrs()),
    }


def _attrs(synthetic: bool, coarsen: int) -> dict[str, str]:
    tc = NbmAnalysisDataset().template_config
    attrs = tc.dataset_attributes.model_dump()
    extra = attrs.pop("extra", {})
    attrs.update(extra)
    attrs["id"] = "noaa-nbm-conus-analysis-sample"
    attrs["title"] = "NOAA NBM CONUS Analysis — short sample (GitHub-hosted)"
    attrs["spatial_coarsening"] = f"{coarsen}x strided from native ~2.5 km grid"
    attrs["hosting"] = "Committed directly to this GitHub repo (free, no object store)"
    if synthetic:
        attrs["data_status"] = (
            "SYNTHETIC demo data — schema/coords are real, values are generated. "
            "Run build_sample_analysis.py --real to publish a real sample."
        )
    else:
        attrs["data_status"] = "Real NBM-derived sample (spatially coarsened)."
    return attrs


def build_synthetic(days: int, coarsen: int, start: datetime) -> xr.Dataset:
    """Build a small synthetic analysis store with the real schema."""
    times = pd.date_range(start=start, periods=days, freq="1D").values
    coords = _base_coords(times, coarsen)
    ny, nx = coords["latitude"].shape
    lat2d = coords["latitude"].values
    rng = np.random.default_rng(42)

    data_vars: dict[str, xr.DataArray] = {}
    for v in variable_set(include_std=False):
        # Latitude- and season-aware smooth fields so the demo plots sensibly.
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
    ds.attrs.update(_attrs(synthetic=True, coarsen=coarsen))
    return ds


def build_real(days: int, coarsen: int, start: datetime) -> xr.Dataset:
    """Fetch real NBM analysis for ``days`` and coarsen spatially."""
    end = start + timedelta(days=days - 1)
    tmp = Path("data/_full_analysis_tmp.zarr")
    NbmAnalysisDataset().backfill(tmp, start, end)
    full = xr.open_zarr(tmp)
    # Stride spatially to match the coarsened sample grid.
    sample = full.isel(y=slice(None, None, coarsen), x=slice(None, None, coarsen))
    sample = sample.compute()
    full.close()
    sample.attrs.update(_attrs(synthetic=False, coarsen=coarsen))
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Fetch real NBM data")
    parser.add_argument("--days", type=int, default=14, help="Number of days")
    parser.add_argument("--coarsen", type=int, default=10, help="Spatial stride factor")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = datetime.fromisoformat(args.start)

    if args.real:
        ds = build_real(args.days, args.coarsen, start)
    else:
        ds = build_synthetic(args.days, args.coarsen, start)

    # Bit-round mantissas before writing — for a demo, ~10 bits is plenty and
    # roughly halves the on-disk size, keeping the committed sample tiny.
    from nbm_to_zarr.base.region_job import RegionJob

    for v in ds.data_vars:
        ds[v].values = RegionJob._apply_bit_rounding(
            ds[v].values.astype(np.float32), keepbits=10
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"compressor": _zstd()} for v in ds.data_vars}
    ds.to_zarr(args.output, mode="w", zarr_format=2, encoding=encoding)

    size_mb = _dir_size_mb(args.output)
    logger.info("Wrote %s (%.2f MB on disk)", args.output, size_mb)
    logger.info("dims=%s vars=%s", dict(ds.sizes), list(ds.data_vars))
    if size_mb > 50:
        logger.warning(
            "Sample is %.1f MB — consider larger --coarsen or fewer --days "
            "to stay comfortably within GitHub's per-file/repo limits.",
            size_mb,
        )


def _zstd():
    from numcodecs import Blosc

    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


if __name__ == "__main__":
    main()
