"""Base dataset orchestrator.

A ``Dataset`` ties a :class:`TemplateConfig` to a :class:`RegionJob` class and
drives the two workflows shared by all dynamical-style reformatters:

- ``operational_update`` — fetch the latest cycle(s) and overwrite a small store.
- ``backfill`` — process a historical range, appending each region slab along
  the append dimension so the store *grows* (this is how the historical
  analysis archive is built).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from nbm_to_zarr.base.region_job import RegionJob
from nbm_to_zarr.base.template_config import TemplateConfig

logger = logging.getLogger(__name__)

# Coordinate attrs that conflict with Zarr encoding and must be stripped.
_ENCODING_ATTRS = frozenset(
    {"dtype", "compressor", "fill_value", "filters", "chunks", "calendar"}
)


class Dataset(ABC):
    """Base class for dataset orchestration."""

    @property
    @abstractmethod
    def template_config(self) -> TemplateConfig:
        """Return the template configuration."""
        ...

    @property
    @abstractmethod
    def region_job_class(self) -> type[RegionJob]:
        """Return the region job class."""
        ...

    @property
    def dataset_id(self) -> str:
        return self.template_config.dataset_attributes.id

    @property
    def append_dim(self) -> str:
        return self.template_config.append_dim

    # --- Workflows ---------------------------------------------------------

    def operational_update(self, output_path: Path) -> None:
        """Fetch the latest cycle and (over)write a compact store."""
        logger.info("Operational update for %s", self.dataset_id)
        jobs = self.region_job_class.operational_update_jobs(
            template_config=self.template_config,
            output_path=output_path,
        )
        for i, job in enumerate(jobs):
            slab = job.process()
            mode = "w" if i == 0 else "a"
            self._write_slab(slab, output_path, mode=mode)

    def backfill(self, output_path: Path, start: datetime, end: datetime) -> None:
        """Process a historical range, appending each slab to grow the store.

        Resumable: existing append-dim values already in the store are skipped.
        """
        logger.info("Backfill %s: %s .. %s", self.dataset_id, start, end)
        existing = self._existing_append_values(output_path)
        jobs = self.region_job_class.backfill_jobs(
            template_config=self.template_config,
            output_path=output_path,
            start=start,
            end=end,
        )
        first_write = not output_path.exists()
        for job in jobs:
            slab = job.process()
            slab = self._drop_existing(slab, existing)
            if slab.sizes.get(self.append_dim, 0) == 0:
                logger.info("Region already present, skipping")
                continue
            if first_write:
                self._write_slab(slab, output_path, mode="w")
                first_write = False
            else:
                self._write_slab(slab, output_path, mode="a-append")
            existing = self._existing_append_values(output_path)

    # --- Zarr I/O ----------------------------------------------------------

    def _existing_append_values(self, output_path: Path) -> set:
        if not output_path.exists():
            return set()
        try:
            ds = xr.open_zarr(output_path)
            vals = set(np.asarray(ds[self.append_dim].values).tolist())
            ds.close()
            return vals
        except Exception:
            return set()

    def _drop_existing(self, slab: xr.Dataset, existing: set) -> xr.Dataset:
        if not existing:
            return slab
        keep = [
            v.tolist() not in existing
            for v in np.asarray(slab[self.append_dim].values)
        ]
        if all(keep):
            return slab
        return slab.isel({self.append_dim: np.array(keep)})

    def _write_slab(self, ds: xr.Dataset, output_path: Path, mode: str) -> None:
        ds = self._clean_for_zarr(ds)
        if mode == "a-append":
            ds.to_zarr(output_path, append_dim=self.append_dim, zarr_format=2)
        else:
            ds.to_zarr(output_path, mode=mode, zarr_format=2)

    def _clean_for_zarr(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.copy()
        for coord in ds.coords:
            ds[coord].attrs = {
                k: v for k, v in ds[coord].attrs.items() if k not in _ENCODING_ATTRS
            }
            # xarray manages units/calendar for datetime-like coords itself.
            if np.issubdtype(ds[coord].dtype, np.datetime64) or np.issubdtype(
                ds[coord].dtype, np.timedelta64
            ):
                ds[coord].attrs.pop("units", None)
        return ds
