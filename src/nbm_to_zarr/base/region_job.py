"""Base region job for data processing.

A ``RegionJob`` processes one *region* of the append dimension (e.g. one init
time, or one month of analysis days), downloading the needed source messages,
reading them into ``(y, x)`` arrays, applying transforms, and assembling an
``xarray.Dataset`` slab ready to be written/appended to the Zarr store.

This generalizes the dynamical.org ``RegionJob`` so the same base supports both
forecast layouts (``append_dim, lead_day, y, x``) and analysis layouts
(``append_dim, y, x``): the concrete subclass tells the base where each source
message lands via :meth:`output_index`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from nbm_to_zarr.base.template_config import DataVariableConfig, TemplateConfig

logger = logging.getLogger(__name__)


class SourceFileCoord(BaseModel):
    """Identifies one source GRIB message to fetch (one valid time / variable)."""

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def url(self) -> str:
        """Return the URL of the source GRIB file."""
        ...

    @abstractmethod
    def index_url(self) -> str | None:
        """Return the URL of the ``.idx`` sidecar, or None if not available."""
        ...


@dataclass
class ProcessingRegion:
    """A contiguous span of the append dimension to process."""

    append_start: datetime
    append_end: datetime


@dataclass
class ReadResult:
    """A read ``(y, x)`` array plus where it belongs in the output slab."""

    data: np.ndarray
    # Index of this value along the append dimension (within the region slab).
    append_index: int
    # Index along the secondary dimension (e.g. lead_day) or None for 3D layouts.
    secondary_index: int | None = None


class RegionJob(ABC):
    """Base class for processing a region of data."""

    def __init__(
        self,
        template_config: TemplateConfig,
        processing_region: ProcessingRegion,
        data_vars: list[DataVariableConfig],
        output_path: Path,
        download_dir: Path | None = None,
    ) -> None:
        self.template_config = template_config
        self.processing_region = processing_region
        self.data_vars = data_vars
        self.output_path = output_path
        self.download_dir = download_dir or Path("/tmp/nbm_downloads")
        self.download_dir.mkdir(parents=True, exist_ok=True)

    # --- Subclass contract -------------------------------------------------

    @abstractmethod
    def region_dim_coords(self) -> dict[str, np.ndarray]:
        """Return dimension coordinate arrays for *just this region's* slab.

        The append dimension must be restricted to this region; spatial and any
        secondary dims should match the full template.
        """
        ...

    @abstractmethod
    def generate_source_file_coords(self) -> list[SourceFileCoord]:
        """List the source messages needed to fill this region's slab."""
        ...

    @abstractmethod
    def read_data(
        self, var: DataVariableConfig, coord: SourceFileCoord
    ) -> ReadResult | None:
        """Fetch + decode one ``(y, x)`` field for ``var`` at ``coord``.

        Returns None to skip (missing init/lead — logged, not fatal).
        """
        ...

    # --- Shared machinery --------------------------------------------------

    def apply_transformations(
        self, data: np.ndarray, var: DataVariableConfig
    ) -> np.ndarray:
        """Apply per-variable transforms (bit rounding for compression)."""
        if var.keepbits is not None:
            data = self._apply_bit_rounding(data, var.keepbits)
        return data

    @staticmethod
    def _apply_bit_rounding(data: np.ndarray, keepbits: int) -> np.ndarray:
        """Round float mantissa to ``keepbits`` bits to help zstd compression.

        Uses the numcodecs/xbitinfo "round to nearest, ties to even" scheme.
        NaNs and infs pass through untouched.
        """
        if data.dtype != np.float32:
            return data
        if keepbits >= 23:  # full float32 mantissa
            return data
        view = data.view(np.int32)
        mask_bits = 23 - keepbits
        # Round half to even.
        half = (1 << (mask_bits - 1)) if mask_bits > 0 else 0
        ones = np.int32(~((1 << mask_bits) - 1))
        rounded = (view + half + ((view >> mask_bits) & 1)) & ones
        out = rounded.view(np.float32).copy()
        nan_inf = ~np.isfinite(data)
        out[nan_inf] = data[nan_inf]
        return out

    def build_region_dataset(self) -> xr.Dataset:
        """Build the empty slab dataset for this region."""
        return self.template_config.build_dataset(self.region_dim_coords())

    def process(self) -> xr.Dataset:
        """Process the region and return the populated slab dataset."""
        region_ds = self.build_region_dataset()
        source_coords = self.generate_source_file_coords()
        logger.info(
            "Region %s..%s: %d source messages × %d variables",
            self.processing_region.append_start,
            self.processing_region.append_end,
            len(source_coords),
            len(self.data_vars),
        )

        has_secondary = len(self.template_config.dimensions) >= 4

        for var in self.data_vars:
            for coord in source_coords:
                try:
                    result = self.read_data(var, coord)
                except Exception:
                    logger.exception("Error reading %s from %s", var.name, coord.url())
                    continue
                if result is None:
                    continue
                data = self.apply_transformations(result.data, var)
                arr = region_ds[var.name].values
                if has_secondary:
                    if result.secondary_index is None:
                        logger.warning("Missing secondary_index for 4D var %s", var.name)
                        continue
                    arr[result.append_index, result.secondary_index, :, :] = data
                else:
                    arr[result.append_index, :, :] = data

        return region_ds

    @classmethod
    @abstractmethod
    def operational_update_jobs(
        cls,
        template_config: TemplateConfig,
        output_path: Path,
    ) -> list[RegionJob]:
        """Create the region jobs for an operational (latest-cycle) update."""
        ...

    @classmethod
    @abstractmethod
    def backfill_jobs(
        cls,
        template_config: TemplateConfig,
        output_path: Path,
        start: datetime,
        end: datetime,
    ) -> list[RegionJob]:
        """Create region jobs to backfill a historical date range."""
        ...
