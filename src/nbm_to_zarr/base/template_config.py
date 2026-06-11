"""Base template configuration for datasets.

Mirrors the dynamical.org reformatter ``TemplateConfig`` contract: a dataset is
described by its dimensions, an append dimension (the one we grow over time),
coordinate configs, and data-variable configs. ``get_template`` materializes an
empty :class:`xarray.Dataset` with the right structure, which both initializes
the on-disk Zarr and documents the schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict


class CoordinateConfig(BaseModel):
    """Configuration for a coordinate variable."""

    model_config = ConfigDict(frozen=True)

    dtype: str
    chunks: dict[str, int] | None = None
    compression: str = "zstd"
    compression_level: int = 3
    units: str | None = None
    long_name: str | None = None
    standard_name: str | None = None


class DataVariableConfig(BaseModel):
    """Configuration for a data variable."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str = "float32"
    chunks: dict[str, int]
    compression: str = "zstd"
    compression_level: int = 3
    units: str | None = None
    long_name: str | None = None
    standard_name: str | None = None
    fill_value: float = float("nan")
    # Bit rounding for compression (mantissa bits to keep); None disables it.
    keepbits: int | None = None


class DatasetAttributes(BaseModel):
    """Dataset-level attributes (written to the Zarr root ``.zattrs``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    description: str
    version: str
    provider: str
    model: str
    variant: str
    # Extra free-form attrs (conventions, attribution, valid-date semantics, …).
    extra: dict[str, str] = {}


class TemplateConfig(BaseModel, ABC):
    """Base configuration for dataset templates."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Dimension names in storage order, e.g. ("init_time", "lead_day", "y", "x").
    dimensions: tuple[str, ...]
    # The dimension we append to over time, e.g. "init_time" or "time".
    append_dim: str
    append_dim_start: datetime
    # Pandas frequency string for the append dimension (e.g. "24h", "1D").
    append_dim_freq: str

    @property
    @abstractmethod
    def dataset_attributes(self) -> DatasetAttributes:
        """Return dataset-level attributes."""
        ...

    @abstractmethod
    def dimension_coordinates(self, append_dim_end: datetime) -> dict[str, np.ndarray]:
        """Return dimension coordinate arrays keyed by dimension name."""
        ...

    @abstractmethod
    def derive_coordinates(
        self, dim_coords: dict[str, np.ndarray]
    ) -> dict[str, xr.DataArray]:
        """Derive non-dimension coordinates (e.g. 2D lat/lon, spatial_ref)."""
        ...

    @property
    @abstractmethod
    def coords(self) -> dict[str, CoordinateConfig]:
        """Return coordinate configurations keyed by coordinate name."""
        ...

    @property
    @abstractmethod
    def data_vars(self) -> list[DataVariableConfig]:
        """Return data variable configurations."""
        ...

    def append_dim_coordinates(self, end: datetime) -> pd.DatetimeIndex:
        """Generate a DatetimeIndex for the append dimension."""
        return pd.date_range(
            start=self.append_dim_start,
            end=end,
            freq=self.append_dim_freq,
        )

    def _coord_attrs(self, name: str) -> dict[str, str]:
        cfg = self.coords.get(name)
        attrs: dict[str, str] = {}
        if cfg is None:
            return attrs
        if cfg.units:
            attrs["units"] = cfg.units
        if cfg.long_name:
            attrs["long_name"] = cfg.long_name
        if cfg.standard_name:
            attrs["standard_name"] = cfg.standard_name
        return attrs

    def _var_attrs(self, var: DataVariableConfig) -> dict[str, str]:
        attrs: dict[str, str] = {}
        if var.units:
            attrs["units"] = var.units
        if var.long_name:
            attrs["long_name"] = var.long_name
        if var.standard_name:
            attrs["standard_name"] = var.standard_name
        return attrs

    def build_dataset(self, dim_coords: dict[str, np.ndarray]) -> xr.Dataset:
        """Assemble an empty (fill-valued) dataset from dimension coordinates.

        Shared by ``get_template`` (full schema) and the region jobs (a single
        append-dim slab), so both produce identical structure.
        """
        coords: dict[str, xr.DataArray] = {}
        for name, values in dim_coords.items():
            coords[name] = xr.DataArray(values, dims=[name], attrs=self._coord_attrs(name))
        coords.update(self.derive_coordinates(dim_coords))

        data_vars: dict[str, xr.DataArray] = {}
        for var in self.data_vars:
            shape = tuple(len(dim_coords[d]) for d in self.dimensions)
            data = np.full(shape, var.fill_value, dtype=var.dtype)
            data_vars[var.name] = xr.DataArray(
                data, dims=self.dimensions, attrs=self._var_attrs(var)
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        attrs = self.dataset_attributes.model_dump()
        extra = attrs.pop("extra", {})
        ds.attrs.update(attrs)
        ds.attrs.update(extra)
        return ds

    def get_template(self, end: datetime) -> xr.Dataset:
        """Create an empty template dataset covering ``append_dim_start``..``end``."""
        dim_coords = self.dimension_coordinates(end)
        return self.build_dataset(dim_coords)

    def template_path(self) -> Path:
        """Return the on-disk path used to store/load the template."""
        templates_dir = Path(__file__).parent.parent / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = self.dataset_attributes.id.replace("-", "_")
        return templates_dir / f"{dataset_id}.zarr"
