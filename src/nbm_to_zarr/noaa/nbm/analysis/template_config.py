"""Template configuration for NBM CONUS historical analysis data.

This mirrors dynamical.org *analysis* products (e.g. ``noaa-gfs-analysis``,
``noaa-hrrr-analysis``): a single best-estimate value per valid ``time``, with no
forecast-lead dimension — layout ``(time, y, x)``.

NBM is a forecast-only product, so the analysis "best estimate" for a calendar
day is taken as **lead-day 1 of that day's 00z init** (the shortest lead valid on
that day). This is the standard construction dynamical uses to derive an analysis
series from a forecast-only model, and it is documented in the dataset attrs so
consumers understand the provenance.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr
from pydantic import ConfigDict

from nbm_to_zarr.base.template_config import (
    CoordinateConfig,
    DatasetAttributes,
    DataVariableConfig,
    TemplateConfig,
)
from nbm_to_zarr.noaa.nbm import grid
from nbm_to_zarr.noaa.nbm.variables import variable_set


class NbmAnalysisTemplateConfig(TemplateConfig):
    """Template configuration for NBM CONUS historical analysis data."""

    model_config = ConfigDict(frozen=True)

    dimensions: tuple[str, ...] = ("time", "y", "x")
    append_dim: str = "time"
    append_dim_start: datetime = datetime(2025, 1, 1, 0, 0, 0)
    append_dim_freq: str = "1D"

    include_std: bool = False

    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            id="noaa-nbm-conus-analysis",
            title="NOAA NBM CONUS Historical Analysis (daily best estimate)",
            description=(
                "National Blend of Models (NBM) CONUS daily historical analysis: "
                "one best-estimate value per calendar day on the native ~2.5 km "
                "Lambert conformal grid. Built from lead-day 1 of each 00z NBM "
                "forecast (shortest lead valid on the day). Variables: "
                "tmean/tmax/tmin (degC), precip (mm), srad (MJ/m^2/day)."
            ),
            version="0.1.0",
            provider="NOAA",
            model="NBM",
            variant="conus-analysis",
            extra={
                "analysis_construction": (
                    "Best estimate for day D = lead_day 1 of the 00z init on day D."
                ),
                "source": "https://noaa-nbm-grib2-pds.s3.amazonaws.com/",
                "license": "CC-BY-4.0 (derived); source NOAA NBM is public domain",
                "attribution": "NOAA NBM via AWS Open Data; layout inspired by dynamical.org",
            },
        )

    def dimension_coordinates(self, append_dim_end: datetime) -> dict[str, np.ndarray]:
        times = self.append_dim_coordinates(append_dim_end).values
        x, y = grid.grid_xy()
        return {"time": times, "y": y, "x": x}

    def derive_coordinates(
        self, dim_coords: dict[str, np.ndarray]
    ) -> dict[str, xr.DataArray]:
        x = dim_coords["x"]
        y = dim_coords["y"]
        lat, lon = grid.grid_latlon(x, y)
        return {
            "latitude": xr.DataArray(
                lat, dims=("y", "x"),
                attrs={"long_name": "Latitude", "standard_name": "latitude", "units": "degrees_north"},
            ),
            "longitude": xr.DataArray(
                lon, dims=("y", "x"),
                attrs={"long_name": "Longitude", "standard_name": "longitude", "units": "degrees_east"},
            ),
            "spatial_ref": xr.DataArray(0, attrs=grid.spatial_ref_attrs()),
        }

    @property
    def coords(self) -> dict[str, CoordinateConfig]:
        return {
            "time": CoordinateConfig(
                dtype="datetime64[ns]", chunks={"time": 1},
                long_name="Valid date (analysis)", standard_name="time",
            ),
            "y": CoordinateConfig(
                dtype="float64", units="meters",
                long_name="Y coordinate (Lambert conformal projection)",
                standard_name="projection_y_coordinate",
            ),
            "x": CoordinateConfig(
                dtype="float64", units="meters",
                long_name="X coordinate (Lambert conformal projection)",
                standard_name="projection_x_coordinate",
            ),
        }

    @property
    def data_vars(self) -> list[DataVariableConfig]:
        chunks = {"time": 1, "y": 512, "x": 512}
        return [v.to_config(chunks) for v in variable_set(self.include_std)]
