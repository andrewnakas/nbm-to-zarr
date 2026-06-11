"""Template configuration for NBM CONUS forecast data.

Layout mirrors dynamical.org forecast zarrs: an ``init_time`` append dimension,
a forecast-lead dimension (here ``lead_day`` 1-11 instead of hourly ``step``,
because the consumer is daily and native hourly leads would be ~1-2 TB), and the
2D Lambert grid (``y``, ``x`` + 2D ``latitude``/``longitude`` + ``spatial_ref``).

Valid-date convention (documented in attrs and matching RiverWatch2's GFS/HRRR
fetchers): a 00z init on calendar day D has ``lead_day = 1`` covering day D.
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

N_LEAD_DAYS = 11


class NbmForecastTemplateConfig(TemplateConfig):
    """Template configuration for NBM CONUS forecast data."""

    model_config = ConfigDict(frozen=True)

    dimensions: tuple[str, ...] = ("init_time", "lead_day", "y", "x")
    append_dim: str = "init_time"
    # 2025 is a single NBM v4.2+ era — no version seam (see plan Context).
    append_dim_start: datetime = datetime(2025, 1, 1, 0, 0, 0)
    append_dim_freq: str = "24h"  # 00z cycle only

    include_std: bool = False

    @property
    def dataset_attributes(self) -> DatasetAttributes:
        return DatasetAttributes(
            id="noaa-nbm-conus-forecast",
            title="NOAA NBM CONUS Forecast (daily, lead days 1-11)",
            description=(
                "National Blend of Models (NBM) CONUS forecast, 00z cycle, "
                "daily-aggregated lead days 1-11 on the native ~2.5 km Lambert "
                "conformal grid. Variables: tmean/tmax/tmin (degC), precip (mm), "
                "srad (MJ/m^2/day). Honest note: past ~day 3 the 2.5 km grid "
                "carries statistically downscaled global-ensemble information — "
                "terrain-aware texture and calibration, not fresh 2.5 km physics."
            ),
            version="0.1.0",
            provider="NOAA",
            model="NBM",
            variant="conus-forecast",
            extra={
                "valid_date_convention": (
                    "A 00z init on calendar day D has lead_day=1 covering day D "
                    "(valid_date = init_date + lead_day - 1)."
                ),
                "source": "https://noaa-nbm-grib2-pds.s3.amazonaws.com/",
                "license": "CC-BY-4.0 (derived); source NOAA NBM is public domain",
                "attribution": "NOAA NBM via AWS Open Data; layout inspired by dynamical.org",
            },
        )

    def dimension_coordinates(self, append_dim_end: datetime) -> dict[str, np.ndarray]:
        init_times = self.append_dim_coordinates(append_dim_end).values
        lead_days = np.arange(1, N_LEAD_DAYS + 1, dtype="int32")
        x, y = grid.grid_xy()
        return {"init_time": init_times, "lead_day": lead_days, "y": y, "x": x}

    def derive_coordinates(
        self, dim_coords: dict[str, np.ndarray]
    ) -> dict[str, xr.DataArray]:
        init_times = dim_coords["init_time"]
        lead_days = dim_coords["lead_day"]
        x = dim_coords["x"]
        y = dim_coords["y"]
        coords: dict[str, xr.DataArray] = {}

        # valid_date[init, lead] = init_date + (lead_day - 1) days.
        init_days = init_times.astype("datetime64[D]")
        valid_date = (
            init_days[:, np.newaxis]
            + (lead_days[np.newaxis, :] - 1).astype("timedelta64[D]")
        )
        coords["valid_date"] = xr.DataArray(
            valid_date,
            dims=("init_time", "lead_day"),
            attrs={"long_name": "Valid calendar date", "standard_name": "time"},
        )

        lat, lon = grid.grid_latlon(x, y)
        coords["latitude"] = xr.DataArray(
            lat, dims=("y", "x"),
            attrs={"long_name": "Latitude", "standard_name": "latitude", "units": "degrees_north"},
        )
        coords["longitude"] = xr.DataArray(
            lon, dims=("y", "x"),
            attrs={"long_name": "Longitude", "standard_name": "longitude", "units": "degrees_east"},
        )
        coords["spatial_ref"] = xr.DataArray(0, attrs=grid.spatial_ref_attrs())
        return coords

    @property
    def coords(self) -> dict[str, CoordinateConfig]:
        return {
            "init_time": CoordinateConfig(
                dtype="datetime64[ns]", chunks={"init_time": 1},
                long_name="Forecast initialization time",
                standard_name="forecast_reference_time",
            ),
            "lead_day": CoordinateConfig(
                dtype="int32", units="days", long_name="Forecast lead day (1-11)",
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
        chunks = {"init_time": 1, "lead_day": N_LEAD_DAYS, "y": 512, "x": 512}
        return [v.to_config(chunks) for v in variable_set(self.include_std)]
