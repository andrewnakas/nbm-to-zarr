"""Tests for forecast/analysis template configs and bit rounding."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from nbm_to_zarr.base.region_job import RegionJob
from nbm_to_zarr.noaa.nbm.analysis import NbmAnalysisDataset
from nbm_to_zarr.noaa.nbm.forecast import NbmForecastDataset


def test_forecast_template_structure() -> None:
    tc = NbmForecastDataset().template_config
    assert tc.dimensions == ("init_time", "lead_day", "y", "x")
    assert tc.append_dim == "init_time"
    ds = tc.get_template(datetime(2025, 1, 3))
    assert dict(ds.sizes)["init_time"] == 3
    assert dict(ds.sizes)["lead_day"] == 11
    assert "valid_date" in ds.coords
    # valid_date convention: lead_day 1 of a 00z init == the init's own day.
    assert str(ds.valid_date.values[0, 0])[:10] == "2025-01-01"


def test_analysis_template_structure() -> None:
    tc = NbmAnalysisDataset().template_config
    assert tc.dimensions == ("time", "y", "x")
    assert tc.append_dim == "time"
    ds = tc.get_template(datetime(2025, 1, 5))
    assert dict(ds.sizes)["time"] == 5
    assert "latitude" in ds.coords and "longitude" in ds.coords
    assert ds.spatial_ref.attrs["grid_mapping_name"] == "lambert_conformal_conic"


def test_dataset_ids() -> None:
    assert NbmForecastDataset().dataset_id == "noaa-nbm-conus-forecast"
    assert NbmAnalysisDataset().dataset_id == "noaa-nbm-conus-analysis"


def test_include_std_adds_channel() -> None:
    base = NbmAnalysisDataset(include_std=False).template_config.data_vars
    std = NbmAnalysisDataset(include_std=True).template_config.data_vars
    assert len(std) == len(base) + 1
    assert any(v.name == "tmean_std" for v in std)


def test_bit_rounding_preserves_nan_and_shrinks() -> None:
    data = np.array([1.23456789, np.nan, -42.5, np.inf], dtype=np.float32)
    rounded = RegionJob._apply_bit_rounding(data, keepbits=10)
    assert np.isnan(rounded[1])
    assert np.isinf(rounded[3])
    # Value is close but not bit-identical (precision dropped).
    assert abs(rounded[0] - data[0]) < 0.01


def test_bit_rounding_full_precision_noop() -> None:
    data = np.array([1.23456789], dtype=np.float32)
    assert RegionJob._apply_bit_rounding(data, keepbits=23)[0] == data[0]
