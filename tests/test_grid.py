"""Tests for NBM grid geometry and idx parsing (no network)."""

from __future__ import annotations

from nbm_to_zarr.noaa.nbm import grid


def test_grid_shape() -> None:
    x, y = grid.grid_xy()
    assert x.shape == (grid.NX,)
    assert y.shape == (grid.NY,)


def test_grid_covers_conus() -> None:
    x, y = grid.grid_xy()
    lat, lon = grid.grid_latlon(x, y)
    assert lat.shape == (grid.NY, grid.NX)
    # CONUS bounds, roughly.
    assert 18 < float(lat.min()) < 22
    assert 53 < float(lat.max()) < 60
    assert -140 < float(lon.min()) < -120
    assert -65 < float(lon.max()) < -55


def test_spatial_ref_attrs() -> None:
    attrs = grid.spatial_ref_attrs()
    assert attrs["grid_mapping_name"] == "lambert_conformal_conic"
    assert "crs_wkt" in attrs


def test_grib_and_idx_urls() -> None:
    u = grid.grib_url("20250101", 0, 24)
    assert u.endswith("blend.20250101/00/core/blend.t00z.core.f024.co.grib2")
    assert grid.idx_url("20250101", 0, 24) == u + ".idx"


SAMPLE_IDX = """\
1:0:d=2025010100:TMP:2 m above ground:24 hour fcst:
2:1500000:d=2025010100:TMAX:2 m above ground:12-24 hour max fcst:
3:3000000:d=2025010100:APCP:surface:18-24 hour acc fcst:
4:4200000:d=2025010100:APCP:surface:prob >0.254:18-24 hour acc fcst:
5:5500000:d=2025010100:DSWRF:surface:24 hour fcst:
6:6800000:d=2025010100:TMP:2 m above ground:24 hour fcst:ens std dev:
"""


def test_parse_idx() -> None:
    entries = grid.parse_idx(SAMPLE_IDX)
    assert len(entries) == 6
    assert entries[0].var == "TMP"
    assert entries[0].level == "2 m above ground"
    assert entries[3].is_prob is True
    assert entries[5].is_ens_std is True


def test_message_byte_range() -> None:
    entries = grid.parse_idx(SAMPLE_IDX)
    start, end = grid.message_byte_range(entries, entries[0])
    assert start == 0
    assert end == 1500000 - 1
    # Last message has no upper bound.
    start, end = grid.message_byte_range(entries, entries[-1])
    assert end is None
