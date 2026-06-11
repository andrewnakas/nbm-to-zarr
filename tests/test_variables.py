"""Tests for NBM variable idx matching and aggregation rules."""

from __future__ import annotations

import numpy as np

from nbm_to_zarr.noaa.nbm.forecast.region_job import NbmForecastRegionJob, lead_hours_for_day
from nbm_to_zarr.noaa.nbm.variables import (
    CORE_VARIABLES,
    Aggregation,
    variable_set,
)


def _var(name: str):
    return next(v for v in variable_set(include_std=True) if v.name == name)


def test_tmean_matches_deterministic_only() -> None:
    tmean = _var("tmean")
    assert tmean.matches("TMP", "2 m above ground", "1:0:d=...:TMP:2 m above ground:24 hour fcst:")
    # Probabilistic line is a trap — must not match.
    assert not tmean.matches("TMP", "2 m above ground", "...:TMP:2 m above ground:prob >x:")
    # ens std dev belongs to a different channel.
    assert not tmean.matches("TMP", "2 m above ground", "...:TMP:2 m above ground:ens std dev:")


def test_tmean_std_matches_only_std() -> None:
    std = _var("tmean_std")
    assert std.matches("TMP", "2 m above ground", "...:TMP:2 m above ground:24 hour fcst:ens std dev:")
    assert not std.matches("TMP", "2 m above ground", "...:TMP:2 m above ground:24 hour fcst:")


def test_precip_selects_single_hour_increment() -> None:
    precip = _var("precip")
    # At f024, NBM publishes BOTH "23-24" (1 h) and "18-24" (6 h) acc windows.
    # We must pick only the 1-hour increment so the 24 hourly pieces tile the
    # day without double counting.
    assert precip.matches("APCP", "surface", "...:APCP:surface:23-24 hour acc fcst:", lead_hour=24)
    assert not precip.matches("APCP", "surface", "...:APCP:surface:18-24 hour acc fcst:", lead_hour=24)
    # Without a lead hour, an accumulation cannot be disambiguated -> no match.
    assert not precip.matches("APCP", "surface", "...:APCP:surface:23-24 hour acc fcst:")
    # Wrong variable form excluded.
    assert not precip.matches("APCP", "surface", "...:APCP:surface:24 hour fcst:", lead_hour=24)


def test_precip_hourly_windows_tile_a_day() -> None:
    # Simulate the real idx for each lead hour of day 1 having a 1 h increment
    # (and at 6/12/18/24 also a multi-hour cumulative window that must be
    # ignored). Exactly one match per lead hour, tiling [0,24].
    precip = _var("precip")
    matched_hours = []
    for lead in lead_hours_for_day(1):
        lines = [f"...:APCP:surface:{lead-1}-{lead} hour acc fcst:"]
        if lead in (6, 12, 18, 24):
            lines.append(f"...:APCP:surface:{lead-6}-{lead} hour acc fcst:")
        hits = [ln for ln in lines if precip.matches("APCP", "surface", ln, lead_hour=lead)]
        assert len(hits) == 1, f"f{lead}: expected 1 match, got {len(hits)}"
        matched_hours.append(lead)
    assert matched_hours == list(range(1, 25))


def test_tmax_tmin_windows() -> None:
    assert _var("tmax").window_contains == "max fcst"
    assert _var("tmin").window_contains == "min fcst"
    assert _var("tmax").aggregation is Aggregation.MAX
    assert _var("tmin").aggregation is Aggregation.MIN


def test_unit_conversions() -> None:
    # Temperature K -> degC.
    assert _var("tmean").offset == -273.15
    # APCP already mm — no scaling.
    assert _var("precip").scale == 1.0
    # DSWRF W/m^2 daily mean -> MJ/m^2/day.
    assert abs(_var("srad").scale - 0.0864) < 1e-9


def test_lead_hours_tile_day_one() -> None:
    hours = lead_hours_for_day(1)
    assert hours[0] > 0 and hours[-1] == 24
    assert all(0 < h <= 24 for h in hours)


def test_lead_hours_non_overlapping_across_days() -> None:
    seen: set[int] = set()
    for d in range(1, 12):
        for h in lead_hours_for_day(d):
            assert h not in seen, f"hour {h} appears in two lead-days"
            seen.add(h)


def test_aggregate_rules() -> None:
    fields = [np.array([[1.0, 2.0]]), np.array([[3.0, 0.0]])]
    assert np.allclose(NbmForecastRegionJob._aggregate(fields, _var("tmean")), [[2.0, 1.0]])
    assert np.allclose(NbmForecastRegionJob._aggregate(fields, _var("tmax")), [[3.0, 2.0]])
    assert np.allclose(NbmForecastRegionJob._aggregate(fields, _var("tmin")), [[1.0, 0.0]])
    assert np.allclose(NbmForecastRegionJob._aggregate(fields, _var("precip")), [[4.0, 2.0]])


def test_core_variables_count() -> None:
    assert len(CORE_VARIABLES) == 5
    assert len(variable_set(include_std=True)) == 6
