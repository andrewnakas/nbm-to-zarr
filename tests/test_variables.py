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


def test_precip_selects_6hour_window() -> None:
    precip = _var("precip")
    # At f024, NBM publishes "23-24" (1 h) and "18-24" (6 h). We take the 6 h.
    assert precip.matches("APCP", "surface", "...:APCP:surface:18-24 hour acc fcst:", lead_hour=24)
    assert not precip.matches("APCP", "surface", "...:APCP:surface:23-24 hour acc fcst:", lead_hour=24)
    # Without a lead hour, an accumulation cannot be disambiguated -> no match.
    assert not precip.matches("APCP", "surface", "...:APCP:surface:18-24 hour acc fcst:")
    # Wrong variable form excluded.
    assert not precip.matches("APCP", "surface", "...:APCP:surface:24 hour fcst:", lead_hour=24)


def test_precip_6hour_windows_tile_each_day() -> None:
    # Four 6 h windows tile each 24 h lead-day exactly (day 1 -> [6,12,18,24]).
    precip = _var("precip")
    for day in range(1, 8):
        leads = precip.accum_leads_for_day(day)
        assert leads == [24 * (day - 1) + 6 * k for k in range(1, 5)]
        # Each lead's 6 h window matches exactly; covered hours tile the day.
        covered = []
        for lead in leads:
            line = f"...:APCP:surface:{lead-6}-{lead} hour acc fcst:"
            assert precip.matches("APCP", "surface", line, lead_hour=lead)
            covered.extend(range(lead - 6, lead))
        assert covered == list(range(24 * (day - 1), 24 * day))


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
    for d in range(1, 8):  # NBM core covers 7 full lead-days
        for h in lead_hours_for_day(d):
            assert h not in seen, f"hour {h} appears in two lead-days"
            seen.add(h)


def test_lead_cadence_matches_real_nbm_core() -> None:
    # Hourly f001-f036, then 3-hourly f039-f177 (no f037/f038).
    from nbm_to_zarr.noaa.nbm.forecast.region_job import AVAILABLE_LEADS

    assert 36 in AVAILABLE_LEADS
    assert 37 not in AVAILABLE_LEADS
    assert 38 not in AVAILABLE_LEADS
    assert 39 in AVAILABLE_LEADS
    assert AVAILABLE_LEADS[-1] == 177
    # Day 7 fully closes (needs f168). Day 8 would need f192 (absent), so it is
    # only partially covered (f171/f174/f177) — which is why we keep 7 days.
    assert 168 in AVAILABLE_LEADS
    assert max(lead_hours_for_day(7)) == 168
    assert lead_hours_for_day(7) == list(range(147, 169, 3))
    assert lead_hours_for_day(8) == [171, 174, 177]  # partial -> excluded
    assert max(AVAILABLE_LEADS) < 192  # day 8 never fully closes


def test_aggregate_rules() -> None:
    fields = [np.array([[1.0, 2.0]]), np.array([[3.0, 0.0]])]
    assert np.allclose(NbmForecastRegionJob._aggregate(fields, _var("tmean")), [[2.0, 1.0]])
    assert np.allclose(NbmForecastRegionJob._aggregate(fields, _var("tmax")), [[3.0, 2.0]])
    assert np.allclose(NbmForecastRegionJob._aggregate(fields, _var("tmin")), [[1.0, 0.0]])
    assert np.allclose(NbmForecastRegionJob._aggregate(fields, _var("precip")), [[4.0, 2.0]])


def test_core_variables_count() -> None:
    assert len(CORE_VARIABLES) == 5
    assert len(variable_set(include_std=True)) == 6
