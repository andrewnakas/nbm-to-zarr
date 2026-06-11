"""NBM daily-aggregate variable definitions, shared by both variants.

Each variable maps to one or more GRIB ``.idx`` lines and an aggregation rule
that turns the matched messages (across a 24 h lead-day window) into one daily
value. The selection patterns avoid the documented traps:

- ``prob`` lines are probabilistic — never deterministic (skipped in grid.py).
- temperatures take ``2 m above ground``; APCP / DSWRF take ``surface``.
- ``ens std dev`` lines are separate uncertainty channels (opt-in).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nbm_to_zarr.base.template_config import DataVariableConfig


class Aggregation(str, Enum):
    """How the messages tiling a 24 h lead-day combine into a daily value."""

    MEAN = "mean"  # average instantaneous fields over the day (tmean, srad)
    MAX = "max"  # daytime maximum (tmax)
    MIN = "min"  # overnight minimum (tmin)
    SUM = "sum"  # accumulate windowed amounts to a 24 h total (precip)


@dataclass(frozen=True)
class NbmVariable:
    """A daily-aggregate NBM variable and how to extract it from GRIB."""

    name: str
    # GRIB ``VAR`` token in the idx (e.g. "TMP", "APCP", "DSWRF").
    grib_var: str
    # GRIB ``level`` token (e.g. "2 m above ground", "surface").
    grib_level: str
    aggregation: Aggregation
    units: str
    long_name: str
    standard_name: str | None = None
    keepbits: int = 12
    # Multiply decoded values by this after aggregation (unit conversion).
    scale: float = 1.0
    # Add this after scaling (e.g. K -> degC is scale=1, offset=-273.15).
    offset: float = 0.0
    # If set, only match idx lines whose window token contains this fragment
    # (e.g. "max fcst" for TMAX, "acc fcst" for APCP).
    window_contains: str | None = None
    # True for ensemble std-dev channels (matched on "ens std dev" in extra).
    is_ens_std: bool = False
    # For accumulations (APCP): NBM publishes several overlapping windows at the
    # same lead (e.g. f024 has both "23-24" 1 h and "18-24" 6 h acc). To sum
    # cleanly to a 24 h total we pick exactly ONE non-overlapping tiling. When
    # set, match only the "(lead-N)-lead hour acc fcst" window of this width N.
    # We use N=6: a 6 h window exists at every 6-hourly lead f006..f174, so four
    # of them tile each 24 h lead-day exactly — and, unlike the 1 h windows,
    # they have no gaps in NBM's 3-hourly lead range (f039+).
    accum_window_hours: int | None = None

    def to_config(self, chunks: dict[str, int]) -> DataVariableConfig:
        return DataVariableConfig(
            name=self.name,
            chunks=chunks,
            units=self.units,
            long_name=self.long_name,
            standard_name=self.standard_name,
            keepbits=self.keepbits,
        )

    def matches(self, var: str, level: str, raw: str, lead_hour: int | None = None) -> bool:
        """True if a parsed idx line corresponds to this variable.

        ``lead_hour`` is required for accumulation variables so the exact window
        ("(lead - accum_window_hours)-lead hour acc fcst") can be selected.
        """
        if var != self.grib_var or level != self.grib_level:
            return False
        if "prob" in raw.lower():
            return False
        if self.is_ens_std != ("ens std dev" in raw.lower()):
            return False
        if self.window_contains and self.window_contains not in raw:
            return False
        if self.accum_window_hours is not None:
            if lead_hour is None:
                return False
            window = f"{lead_hour - self.accum_window_hours}-{lead_hour} hour acc fcst"
            if window not in raw:
                return False
        return True

    def accum_leads_for_day(self, day: int) -> list[int] | None:
        """For accumulation vars, the leads whose windows tile lead-day ``day``.

        Returns the 6-hourly leads (e.g. day 1 -> [6, 12, 18, 24]) or None for
        non-accumulation variables.
        """
        if self.accum_window_hours is None:
            return None
        n = self.accum_window_hours
        lo, hi = 24 * (day - 1), 24 * day
        return list(range(lo + n, hi + 1, n))


# Core daily-aggregate variables (the bottom rung — matches the plan §1).
CORE_VARIABLES: list[NbmVariable] = [
    NbmVariable(
        name="tmean",
        grib_var="TMP",
        grib_level="2 m above ground",
        aggregation=Aggregation.MEAN,
        units="degC",
        long_name="Daily mean 2 m temperature",
        standard_name="air_temperature",
        scale=1.0,
        offset=-273.15,
    ),
    NbmVariable(
        name="tmax",
        grib_var="TMAX",
        grib_level="2 m above ground",
        aggregation=Aggregation.MAX,
        units="degC",
        long_name="Daily maximum 2 m temperature",
        standard_name="air_temperature",
        window_contains="max fcst",
        scale=1.0,
        offset=-273.15,
    ),
    NbmVariable(
        name="tmin",
        grib_var="TMIN",
        grib_level="2 m above ground",
        aggregation=Aggregation.MIN,
        units="degC",
        long_name="Daily minimum 2 m temperature",
        standard_name="air_temperature",
        window_contains="min fcst",
        scale=1.0,
        offset=-273.15,
    ),
    NbmVariable(
        name="precip",
        grib_var="APCP",
        grib_level="surface",
        aggregation=Aggregation.SUM,
        units="mm",
        long_name="Daily total precipitation",
        standard_name="precipitation_amount",
        window_contains="acc fcst",
        accum_window_hours=6,
        keepbits=10,
        # APCP is already kg/m^2 == mm; no rate conversion (unlike GFS).
    ),
    NbmVariable(
        name="srad",
        grib_var="DSWRF",
        grib_level="surface",
        aggregation=Aggregation.MEAN,
        units="MJ m-2 day-1",
        long_name="Daily mean downward shortwave radiation as daily energy",
        standard_name="surface_downwelling_shortwave_flux_in_air",
        keepbits=10,
        # Daily-mean W/m^2 -> MJ/m^2/day: x 0.0864.
        scale=0.0864,
    ),
]

# Optional ensemble-spread uncertainty channels (the plan's *_std, opt-in).
STD_VARIABLES: list[NbmVariable] = [
    NbmVariable(
        name="tmean_std",
        grib_var="TMP",
        grib_level="2 m above ground",
        aggregation=Aggregation.MEAN,
        units="degC",
        long_name="Daily mean 2 m temperature ensemble standard deviation",
        keepbits=10,
        is_ens_std=True,
        # std dev of Kelvin == std dev of Celsius (no offset).
    ),
]


def variable_set(include_std: bool) -> list[NbmVariable]:
    return CORE_VARIABLES + (STD_VARIABLES if include_std else [])
