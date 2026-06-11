"""Region job for NBM CONUS forecast data.

For each init time, for each lead-day d (1-11), for each variable, this selects
the GRIB messages whose forecast windows tile the 24 h block
``[24*(d-1), 24*d]``, byte-range-fetches them, and aggregates them into one daily
value (mean/max/min/sum) on the Lambert grid.

NBM lead cadence: hourly to ~f036, 3-hourly to ~f192, 6-hourly to f264 (verify
per init via the idx; missing leads are logged and skipped, never fatal).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
from pydantic import ConfigDict

from nbm_to_zarr.base.region_job import (
    ProcessingRegion,
    RegionJob,
    SourceFileCoord,
)
from nbm_to_zarr.base.template_config import DataVariableConfig, TemplateConfig
from nbm_to_zarr.noaa.nbm import grid
from nbm_to_zarr.noaa.nbm.forecast.template_config import N_LEAD_DAYS
from nbm_to_zarr.noaa.nbm.variables import Aggregation, NbmVariable, variable_set

logger = logging.getLogger(__name__)

CYCLE = 0  # 00z only


def lead_hours_for_day(day: int) -> list[int]:
    """Forecast lead hours that fall within lead-day ``day``'s 24 h window.

    Uses NBM's native cadence: hourly to f036, then 3-hourly to f192, then
    6-hourly to f264. The exact set present is intersected with the idx at
    fetch time, so over-listing here is harmless.
    """
    lo, hi = 24 * (day - 1), 24 * day
    hours: list[int] = []
    for h in range(0, 265):
        if h < 36:
            step_ok = True
        elif h <= 192:
            step_ok = h % 3 == 0
        else:
            step_ok = h % 6 == 0
        if step_ok and lo < h <= hi:
            hours.append(h)
    return hours


class NbmForecastSourceFileCoord(SourceFileCoord):
    """One NBM CONUS GRIB file (one init + lead hour)."""

    model_config = ConfigDict(frozen=True)

    init_time: datetime
    lead_hour: int

    def url(self) -> str:
        return grid.grib_url(self.init_time.strftime("%Y%m%d"), CYCLE, self.lead_hour)

    def index_url(self) -> str | None:
        return grid.idx_url(self.init_time.strftime("%Y%m%d"), CYCLE, self.lead_hour)


class NbmForecastRegionJob(RegionJob):
    """Process one NBM forecast init into a daily-lead slab."""

    def __init__(self, *args: object, include_std: bool = False, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._include_std = include_std
        self._client = httpx.Client(follow_redirects=True)
        # init_time -> index along the region's append dimension.
        self._init_index = {
            np.datetime64(t).astype("datetime64[ns]"): i
            for i, t in enumerate(self._init_times())
        }
        # Cache of parsed idx per (init, lead_hour) within this region.
        self._idx_cache: dict[tuple[datetime, int], list[grid.IdxEntry] | None] = {}

    # --- region geometry ---------------------------------------------------

    def _init_times(self) -> list[datetime]:
        times: list[datetime] = []
        t = self.processing_region.append_start
        while t <= self.processing_region.append_end:
            times.append(t)
            t += timedelta(hours=24)
        return times

    def region_dim_coords(self) -> dict[str, np.ndarray]:
        x, y = grid.grid_xy()
        return {
            "init_time": np.array(self._init_times(), dtype="datetime64[ns]"),
            "lead_day": np.arange(1, N_LEAD_DAYS + 1, dtype="int32"),
            "y": y,
            "x": x,
        }

    def _nbm_vars(self) -> list[NbmVariable]:
        return variable_set(self._include_std)

    def _nbm_var_for(self, var: DataVariableConfig) -> NbmVariable:
        return next(v for v in self._nbm_vars() if v.name == var.name)

    # --- source listing ----------------------------------------------------

    def generate_source_file_coords(self) -> list[NbmForecastSourceFileCoord]:
        coords: list[NbmForecastSourceFileCoord] = []
        for init in self._init_times():
            leads: set[int] = set()
            for day in range(1, N_LEAD_DAYS + 1):
                leads.update(lead_hours_for_day(day))
            for lead in sorted(leads):
                coords.append(
                    NbmForecastSourceFileCoord(init_time=init, lead_hour=lead)
                )
        return coords

    # --- per-message reads (cached idx, byte-range message fetch) ----------

    def _get_idx(self, coord: NbmForecastSourceFileCoord) -> list[grid.IdxEntry] | None:
        key = (coord.init_time, coord.lead_hour)
        if key not in self._idx_cache:
            self._idx_cache[key] = grid.fetch_idx(self._client, coord.index_url() or "")
        return self._idx_cache[key]

    def _fetch_field(
        self, coord: NbmForecastSourceFileCoord, nbm_var: NbmVariable
    ) -> np.ndarray | None:
        entries = self._get_idx(coord)
        if entries is None:
            return None
        match = next(
            (
                e
                for e in entries
                if nbm_var.matches(e.var, e.level, e.raw, lead_hour=coord.lead_hour)
            ),
            None,
        )
        if match is None:
            return None
        start, end = grid.message_byte_range(entries, match)
        raw = grid.fetch_message_bytes(self._client, coord.url(), start, end)
        if raw is None:
            return None
        return grid.decode_message(raw)

    def read_data(
        self, var: DataVariableConfig, coord: NbmForecastSourceFileCoord
    ) -> None:
        """Unused: daily aggregation needs all leads of a day together, so this
        job overrides :meth:`process`. Defined only to satisfy the base class."""
        raise NotImplementedError

    # --- override process to aggregate per lead-day ------------------------

    def process(self):  # type: ignore[override]
        region_ds = self.build_region_dataset()
        # Pair each NBM variable spec with its on-disk DataVariableConfig (which
        # carries keepbits for the compression transform).
        var_pairs = list(zip(self._nbm_vars(), self.data_vars, strict=True))
        logger.info(
            "NBM forecast region %s..%s: %d inits × %d vars × %d lead-days",
            self.processing_region.append_start,
            self.processing_region.append_end,
            len(self._init_times()),
            len(var_pairs),
            N_LEAD_DAYS,
        )
        for init in self._init_times():
            init64 = np.datetime64(init).astype("datetime64[ns]")
            init_idx = self._init_index[init64]
            for day in range(1, N_LEAD_DAYS + 1):
                leads = lead_hours_for_day(day)
                for nbm_var, var_cfg in var_pairs:
                    fields: list[np.ndarray] = []
                    for lead in leads:
                        coord = NbmForecastSourceFileCoord(
                            init_time=init, lead_hour=lead
                        )
                        field = self._fetch_field(coord, nbm_var)
                        if field is not None:
                            fields.append(field)
                    if not fields:
                        continue
                    daily = self._aggregate(fields, nbm_var)
                    daily = daily * nbm_var.scale + nbm_var.offset
                    daily = self.apply_transformations(daily.astype(np.float32), var_cfg)
                    region_ds[nbm_var.name].values[init_idx, day - 1, :, :] = daily
        self._client.close()
        return region_ds

    @staticmethod
    def _aggregate(fields: list[np.ndarray], nbm_var: NbmVariable) -> np.ndarray:
        stack = np.stack(fields, axis=0)
        if nbm_var.aggregation is Aggregation.MEAN:
            return np.nanmean(stack, axis=0)
        if nbm_var.aggregation is Aggregation.MAX:
            return np.nanmax(stack, axis=0)
        if nbm_var.aggregation is Aggregation.MIN:
            return np.nanmin(stack, axis=0)
        if nbm_var.aggregation is Aggregation.SUM:
            return np.nansum(stack, axis=0)
        raise ValueError(f"Unknown aggregation {nbm_var.aggregation}")

    # --- job factories -----------------------------------------------------

    @classmethod
    def _make_job(
        cls,
        template_config: TemplateConfig,
        output_path: Path,
        start: datetime,
        end: datetime,
    ) -> NbmForecastRegionJob:
        include_std = getattr(template_config, "include_std", False)
        return cls(
            template_config=template_config,
            processing_region=ProcessingRegion(append_start=start, append_end=end),
            data_vars=template_config.data_vars,
            output_path=output_path,
            include_std=include_std,
        )

    @classmethod
    def operational_update_jobs(
        cls, template_config: TemplateConfig, output_path: Path
    ) -> list[NbmForecastRegionJob]:
        now = datetime.now(UTC).replace(tzinfo=None)
        init = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # NBM 00z full run is available with several hours' latency; if it's
        # early in the UTC day, fall back to yesterday's 00z.
        if now.hour < 8:
            init -= timedelta(days=1)
        return [cls._make_job(template_config, output_path, init, init)]

    @classmethod
    def backfill_jobs(
        cls,
        template_config: TemplateConfig,
        output_path: Path,
        start: datetime,
        end: datetime,
    ) -> list[NbmForecastRegionJob]:
        # One job per init (keeps memory bounded and makes resume granular).
        jobs: list[NbmForecastRegionJob] = []
        t = start
        while t <= end:
            jobs.append(cls._make_job(template_config, output_path, t, t))
            t += timedelta(hours=24)
        return jobs
