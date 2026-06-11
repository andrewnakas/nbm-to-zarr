"""Region job for NBM CONUS historical analysis data.

Produces a ``(time, y, x)`` best-estimate daily series. For each calendar day,
the analysis value is the lead-day-1 daily aggregate of that day's 00z init —
i.e. the same per-day aggregation the forecast job does for ``lead_day == 1``,
written to a flat ``time`` series instead of an ``(init_time, lead_day)`` grid.

The download/decode path (idx + byte-range + eccodes) is shared via
``nbm.grid``; only the output layout differs from the forecast job.
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
from nbm_to_zarr.base.template_config import TemplateConfig
from nbm_to_zarr.noaa.nbm import grid
from nbm_to_zarr.noaa.nbm.forecast.region_job import (
    NbmForecastRegionJob,
    lead_hours_for_day,
)
from nbm_to_zarr.noaa.nbm.variables import NbmVariable, variable_set

logger = logging.getLogger(__name__)

CYCLE = 0  # 00z only


class NbmAnalysisSourceFileCoord(SourceFileCoord):
    """One NBM CONUS GRIB file contributing to a day's analysis value."""

    model_config = ConfigDict(frozen=True)

    valid_date: datetime
    lead_hour: int

    def url(self) -> str:
        return grid.grib_url(self.valid_date.strftime("%Y%m%d"), CYCLE, self.lead_hour)

    def index_url(self) -> str | None:
        return grid.idx_url(self.valid_date.strftime("%Y%m%d"), CYCLE, self.lead_hour)


class NbmAnalysisRegionJob(RegionJob):
    """Process a span of analysis days into a (time, y, x) slab."""

    def __init__(self, *args: object, include_std: bool = False, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._include_std = include_std
        self._client = httpx.Client(follow_redirects=True)
        self._idx_cache: dict[tuple[datetime, int], list[grid.IdxEntry] | None] = {}

    # --- region geometry ---------------------------------------------------

    def _days(self) -> list[datetime]:
        days: list[datetime] = []
        t = self.processing_region.append_start
        while t <= self.processing_region.append_end:
            days.append(t)
            t += timedelta(days=1)
        return days

    def region_dim_coords(self) -> dict[str, np.ndarray]:
        x, y = grid.grid_xy()
        return {
            "time": np.array(self._days(), dtype="datetime64[ns]"),
            "y": y,
            "x": x,
        }

    def _nbm_vars(self) -> list[NbmVariable]:
        return variable_set(self._include_std)

    # --- source listing (lead-day-1 hours of each day's 00z init) ----------

    def generate_source_file_coords(self) -> list[NbmAnalysisSourceFileCoord]:
        coords: list[NbmAnalysisSourceFileCoord] = []
        for day in self._days():
            for lead in lead_hours_for_day(1):
                coords.append(
                    NbmAnalysisSourceFileCoord(valid_date=day, lead_hour=lead)
                )
        return coords

    # --- fetch (reuse forecast job's idx/byte-range helpers) ---------------

    def _get_idx(self, coord: NbmAnalysisSourceFileCoord) -> list[grid.IdxEntry] | None:
        key = (coord.valid_date, coord.lead_hour)
        if key not in self._idx_cache:
            self._idx_cache[key] = grid.fetch_idx(self._client, coord.index_url() or "")
        return self._idx_cache[key]

    def _fetch_field(
        self, coord: NbmAnalysisSourceFileCoord, nbm_var: NbmVariable
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

    def read_data(self, var, coord):  # type: ignore[override]
        """Unused — see :meth:`process` (daily aggregation is per-day)."""
        raise NotImplementedError

    # --- process: one best-estimate value per day --------------------------

    def process(self):  # type: ignore[override]
        region_ds = self.build_region_dataset()
        var_pairs = list(zip(self._nbm_vars(), self.data_vars, strict=True))
        day_index = {
            np.datetime64(d).astype("datetime64[ns]"): i
            for i, d in enumerate(self._days())
        }
        leads = lead_hours_for_day(1)
        logger.info(
            "NBM analysis region %s..%s: %d days × %d vars",
            self.processing_region.append_start,
            self.processing_region.append_end,
            len(self._days()),
            len(var_pairs),
        )
        for day in self._days():
            day_idx = day_index[np.datetime64(day).astype("datetime64[ns]")]
            for nbm_var, var_cfg in var_pairs:
                fields: list[np.ndarray] = []
                for lead in leads:
                    coord = NbmAnalysisSourceFileCoord(valid_date=day, lead_hour=lead)
                    field = self._fetch_field(coord, nbm_var)
                    if field is not None:
                        fields.append(field)
                if not fields:
                    continue
                daily = NbmForecastRegionJob._aggregate(fields, nbm_var)
                daily = daily * nbm_var.scale + nbm_var.offset
                daily = self.apply_transformations(daily.astype(np.float32), var_cfg)
                region_ds[nbm_var.name].values[day_idx, :, :] = daily
        self._client.close()
        return region_ds

    # --- job factories -----------------------------------------------------

    @classmethod
    def _make_job(
        cls,
        template_config: TemplateConfig,
        output_path: Path,
        start: datetime,
        end: datetime,
    ) -> NbmAnalysisRegionJob:
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
    ) -> list[NbmAnalysisRegionJob]:
        # The most recent fully-available analysis day (yesterday, UTC).
        now = datetime.now(UTC).replace(tzinfo=None)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        day = today - timedelta(days=1)
        return [cls._make_job(template_config, output_path, day, day)]

    @classmethod
    def backfill_jobs(
        cls,
        template_config: TemplateConfig,
        output_path: Path,
        start: datetime,
        end: datetime,
    ) -> list[NbmAnalysisRegionJob]:
        # One job per ~30-day month keeps each slab modest while limiting the
        # number of append operations.
        jobs: list[NbmAnalysisRegionJob] = []
        chunk = timedelta(days=30)
        t = start
        while t <= end:
            chunk_end = min(t + chunk - timedelta(days=1), end)
            jobs.append(cls._make_job(template_config, output_path, t, chunk_end))
            t = chunk_end + timedelta(days=1)
        return jobs
