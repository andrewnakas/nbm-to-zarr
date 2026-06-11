"""NBM CONUS forecast dataset orchestrator."""

from __future__ import annotations

from nbm_to_zarr.base.dataset import Dataset
from nbm_to_zarr.noaa.nbm.forecast.region_job import NbmForecastRegionJob
from nbm_to_zarr.noaa.nbm.forecast.template_config import NbmForecastTemplateConfig


class NbmForecastDataset(Dataset):
    """NBM CONUS forecast dataset (per-init, daily lead days 1-11)."""

    def __init__(self, include_std: bool = False) -> None:
        self._tc = NbmForecastTemplateConfig(include_std=include_std)

    @property
    def template_config(self) -> NbmForecastTemplateConfig:
        return self._tc

    @property
    def region_job_class(self) -> type[NbmForecastRegionJob]:
        return NbmForecastRegionJob
