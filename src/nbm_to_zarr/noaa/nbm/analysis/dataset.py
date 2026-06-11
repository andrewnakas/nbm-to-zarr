"""NBM CONUS historical analysis dataset orchestrator."""

from __future__ import annotations

from nbm_to_zarr.base.dataset import Dataset
from nbm_to_zarr.noaa.nbm.analysis.region_job import NbmAnalysisRegionJob
from nbm_to_zarr.noaa.nbm.analysis.template_config import NbmAnalysisTemplateConfig


class NbmAnalysisDataset(Dataset):
    """NBM CONUS historical analysis dataset (daily best estimate)."""

    def __init__(self, include_std: bool = False) -> None:
        self._tc = NbmAnalysisTemplateConfig(include_std=include_std)

    @property
    def template_config(self) -> NbmAnalysisTemplateConfig:
        return self._tc

    @property
    def region_job_class(self) -> type[NbmAnalysisRegionJob]:
        return NbmAnalysisRegionJob
