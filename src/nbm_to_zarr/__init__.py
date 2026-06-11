"""NBM to Zarr — reformat NOAA National Blend of Models into Zarr.

Follows the dynamical.org reformatter architecture (TemplateConfig / RegionJob /
Dataset), with two dataset variants that mirror dynamical's forecast and analysis
products:

- ``noaa-nbm-conus-forecast``  — per-init forecast, daily-aggregated lead days 1-7
- ``noaa-nbm-conus-analysis``  — best-estimate-at-valid-time daily series
"""

__version__ = "0.1.0"
