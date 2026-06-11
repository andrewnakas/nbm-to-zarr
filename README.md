# NBM to Zarr

> Reformat NOAA's **National Blend of Models (NBM)** into Zarr — a **forecast**
> product and a matching **historical analysis** product — following the
> [dynamical.org](https://dynamical.org) reformatter architecture. The short
> analysis sample is hosted **for free, directly in this GitHub repo**.

NBM is NOAA's 2.5 km calibrated CONUS blend with leads out to 264 h (11 days),
but it only exists as per-init GRIB2 on AWS — nobody has reformatted it to Zarr
(dynamical.org doesn't carry it). This project does, with the **same
`TemplateConfig` / `RegionJob` / `Dataset` structure** dynamical uses, so both
the forecast and the analysis variants are drop-in familiar.

## Two datasets, both dynamical-style

| Dataset ID | Layout | What it is |
|---|---|---|
| `noaa-nbm-conus-forecast` | `(init_time, lead_day, y, x)` | Per-init forecast, daily-aggregated lead days 1–11 |
| `noaa-nbm-conus-analysis` | `(time, y, x)` | Daily **best-estimate-at-valid-time** series |

Both share the native NBM Lambert-conformal grid (`y`, `x` + 2D
`latitude`/`longitude` + `spatial_ref`), the same variables, the same GRIB
byte-range fetch path, and the same `eccodes` decode — exactly mirroring how
dynamical exposes paired forecast/analysis products (e.g. `noaa-gfs-forecast` ↔
`noaa-gfs-analysis`).

**Variables** (daily aggregates): `tmean`, `tmax`, `tmin` (°C), `precip` (mm),
`srad` (MJ/m²/day), and optional `tmean_std` (ensemble spread, `--include-std`).

### How the analysis is constructed

NBM is forecast-only, so the analysis "best estimate" for calendar day *D* is
**lead-day 1 of that day's 00z init** (the shortest lead valid on the day). This
is the standard dynamical construction for deriving an analysis series from a
forecast-only model, and it's recorded in the dataset's
`analysis_construction` attribute.

### Honesty note (baked into the dataset attrs)

Past ~day 3, NBM's 2.5 km grid carries statistically downscaled *global-ensemble*
information — terrain-aware texture and calibration, not fresh 2.5 km physics.
It's still the best available product of its kind; we just don't oversell it.

## Hosting: a short analysis dataset, free on GitHub

You asked whether the analysis data could live somewhere free instead of an
external object store. **Yes — and it's already wired up.** The full-resolution
NBM grid is ~75 MB/day, too big to keep many days of in git. But a **spatially
coarsened, short** analysis Zarr is a few MB, which fits comfortably in a git
repo (well under GitHub's 100 MB/file cap), with **no object store, no Git LFS,
and no egress bills**.

```bash
# Build the committed sample (synthetic by default — runs anywhere, no GRIB libs)
python scripts/build_sample_analysis.py --days 14 --coarsen 10
# → data/noaa-nbm-conus-analysis-sample.zarr  (~3.6 MB, committed to the repo)

# Build a REAL coarsened sample (needs network + eccodes)
python scripts/build_sample_analysis.py --real --days 14 --coarsen 10
```

The committed sample ships **synthetic values with a real schema/coords** so the
repo always has a working, openable demo and tests run offline; the `data_status`
attr says which it is. A GitHub Actions job rebuilds it daily, regenerates the
catalog, and (via GitHub Pages) publishes a discoverable catalog page.

> Where to host the *full* 1-year archive (8–90 GB) when you want it: a
> HuggingFace dataset repo, Cloudflare R2, or Source Cooperative (where
> dynamical hosts). GitHub is the right home only for the **short** sample —
> which is exactly what this repo commits.

```python
import xarray as xr

# Open the GitHub-hosted short analysis sample
url = "https://raw.githubusercontent.com/andrewnakas/nbm-to-zarr/main/data/noaa-nbm-conus-analysis-sample.zarr"
ds = xr.open_zarr(url)            # or a local path
print(ds)                         # (time, y, x) with tmean/tmax/tmin/precip/srad
ds["tmean"].isel(time=0).plot()   # terrain-aware CONUS field
```

## Architecture

```
src/nbm_to_zarr/
├── base/                     # dynamical-style framework (shared)
│   ├── template_config.py    #   dimensions, coords, variables, get_template()
│   ├── region_job.py         #   fetch → decode → transform → assemble slab
│   └── dataset.py            #   operational_update() + backfill() (append-grows)
└── noaa/nbm/
    ├── grid.py               # Lambert grid, idx parsing, byte-range GRIB fetch
    ├── variables.py          # daily-aggregate var specs + idx matching rules
    ├── forecast/             # (init_time, lead_day, y, x)
    │   ├── template_config.py
    │   ├── region_job.py
    │   └── dataset.py
    └── analysis/             # (time, y, x) best estimate
        ├── template_config.py
        ├── region_job.py
        └── dataset.py
```

The forecast and analysis variants differ **only in how they aggregate** the
fetched fields (per `(init, lead_day)` vs. per valid day) — the grid, fetch, and
decode code is shared in `noaa/nbm/`.

## CLI

```bash
pip install -e .

nbm list-datasets
nbm info noaa-nbm-conus-analysis

# Materialize the empty schema (template) for either dataset
nbm update-template noaa-nbm-conus-forecast --end 2025-12-31

# Operational update (latest cycle/day) — needs eccodes
nbm operational-update noaa-nbm-conus-analysis -o data/nbm-analysis.zarr

# Backfill a historical range (grows the store along the append dim; resumable)
nbm backfill noaa-nbm-conus-analysis -o data/nbm-analysis.zarr \
    --start 2025-01-01 --end 2025-01-31
```

`eccodes` is only needed to decode real GRIB. The template, sample (synthetic),
catalog, and tests all run without it.

## Data source

- **Bucket** (public, no auth): `https://noaa-nbm-grib2-pds.s3.amazonaws.com/`
- **Layout**: `blend.YYYYMMDD/CC/core/blend.tCCz.core.fNNN.co.grib2` (+ `.idx`)
- **Grid**: Lambert conformal ~2.54 km, 2345 × 1597, CONUS

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # 21 tests, no network/GRIB needed
ruff check src/
```

## Credits & license

- Architecture inspired by [dynamical.org/reformatters](https://github.com/dynamical-org/reformatters)
- Data: NOAA NBM via the [AWS Open Data Program](https://registry.opendata.aws/) (public domain)
- Code: MIT. Derived Zarr data: CC-BY-4.0 — please credit NOAA + AWS Open Data.
