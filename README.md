# NBM to Zarr

> Reformat NOAA's **National Blend of Models (NBM)** into Zarr — a **forecast**
> product and a matching **historical analysis** product — following the
> [dynamical.org](https://dynamical.org) reformatter architecture. The short
> full-res analysis sample is hosted **for free on GitHub** (the `data` branch).

NBM is NOAA's 2.5 km calibrated CONUS blend; its `core` product runs to f177
(~7.4 days), but it only exists as per-init GRIB2 on AWS — nobody has reformatted it to Zarr
(dynamical.org doesn't carry it). This project does, with the **same
`TemplateConfig` / `RegionJob` / `Dataset` structure** dynamical uses, so both
the forecast and the analysis variants are drop-in familiar.

## Two datasets, both dynamical-style

| Dataset ID | Layout | What it is |
|---|---|---|
| `noaa-nbm-conus-forecast` | `(init_time, lead_day, y, x)` | Per-init forecast, daily-aggregated lead days 1–7 |
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

## Hosting: a short, full-resolution analysis dataset, free on GitHub

The analysis data lives **on GitHub itself** — no external object store. We keep
it at **full native resolution** (~2.5 km, 2345 × 1597) but **short**: a rolling
**last 7 days** (~110 MB of real data). Zarr splits each field into many small
zstd-compressed chunk files (512 × 512, each well under 1 MB), so the per-file
100 MB cap is never near — **no object store, no Git LFS, no egress bills**.

To avoid growing `main`'s history, CI force-pushes the store to a dedicated
**`data` branch** as a single fresh commit each day (one snapshot, no
accumulation). `main` stays code + catalog only. Open the data straight from
that branch's raw URL (see the example below).

```bash
# Build the committed sample (synthetic by default — runs anywhere, no GRIB libs)
python scripts/build_sample_analysis.py --days 7
# → data/noaa-nbm-conus-analysis-sample.zarr  (full-res, last 7 days)

# Build it from REAL NBM data (needs network + eccodes)
python scripts/build_sample_analysis.py --real --days 7 --end 2026-06-10
```

The synthetic build ships **real schema/coords/grid with generated values**, so
tests run offline and you can produce a store anywhere; the `data_status` attr
says which it is. The GitHub Actions job builds the **real** sample daily,
force-pushes it to the `data` branch, regenerates the catalog, and publishes a
Pages catalog. Each run *replaces* the store (rolling window), so it stays bounded.

**Long-term archive:** the full multi-year analysis lives in a HuggingFace
dataset repo, not GitHub. See **[RUNNING_THE_BACKFILL.md](RUNNING_THE_BACKFILL.md)**
to run the resumable week-by-week backfill to
[`nakas/nbm-conus-analysis`](https://huggingface.co/datasets/nakas/nbm-conus-analysis)
(48 weeks done, ~250 remaining to the 2020-10 NBM data floor).

> The **forecast** store — `(init_time, lead_day 1-7, y, x)`, ~7× the analysis
> size — is too large for git. Build it with `scripts/build_forecast.py` to a
> gitignored path and host it externally (HuggingFace / R2 / Source Cooperative)
> when you want the full archive.

```python
import xarray as xr

# Open the GitHub-hosted short analysis sample
url = "https://raw.githubusercontent.com/andrewnakas/nbm-to-zarr/data/data/noaa-nbm-conus-analysis-sample.zarr"
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
pytest -q          # 23 tests, no network/GRIB needed
ruff check src/
```

## Credits & license

- Architecture inspired by [dynamical.org/reformatters](https://github.com/dynamical-org/reformatters)
- Data: NOAA NBM via the [AWS Open Data Program](https://registry.opendata.aws/) (public domain)
- Code: MIT. Derived Zarr data: CC-BY-4.0 — please credit NOAA + AWS Open Data.
