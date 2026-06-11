# Plan: NBM → Zarr reformatter (one-year archive, dynamical.org-style)

**For: a fresh Claude session. This document is self-contained — no other context
needed. Ask the user only for the hosting decision (§5) and an HF token if
HuggingFace is chosen.**

## Context

RiverWatch2 (this repo) runs an LSTM streamflow forecaster whose decoder is
driven by weather *forecasts*. We already extract archived GFS (0.25°) and HRRR
(3 km, ≤48 h) forecasts at gauge points from dynamical.org's zarr stores
(`scripts/fetch_gfs_forcings.py`, `scripts/fetch_hrrr_forcings.py`). The gap:
no high-resolution forecast source beyond 48 h. NOAA's **NBM (National Blend of
Models)** fills it — 2.5 km CONUS, calibrated blend, leads to 264 h (11 days) —
but exists only as per-init GRIB2 on AWS, which nobody has reformatted to zarr
(dynamical.org doesn't carry it). This project builds that reformatter and
publishes a one-year (2025) NBM forecast zarr, both as a public artifact and as
the decoder-forcing source for RiverWatch2's next experiment.

Important honesty note baked into the README you'll write: past day ~3, NBM's
2.5 km grid carries statistically downscaled *global-ensemble* information —
terrain-aware texture and calibration, not fresh 3 km physics. That's still the
best available product of its kind; just don't oversell it.

## Verified facts (checked 2026-06-11 — trust these, they're load-bearing)

- Bucket: `https://noaa-nbm-grib2-pds.s3.amazonaws.com/` (public, no auth).
  Layout: `blend.YYYYMMDD/CC/core/blend.tCCz.core.fNNN.{co,ak,hi,pr,gu}.grib2`
  plus `.idx` sidecars. CONUS (`co`) file ≈ 160 MB, idx ≈ 21 KB / ~163 messages.
- Confirmed present in a 2025 CONUS core idx (f024): `TMP:2 m above ground`,
  `TMAX:2 m above ground:12-24 hour max fcst`, `APCP:surface:<window> acc fcst`
  (deterministic lines have NO `prob` suffix — the prob lines are a trap),
  `DSWRF:surface` (yes, NBM has shortwave — earlier assumptions it didn't are
  wrong), and `ens std dev` variants (TMP, DPT, TMAX…) — free uncertainty
  channels worth including.
- APCP accumulation windows VARY BY LEAD (at f024 you'll see `23-24`, `18-24`,
  `12-24 hour acc` deterministic lines). You must pick non-overlapping windows
  that tile each 24 h lead-day exactly — write an inventory step first (§3.1).
- TMIN appears at leads whose 12 h min-window closes there (expect ~f036 for
  day-1 overnight); confirm via inventory, don't assume.
- Lead hours: hourly to ~f036, 3-hourly to ~f192, 6-hourly to f264. Verify by
  listing one init (`?list-type=2&prefix=blend.20250101/00/core/`).
- Grid: Lambert conformal ~2.54 km, ~2345×1597. Read the true CRS/2D lat-lon
  from the first decoded message; store them as coords like dynamical's HRRR
  zarr does (`y`, `x` dims + 2D `latitude`/`longitude` + `spatial_ref`).
- 2025 is a single NBM version era (v4.2+) — no version seam. That's why the
  year is 2025 and not earlier.
- dynamical.org's pipeline is open source: `github.com/dynamical-org/reformatters`
  (BSD-3, has `dataset_integration_guide.md`). Build standalone-first (their
  framework assumes their k8s operational setup); structure code so a later
  upstream PR is easy. Mention in README; email/issue to dynamical offering it
  is encouraged.

## 1. Product definition (the size ladder — build bottom rung first)

All variants: 00z cycle only, CONUS only, calendar 2025, **daily-aggregated
lead days 1–11** (the consumer is daily; native hourly leads would be ~1–2 TB).

Variables (daily aggregates from GRIB messages):
`tmean` (TMP 2m → daily mean, °C), `tmax`/`tmin` (TMAX/TMIN 12 h windows, °C),
`precip` (APCP windows tiled to 24 h sum, mm), `srad` (DSWRF → daily mean
W/m² × 0.0864 = MJ/m²/day), plus `tmean_std` (TMP ens std dev, daily mean) if
budget allows.

- **Rung A — weekly inits (52)**: ~8–12 GB zarr. Proves the pipeline end-to-end,
  uploads in hours. SHIP THIS FIRST.
- **Rung B — daily inits (365)**: ~60–90 GB. Same code, longer backfill
  (~500 GB streamed, ~1–2 days). Append along `init_time` — rung A's store
  grows into rung B; design chunking for that from day one.
- **Rung C (optional, RiverWatch2-specific)**: point-extracted csv.gz at the
  1,893 gauges (mirror `riverwatch2/data/mblstm/gfs_fcst/` format exactly:
  `station_id, lead_day, temperature_2m_mean, temperature_2m_max,
  temperature_2m_min, precipitation_sum, shortwave_radiation_sum`, one file per
  init named `<YYYY-MM-DD>.csv.gz`, station_id as zero-padded string). ~150 MB
  for the year — this one DOES fit in a GitHub repo.

## 2. Repo + hosting

- New public GitHub repo (suggest `nbm-zarr`): code, README, manifest. The
  zarr itself does NOT go on GitHub — 8–90 GB of many small chunk files is the
  wrong shape for git (100 MB file cap, ~5 GB repo practicality, LFS costs).
- Zarr hosting (ask the user to pick; recommend the first):
  1. **HuggingFace dataset repo** (free, public, TB-scale OK; needs user's HF
     account + write token; upload via `huggingface_hub.HfApi.upload_large_folder`).
  2. Cloudflare R2 (free egress; ~$1.5/100 GB-month; needs account + key).
  3. Source Cooperative (where dynamical hosts; requires an application —
     right long-term home, wrong for this week).
- Attribution: NOAA NBM is public domain; the derived zarr CC-BY-4.0 with
  credit to NOAA + AWS Open Data Program; note the dynamical.org-inspired
  layout.

## 3. Pipeline (standalone Python; ~400 lines total is the right size)

Dependencies: `requests`, `numpy`, `pandas`, `xarray`, `zarr>=3`, `eccodes`
(decode GRIB messages from bytes; `cfgrib` works but is file-oriented — with
byte-range reads, raw `eccodes.codes_new_from_message` is simpler).

### 3.1 `inventory.py` — element map (DO THIS FIRST)
For one init (2025-01-01 00z), fetch every lead's `.idx`, parse
`msg#:byte_offset:date:VAR:level:window:extra`, and emit a table of which
(VAR, window) deterministic lines exist at which lead hour. From it, hard-code
the **message-selection spec**: for each lead-day d (1..11) and variable, the
exact (lead_hour, idx-line-pattern) list whose windows tile [24(d-1), 24d].
Unit test this spec against a second init (mid-year, e.g. 2025-07-01) to catch
seasonal/version drift. Watch for: `prob` suffixed lines (skip), `ens std dev`
(separate channel), duplicate VAR at different levels (take `2 m above ground`
for temps, `surface` for APCP/DSWRF).

### 3.2 `fetch.py` — byte-range GRIB reader
`idx` line N's offset and line N+1's offset bound message N → one HTTP
`Range: bytes=a-b` GET per needed message (~5–15 MB each; ~40–60 messages
≈ 1–1.5 GB per init). Decode with eccodes from bytes; reshape to (y, x) using
Ny/Nx from the message. Retry w/ backoff; a missing init or lead → log to
`manifest.json` and skip (NBM has occasional gaps — do not fail the run).

### 3.3 `build.py` — aggregate + write zarr
Per init: assemble `(lead_day: 11, y, x)` float32 arrays per variable, apply
unit conversions (K→°C, W/m²→MJ/m²/day ×0.0864; APCP is already mm — kg/m² ≡ mm,
NO rate conversion unlike GFS), then append one `init_time` slab to the zarr
store. Chunking: `init_time=1, lead_day=11, y=~512, x=~512`, blosc-zstd-3.
Coordinates: `init_time`, `lead_day` (1..11), 2D `latitude`/`longitude`,
`spatial_ref` attrs, and a `valid_date = init_date + lead_day - 1` convention
documented in attrs (**a 00z init on day D has lead-day 1 = calendar day D**
— this matches RiverWatch2's GFS/HRRR fetchers; issue date t0 = D−1).
Resumable: on start, read existing `init_time` values and skip done inits.
Disk hygiene: write locally under a scratch dir, upload-and-prune every N
inits if local disk < 50 GB free (this machine has had disk-full incidents —
check `df` before starting and keep ≥20 GB headroom at all times).

### 3.4 `upload.py` — push to host
HF: `upload_large_folder` (handles resume/dedup). Run after rung A completes,
then incrementally (HF dedups unchanged chunks). Verify post-upload by opening
the remote zarr with `xarray.open_zarr` + spot-checking one field.

### 3.5 Validation gate (before announcing anything)
- Open the published zarr fresh; check dims, dtypes, no all-NaN slabs.
- Cross-check ~50 random (init, station, lead-day) samples against
  RiverWatch2's existing extractions: GFS csvs (`data/mblstm/gfs_fcst/`) for
  the same dates — correlation should be high (temps r>0.95) but NOT identical
  (different models); and against Daymet observations in
  `data/mblstm/corpus/<id>.csv.gz` at lead-day 1 (temps r≈0.95+, precip scale
  within ~2×). Sanity ranges: tmean −40..45 °C, precip 0..400 mm/day,
  srad 0..35 MJ/m²/day.
- README with: coverage, variables/units, the valid-date convention, the
  honesty note from Context, a 10-line xarray usage example, and attribution.

## 4. RiverWatch2 consumption (small follow-up PR in this repo)

`scripts/fetch_nbm_forcings.py`: open the published zarr, KDTree the 1,893
stations from `data/stations_40_enriched.json` onto the Lambert grid (copy the
pattern in `scripts/fetch_hrrr_forcings.py::grid_index` verbatim), write
`data/mblstm/nbm_fcst/<init>.csv.gz` in the exact GFS csv schema. Then the
existing machinery (`backtest_mblstm.py --gfs`, `train_mblstm.py
--gfs-finetune`) gains an NBM source with ~20-line changes (a `--src-dir`
style flag), and the hybrid overlay logic (HRRR d1-2) extends to
NBM d1-11 + GEFS/GFS d12-14.

## 5. Open decisions for the user (ask before starting)

1. Hosting: HF dataset repo (recommended) / R2 / Source Coop application?
2. Include `*_std` uncertainty channels (≈ +20–40 % size)? Recommend yes at
   rung A, decide for B from measured size.
3. Repo name + whether to file the dynamical.org issue offering upstreaming.

## Effort estimate

Inventory + fetch + build: ~1 day of focused work. Rung A backfill: ~3–6 h
streaming (52 inits × ~1.5 GB at ~30 MB/s) + upload. Rung B: ~1–2 days
unattended backfill. Biggest schedule risks: APCP window-tiling logic (§3.1)
and eccodes install friction (use `brew install eccodes` / conda-forge wheel).
