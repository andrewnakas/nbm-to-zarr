# Running the NBM analysis backfill (on another machine, e.g. openclaw)

The backfill builds NBM CONUS daily analysis **week by week**, uploads each week
to the HuggingFace dataset repo `nakas/nbm-conus-analysis`, then prunes the local
copy — so it only needs ~1 week (~170 MB) of disk at a time. It's **resumable**:
it checks HuggingFace and skips any week already uploaded.

**Current state (2026-06-11):** 48 weeks done, `2025-W29 … 2026-W24` (~11 months).
~250 weeks remain to reach the data floor (NBM on AWS starts **2020-10-01**).
Running the command below picks up exactly where it left off.

## Prerequisites

- Python 3.12+ (3.14 works)
- **eccodes** system library (for GRIB decode). On Debian/Ubuntu:
  ```bash
  sudo apt-get update && sudo apt-get install -y libeccodes0 libeccodes-dev
  ```
  On macOS: `brew install eccodes`
- A HuggingFace **write token** with access to `nakas/nbm-conus-analysis`
  (https://huggingface.co/settings/tokens). Rotate the one used earlier — it
  appeared in a chat log.
- ~5 GB free disk (it stays bounded at ~1 week, but give headroom).

## Setup

```bash
git clone https://github.com/andrewnakas/nbm-to-zarr.git
cd nbm-to-zarr

python -m venv .venv && . .venv/bin/activate
pip install -e .
pip install "huggingface_hub>=0.20" fsspec aiohttp   # upload + remote-open deps
```

## Run the backfill (resumes automatically)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx   # your write token
PYTHONPATH=src python scripts/backfill_to_hf.py analysis \
    --repo nakas/nbm-conus-analysis \
    --start 2020-10-01 --end 2026-06-10
```

It walks **newest → oldest**, skips the 48 weeks already on HF, and uploads each
new week under `YYYY-Www/`. Expect ~6–10 min per week (serial fetch), so the full
remaining ~250 weeks is ~1.5–2 days unattended. Safe to Ctrl-C and re-run — it
resumes from the next missing week.

### Run it detached (survives logout)

```bash
nohup env HF_TOKEN=$HF_TOKEN PYTHONPATH=src \
    python scripts/backfill_to_hf.py analysis \
    --repo nakas/nbm-conus-analysis --start 2020-10-01 --end 2026-06-10 \
    > backfill.log 2>&1 &
tail -f backfill.log | grep -E "uploaded|building|failed"
```

### Optional: forecast archive too

The forecast store is ~7× larger. Same idea, different repo:

```bash
PYTHONPATH=src python scripts/backfill_to_hf.py forecast \
    --repo nakas/nbm-conus-forecast --start 2020-10-01 --end 2026-06-10
```

## Verify what's on HuggingFace

```bash
python - <<'PY'
import xarray as xr
ds = xr.open_zarr("https://huggingface.co/datasets/nakas/nbm-conus-analysis/resolve/main/2026-W24")
print(ds)            # (time, y, x) full-res 2345x1597, tmean/tmax/tmin/precip/srad
PY
```

## Notes

- **Disk-bounded:** one week of scratch (`data/_backfill_week.zarr`) is built,
  uploaded, then deleted before the next week. Peak local use ≈ one week.
- **Direction:** add `--forward` to go oldest→newest instead of the default
  newest→oldest.
- **Data floor:** dates before 2020-10-01 simply 404 and are skipped (NBM's AWS
  archive doesn't go earlier — 2013 is not available from this source).
- Each week is an independent Zarr; concatenate along `time` to span the archive.
