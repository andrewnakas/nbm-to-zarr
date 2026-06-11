#!/usr/bin/env python3
"""Generate the data catalog (JSON + HTML) for GitHub Pages.

Scans the committed Zarr stores under ``data/`` and emits ``catalog/catalog.json``
plus a simple ``catalog/index.html``, mirroring how the ak_hrrr_to_zarr / dynamical
reformatters expose a discoverable catalog. The sample analysis store is hosted in
this same repo, so its ``zarr_url`` points at GitHub's raw content.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import xarray as xr

REPO = "andrewnakas/nbm-to-zarr"  # update if the repo is named differently
DATA_BRANCH = "data"  # full-res sample is force-pushed to this branch
DATA_DIR = Path("data")
CATALOG_DIR = Path("catalog")


def _dataset_entry(zarr_path: Path) -> dict:
    ds = xr.open_zarr(zarr_path)
    try:
        time_dim = "time" if "time" in ds.coords else "init_time"
        entry = {
            "id": str(ds.attrs.get("id", zarr_path.stem)),
            "title": str(ds.attrs.get("title", zarr_path.stem)),
            "description": str(ds.attrs.get("description", "")),
            "provider": str(ds.attrs.get("provider", "NOAA")),
            "model": str(ds.attrs.get("model", "NBM")),
            "variant": str(ds.attrs.get("variant", "")),
            "data_status": str(ds.attrs.get("data_status", "")),
            "hosting": str(ds.attrs.get("hosting", "")),
            "dimensions": {k: int(v) for k, v in ds.sizes.items()},
            "variables": list(ds.data_vars),
            "coordinates": list(ds.coords),
            "temporal_extent": {
                "start": str(ds[time_dim].min().values),
                "end": str(ds[time_dim].max().values),
            },
            "zarr_path": str(zarr_path),
            "zarr_url": f"https://raw.githubusercontent.com/{REPO}/{DATA_BRANCH}/{zarr_path}",
        }
    finally:
        ds.close()
    return entry


def generate() -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    catalog = {
        "title": "NBM to Zarr — Data Catalog",
        "description": "NOAA NBM CONUS forecast & historical analysis reformatted to Zarr.",
        "generated": datetime.now(UTC).isoformat(),
        "datasets": [],
    }

    for zarr_path in sorted(DATA_DIR.glob("*.zarr")):
        # Only the published sample is cataloged — skip scratch (_-prefixed) and
        # the large forecast store, which are not hosted on the data branch.
        if zarr_path.name.startswith("_") or "forecast" in zarr_path.name:
            continue
        if not (zarr_path / ".zmetadata").exists() and not (zarr_path / "zarr.json").exists():
            # Fall back: any dir with a .zarray inside is a v2 store.
            if not any(zarr_path.rglob(".zarray")):
                continue
        try:
            catalog["datasets"].append(_dataset_entry(zarr_path))
        except Exception as exc:
            catalog["datasets"].append({"id": zarr_path.stem, "error": str(exc)})

    (CATALOG_DIR / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (CATALOG_DIR / "index.html").write_text(_html(catalog))
    print(f"Wrote {CATALOG_DIR/'catalog.json'} and {CATALOG_DIR/'index.html'}")


def _html(catalog: dict) -> str:
    cards = []
    for d in catalog["datasets"]:
        if "error" in d:
            cards.append(f"<div class='card'><h2>{d['id']}</h2><p class='err'>{d['error']}</p></div>")
            continue
        dims = "".join(f"<span class='tag'>{k}={v}</span>" for k, v in d["dimensions"].items())
        vars_ = "".join(f"<span class='tag'>{v}</span>" for v in d["variables"])
        status = f"<p class='status'>{d['data_status']}</p>" if d.get("data_status") else ""
        cards.append(
            f"""<div class='card'>
  <h2>{d['title']}</h2>
  <p>{d['description']}</p>
  {status}
  <p><b>Provider:</b> {d['provider']} &nbsp; <b>Model:</b> {d['model']} &nbsp; <b>Variant:</b> {d['variant']}</p>
  <p><b>Coverage:</b> {d['temporal_extent']['start']} → {d['temporal_extent']['end']}</p>
  <p><b>Dimensions:</b> {dims}</p>
  <p><b>Variables:</b> {vars_}</p>
  <p><b>Hosting:</b> {d.get('hosting','')}</p>
  <p><a href='{d['zarr_url']}'>Zarr store (raw GitHub)</a></p>
</div>"""
        )
    body = "\n".join(cards)
    return f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{catalog['title']}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1000px;margin:0 auto;padding:20px;background:#f5f5f5;color:#1e293b}}
header{{background:#0f766e;color:#fff;padding:1.5rem;border-radius:8px}}
.card{{background:#fff;border-radius:8px;padding:1.5rem;margin:1.5rem 0;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.tag{{display:inline-block;background:#ccfbf1;color:#115e59;padding:.15rem .6rem;border-radius:4px;margin:.15rem;font-family:monospace;font-size:.85rem}}
.status{{background:#fef9c3;padding:.5rem .75rem;border-radius:4px;font-size:.9rem}}
.err{{color:#b91c1c}} a{{color:#0f766e}}
</style></head><body>
<header><h1>{catalog['title']}</h1><p>{catalog['description']}</p>
<small>Updated {catalog['generated']}</small></header>
{body}
<footer style='text-align:center;color:#64748b;margin-top:2rem'>
<p>Generated by nbm-to-zarr · Data: NOAA NBM via AWS Open Data · Layout inspired by dynamical.org</p></footer>
</body></html>"""


if __name__ == "__main__":
    generate()
