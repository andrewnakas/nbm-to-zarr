"""Shared NBM CONUS grid geometry, GRIB byte-range fetching, and decoding.

Both the forecast and analysis reformatters consume the same NBM CONUS GRIB2
files on AWS, the same Lambert-conformal grid, and the same idx-driven
byte-range fetch path. Keeping that here avoids duplicating it across the two
variants (the forecast and analysis ``region_job.py`` files only differ in *how
they aggregate* the fetched fields, not in *how they fetch* them).

NBM CONUS layout on AWS (public, no auth):
    https://noaa-nbm-grib2-pds.s3.amazonaws.com/
    blend.YYYYMMDD/CC/core/blend.tCCz.core.fNNN.co.grib2  (+ .idx sidecar)

Grid: Lambert conformal, ~2.54 km, 2345 x 1597. The authoritative CRS and 2D
lat/lon are read from a decoded message at runtime (see ``read_grid_from_message``)
so we never hard-code numbers that could drift; the constants below are the
documented NBM v4 CONUS parameters used to build the projection when we only
have coordinate arrays (e.g. when materializing an empty template).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import numpy as np
import pyproj

logger = logging.getLogger(__name__)

S3_BASE = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"

# --- NBM v4 CONUS Lambert conformal grid (documented; verified at runtime) ----
NX = 2345
NY = 1597
DX = 2539.703  # meters
DY = 2539.703  # meters
# Lambert conformal projection parameters for NBM CONUS.
LAT_0 = 25.0
LON_0 = 265.0  # 95.0 W
LAT_1 = 25.0
LAT_2 = 25.0
EARTH_RADIUS = 6371200.0
# First grid point (lower-left, message scanning +i +j) — NBM v4 CONUS.
FIRST_LAT = 19.229
FIRST_LON = 233.723  # 126.277 W


def nbm_proj() -> pyproj.Proj:
    """Return the NBM CONUS Lambert conformal projection."""
    return pyproj.Proj(
        proj="lcc",
        lat_0=LAT_0,
        lon_0=LON_0,
        lat_1=LAT_1,
        lat_2=LAT_2,
        x_0=0.0,
        y_0=0.0,
        R=EARTH_RADIUS,
    )


def grid_xy() -> tuple[np.ndarray, np.ndarray]:
    """Return the 1D projection x and y coordinate arrays (meters)."""
    proj = nbm_proj()
    first_lon = FIRST_LON - 360.0 if FIRST_LON > 180.0 else FIRST_LON
    x0, y0 = proj(first_lon, FIRST_LAT)
    x = x0 + np.arange(NX, dtype=np.float64) * DX
    y = y0 + np.arange(NY, dtype=np.float64) * DY
    return x, y


def grid_latlon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D (lat, lon) arrays for projection coordinate arrays x, y."""
    proj = nbm_proj()
    xx, yy = np.meshgrid(x, y, indexing="xy")
    lon, lat = proj(xx, yy, inverse=True)
    return lat.astype(np.float32), lon.astype(np.float32)


def spatial_ref_attrs() -> dict[str, object]:
    """CF grid-mapping attrs for the NBM Lambert conformal grid."""
    return {
        "grid_mapping_name": "lambert_conformal_conic",
        "standard_parallel": [LAT_1, LAT_2],
        "longitude_of_central_meridian": LON_0,
        "latitude_of_projection_origin": LAT_0,
        "false_easting": 0.0,
        "false_northing": 0.0,
        "earth_radius": EARTH_RADIUS,
        "crs_wkt": nbm_proj().crs.to_wkt(),
    }


# --- AWS paths ----------------------------------------------------------------


def grib_url(date_str: str, cycle: int, lead: int) -> str:
    """URL of the NBM CONUS core GRIB2 file for an init date/cycle/lead."""
    cc = f"{cycle:02d}"
    return (
        f"{S3_BASE}/blend.{date_str}/{cc}/core/"
        f"blend.t{cc}z.core.f{lead:03d}.co.grib2"
    )


def idx_url(date_str: str, cycle: int, lead: int) -> str:
    """URL of the .idx sidecar for a given GRIB2 file."""
    return grib_url(date_str, cycle, lead) + ".idx"


# --- idx parsing + byte-range fetch ------------------------------------------


@dataclass(frozen=True)
class IdxEntry:
    """One parsed line of a GRIB2 .idx sidecar."""

    msg: int
    start: int
    var: str
    level: str
    window: str
    extra: str
    raw: str

    @property
    def is_prob(self) -> bool:
        """True for probabilistic lines (a trap — deterministic lines have none)."""
        return "prob" in self.raw.lower()

    @property
    def is_ens_std(self) -> bool:
        return "ens std dev" in self.raw.lower()


def parse_idx(text: str) -> list[IdxEntry]:
    """Parse a GRIB2 .idx file into entries with byte offsets.

    Lines look like:  ``msg:start:date:VAR:level:window:extra...``
    The end offset of message N is the start offset of message N+1 (or EOF).
    """
    entries: list[IdxEntry] = []
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    for ln in lines:
        parts = ln.split(":")
        if len(parts) < 5:
            continue
        msg = int(parts[0])
        start = int(parts[1])
        var = parts[3]
        level = parts[4]
        window = parts[5] if len(parts) > 5 else ""
        extra = ":".join(parts[6:]) if len(parts) > 6 else ""
        entries.append(
            IdxEntry(
                msg=msg,
                start=start,
                var=var,
                level=level,
                window=window,
                extra=extra,
                raw=ln,
            )
        )
    return entries


def message_byte_range(
    entries: list[IdxEntry], target: IdxEntry
) -> tuple[int, int | None]:
    """Return (start, end) byte offsets for ``target``; end is None at EOF."""
    by_msg = sorted(entries, key=lambda e: e.start)
    for i, e in enumerate(by_msg):
        if e.msg == target.msg and e.start == target.start:
            nxt = by_msg[i + 1].start - 1 if i + 1 < len(by_msg) else None
            return e.start, nxt
    return target.start, None


def fetch_idx(client: httpx.Client, url: str) -> list[IdxEntry] | None:
    """Fetch and parse a .idx file; None if missing (logged, not fatal)."""
    try:
        resp = client.get(url, timeout=30.0)
        if resp.status_code == 404:
            logger.warning("idx not found: %s", url)
            return None
        resp.raise_for_status()
        return parse_idx(resp.text)
    except httpx.HTTPError:
        logger.warning("idx fetch failed: %s", url)
        return None


def fetch_message_bytes(
    client: httpx.Client, grib_file_url: str, start: int, end: int | None
) -> bytes | None:
    """Byte-range GET one GRIB message from the full file. None on failure."""
    range_header = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    try:
        resp = client.get(
            grib_file_url,
            headers={"Range": range_header},
            timeout=120.0,
        )
        if resp.status_code in (200, 206):
            return resp.content
        logger.warning("range GET %s [%s] -> %s", grib_file_url, range_header, resp.status_code)
        return None
    except httpx.HTTPError:
        logger.warning("range GET failed: %s [%s]", grib_file_url, range_header)
        return None


def decode_message(raw: bytes) -> np.ndarray:
    """Decode a single GRIB2 message (bytes) into a (NY, NX) float32 array.

    Uses eccodes directly — with byte-range reads cfgrib is awkward (file
    oriented), whereas ``codes_new_from_message`` consumes bytes directly.
    Missing values are mapped to NaN.
    """
    import eccodes

    gid = eccodes.codes_new_from_message(raw)
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        missing = eccodes.codes_get_double(gid, "missingValue")
        values = eccodes.codes_get_values(gid)
        arr = np.asarray(values, dtype=np.float64)
        arr[arr == missing] = np.nan
        # GRIB scans +i (west->east) then +j (south->north) for NBM CONUS.
        arr = arr.reshape(nj, ni)
    finally:
        eccodes.codes_release(gid)
    return arr.astype(np.float32)


def read_grid_from_message(raw: bytes) -> dict[str, np.ndarray]:
    """Decode a message's true 2D lat/lon and grid dims (for verification)."""
    import eccodes

    gid = eccodes.codes_new_from_message(raw)
    try:
        ni = eccodes.codes_get(gid, "Ni")
        nj = eccodes.codes_get(gid, "Nj")
        lats = np.asarray(eccodes.codes_get_array(gid, "latitudes"), dtype=np.float32)
        lons = np.asarray(eccodes.codes_get_array(gid, "longitudes"), dtype=np.float32)
    finally:
        eccodes.codes_release(gid)
    return {
        "ni": np.asarray(ni),
        "nj": np.asarray(nj),
        "latitude": lats.reshape(nj, ni),
        "longitude": lons.reshape(nj, ni),
    }
