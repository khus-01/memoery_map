"""
metadata_engine.py - Extract EXIF metadata for GPS + time-based clustering
Uses pillow_heif opener registration to preserve EXIF on HEIC files
"""

import io
import math
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pillow_heif

# Register HEIC opener with PIL — this preserves EXIF data
pillow_heif.register_heif_opener()


# ── EXIF Extraction ────────────────────────────────────────────────────────────

def extract_metadata(raw_bytes: bytes, filename: str) -> dict:
    result = {
        "datetime": None,
        "lat":      None,
        "lon":      None,
        "alt":      None,
    }
    try:
        # Open via BytesIO — works for BOTH HEIC and JPEG now
        # because pillow_heif.register_heif_opener() hooks into PIL
        pil    = Image.open(io.BytesIO(raw_bytes))
        result = _extract_pil(pil, result)

        if result["datetime"]:
            dt_str = result["datetime"].strftime("%Y-%m-%d %H:%M")
        else:
            dt_str = "no datetime"
        gps_str = f"{result['lat']:.4f},{result['lon']:.4f}" \
                  if result["lat"] else "no GPS"

        print(f"    Metadata: {dt_str} | {gps_str}")

    except Exception as e:
        print(f"    Metadata extract error ({filename}): {e}")

    return result


def _extract_pil(pil: Image.Image, result: dict) -> dict:
    try:
        exif_data = pil.getexif()
        if not exif_data:
            print("    No EXIF data found")
            return result

        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

        # ── Datetime ──────────────────────────────────────────────────────────
        for dt_tag in ["DateTimeOriginal", "DateTime", "DateTimeDigitized"]:
            if dt_tag in exif:
                try:
                    result["datetime"] = datetime.strptime(
                        str(exif[dt_tag])[:19], "%Y:%m:%d %H:%M:%S"
                    )
                    break
                except Exception:
                    continue

        # ── GPS ───────────────────────────────────────────────────────────────
        gps_raw = exif_data.get_ifd(0x8825)
        if gps_raw:
            gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            lat = _parse_gps(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
            lon = _parse_gps(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
            if lat is not None and lon is not None:
                result["lat"] = lat
                result["lon"] = lon
            alt_raw = gps.get("GPSAltitude")
            if alt_raw:
                result["alt"] = float(alt_raw)

    except Exception as e:
        print(f"    PIL EXIF parse error: {e}")

    return result


def _parse_gps(coord, ref) -> float | None:
    if not coord or not ref:
        return None
    try:
        def to_float(v):
            if isinstance(v, tuple):
                return v[0] / v[1] if v[1] else 0
            return float(v)
        d       = to_float(coord[0])
        m       = to_float(coord[1])
        s       = to_float(coord[2])
        decimal = d + (m / 60.0) + (s / 3600.0)
        if str(ref).strip() in ["S", "W"]:
            decimal = -decimal
        return decimal
    except Exception:
        return None


# ── Distance + Time helpers ────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = math.sin(dlat / 2) ** 2 + \
           math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
           math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def time_diff_hours(dt1: datetime, dt2: datetime) -> float:
    return abs((dt1 - dt2).total_seconds()) / 3600.0


def same_event(meta1: dict, meta2: dict,
               time_thresh_hrs: float = 8.0,
               gps_thresh_km:   float = 8.0) -> bool:
    """
    Same event = same calendar date + within 8km GPS range.
    8km threshold:
      - Splits zip-line location (6km away) → own cluster
      - Keeps same-park photos (<1km) → same cluster
    """
    dt1, dt2 = meta1.get("datetime"), meta2.get("datetime")
    if not dt1 or not dt2:
        return False

    # Hard rule: must be same calendar date
    if dt1.date() != dt2.date():
        return False

    # GPS check if both photos have location
    if all(meta1.get(k) for k in ["lat", "lon"]) and \
       all(meta2.get(k) for k in ["lat", "lon"]):
        dist = haversine_km(meta1["lat"], meta1["lon"],
                            meta2["lat"], meta2["lon"])
        if dist > gps_thresh_km:
            return False

    return True
