"""
caption_engine.py - Smart rule-based event naming
Zero heavy dependencies — no PyTorch, no Gemini, no API keys
Uses photo colors + count to generate meaningful names
"""

import httpx
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


# ── Color → scene mapping ──────────────────────────────────────────────────────

def _dominant_color_name(img_rgb: np.ndarray) -> str:
    """Detect dominant color mood from image."""
    img_small = cv2.resize(img_rgb, (50, 50))
    hsv       = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0].mean()   # hue
    s = hsv[:, :, 1].mean()   # saturation
    v = hsv[:, :, 2].mean()   # brightness

    if s < 40:
        if v > 180: return "Bright Moments"
        if v < 80:  return "Night Out"
        return "Cloudy Day"

    if h < 15 or h > 165:     return "Golden Hour"
    if 15 <= h < 35:           return "Warm Evening"
    if 35 <= h < 75:           return "Garden Outing"
    if 75 <= h < 135:          return "Blue Sky Day"
    if 135 <= h < 165:         return "Teal Vibes"
    return "Colorful Outing"


def _get_image_from_url(url: str) -> np.ndarray | None:
    try:
        raw   = httpx.get(url, timeout=10).content
        image = Image.open(BytesIO(raw)).convert("RGB")
        return np.array(image)
    except Exception:
        return None


# ── Size → occasion mapping ────────────────────────────────────────────────────

SIZE_NAMES = {
    1:  "Solo Moment",
    2:  "Quick Visit",
    3:  "Small Outing",
    4:  "Group Hangout",
    5:  "Day Out",
}

def _size_hint(count: int) -> str:
    if count <= 5:
        return SIZE_NAMES.get(count, "Day Out")
    if count <= 8:
        return "Fun Gathering"
    if count <= 12:
        return "Big Day Out"
    return "Special Event"


# ── Main function ──────────────────────────────────────────────────────────────

def generate_event_name(photo_urls: list, fallback: str = None) -> str:
    if not photo_urls:
        return fallback or "Untitled Event"

    try:
        # Sample up to 3 photos for color analysis
        color_names = []
        for url in photo_urls[:3]:
            img = _get_image_from_url(url)
            if img is not None:
                color_names.append(_dominant_color_name(img))

        size_hint = _size_hint(len(photo_urls))

        if color_names:
            # Pick the most common color name
            name = max(set(color_names), key=color_names.count)
        else:
            name = size_hint

        print(f"  Named: '{name}'")
        return name

    except Exception as e:
        print(f"  Naming failed ({e}), using fallback: {fallback}")
        return fallback or "Untitled Event"


# ── Batch naming ───────────────────────────────────────────────────────────────

def generate_all_event_names(clusters: dict) -> dict:
    renamed    = {}
    used_names = {}

    for key, urls in clusters.items():
        ai_name = generate_event_name(urls, fallback=key)

        if ai_name in used_names:
            used_names[ai_name] += 1
            ai_name = f"{ai_name} {used_names[ai_name]}"
        else:
            used_names[ai_name] = 1

        renamed[ai_name] = urls
        print(f"  '{key}' → '{ai_name}'")

    return renamed
