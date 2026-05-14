"""
outfit_engine.py - Outfit-first clustering (matches your PDF output)
Uses ResNet-style color histogram on torso region.
"""

import cv2
import numpy as np


def get_outfit_signature(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract a robust outfit signature from torso region.
    Returns a 48-dim color histogram (HSV) — much more reliable than single color.
    """
    h, w = img_bgr.shape[:2]

    # Torso region: rows 30%-75%, cols 20%-80%
    y1, y2 = int(h * 0.30), int(h * 0.75)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    crop    = img_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return np.zeros(48)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 3-channel histogram: H(16 bins), S(16 bins), V(16 bins)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    signature = np.concatenate([hist_h, hist_s, hist_v])

    # Normalize
    norm = np.linalg.norm(signature)
    if norm > 0:
        signature = signature / norm

    return signature


def outfit_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Cosine distance between two outfit signatures. 0=identical, 1=completely different."""
    dot = np.dot(sig1, sig2)
    return float(1.0 - dot)


def is_same_outfit(sig1: np.ndarray, sig2: np.ndarray, threshold: float = 0.35) -> bool:
    """
    threshold=0.15 → strict (same outfit)
    threshold=0.25 → loose (similar outfit)
    threshold=0.35 → very loose (similar colors/tones) — ADJUSTED for better grouping
    """
    return outfit_distance(sig1, sig2) <= threshold
