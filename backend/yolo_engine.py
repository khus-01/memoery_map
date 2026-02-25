"""
yolo_engine.py - Pure OpenCV scene detection + feature extraction
Zero torch, zero onnxruntime, zero DLL issues
Uses color analysis + edge detection for scene understanding
"""

import cv2
import numpy as np


# ── Scene detection via color + brightness analysis ───────────────────────────

SCENE_NAMES = {
    "beach":       "Beach Day",
    "sunset":      "Golden Hour",
    "nature":      "Nature Outing",
    "food":        "Food Trail",
    "night":       "Night Out",
    "indoor":      "Indoor Moments",
    "city":        "City Exploration",
    "bright":      "Bright Day Out",
    "person":      "Memories",
}


def detect_objects(img_bgr: np.ndarray) -> dict:
    """
    Scene detection using HSV color analysis — no ML model needed.
    Detects: beach, sunset, night, nature, indoor, bright day
    """
    try:
        img_small = cv2.resize(img_bgr, (128, 128))
        hsv       = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

        h_ch = hsv[:, :, 0].astype(np.float32)
        s_ch = hsv[:, :, 1].astype(np.float32)
        v_ch = hsv[:, :, 2].astype(np.float32)

        mean_h = float(h_ch.mean())
        mean_s = float(s_ch.mean())
        mean_v = float(v_ch.mean())

        # Bottom half vs top half brightness (horizon detection)
        top_v    = float(v_ch[:64, :].mean())
        bottom_v = float(v_ch[64:, :].mean())

        # Blue ratio (sky / water)
        blue_mask  = ((h_ch >= 100) & (h_ch <= 130) & (s_ch > 50)).astype(np.uint8)
        blue_ratio = float(blue_mask.mean())

        # Orange/red ratio (sunset / food)
        orange_mask  = ((h_ch <= 20) & (s_ch > 100)).astype(np.uint8)
        orange_ratio = float(orange_mask.mean())

        # Green ratio (nature)
        green_mask  = ((h_ch >= 35) & (h_ch <= 85) & (s_ch > 40)).astype(np.uint8)
        green_ratio = float(green_mask.mean())

        # Determine scene
        scene_type = "person"   # default
        labels     = ["person"] # default — assume person present

        if mean_v < 60:
            scene_type = "night"
            labels     = ["night", "dark"]

        elif orange_ratio > 0.15 and mean_v > 150:
            scene_type = "sunset"
            labels     = ["sunset", "orange", "sky"]

        elif blue_ratio > 0.25 and bottom_v > top_v:
            scene_type = "beach"
            labels     = ["beach", "water", "sky"]

        elif green_ratio > 0.30:
            scene_type = "nature"
            labels     = ["nature", "trees", "outdoor"]

        elif mean_s < 30 and mean_v > 180:
            scene_type = "indoor"
            labels     = ["indoor", "bright"]

        elif blue_ratio > 0.15:
            scene_type = "city"
            labels     = ["city", "sky", "outdoor"]

        elif mean_v > 180:
            scene_type = "bright"
            labels     = ["bright", "outdoor"]

        return {
            "labels":     labels,
            "scene_type": scene_type,
            "has_person": True,   # face_recognition handles this separately
            "confidence": {scene_type: 0.8},
        }

    except Exception as e:
        print(f"    Scene detection error: {e}")
        return {"labels": [], "scene_type": "unknown",
                "has_person": False, "confidence": {}}


# ── OpenCV visual feature extraction ──────────────────────────────────────────

def extract_cnn_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    464-dim visual feature vector — pure OpenCV.
    HSV histogram (192) + LBP texture (256) + Edge grid (16)
    """
    try:
        img_small = cv2.resize(img_bgr, (128, 128))

        # 1. HSV Color Histogram (192 dims)
        hsv    = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
        color_feat = np.concatenate([hist_h, hist_s, hist_v])

        # 2. LBP Texture (256 dims)
        gray     = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        lbp      = _compute_lbp(gray)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        lbp_feat = lbp_hist.astype(np.float32)

        # 3. Edge density grid 4x4 (16 dims)
        edges     = cv2.Canny(gray, 50, 150)
        h, w      = edges.shape
        edge_feat = []
        for r in range(4):
            for c in range(4):
                cell = edges[r * h // 4:(r + 1) * h // 4,
                             c * w // 4:(c + 1) * w // 4]
                edge_feat.append(float(cell.mean()))
        edge_feat = np.array(edge_feat)

        # Combine + normalize
        features = np.concatenate([color_feat, lbp_feat, edge_feat])
        norm     = np.linalg.norm(features)
        return features / norm if norm > 0 else features

    except Exception as e:
        print(f"    Feature extraction error: {e}")
        return np.zeros(464)


def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    lbp    = np.zeros((gray.shape[0] - 2, gray.shape[1] - 2), dtype=np.uint8)
    center = gray[1:-1, 1:-1]
    for i, nb in enumerate([
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
        gray[1:-1, 2:],   gray[2:,   2:],   gray[2:,   1:-1],
        gray[2:,   0:-2], gray[1:-1, 0:-2],
    ]):
        lbp |= ((nb >= center).astype(np.uint8) << i)
    return lbp


# ── Smart cluster naming ───────────────────────────────────────────────────────

def get_cluster_name_from_yolo(all_labels: list,
                                fallback: str = "Memories") -> str:
    from collections import Counter
    if not all_labels:
        return fallback

    counts     = Counter(all_labels)
    top_label  = counts.most_common(1)[0][0]
    return SCENE_NAMES.get(top_label, fallback)
