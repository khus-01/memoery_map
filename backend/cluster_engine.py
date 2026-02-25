"""
cluster_engine.py - Outfit-first + DBSCAN face clustering
"""

import numpy as np
import face_recognition
import cv2
from sklearn.cluster import DBSCAN
from outfit_engine import get_outfit_signature


def get_ml_status() -> dict:
    return {
        "ready":   True,
        "issues":  [],
        "people":  [],
        "message": "Outfit-based auto-clustering active.",
    }


def get_face_embeddings(img_bgr: np.ndarray) -> list:
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(img_rgb, model="hog", number_of_times_to_upsample=1)
    encodings = face_recognition.face_encodings(img_rgb, locations, num_jitters=2)
    return [
        {"embedding": enc.tolist(), "box": list(loc)}
        for enc, loc in zip(encodings, locations)
    ]


def get_photo_outfit(img_bgr: np.ndarray) -> list:
    """Returns 48-dim outfit signature as list for JSON serialization."""
    return get_outfit_signature(img_bgr).tolist()


def cluster_faces(all_embeddings: list, eps: float = 0.50, min_samples: int = 1) -> list:
    if not all_embeddings:
        return []
    X      = np.array(all_embeddings)
    labels = DBSCAN(eps=eps, min_samples=min_samples,
                    metric="euclidean", n_jobs=-1).fit_predict(X)
    return labels.tolist()
