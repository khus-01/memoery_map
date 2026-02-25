"""
ml_cache.py - Cache face embeddings + CNN features + outfit signatures by photo hash
"""

import hashlib
import json
import os

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_cache.json")


def get_photo_hash(raw_bytes: bytes) -> str:
    return hashlib.md5(raw_bytes).hexdigest()


def load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_cached_ml(raw_bytes: bytes) -> dict | None:
    photo_hash = get_photo_hash(raw_bytes)
    cache      = load_cache()
    return cache.get(photo_hash, None)


def save_ml_to_cache(raw_bytes: bytes, embeddings: list,
                     cnn_features: list, url: str):
    photo_hash        = get_photo_hash(raw_bytes)
    cache             = load_cache()
    cache[photo_hash] = {
        "embeddings":   embeddings,
        "cnn_features": cnn_features,
        "url":          url,
    }
    save_cache(cache)


def get_cache_stats() -> dict:
    cache = load_cache()
    return {
        "total_cached_photos": len(cache),
        "cache_file":          CACHE_FILE,
    }
