import json, random, re, os
from pathlib import Path
from collections import defaultdict
import warnings

import cv2
import numpy as np
from PIL import Image
import io
import urllib.request

# Suppress deprecation warning for google.generativeai (will migrate to google-genai in future)
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

try:
    # Using google-generativeai (deprecated but still functional)
    # Will migrate to google-genai in future version
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
        model = None
except Exception:
    GEMINI_AVAILABLE = False
    model = None

from caption_engine import generate_all_event_names


# ── Decoration themes ───────────────────────────────────────────────────────────
THEMES = {
    "beach":      {"bg": "#e8f4f8", "stickers": ["🌊","🌴","🐚","☀️"], "border": "wave"},
    "wedding":    {"bg": "#fff5f5", "stickers": ["💍","🌸","🥂","💐"], "border": "elegant"},
    "birthday":   {"bg": "#fff9e6", "stickers": ["🎂","🎉","🎈","🎁"], "border": "festive"},
    "graduation": {"bg": "#f0f4ff", "stickers": ["🎓","📚","🏆","✨"], "border": "formal"},
    "travel":     {"bg": "#f0fff4", "stickers": ["✈️","🗺️","📸","🌍"], "border": "adventure"},
    "party":      {"bg": "#fdf0ff", "stickers": ["🎊","🍾","💃","🎵"], "border": "festive"},
    "family":     {"bg": "#f5f5f0", "stickers": ["❤️","🏡","🌿","😊"], "border": "soft"},
    "other":      {"bg": "#f9f9f9", "stickers": ["📷","✨","🌟","💫"], "border": "minimal"},
}

CAPTIONS_MAP = {
    "beach":      "Sun, sand, and perfect memories 🌊",
    "birthday":   "Another year, another celebration 🎂",
    "wedding":    "Love in every frame 💍",
    "travel":     "Adventure awaits at every turn ✈️",
    "graduation": "The journey begins here 🎓",
    "party":      "Good times and great company 🎉",
    "family":     "Family is everything ❤️",
    "other":      "Moments to remember forever ✨",
}


# ── Score photo quality ─────────────────────────────────────────────────────────
def score_photo_url(url: str) -> float:
    """
    Score a photo from URL by analyzing:
    - Sharpness (Laplacian variance)
    - Lighting (brightness histogram)
    - Composition (rule of thirds, centering)
    - Face detection (prefers photos with clear faces)
    """
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            img_data = r.read()
        
        # Convert to OpenCV format
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return 0.0
        
        # Normalize to standard size for fair comparison
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (640, 480))
        
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # 1. SHARPNESS (Laplacian variance) - how crisp is the image
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
        sharpness = min(sharpness / 2.0, 1.0)  # Normalize to 0-1
        
        # 2. BRIGHTNESS - not too dark, not too blown out
        brightness = gray.mean() / 255
        brightness_score = 1.0 - abs(brightness - 0.5) * 0.5  # Sweet spot at 0.5
        brightness_score = max(0, brightness_score)
        
        # 3. CONTRAST - Michelson contrast for visual appeal
        img_min = gray.min() / 255
        img_max = gray.max() / 255
        if img_max + img_min > 0:
            contrast = (img_max - img_min) / (img_max + img_min)
        else:
            contrast = 0
        contrast_score = min(contrast, 1.0)
        
        # 4. FACE DETECTION - prefer photos with faces
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        face_score = 0.0
        if len(faces) > 0:
            face_score = min(len(faces) / 3.0, 1.0)  # Bonus for multiple faces
            # Prefer centered faces
            for (x, y, fw, fh) in faces:
                face_center_x = (x + fw/2) / w
                face_center_y = (y + fh/2) / h
                # Rule of thirds: faces near center (0.3-0.7) score higher
                center_bonus = 1.0 - abs(face_center_x - 0.5) * 0.3
                face_score += center_bonus
            face_score = min(face_score / len(faces), 1.0)
        
        # 5. COMPOSITION - prefer landscape/portrait ratio (not extreme aspect ratios)
        ratio = w / h if h > 0 else 1
        ratio_score = 1.0 if 0.4 < ratio < 2.5 else 0.5
        
        # FINAL SCORE: Weighted combination
        # Sharpness is most important, then face detection, then lighting
        final_score = (
            sharpness * 0.35 +
            face_score * 0.30 +
            brightness_score * 0.15 +
            contrast_score * 0.15 +
            ratio_score * 0.05
        )
        
        return round(max(0, min(final_score, 1.0)), 3)
    
    except Exception as e:
        print(f"    [scoring error] {url[:50]}... — {e}")
        return 0.0


def score_photo_path(img_path: str) -> float:
    """Score a photo from a local file path using CV2."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return 0.5
        gray        = cv2.Laplacian(img, cv2.COLOR_BGR2GRAY).var() / 1000
        brightness  = 1 - abs(img.mean() - 128) / 128
        h, w        = img.shape[:2]
        ratio_score = 1.0 if 0.5 < w / h < 2.0 else 0.5
        return round(min((gray * 0.5 + brightness * 0.3 + ratio_score * 0.2), 1.0), 3)
    except Exception:
        return 0.5


def select_best_photo(photo_urls: list) -> tuple:
    """
    Select the best photo from a list of URLs.
    Returns: (best_url, score)
    Prefers: sharp, well-lit, with faces, good composition
    """
    if not photo_urls:
        return (None, 0.0)
    
    print(f"    Analyzing {len(photo_urls)} photos to find the best...")
    
    scores = {}
    for url in photo_urls:
        score = score_photo_url(url)
        scores[url] = score
        print(f"      ✓ {score:.3f} → {url.split('/')[-1][:30]}")
    
    best_url = max(photo_urls, key=lambda u: scores.get(u, 0.0))
    best_score = scores[best_url]
    
    print(f"    ★ BEST: {best_score:.3f} ⭐\n")
    
    return (best_url, best_score)


# ── Detect events using Gemini (optional) ──────────────────────────────────────
def detect_events_from_urls(photo_urls: list) -> list:
    """
    Try Gemini event detection; fall back to single 'other' group if unavailable.
    Works with URLs only — no local file paths needed.
    """
    if not GEMINI_AVAILABLE or not model:
        return [{"event": "other", "mood": "cheerful",
                 "indices": list(range(len(photo_urls)))}]

    try:
        parts  = []
        sample = photo_urls[:6]
        for url in sample:
            try:
                import urllib.request, io
                with urllib.request.urlopen(url, timeout=5) as r:
                    img = Image.open(io.BytesIO(r.read()))
                    img.thumbnail((512, 512))
                    parts.append(img)
            except Exception:
                pass

        if not parts:
            raise ValueError("No images loaded")

        prompt = """
        Look at these photos and group them by event/occasion.
        Return ONLY valid JSON like this (no extra text):
        {
          "groups": [
            {"event": "beach", "mood": "adventurous", "indices": [0,1,2]},
            {"event": "birthday", "mood": "joyful", "indices": [3,4]}
          ]
        }
        Event types allowed: beach, wedding, birthday, graduation, travel, party, family, other
        If all photos are the same event, return one group with all indices.
        """
        response = model.generate_content([prompt] + parts)
        raw      = response.text.strip()
        match    = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("groups", [])
    except Exception as e:
        print(f"  Gemini event detection skipped: {e}")

    return [{"event": "other", "mood": "cheerful",
             "indices": list(range(len(photo_urls)))}]


# ── Build pages from a list of URLs ────────────────────────────────────────────
def build_pages_from_urls(photo_urls: list) -> dict:
    """
    Core page builder — works with Cloudinary URLs only.
    Called by auto_generate_book().
    Returns { pages: [...], events: [...] }
    """
    if not photo_urls:
        return {"pages": [], "events": []}

    # Detect event groups
    groups = detect_events_from_urls(photo_urls)

    pages = []
    for group in groups:
        indices = group.get("indices", [])
        event   = group.get("event", "other")
        mood    = group.get("mood", "cheerful")
        theme   = THEMES.get(event, THEMES["other"])
        caption = CAPTIONS_MAP.get(event, "Moments to remember ✨")

        group_urls = [photo_urls[i] for i in indices if i < len(photo_urls)]

        # Split into pages of max 4 photos
        for i in range(0, max(len(group_urls), 1), 4):
            chunk = group_urls[i:i + 4]
            count = len(chunk)

            if   count == 1: layout = "single"
            elif count == 2: layout = "two-column"
            elif count == 3: layout = "three-mixed"
            else:            layout = "grid-2x2"

            pages.append({
                "photos":    chunk,
                "layout":    layout,
                "caption":   caption,
                "bg_color":  theme["bg"],
                "stickers":  random.sample(theme["stickers"], min(2, len(theme["stickers"]))),
                "border":    theme["border"],
                "event_tag": event,
                "texts":     [{"content": caption, "x": 40, "y": 20,
                               "fontSize": 18, "color": "#333"}],
            })

    return {"pages": pages, "events": groups}


# ── auto_generate_book — called by main.py /auto-generate endpoint ──────────────
def auto_generate_book(username: str, book_id: str) -> dict:
    """
    Loads book data from database.json and generates pages.
    Returns { pages, events } or { pages: [] } if book/photos not found.
    """
    import json as _json

    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    DB_PHOTOS = os.path.join(BASE_DIR, "database.json")

    try:
        with open(DB_PHOTOS, "r", encoding="utf-8") as f:
            db = _json.load(f)
    except Exception:
        return {"pages": []}

    user = db.get(username, {})
    book = user.get("books", {}).get(book_id, {})

    if not book:
        return {"pages": []}

    # Collect all photo URLs from clusters + extras
    all_urls = []
    for urls in book.get("clusters", {}).values():
        all_urls.extend(urls)
    all_urls.extend(book.get("extras", []))

    if not all_urls:
        return {"pages": []}

    print(f"  auto_generate_book: {len(all_urls)} photos for {username}/{book_id}")
    return build_pages_from_urls(all_urls)


# ── Legacy: generate_book_from_photos (kept for compatibility) ─────────────────
def generate_book_from_photos(photo_paths: list, photo_urls: list) -> dict:
    """Legacy function — use build_pages_from_urls() for new code."""
    return build_pages_from_urls(photo_urls)