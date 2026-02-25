"""
main.py - MemoryMap Photo Clustering API
Multi-book support: users can create, list, select, rename, delete books
Primary:   GPS + Timestamp metadata clustering (same date + 8km range)
Secondary: CNN visual features + DBSCAN (only for no-metadata photos)
Tertiary:  Face DBSCAN identity check + same-day primary-person merge
"""

import io
import json
import os
import uuid
from datetime import datetime
from typing import List, Optional
from collections import defaultdict

import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
import pillow_heif
import uvicorn
from dotenv import load_dotenv
from fastapi import Body, File, Form, HTTPException, UploadFile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sklearn.cluster import DBSCAN

import cluster_engine as ml
from caption_engine import generate_all_event_names
from metadata_engine import extract_metadata, same_event as metadata_same_event
from ml_cache import get_cached_ml, save_ml_to_cache, get_cache_stats
from yolo_engine import detect_objects, extract_cnn_features

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PHOTOS = os.path.join(BASE_DIR, "database.json")
DB_USERS  = os.path.join(BASE_DIR, "users.json")


# ── Helpers ────────────────────────────────────────────────────────────────────

def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    if not os.path.exists(filename):
        return {} if filename == DB_PHOTOS else []
    with open(filename, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if data else ({} if filename == DB_PHOTOS else [])
        except Exception:
            return {} if filename == DB_PHOTOS else []


def now_str() -> str:
    return datetime.now().isoformat()


def get_user_db(username: str) -> dict:
    """Get full user record, auto-migrate old format if needed."""
    db   = load_json(DB_PHOTOS)
    user = db.get(username, {})

    # ── Auto-migrate old format (clusters/extras) → new multi-book format ──
    if "clusters" in user or "extras" in user:
        print(f"  Migrating {username} to multi-book format...")
        old_clusters = user.get("clusters", {})
        old_extras   = user.get("extras", [])
        book_id      = f"book_{uuid.uuid4().hex[:8]}"
        user = {
            "books": {
                book_id: {
                    "id":         book_id,
                    "name":       "My First Book",
                    "created_at": now_str(),
                    "updated_at": now_str(),
                    "clusters":   old_clusters,
                    "extras":     old_extras,
                }
            }
        }
        db[username] = user
        save_json(DB_PHOTOS, db)

    if "books" not in user:
        user["books"] = {}

    return user


def save_user_db(username: str, user_data: dict):
    db           = load_json(DB_PHOTOS)
    db[username] = user_data
    save_json(DB_PHOTOS, db)


# ── Auth ───────────────────────────────────────────────────────────────────────

@app.post("/signup/")
async def signup(user: dict = Body(...)):
    users = load_json(DB_USERS)
    users.append(user)
    save_json(DB_USERS, users)
    return {"status": "success"}


@app.post("/login/")
async def login(credentials: dict = Body(...)):
    users = load_json(DB_USERS)
    for u in users:
        if u.get("username") == credentials.get("username") and \
           u.get("password") == credentials.get("password"):
            return {"status": "success", "user": u["username"]}
    raise HTTPException(status_code=401, detail="Invalid credentials")


# ── Book Management Endpoints ──────────────────────────────────────────────────

@app.get("/books/{username}")
def list_books(username: str):
    """List all books for a user (id, name, photo count, dates)."""
    user = get_user_db(username)

    book_list = []
    for book_id, book in user["books"].items():
        total_photos = sum(len(v) for v in book.get("clusters", {}).values()) \
                       + len(book.get("extras", []))
        book_list.append({
            "id":           book_id,
            "name":         book.get("name", "Untitled Book"),
            "created_at":   book.get("created_at", ""),
            "updated_at":   book.get("updated_at", ""),
            "total_photos": total_photos,
            "event_count":  len(book.get("clusters", {})),
        })

    # Sort by updated_at descending (most recent first)
    book_list.sort(key=lambda b: b["updated_at"], reverse=True)
    return {"status": "success", "books": book_list}


@app.post("/books/{username}/create")
def create_book(username: str, body: dict = Body(...)):
    """Create a new empty book. Returns the new book_id."""
    name    = body.get("name", "Untitled Book").strip() or "Untitled Book"
    book_id = f"book_{uuid.uuid4().hex[:8]}"

    user = get_user_db(username)
    user["books"][book_id] = {
        "id":         book_id,
        "name":       name,
        "created_at": now_str(),
        "updated_at": now_str(),
        "clusters":   {},
        "extras":     [],
    }
    save_user_db(username, user)
    print(f"  Created book '{name}' ({book_id}) for {username}")
    return {"status": "success", "book_id": book_id, "name": name}


@app.get("/books/{username}/{book_id}")
def get_book(username: str, book_id: str):
    """Get full content of a specific book (clusters + extras)."""
    user = get_user_db(username)
    book = user["books"].get(book_id)

    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    return {"status": "success", "book": book}


@app.patch("/books/{username}/{book_id}/rename")
def rename_book(username: str, book_id: str, body: dict = Body(...)):
    """Rename a book."""
    new_name = body.get("name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    user = get_user_db(username)
    if book_id not in user["books"]:
        raise HTTPException(status_code=404, detail="Book not found")

    user["books"][book_id]["name"]       = new_name
    user["books"][book_id]["updated_at"] = now_str()
    save_user_db(username, user)

    return {"status": "success", "book_id": book_id, "name": new_name}


@app.delete("/books/{username}/{book_id}")
def delete_book(username: str, book_id: str):
    """Delete a book permanently."""
    user = get_user_db(username)
    if book_id not in user["books"]:
        raise HTTPException(status_code=404, detail="Book not found")

    deleted_name = user["books"][book_id].get("name", book_id)
    del user["books"][book_id]
    save_user_db(username, user)

    print(f"  Deleted book '{deleted_name}' ({book_id}) for {username}")
    return {"status": "success", "message": f"'{deleted_name}' deleted"}


@app.patch("/books/{username}/{book_id}/rename-event")
def rename_event(username: str, book_id: str, body: dict = Body(...)):
    """Rename a specific event/cluster inside a book."""
    old_name = body.get("old_name", "").strip()
    new_name = body.get("new_name", "").strip()

    if not old_name or not new_name:
        raise HTTPException(status_code=400, detail="old_name and new_name required")

    user = get_user_db(username)
    if book_id not in user["books"]:
        raise HTTPException(status_code=404, detail="Book not found")

    clusters = user["books"][book_id].get("clusters", {})
    if old_name not in clusters:
        raise HTTPException(status_code=404, detail="Event not found")

    # Rename by rebuilding dict (preserve order)
    new_clusters = {}
    for k, v in clusters.items():
        new_clusters[new_name if k == old_name else k] = v

    user["books"][book_id]["clusters"]   = new_clusters
    user["books"][book_id]["updated_at"] = now_str()
    save_user_db(username, user)

    return {"status": "success", "old_name": old_name, "new_name": new_name}


# ── Upload + Auto Cluster into specific book ───────────────────────────────────

@app.post("/upload-photos/")
async def upload_photos(
    files:    List[UploadFile] = File(...),
    username: str              = Form(...),
    book_id:  Optional[str]    = Form(None),   # ← if None, create new book
    book_name: Optional[str]   = Form(None),   # ← name for new book
):
    print(f"\n{'=' * 70}")
    print(f"UPLOAD | user={username} | files={len(files)} | book={book_id}")
    print(f"{'=' * 70}\n")

    # ── Resolve book ───────────────────────────────────────────────────────────
    user = get_user_db(username)

    if not book_id:
        # Create a new book automatically
        book_id   = f"book_{uuid.uuid4().hex[:8]}"
        name      = (book_name or "").strip() or \
                    f"Book {len(user['books']) + 1}"
        user["books"][book_id] = {
            "id":         book_id,
            "name":       name,
            "created_at": now_str(),
            "updated_at": now_str(),
            "clusters":   {},
            "extras":     [],
        }
        print(f"  Auto-created book: '{name}' ({book_id})")
    elif book_id not in user["books"]:
        raise HTTPException(status_code=404,
                            detail=f"Book '{book_id}' not found")

    # ── STEP 1: Extract everything from each photo ─────────────────────────────
    processed  = []
    cache_hits = 0
    cache_miss = 0
    print("STEP 1: Metadata + Scene Detection + CNN + Face embeddings\n")

    for idx, file in enumerate(files):
        try:
            raw = await file.read()

            meta    = extract_metadata(raw, file.filename)
            dt_str  = meta["datetime"].strftime("%Y-%m-%d %H:%M") \
                      if meta["datetime"] else "no datetime"
            gps_str = f"{meta['lat']:.4f},{meta['lon']:.4f}" \
                      if meta["lat"] else "no GPS"

            if file.filename.lower().endswith(".heic"):
                heif = pillow_heif.read_heif(raw)
                pil  = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
            else:
                pil = Image.open(io.BytesIO(raw))

            pil_rgb      = pil.convert("RGB")
            img_np       = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
            scene_result = detect_objects(img_np)
            cached       = get_cached_ml(raw)

            if cached:
                cache_hits += 1
                print(f"  [{idx+1}/{len(files)}] {file.filename} → "
                      f"CACHE HIT ✓ | {dt_str} | {gps_str} | "
                      f"scene={scene_result['scene_type']}")
                processed.append({
                    "url":          cached["url"],
                    "filename":     file.filename,
                    "embeddings":   cached["embeddings"],
                    "cnn_features": cached.get("cnn_features",
                                               np.zeros(464).tolist()),
                    "face_count":   len(cached["embeddings"]),
                    "meta":         meta,
                    "yolo":         scene_result,
                })
                continue

            cache_miss   += 1
            face_data     = ml.get_face_embeddings(img_np)
            embeddings    = [f["embedding"] for f in face_data]
            cnn_features  = extract_cnn_features(img_np)

            buf = io.BytesIO()
            pil_rgb.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            result = cloudinary.uploader.upload(buf.getvalue(),
                                                folder="memorymap")
            url = result.get("secure_url")
            save_ml_to_cache(raw, embeddings, cnn_features.tolist(), url)

            processed.append({
                "url":          url,
                "filename":     file.filename,
                "embeddings":   embeddings,
                "cnn_features": cnn_features.tolist(),
                "face_count":   len(embeddings),
                "meta":         meta,
                "yolo":         scene_result,
            })
            print(f"  [{idx+1}/{len(files)}] {file.filename} → "
                  f"{len(embeddings)} face(s) | {dt_str} | {gps_str} | "
                  f"scene={scene_result['scene_type']}")

        except Exception as exc:
            print(f"  [{idx+1}/{len(files)}] ERROR {file.filename}: {exc}")

    print(f"\n  Cache hits: {cache_hits} | ML computed: {cache_miss}")

    if not processed:
        return {"status": "error", "message": "No photos could be processed"}

    # ── STEP 2: Metadata clustering ────────────────────────────────────────────
    print(f"\nSTEP 2: Metadata clustering (same date + 8km GPS)\n")

    meta_clusters = []
    no_meta       = []

    for photo_idx, photo in enumerate(processed):
        meta = photo["meta"]
        if not meta["datetime"]:
            no_meta.append(photo_idx)
            continue

        matched = None
        for cluster in meta_clusters:
            if metadata_same_event(meta, cluster["anchor_meta"],
                                   time_thresh_hrs=8.0,
                                   gps_thresh_km=8.0):
                matched = cluster
                break

        if matched:
            matched["photos"].append(photo_idx)
        else:
            meta_clusters.append({
                "photos":      [photo_idx],
                "anchor_meta": meta,
            })

    print(f"  Metadata clusters: {len(meta_clusters)} | "
          f"No metadata: {len(no_meta)}")

    # ── STEP 3: DBSCAN on CNN visual embeddings ────────────────────────────────
    print(f"\nSTEP 3: DBSCAN on CNN visual embeddings\n")

    all_cnn    = np.array([p["cnn_features"] for p in processed])
    dbscan     = DBSCAN(eps=0.25, min_samples=1, metric="cosine", n_jobs=-1)
    cnn_labels = dbscan.fit_predict(all_cnn)

    print(f"  Visual groups: "
          f"{len(set(cnn_labels)) - (1 if -1 in cnn_labels else 0)}")

    # ── STEP 4: Keep metadata clusters intact ─────────────────────────────────
    print(f"\nSTEP 4: Building base clusters (metadata intact)\n")

    final_clusters = [mc["photos"] for mc in meta_clusters]

    if no_meta:
        no_meta_set = set(no_meta)
        cnn_no_meta = defaultdict(list)
        for p_idx in no_meta:
            cnn_no_meta[cnn_labels[p_idx]].append(p_idx)

        for cnn_group in cnn_no_meta.values():
            merged = False
            for fc in final_clusters:
                fc_cnn = [cnn_labels[i] for i in fc if i not in no_meta_set]
                if fc_cnn and max(set(fc_cnn), key=fc_cnn.count) == \
                              cnn_labels[cnn_group[0]]:
                    fc.extend(cnn_group)
                    merged = True
                    break
            if not merged:
                final_clusters.append(cnn_group)

    # ── STEP 4.5: Merge clusters sharing primary person + same date ────────────
    print(f"\nSTEP 4.5: Primary-person same-day merge\n")

    all_emb_temp = []
    all_idx_temp = []
    for photo_idx, photo in enumerate(processed):
        for emb in photo["embeddings"]:
            all_emb_temp.append(emb)
            all_idx_temp.append(photo_idx)

    if all_emb_temp:
        temp_labels    = ml.cluster_faces(all_emb_temp)
        temp_photo_set = defaultdict(set)
        for label, pidx in zip(temp_labels, all_idx_temp):
            if label != -1:
                temp_photo_set[label].add(pidx)

        if temp_photo_set:
            primary_lbl = max(temp_photo_set,
                              key=lambda l: len(temp_photo_set[l]))
            primary_set = temp_photo_set[primary_lbl]

            merged_any = True
            while merged_any:
                merged_any   = False
                new_clusters = []
                used         = set()

                for i, g1 in enumerate(final_clusters):
                    if i in used:
                        continue
                    merged = list(g1)

                    for j, g2 in enumerate(final_clusters):
                        if j <= i or j in used:
                            continue
                        if not (set(g1) & primary_set) or \
                           not (set(g2) & primary_set):
                            continue

                        dt1 = next((processed[idx]["meta"]["datetime"]
                                    for idx in g1
                                    if processed[idx]["meta"]["datetime"]),
                                   None)
                        dt2 = next((processed[idx]["meta"]["datetime"]
                                    for idx in g2
                                    if processed[idx]["meta"]["datetime"]),
                                   None)

                        if dt1 and dt2 and dt1.date() == dt2.date():
                            merged.extend(g2)
                            used.add(j)
                            merged_any = True
                            print(f"  Merged {i}+{j} "
                                  f"(primary person, {dt1.date()})")

                    new_clusters.append(merged)
                    used.add(i)

                final_clusters = new_clusters

    print(f"  Final cluster count: {len(final_clusters)}")

    # ── STEP 5: Face identity ──────────────────────────────────────────────────
    print(f"\nSTEP 5: Face identity check\n")

    all_embeddings    = []
    all_photo_indices = []
    for photo_idx, photo in enumerate(processed):
        for emb in photo["embeddings"]:
            all_embeddings.append(emb)
            all_photo_indices.append(photo_idx)

    primary_photos = set(range(len(processed)))

    if all_embeddings:
        face_labels     = ml.cluster_faces(all_embeddings)
        label_photo_set = defaultdict(set)
        for label, photo_idx in zip(face_labels, all_photo_indices):
            if label != -1:
                label_photo_set[label].add(photo_idx)

        if label_photo_set:
            primary_label  = max(label_photo_set,
                                 key=lambda l: len(label_photo_set[l]))
            primary_photos = label_photo_set[primary_label]
            print(f"  Unique people: {len(label_photo_set)} | "
                  f"Primary in {len(primary_photos)} photo(s)")

    # ── STEP 6: Build clusters + Extras ───────────────────────────────────────
    print(f"\nSTEP 6: Building clusters + separating Extras\n")

    clusters            = {}
    cluster_yolo_labels = {}
    extras              = []

    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        n    = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / n) if n > 0 else 0.0

    for i, group in enumerate(final_clusters):
        face_photos     = [idx for idx in group
                           if processed[idx]["face_count"] > 0]
        faceless_photos = [idx for idx in group
                           if processed[idx]["face_count"] == 0]

        if not face_photos:
            for idx in faceless_photos:
                extras.append(processed[idx]["url"])
            continue

        face_cnn_avg    = np.mean([processed[idx]["cnn_features"]
                                   for idx in face_photos], axis=0)
        keep_with_event = []
        send_to_extras  = []

        for idx in faceless_photos:
            sim = cosine_sim(processed[idx]["cnn_features"], face_cnn_avg)
            if sim >= 0.80:
                keep_with_event.append(idx)
            else:
                send_to_extras.append(idx)

        event_photos             = face_photos + keep_with_event
        key                      = f"Event_{i + 1}"
        urls                     = [processed[idx]["url"]
                                    for idx in event_photos]
        all_labels               = []
        for idx in event_photos:
            all_labels.extend(processed[idx]["yolo"]["labels"])

        clusters[key]            = urls
        cluster_yolo_labels[key] = all_labels

        for idx in send_to_extras:
            extras.append(processed[idx]["url"])

        print(f"  {key}: {len(urls)} photo(s) | "
              f"faces={len(face_photos)} | "
              f"candid={len(keep_with_event)}")

    # ── STEP 7: Smart naming ───────────────────────────────────────────────────
    if clusters:
        print(f"\nSTEP 7: Naming {len(clusters)} cluster(s)...\n")
        try:
            clusters = generate_all_event_names(
                clusters,
                cluster_yolo_labels=cluster_yolo_labels,
            )
        except Exception as e:
            print(f"  Naming skipped: {e}")

    # ── Save into specific book ────────────────────────────────────────────────
    user["books"][book_id]["clusters"]   = clusters
    user["books"][book_id]["extras"]     = extras
    user["books"][book_id]["updated_at"] = now_str()
    save_user_db(username, user)

    print(f"\n{'=' * 70}")
    print(f"SAVED → book '{user['books'][book_id]['name']}' ({book_id})")
    print(f"  Events: {len(clusters)} | Extras: {len(extras)}")
    print(f"{'=' * 70}\n")

    return {
        "status":  "success",
        "book_id": book_id,
        "name":    user["books"][book_id]["name"],
        "data":    {"clusters": clusters, "extras": extras},
    }


# ── Legacy endpoint (backwards compatible) ────────────────────────────────────

@app.get("/photos/{username}")
def get_photos(username: str):
    """Legacy: returns most recently updated book."""
    user  = get_user_db(username)
    books = user.get("books", {})
    if not books:
        return {"clusters": {}, "extras": []}

    latest = max(books.values(), key=lambda b: b.get("updated_at", ""))
    return {"clusters": latest.get("clusters", {}),
            "extras":   latest.get("extras", [])}


# ── Other Endpoints ────────────────────────────────────────────────────────────

@app.get("/model-info")
def model_info():
    status = ml.get_ml_status()
    return {"ready": status["ready"], "message": status["message"],
            "issues": status["issues"]}


@app.get("/cache-stats")
def cache_stats():
    return get_cache_stats()


@app.delete("/cache-clear")
def cache_clear():
    from ml_cache import CACHE_FILE
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    return {"status": "success", "message": "ML cache cleared"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
