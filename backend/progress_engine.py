import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "database.json"

def _load_db():
    with open(DB_PATH, "r") as f:
        return json.load(f)

def _save_db(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)

def save_progress(username: str, book_id: str, pages: list):
    db  = _load_db()
    key = f"{username}_{book_id}"

    if "book_progress" not in db:
        db["book_progress"] = {}

    db["book_progress"][key] = {
        "username":   username,
        "book_id":    book_id,
        "pages":      pages,
        "updated_at": datetime.now().isoformat(),
        "status":     "draft",
    }
    _save_db(db)

def load_progress(username: str, book_id: str):
    db  = _load_db()
    key = f"{username}_{book_id}"
    progress = db.get("book_progress", {}).get(key)
    if progress:
        return {"pages": progress["pages"], "status": progress["status"]}
    return {"pages": [], "status": "new"}