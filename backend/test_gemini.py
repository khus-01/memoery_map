# test_gemini.py  — run with: python test_gemini.py
import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
print(f"Key found:   {bool(key)}")
print(f"Key value:   '{key}'")
print(f"Starts with: {key[:8] if key else 'NONE'}")
print(f"Length:      {len(key) if key else 0}")
