import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY", "")
print(f"API Key found: {bool(key)}")
print(f"Key length: {len(key) if key else 0}")

try:
    import google.genai as genai
    if key:
        client = genai.Client(api_key=key)
        model = client.models.get("models/gemini-1.5-flash")
        print("✓ google.genai Client created successfully")
        print(f"✓ Model: {model.name if hasattr(model, 'name') else 'gemini-1.5-flash'}")
    else:
        print("✗ No API key available")
except Exception as e:
    print(f"✗ Error: {e}")
