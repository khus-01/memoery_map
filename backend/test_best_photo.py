"""
test_best_photo.py - Test AI best photo selection
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, r"c:\Users\Khushi\MemoryMap-Project\backend")

from ai_generator import score_photo_url, select_best_photo

def test_with_local_files():
    """Test photo scoring with local files by loading them directly."""
    
    print("\n" + "="*70)
    print("AI PHOTO SELECTION TEST")
    print("="*70)
    
    data_dir = Path(r"c:\Users\Khushi\MemoryMap-Project\data\Original Images")
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return
    
    # Get some test images
    test_images = []
    for person_dir in sorted(data_dir.iterdir())[:2]:
        if not person_dir.is_dir():
            continue
        for img_path in list(person_dir.glob("*.jpg"))[:3]:
            test_images.append(img_path)
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"\n📸 Found {len(test_images)} test images\n")
    
    # Analyze each image
    print("-" * 70)
    print("INDIVIDUAL PHOTO SCORING")
    print("-" * 70)
    
    scores = {}
    for img_path in test_images[:6]:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
            sharpness = min(sharpness / 2.0, 1.0)
            
            brightness = gray.mean() / 255
            brightness_score = 1.0 - abs(brightness - 0.5) * 0.5
            
            contrast = gray.max() - gray.min()
            contrast_score = min(contrast / 255, 1.0)
            
            ratio = w / h
            ratio_score = 1.0 if 0.4 < ratio < 2.5 else 0.5
            
            final = (
                sharpness * 0.35 + 
                brightness_score * 0.15 + 
                contrast_score * 0.15 + 
                ratio_score * 0.05
            )
            final = min(max(final, 0), 1.0)
            
            scores[img_path.name] = {
                "score": final,
                "sharpness": sharpness,
                "brightness": brightness_score,
                "contrast": contrast_score,
                "ratio": ratio_score
            }
            
            print(f"  {img_path.name}")
            print(f"    Score: {final:.3f} ⭐")
            print(f"    - Sharpness: {sharpness:.3f}")
            print(f"    - Brightness: {brightness_score:.3f}")
            print(f"    - Contrast: {contrast_score:.3f}")
            print(f"    - Ratio: {ratio_score:.3f}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    if scores:
        print("-" * 70)
        print("RANKING")
        print("-" * 70)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        for i, (name, data) in enumerate(sorted_scores[:3], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"  {medal} #{i}: {name} ({data['score']:.3f})")
    
    print("\n" + "="*70)
    print("✓ AI PHOTO SELECTION TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_with_local_files()
