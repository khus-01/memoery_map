"""
test_outfit_clustering.py - Test outfit-first clustering logic
"""

import os
import cv2
import numpy as np
from pathlib import Path
from outfit_engine import get_outfit_signature, is_same_outfit

def test_outfit_signatures():
    """Test outfit signature extraction from real images."""
    
    # Find celebrity images
    data_dir = Path(r"c:\Users\Khushi\MemoryMap-Project\data\Original Images")
    
    if not data_dir.exists():
        print("❌ Data directory not found")
        return
    
    print("\n" + "="*70)
    print("OUTFIT CLUSTERING TEST")
    print("="*70)
    
    # Get first 2-3 images per person
    results = {}
    
    for person_dir in sorted(data_dir.iterdir())[:5]:  # Test 5 celebrities
        if not person_dir.is_dir():
            continue
        
        person_name = person_dir.name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        
        if not images:
            continue
        
        results[person_name] = []
        
        for img_path in images[:3]:  # 3 images per person
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                sig = get_outfit_signature(img)
                results[person_name].append({
                    "path": img_path.name,
                    "signature": sig
                })
                
            except Exception as e:
                print(f"  Error loading {img_path.name}: {e}")
        
        if results[person_name]:
            print(f"\n👤 {person_name}: {len(results[person_name])} image(s) processed")
    
    # Test cross-person outfit similarity
    print("\n" + "-"*70)
    print("CROSS-PERSON OUTFIT SIMILARITY TEST")
    print("-"*70)
    
    all_images = []
    for person, imgs in results.items():
        for img_data in imgs:
            all_images.append((person, img_data))
    
    # Test threshold tuning
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    for threshold in thresholds:
        clusters = {}
        cluster_id = 0
        matched = set()
        
        for i, (person1, data1) in enumerate(all_images):
            if i in matched:
                continue
            
            cluster = [i]
            matched.add(i)
            
            for j, (person2, data2) in enumerate(all_images):
                if j <= i or j in matched:
                    continue
                
                if is_same_outfit(data1["signature"], data2["signature"], threshold=threshold):
                    cluster.append(j)
                    matched.add(j)
            
            clusters[cluster_id] = cluster
            cluster_id += 1
        
        # Analyze clusters
        same_person_groupings = 0
        cross_person_groupings = 0
        
        for cluster in clusters.values():
            people_in_cluster = set(all_images[i][0] for i in cluster)
            if len(people_in_cluster) == 1:
                same_person_groupings += 1
            elif len(people_in_cluster) > 1:
                cross_person_groupings += 1
        
        print(f"\nThreshold {threshold}:")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Single-person: {same_person_groupings}")
        print(f"  Multi-person: {cross_person_groupings}")
    
    # Check if threshold=0.15 is working well
    print("\n" + "-"*70)
    print("DETAILED ANALYSIS (threshold=0.15 - strict matching)")
    print("-"*70)
    
    threshold = 0.15
    clusters = {}
    cluster_id = 0
    matched = set()
    photos_per_person = {}
    
    for i, (person, data) in enumerate(all_images):
        if i in matched:
            continue
        
        if person not in photos_per_person:
            photos_per_person[person] = []
        
        cluster = [i]
        matched.add(i)
        photos_per_person[person].append(cluster_id)
        
        for j, (person2, data2) in enumerate(all_images):
            if j <= i or j in matched:
                continue
            
            if is_same_outfit(data["signature"], data2["signature"], threshold=threshold):
                cluster.append(j)
                matched.add(j)
                if person2 not in photos_per_person:
                    photos_per_person[person2] = []
                photos_per_person[person2].append(cluster_id)
        
        clusters[cluster_id] = cluster
        cluster_id += 1
    
    for person, cluster_ids in photos_per_person.items():
        unique_clusters = len(set(cluster_ids))
        total_imgs = len(cluster_ids)
        print(f"  {person}: {total_imgs} images → {unique_clusters} outfit(s)")
    
    print("\n" + "="*70)
    print("✓ OUTFIT CLUSTERING TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_outfit_signatures()
