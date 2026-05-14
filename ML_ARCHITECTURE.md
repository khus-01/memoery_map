# MemoryMap ML Architecture & Complete Data Flow

## Executive Summary

Your MemoryMap system implements a **multi-stage intelligent photo clustering pipeline** that transforms raw photo uploads into organized events with automatic best-photo selection. Rather than using traditional deep learning models (CNNs/RNNs), you've built a **feature-engineered ML system** combining classical computer vision techniques with statistical clustering.

**Core Philosophy:** Outfit-first clustering (clothing similarity) → Time/Location refinement → Identity matching → Smart photo selection

---

## 1. SYSTEM OVERVIEW: 7.5-STEP PIPELINE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHOTO UPLOAD ENTERS SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Extract All ML Features from Each Photo                         │
│  • Outfit signature (HSV histogram on torso)                            │
│  • Face embeddings (128-dim vectors from face_recognition)             │
│  • CNN features (464-dim: color + texture + edges)                     │
│  • Scene type (beach, sunset, nature, food, night, indoor, city, ...)  │
│  • Metadata (GPS coordinates, datetime from EXIF)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1.5: OUTFIT-FIRST CLUSTERING (PRIMARY GROUPING)                    │
│  Greedy matching: Each photo compared to existing outfit signatures     │
│  → Groups photos by clothing/appearance regardless of date/location    │
│  → Result: Outfit_A, Outfit_B, Outfit_C, etc.                         │
│  → Filter: Only merge if cosine distance ≤ 0.15                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: TIME + GPS CLUSTERING (WITHIN EACH OUTFIT)                      │
│  Secondary grouping: Within each outfit, cluster by proximity           │
│  → Same outfit + similar time (± 1 hour) + close location              │
│  → Result: Outfit_A Event 1, Outfit_A Event 2, etc.                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: CNN FEATURE CLUSTERING (VISUAL SIMILARITY)                      │
│  DBSCAN on 464-dim feature vectors (eps=0.25)                          │
│  → Finds visually similar photos within each outfit+time group        │
│  → Merges DBSCAN sub-clusters (same person, same visual content)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: FACE RECOGNITION CLUSTERING (IDENTITY MATCHING)                 │
│  DBSCAN on 128-dim face embeddings (eps=0.50)                          │
│  → Clusters faces by identity (same person detected)                  │
│  → If person appears in 2+ outfit groups → merge outfit groups        │
│  → Validates outfit similarity before merging                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4.5: OUTFIT-AWARE PERSON + DATE MERGE (VALIDATION)                │
│  Prevents merging photos of same person in DIFFERENT outfits           │
│  → Checks: Same person + Recent date → but outfit different?          │
│  → If different outfit → DO NOT MERGE (prevents cross-outfit merge)   │
│  → Result: Clean events with consistent clothing                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: SCENE DETECTION & AUTO-NAMING                                   │
│  OpenCV color analysis detects scene type (no ML model)                │
│  → Beach (high blue ratio), Sunset (red/orange ratio)                 │
│  → Nature (green ratio), Food (brown/orange ratio)                    │
│  → Night (low brightness), Indoor (gray ratio)                        │
│  → City (edges/contrast), Bright (high mean brightness)               │
│  → Result: Event named "Beach Outing", "Dinner Time", etc.            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 6: ML CACHE UPDATE                                                 │
│  Store computed features (face embeddings, CNN features) by photo hash │
│  → Avoids recomputing expensive operations on duplicate uploads       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 7.5: AI BEST PHOTO SELECTION                                       │
│  Multi-criteria scoring for each photo in event:                       │
│  → Sharpness (35%): Laplacian variance (edge blur detect)             │
│  → Face detection (30%): Number of detected faces                     │
│  → Brightness (15%): Euclidean distance from ideal 0.5                │
│  → Contrast (15%): max(pixel) - min(pixel)                           │
│  → Composition (5%): Aspect ratio favorability                        │
│                                                                        │
│  Final Score = weighted average of all 5 criteria                      │
│  → Result: best_photo URL and best_photo_score stored in cluster     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ RETURN: Events Array                                                    │
│ [{                                                                      │
│    "event_name": "Beach Outing",                                       │
│    "photos": [url1, url2, url3, ...],                                 │
│    "best_photo": url2,                                                │
│    "best_photo_score": 0.87,                                          │
│    "timestamp": "2024-01-15T10:30:00"                                 │
│ }, ...]                                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DETAILED FEATURE ENGINEERING ENGINES

### 2.1 OUTFIT ENGINE (`outfit_engine.py`)

**Purpose:** Extract and compare clothing signatures to group photos by outfit.

**Algorithm:** HSV Color Histogram on Torso Region

```
INPUT: Photo (BGR image)
  ↓
STEP 1: Extract Torso Region (30-75% height, 20-80% width)
  • Why torso only? Face varies by angle/expression, legs hidden
  • Focus on shirt/dress/chest area for outfit identification
  ↓
STEP 2: Convert to HSV Color Space
  • RGB → HSV (better for clothing color analysis)
  • H: Color hue (0-180°) - red, blue, green, etc.
  • S: Saturation (0-255) - vivid vs pale
  • V: Value/Brightness (0-255) - light vs dark
  ↓
STEP 3: Compute 3 Histograms (48 dimensions total)
  • Hue histogram: 16 bins (captures color palette)
  • Saturation histogram: 16 bins (captures vividness)
  • Value histogram: 16 bins (captures brightness)
  → Each bin = proportion of pixels in that range
  ↓
STEP 4: L2 Normalization
  • Normalize to unit length: divide by sqrt(sum of squares)
  • Makes comparison scale-invariant
  ↓
OUTPUT: 48-dimensional signature vector
```

**Comparison:** Cosine Similarity Distance
```
distance = 1 - cosine_similarity(outfit_A, outfit_B)

Examples:
• Same outfit (red shirt): distance ≈ 0.02 (very similar)
• Different outfit (blue vs red): distance ≈ 0.85 (very different)
• Same outfit, different lighting: distance ≈ 0.08 (still grouped)

Threshold = 0.15
→ distance ≤ 0.15: SAME OUTFIT ✓
→ distance > 0.15: DIFFERENT OUTFIT ✗
```

**Code Example:**
```python
def get_outfit_signature(img_bgr):
    h, w = img_bgr.shape[:2]
    # Extract torso (center 30-75% height, 20-80% width)
    y_start, y_end = int(0.30 * h), int(0.75 * h)
    x_start, x_end = int(0.20 * w), int(0.80 * w)
    torso = img_bgr[y_start:y_end, x_start:x_end]
    
    # HSV histograms
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    
    # Concatenate (48 dims) and normalize
    signature = np.concatenate([h_hist, s_hist, v_hist])
    return signature / np.linalg.norm(signature)
```

---

### 2.2 FACE RECOGNITION ENGINE (`cluster_engine.py`)

**Purpose:** Identify and cluster photos by person identity.

**Algorithm:** dlib Face Detection + Face Embedding Generation + DBSCAN Clustering

**Part A: Face Detection & Encoding**
```
INPUT: Photo (BGR image)
  ↓
STEP 1: Detect Faces (HOG + dlib)
  • HOG (Histogram of Oriented Gradients)
  • Sliding window detector finds face bounding boxes
  • Returns: list of face locations/bounding boxes
  ↓
STEP 2: Generate Face Embeddings
  • For each detected face:
    - Extract face region (aligned)
    - Feed through CNN model (pre-trained on VGGFace2/CelebA)
    - Output: 128-dimensional encoding vector
  • Each person has consistent embeddings (angle, lighting invariant)
  ↓
OUTPUT: List of 128-dim vectors (one per face detected)
```

**Example Data Transformations:**
```
Photo: "beach_vacation_1.jpg"
-face_locations: [(x1, y1, x2, y2), (x3, y3, x4, y4)]
   ↓
-face_encodings:
   [0.245, -0.156, 0.089, ..., -0.023]  ← Person A encoding (128 dims)
   [0.198, -0.145, 0.112, ..., -0.034]  ← Person B encoding (128 dims)
```

**Part B: DBSCAN Clustering**
```
ALGORITHM: Density-Based Spatial Clustering
Parameters: eps=0.50, min_samples=1

1. Distance Metric: Euclidean distance in 128-dim embedding space
2. Core Idea:
   • Points within eps distance of each other → same cluster
   • Dense regions identified automatically
   • Outliers (noise points) handled gracefully

Example:
  Person_1_photo_1.jpg embedding: [0.12, 0.34, ...]  ─┐
  Person_1_photo_5.jpg embedding: [0.11, 0.33, ...]  ─┤ CLUSTER_0 (Person 1)
  Person_1_photo_8.jpg embedding: [0.13, 0.35, ...]  ─┘
  
  Person_2_photo_3.jpg encoding:  [0.89, 0.76, ...]  ─┐
  Person_2_photo_7.jpg encoding:  [0.88, 0.77, ...]  ─┤ CLUSTER_1 (Person 2)
  Person_2_photo_9.jpg encoding:  [0.90, 0.75, ...]  ─┘
  
Result: All photos of the same person clustered together
```

---

### 2.3 CNN FEATURE ENGINE (`yolo_engine.py` - extract_cnn_features)

**Purpose:** Extract 464-dimensional visual feature vectors for image similarity comparison.

**Algorithm:** Multi-component feature extraction (pure OpenCV, no neural networks)

```
INPUT: Photo (BGR image, resized to 128×128)
  ↓
COMPONENT 1: HSV Color Histogram (192 dimensions)
  • Split into H, S, V channels separately
  • Hue: 64 bins → captures color palette
  • Saturation: 64 bins → captures color intensity
  • Value: 64 bins → captures brightness distribution
  → Result: 192-dim color feature vector
  
  Example: Beach photo would have:
    High blue bins (sky/water)
    High yellow bins (sand)
    Moderate brightness values
  ↓
COMPONENT 2: Local Binary Pattern (LBP) Texture (256 dimensions)
  • For each pixel, compare to 8 neighbors
  • Creates binary pattern (0/1 for each neighbor)
  • 8-bit patterns → 256 possible values
  • Histogram of all patterns in image
  → Result: 256-dim texture feature vector
  
  Example: Sandy beach would have:
    High counts in "smooth" patterns (sandy texture)
    Forest would have "complex" patterns (leaf texture)
  ↓
COMPONENT 3: Edge Density Grid (16 dimensions)
  • Canny edge detection on grayscale
  • Divide image into 4×4 grid (16 cells)
  • For each cell: mean edge pixel density
  → Result: 16-dim edge/structure feature vector
  
  Example: Beach (simple) vs city (complex edges)
    Beach: low edge density (smooth)
    City: high edge density (buildings, lines)
  ↓
COMBINATION: 192 + 256 + 16 = 464 DIMENSIONS
  ↓
NORMALIZATION: L2 norm (divide by magnitude)
  • Makes features scale-invariant
  • Euclidean distance meaningful
  ↓
OUTPUT: 464-dimensional feature vector
```

**DBSCAN Clustering on CNN Features:**
```
Parameters: eps=0.25, min_samples=1

Purpose: Find visually similar photos
Example:
  "person_beach_day1_photo1": [0.05, 0.12, ..., 0.34]  ─┐
  "person_beach_day1_photo2": [0.06, 0.11, ..., 0.35]  ─┤ VISUAL_CLUSTER_0
  "person_beach_day1_photo3": [0.04, 0.13, ..., 0.33]  ─┘
    ↓
    Same outfit, same location, taken seconds apart
    → Visually identical/similar

  "person_beach_day1_photo4": [0.89, 0.76, ..., 0.02]  ─┐
  "person_beach_day1_photo5": [0.88, 0.75, ..., 0.03]  ─┤ VISUAL_CLUSTER_1
    ↓
    Different pose/angle, still same outfit/location
    → Visually different but still same event
```

---

### 2.4 METADATA ENGINE (`metadata_engine.py`)

**Purpose:** Extract GPS coordinates and datetime from photo EXIF data.

**Algorithm:** EXIF Parsing (Exchangeable Image File Format)

```
INPUT: Photo file (.jpg or .heic)
  ↓
STEP 1: Read EXIF Data
  • JPEG: Built-in EXIF headers
  • HEIC: Via pillow_heif library (Apple format)
  ↓
STEP 2: Extract Tags
  • DateTimeOriginal: "2024-01-15 14:30:45" (when photo taken)
  • GPSInfo: GPS coordinates (latitude, longitude)
  ↓
STEP 3: Parse GPS Fractions
  • EXIF stores GPS as fractions: (degrees, minutes, seconds)
  • Conversion: 40 + 45/60 + 30/3600 = 40.75833° latitude
  ↓
OUTPUT:
  {
    "datetime": "2024-01-15T14:30:45",
    "latitude": 40.7128,
    "longitude": -74.0060
  }
```

**GPS Distance Calculation:**
```
Haversine formula → Great-circle distance
distance = 2 * R * arcsin(sqrt(sin²((lat2-lat1)/2) + cos(lat1)*cos(lat2)*sin²((lon2-lon1)/2)))
where R = 6371 km (Earth radius)

Example:
  Photo 1: lat=28.6139, lon=77.2090 (Delhi, India)
  Photo 2: lat=28.6145, lon=77.2095
  → Distance ≈ 0.076 km = 76 meters
  → Will be grouped together if time is close (±1 hour)
```

---

### 2.5 SCENE DETECTION ENGINE (`yolo_engine.py` - detect_objects)

**Purpose:** Automatically detect scene/location type for event naming.

**Algorithm:** HSV Color Ratio Analysis (pure OpenCV, no ML models)

```
SCENES DETECTED:

1. BEACH
   Indicators: High blue ratio (sky/water), sandy color
   Code: ratio_blue = count_blue_pixels / total_pixels
   Threshold: ratio_blue > 0.25 → "Beach Outing"

2. SUNSET
   Indicators: High red/orange ratio
   Code: ratio_red_orange = count(red|orange) / total_pixels
   Threshold: ratio_red_orange > 0.20 → "Sunset"

3. NATURE / FOREST
   Indicators: High green ratio
   Code: ratio_green = count_green_pixels / total_pixels
   Threshold: ratio_green > 0.30 → "Nature Walk"

4. FOOD / RESTAURANT
   Indicators: Brown/orange (food colors) + darker brightness
   Code: ratio_brown_orange > 0.15 AND mean_brightness < 100 → "Dinner"

5. NIGHT
   Indicators: Very low brightness, high dark pixels
   Code: mean_brightness < 60 → "Night Out"

6. INDOOR
   Indicators: Gray/neutral colors, moderate brightness
   Code: high_gray_pixels AND 80 < mean_brightness < 150 → "Indoors"

7. CITY
   Indicators: High edge density (buildings), contrast
   Code: edge_density > threshold AND std_dev > 50 → "City Tour"

8. BRIGHT DAYLIGHT
   Indicators: High overall brightness
   Code: mean_brightness > 180 → "Bright Day"
```

**HSV-based Color Extraction:**
```
HSV Space:
  H (Hue):        0° = Red, 60° = Yellow, 120° = Green, 180° = Cyan, ...
  S (Saturation): 0 = Gray, 255 = Vivid
  V (Value):      0 = Black, 255 = White

Beach Detection Example:
  For each pixel:
    if H in [100-130] (blues/cyans) → count as water/sky
    if H in [20-40] (yellows) AND S > 100 → count as sand
  ratio_beach = (water_pixels + sand_pixels) / total_pixels
  if ratio_beach > 0.25 → BEACH LOCATION DETECTED
```

---

### 2.6 AI BEST PHOTO SELECTION (`ai_generator.py`)

**Purpose:** Automatically select the best/sharpest photo from each event.

**Algorithm:** Multi-Criteria Scoring

```
SCORING CRITERIA (Weighted):

1. SHARPNESS (35% weight)
   Method: Laplacian variance (measures blur)
   Code: laplacian = cv2.Laplacian(gray, cv2.CV_64F)
         sharpness_score = np.var(laplacian)
   
   Sharp photo: score = 200-500
   Blurry photo: score = 10-50
   
   Normalized: score / max_possible → [0, 1]

2. FACE DETECTION (30% weight)
   Method: Haar Cascade face detector
   Code: faces = cascade.detectMultiScale(gray, ...)
   
   Score based on:
   • Number of faces (more is better for group photos)
   • Face sizes (larger = clearer faces)
   
   Score = min(num_faces, 3) / 3 → [0, 1]

3. BRIGHTNESS (15% weight)
   Method: Average pixel intensity
   Code: mean_brightness = np.mean(gray)
   
   Optimal brightness: 128 (middle of 0-255)
   Score = 1 - |mean_brightness - 128| / 128
   
   Too dark (brightness=30): score ≈ 0.77
   Perfect (brightness=128): score = 1.0
   Too bright (brightness=220): score ≈ 0.28

4. CONTRAST (15% weight)
   Method: pixel range (max - min)
   Code: contrast = np.max(gray) - np.min(gray)
   
   Normalized: contrast / 255 → [0, 1]
   
   Low contrast (flat): score ≈ 0.3
   High contrast (vivid): score ≈ 0.8

5. COMPOSITION (5% weight)
   Method: Aspect ratio favorability
   Code: aspect_ratio = width / height
   
   Score preferences:
   • 16:9 (1.78): good for landscape
   • 4:3 (1.33): classic
   • 1:1 (1.0): square
   
   Score = 1.0 if aspect_ratio in [1.0, 1.78] else 0.8
```

**Final Scoring Formula:**
```
best_photo_score = 
  0.35 * sharpness_score +
  0.30 * face_detection_score +
  0.15 * brightness_score +
  0.15 * contrast_score +
  0.05 * composition_score

Result: score ∈ [0, 1] (0 = worst, 1 = perfect)

Example:
Photo 1: sharp=0.95, faces=1.0, bright=0.90, contrast=0.85, compos=1.0
  → score = 0.35*0.95 + 0.30*1.0 + 0.15*0.90 + 0.15*0.85 + 0.05*1.0
  → score = 0.333 + 0.300 + 0.135 + 0.128 + 0.050 = 0.946 ✓ BEST

Photo 2: sharp=0.40, faces=0.5, bright=0.50, contrast=0.60, compos=0.8
  → score = 0.35*0.40 + 0.30*0.5 + 0.15*0.50 + 0.15*0.60 + 0.05*0.8
  → score = 0.140 + 0.150 + 0.075 + 0.090 + 0.040 = 0.495 ✗ NOT SELECTED
```

---

## 3. DATA FLOW EXAMPLE: REAL SCENARIO

**Scenario:** User uploads 12 beach photos: 3 different outfits, 2 people, taken on different days

### Input Photos:
```
1. beach_day1_1.jpg - Person A, Red Shirt, 2024-01-15 10:00, GPS(28.6, 77.2)
2. beach_day1_2.jpg - Person A, Red Shirt, 2024-01-15 10:05, GPS(28.6, 77.2)
3. beach_day1_3.jpg - Person A, Red Shirt, 2024-01-15 10:15, GPS(28.6, 77.2)
4. beach_day2_1.jpg - Person A, Blue Shirt, 2024-01-16 15:00, GPS(28.6, 77.2)
5. beach_day2_2.jpg - Person A, Blue Shirt, 2024-01-16 15:10, GPS(28.6, 77.2)
6. beach_day3_1.jpg - Person B, Red Shirt, 2024-01-17 09:00, GPS(28.7, 77.3)
7. beach_day3_2.jpg - Person B, Red Shirt, 2024-01-17 09:20, GPS(28.7, 77.3)
8. beach_day3_3.jpg - Person B, Red Shirt, 2024-01-17 09:35, GPS(28.7, 77.3)
9. beach_day1_4.jpg - Person A, Green Shirt, 2024-01-15 16:00, GPS(28.61, 77.21)
10. beach_day1_5.jpg - Person A, Green Shirt, 2024-01-15 16:10, GPS(28.61, 77.21)
11. beach_day2_3.jpg - Person B, Blue Shirt, 2024-01-16 14:55, GPS(28.6, 77.2)
12. beach_day2_4.jpg - Person B, Blue Shirt, 2024-01-16 14:58, GPS(28.6, 77.2)
```

### STEP 1: Extract Features
**Outfit Engine Output:**
```
Photo 1-3:   outfit_sig_red_shirt_a = [0.12, 0.34, ..., 0.45]
Photo 4-5:   outfit_sig_blue_shirt_a = [0.67, 0.23, ..., 0.34]
Photo 6-8:   outfit_sig_red_shirt_b = [0.13, 0.35, ..., 0.46]  (similar to 1-3!)
Photo 9-10:  outfit_sig_green_shirt = [0.78, 0.12, ..., 0.56]
Photo 11-12: outfit_sig_blue_shirt_b = [0.68, 0.24, ..., 0.35]  (similar to 4-5!)
```

**Face Recognition Output:**
```
Photo 1-5, 9-10: face_encoding_person_a = [0.24, -0.15, 0.08, ...]
Photo 6-8, 11-12: face_encoding_person_b = [0.89, -0.02, 0.19, ...]
```

**Metadata Output:**
```
Photo 1-3: timestamp=2024-01-15T10:00, gps=(28.6, 77.2)
Photo 4-5: timestamp=2024-01-16T15:00, gps=(28.6, 77.2)
Photo 6-8: timestamp=2024-01-17T09:00, gps=(28.7, 77.3)
Photo 9-10: timestamp=2024-01-15T16:00, gps=(28.61, 77.21)
Photo 11-12: timestamp=2024-01-16T14:55, gps=(28.6, 77.2)
```

### STEP 1.5: Outfit-First Clustering
```
Greedy Matching:
Photo 1 (red_shirt_a) → CREATE outfit_cluster_red_a

Photo 2 (red_shirt_a):
  distance to outfit_cluster_red_a = 0.07 ≤ 0.15
  → ADD to outfit_cluster_red_a ✓

Photo 3 (red_shirt_a):
  distance to outfit_cluster_red_a = 0.08 ≤ 0.15
  → ADD to outfit_cluster_red_a ✓

Photo 4 (blue_shirt_a) → CREATE outfit_cluster_blue_a

Photo 5 (blue_shirt_a):
  distance to outfit_cluster_blue_a = 0.06 ≤ 0.15
  → ADD to outfit_cluster_blue_a ✓

Photo 6 (red_shirt_b):
  distance to outfit_cluster_red_a = 0.09 ≤ 0.15
  → ADD to outfit_cluster_red_a ✓ (SAME OUTFIT, DIFFERENT PERSON!)

Photo 7 (red_shirt_b):
  distance to outfit_cluster_red_a = 0.08 ≤ 0.15
  → ADD to outfit_cluster_red_a ✓

Photo 8 (red_shirt_b):
  distance to outfit_cluster_red_a = 0.10 ≤ 0.15
  → ADD to outfit_cluster_red_a ✓

Photo 9 (green_shirt) → CREATE outfit_cluster_green

Photo 10 (green_shirt):
  distance to outfit_cluster_green = 0.05 ≤ 0.15
  → ADD to outfit_cluster_green ✓

Photo 11 (blue_shirt_b):
  distance to outfit_cluster_blue_a = 0.07 ≤ 0.15
  → ADD to outfit_cluster_blue_a ✓

Photo 12 (blue_shirt_b):
  distance to outfit_cluster_blue_a = 0.06 ≤ 0.15
  → ADD to outfit_cluster_blue_a ✓

RESULT AFTER STEP 1.5:
  outfit_cluster_red_a: [photo1, photo2, photo3, photo6, photo7, photo8]
  outfit_cluster_blue_a: [photo4, photo5, photo11, photo12]
  outfit_cluster_green: [photo9, photo10]
```

### STEP 2: Time + GPS Clustering (Within Each Outfit)
```
FOR outfit_cluster_red_a:

Sub-cluster by time + GPS proximity:

  photo1 (2024-01-15 10:00, 28.6, 77.2)   ─┐
  photo2 (2024-01-15 10:05, 28.6, 77.2)   ─┤ Red Outfit Group 1 (Jan 15 morning)
  photo3 (2024-01-15 10:15, 28.6, 77.2)   ─┘
  
  photo9 (2024-01-15 16:00, 28.61, 77.21) ┐ Red Outfit Group 2 (Jan 15 afternoon)
  photo10 (2024-01-15 16:10, 28.61, 77.21)┘

  photo6 (2024-01-17 09:00, 28.7, 77.3)   │
  photo7 (2024-01-17 09:20, 28.7, 77.3)   ├ Red Outfit Group 3 (Jan 17)
  photo8 (2024-01-17 09:35, 28.7, 77.3)   │

RESULT AFTER STEP 2:
  red_outfit_timegroup1: [photo1, 2, 3]
  red_outfit_timegroup2: [photo9, 10]
  red_outfit_timegroup3: [photo6, 7, 8]
  
  blue_outfit_timegroup1: [photo4, 5]
  blue_outfit_timegroup2: [photo11, 12]
  
  green_outfit_timegroup: [photo9, 10]
```

### STEP 3: CNN Feature Clustering
```
FOR red_outfit_timegroup1:
  photo1 CNN features: [0.05, 0.12, ..., 0.34] ─┐
  photo2 CNN features: [0.06, 0.11, ..., 0.35] ─┤ VISUAL_SIMILAR (same pose/angle)
  photo3 CNN features: [0.04, 0.13, ..., 0.33] ─┘
  
  photo9 CNN features: [0.52, 0.67, ..., 0.12] ┐ DIFFERENT VISUAL CONTENT
  photo10 CNN features: [0.51, 0.68, ..., 0.13] (different part of beach)
  
  → red_outfit_visual_A: [photo1, 2, 3] (same pose at same spot)
  → red_outfit_visual_B: [photo9, 10] (different pose at different spot)
```

### STEP 4: Face Recognition Clustering
```
Face embeddings:
  Person A: [0.24, -0.15, 0.08, ...]
  Person B: [0.89, -0.02, 0.19, ...]

DBSCAN clustering (eps=0.50):
  photo1-5, 9-10 (Person A) → CLUSTER_PERSON_A ✓
  photo6-8, 11-12 (Person B) → CLUSTER_PERSON_B ✓

Check: If same person appears in different outfit groups → MERGE?
  photo1-5, 9-10 (Person A) in:
    - red_outfit_timegroup1
    - red_outfit_timegroup2  
    - blue_outfit_timegroup1
  
  Should we merge blue + red? NO! Different outfits!
  → STEP 4.5 validation prevents merge
```

### STEP 4.5: Outfit-Aware Validation
```
Query: Can we merge photo4 (Person A, blue_shirt) with photo1 (Person A, red_shirt)?

Checks:
  ✓ Same person? Yes (face embeddings within eps)
  ✓ Recent date? Yes (Jan 15 vs Jan 16 = 1 day apart)
  ✓ Same outfit? NO! (red vs blue > 0.15 distance)
  
  → DO NOT MERGE ✓

Result: Keep as SEPARATE EVENTS
  Event "Beach Outing Red": [photo1, 2, 3]
  Event "Beach Outing Blue": [photo4, 5]
  Event "Beach Outing Red #2": [photo6, 7, 8]
```

### STEP 5: Scene Detection & Naming
```
For each event, analyze dominant colors:
  High blue ratio (sky/water)
  Sandy color (yellow/brown)
  → SCENE = "Beach"
  
  Bright daylight detected
  → EVENT_NAME = "Beach Outing" ✓
```

### STEP 7.5: AI Best Photo Selection
```
FOR Event "Beach Outing Red" [photo1, 2, 3]:

Photo 1:
  Sharpness: Laplacian variance = 245 → score = 0.95
  Face(s): 1 detected → score = 0.33
  Brightness: mean = 135 → score = 0.89
  Contrast: 180 → score = 0.71
  Composition: 16:9 → score = 0.95
  → Final score = 0.35*0.95 + 0.30*0.33 + 0.15*0.89 + 0.15*0.71 + 0.05*0.95 = 0.676

Photo 2:
  Sharpness: Laplacian variance = 320 → score = 1.0
  Face(s): 1 detected → score = 0.33
  Brightness: mean = 140 → score = 0.90
  Contrast: 190 → score = 0.75
  Composition: 16:9 → score = 0.95
  → Final score = 0.35*1.0 + 0.30*0.33 + 0.15*0.90 + 0.15*0.75 + 0.05*0.95 = 0.714

Photo 3:
  Sharpness: Laplacian variance = 180 → score = 0.82
  Face(s): 0 detected → score = 0.0 (turned away)
  Brightness: mean = 145 → score = 0.87
  Contrast: 175 → score = 0.69
  Composition: 16:9 → score = 0.95
  → Final score = 0.35*0.82 + 0.30*0.0 + 0.15*0.87 + 0.15*0.69 + 0.05*0.95 = 0.506

BEST PHOTO: photo2 (score = 0.714) ✓
```

### FINAL OUTPUT:
```json
[
  {
    "event_name": "Beach Outing",
    "photos": ["url_photo1", "url_photo2", "url_photo3"],
    "best_photo": "url_photo2",
    "best_photo_score": 0.714,
    "timestamp": "2024-01-15T10:00:00"
  },
  {
    "event_name": "Beach Outing",
    "photos": ["url_photo4", "url_photo5", "url_photo11", "url_photo12"],
    "best_photo": "url_photo4",
    "best_photo_score": 0.695,
    "timestamp": "2024-01-16T14:55:00"
  },
  {
    "event_name": "Beach Outing",
    "photos": ["url_photo6", "url_photo7", "url_photo8"],
    "best_photo": "url_photo7",
    "best_photo_score": 0.681,
    "timestamp": "2024-01-17T09:20:00"
  },
  {
    "event_name": "Beach Outing",
    "photos": ["url_photo9", "url_photo10"],
    "best_photo": "url_photo10",
    "best_photo_score": 0.702,
    "timestamp": "2024-01-15T16:10:00"
  }
]
```

**Key Observations:**
1. **Outfit Integrity:** Person A's red shirt photos (photo1-3) stayed separate from blue shirt (photo4-5) despite same person ✓
2. **Time Clustering:** Same outfit on same day grouped together, different days kept separate ✓
3. **Cross-event Validation:** Person B's red shirt grouped with outfit (not person) ✓
4. **Best Photo:** Sharpness + face detection weighted highest → selected clear, frontal photos ✓

---

## 4. ML CACHING MECHANISM (`ml_cache.py`)

**Purpose:** Avoid recomputing expensive ML operations on duplicate uploads.

```
Cache Structure:
{
  "photo_hash_abc123": {
    "face_embeddings": [[0.24, -0.15, ...], [...]],
    "cnn_features": [0.05, 0.12, ...],
    "outfit_signature": [0.12, 0.34, ...],
    "timestamp": 1705327845
  }
}

Workflow:
1. User uploads photo → compute SHA256 hash of image file
2. Check if hash exists in ml_cache.json
3. If EXISTS:
   → Load cached features immediately (avoids re-detection)
   → Use for clustering
4. If NOT EXISTS:
   → Extract all features (face, CNN, outfit)
   → Store in cache
   → Use for clustering

Benefits:
• If user uploads same photo twice → 90% faster second time
• Reduces redundant face detection, embedding generation
• Maintains ML consistency across uploads
```

---

## 5. TECHNOLOGY STACK SUMMARY

| Component | Technology | Why Chosen |
|-----------|-----------|-----------|
| **Face Detection** | dlib (HOG) | Lightweight, accurate, no GPU needed |
| **Face Encoding** | face_recognition (pre-trained CNN) | Fast (128-dim), robust to angle/lighting |
| **Clustering** | scikit-learn DBSCAN | Finds variable-density clusters, no K selection |
| **Color Analysis** | OpenCV HSV histograms | Domain-specific for clothing, no ML overhead |
| **Texture Features** | Local Binary Patterns (LBP) | Fast, rotation-invariant, good for image similarity |
| **Edge Detection** | Canny edge detector | Efficient boundary detection |
| **Metadata** | PIL + pillow_heif | Parse EXIF, HEIC support |
| **Feature Similarity** | Cosine distance (normalized L2) | Scale-invariant, interpretable |
| **Photo Scoring** | OpenCV metrics (Laplacian, histogram) | No ML model needed, interpretable scores |
| **Scene Detection** | HSV color ratios | Domain knowledge (blue = beach, green = nature) |
| **Caching** | JSON file + SHA256 hashing | Fast lookups, no external DB needed |

---

## 6. KEY DESIGN DECISIONS

### Why Outfit-First Clustering?
Traditional clustering groups by time/location first. Problems:
- Same outfit across 2 days = different events (WRONG)
- 4 outfits in 1 hour = same event (WRONG)

**Solution:** Outfit as primary clustering key ensures photos group together regardless of date/location if clothing is similar. Then refine by time/location. This matches human memory ("wearing my red shirt that day").

### Why DBSCAN Instead of K-Means?
- **K-Means:** Requires knowing K in advance (number of clusters)
- **DBSCAN:** Density-based, finds natural clusters in data
  - Handles variable cluster sizes (5-face cluster + 1-face outlier)
  - Soft boundaries (faces at threshold get isolated)

### Why 464-Dimensional CNN Features?
- Combines multi-scale information:
  - **Color** (192 dims): Global color palette
  - **Texture** (256 dims): Local patterns (fabric, surface)
  - **Edges** (16 dims): Structure/composition
- No neural network training needed (pre-computed patterns)
- Lightweight + interpretable vs. deep learning

### Why Multi-Criteria Photo Scoring?
No single metric defines "best photo." Weighting captures priorities:
- **Sharpness (35%):** Core requirement (blurry = useless)
- **Faces (30%):** Recognize people in event
- **Brightness (15%):** Avoid dark/washed photos
- **Contrast (15%):** Vivid > flat
- **Composition (5%):** Minor preference (aspect ratio)

---

## 7. LIMITATIONS & FUTURE IMPROVEMENTS

**Current Limitations:**
1. **Outfit Detection:** Color-based only (doesn't detect style, fit)
   - Fix: Add texture descriptors (checks, stripes, patterns)
2. **Face Clustering:** eps=0.50 may be sensitive to lighting variations
   - Fix: Implement angle/lighting invariance in face encoding
3. **Scene Detection:** Heuristic color thresholds (not learned)
   - Fix: Train lightweight classifier on scene examples
4. **Best Photo:** Laplacian variance can fail on some filter types
   - Fix: Multi-metric sharpness (FFT analysis, edge-based)

**Potential Enhancements:**
1. **Object Detection:** Add person pose (standing vs sitting) to outfit matching
2. **Activity Recognition:** Detect activity type (hiking, swimming, eating) for better event naming
3. **Clothing Attributes:** Detect "formal" vs "casual" to group outfit similarity
4. **ML Personalization:** Learn user preferences (which photos they keep/delete) → tune weights
5. **Temporal Coherence:** Use ML to detect "outfit changes" within a day → multi-outfit events

---

## 8. QUICK REFERENCE: ALGORITHM PARAMETERS

```python
# Outfit clustering threshold (cosine distance)
OUTFIT_THRESHOLD = 0.15  # [0=identical, 1=completely different]

# Face DBSCAN parameters
FACE_DBSCAN_EPS = 0.50   # Euclidean distance in 128-dim space
FACE_DBSCAN_MIN_SAMPLES = 1

# CNN feature DBSCAN parameters
CNN_DBSCAN_EPS = 0.25    # Stricter than faces (more visual similarity needed)
CNN_DBSCAN_MIN_SAMPLES = 1

# Time clustering tolerance
TIME_TOLERANCE = 1 hour  # Photos within 1 hour of each other may be same event

# GPS distance tolerance
GPS_TOLERANCE = 1 km     # Photos within 1 km of each other may be same event

# Photo scoring weights
SHARPNESS_WEIGHT = 0.35
FACE_DETECTION_WEIGHT = 0.30
BRIGHTNESS_WEIGHT = 0.15
CONTRAST_WEIGHT = 0.15
COMPOSITION_WEIGHT = 0.05

# Optimal brightness level
OPTIMAL_BRIGHTNESS = 128  # out of 255
```

---

## 9. VISUAL SYSTEM DIAGRAM

```
┌──────────────────────────────────────────────────────────────────────┐
│                         USER UPLOADS 12 PHOTOS                       │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
     ┌────────────────────────────┼────────────────────────────┐
     ↓                            ↓                            ↓
OUTFIT ENGINE            FACE RECOGNITION             CNN FEATURES
 HSV Histogram          + Face Encoding            + LBP Texture
 (48 dims)              (128 dims)                  + Edge Grid
     ↓                            ↓                            ↓
[0.12,0.34..] → DBSCAN      [0.24,-0.15..] → DBSCAN     [0.05,0.12..] → DBSCAN
                                  ↓                            ↓
                           PERSON CLUSTERS             VISUAL CLUSTERS
     ↓
  OUTFIT
 CLUSTERS
     ↓
     └────────────────────────────┼────────────────────────────┘
                                  ↓
                    TIME + GPS CLUSTERING
                    (1 hour ± 1 km window)
                                  ↓
                        REFINED EVENTS
                    [photo1, photo2, ...]
                                  ↓
                        ┌─────────┼─────────┐
                        ↓         ↓         ↓
                    SCENE      METADATA   AI SCORING
                 DETECTION    (GPS/TIME)  (Sharpness,
                (Beach/Food)             Faces, Brightness)
                        ↓         ↓         ↓
                        └─────────┼─────────┘
                                  ↓
                        ┌──────────────────────┐
                        │  FINAL EVENTS OUTPUT │
                        │ {name, photos,       │
                        │  best_photo, score}  │
                        └──────────────────────┘
```

---

This comprehensive architecture shows how MemoryMap leverages classical computer vision + statistical ML to automatically organize photos into semantically meaningful events without requiring deep neural networks or expensive cloud APIs.
