# Leaf Pipeline

Image processing pipeline for grape leaf morphometric analysis. Detects leaves on squared-paper backgrounds, crops to a tight field of view, finds 8 anatomical landmarks, and calibrates pixel-to-centimeter ratios from the background grid.


## Pipeline Stages

| Stage | Flag | Description |
|-------|------|-------------|
| **Crop** | `--mode crop` | Detects the leaf via HSV color segmentation, creates a tight bounding-box crop with adaptive padding |
| **Landmark** | `--mode landmark` | Detects 8 landmarks: peciolar sinus (PEC) + lobe tips L1, L2, L3, L4 (left/right) |
| **Pixel2cm** | `--mode pixel2cm` | Measures px/cm by counting grid lines on the squared-paper background (10 lines = 1 cm) |
| **All** | `--mode all` | Runs crop, then landmark, then pixel2cm in sequence |


## Directory Structure

```
leaf_pipeline/
  leaf_pipeline.py       # Main script
  README.md
  requirements.txt
  .gitignore
  data/
    raw/                 # Raw images (input)
      tempranillo/
        Temp 3.jpeg
      syrah/
        Syrah 1.jpeg
    processed/           # Processed outputs
      {variety}/
        cropped/         # Cropped images
        landmarks/       # Annotated images + landmarks.csv
        pixel2cm/        # pixel2cm.csv + pixel2cm_detail.csv
      all_dataset/       # Batch mode consolidated output
        cropped/
        landmarks/
        pixel2cm/
```


## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Organize raw images

Place images inside `data/raw/{variety}/`, one subdirectory per variety:

```bash
mkdir -p data/raw/tempranillo
mkdir -p data/raw/syrah
# Copy your images here
```

### 3. Run the pipeline

**Crop only:**
```bash
python leaf_pipeline.py --mode crop --input tempranillo
```

**Full pipeline for one variety:**
```bash
python leaf_pipeline.py --mode crop --input tempranillo
python leaf_pipeline.py --mode landmark --input tempranillo --from-cropped
python leaf_pipeline.py --mode pixel2cm --input tempranillo --from-cropped
```

**Full pipeline for all varieties (batch mode):**
```bash
python leaf_pipeline.py --mode all --input all-dataset
```

### 4. View outputs

- **Cropped images:** `data/processed/{variety}/cropped/`
- **Landmarks:** `data/processed/{variety}/landmarks/landmarks.csv`
- **Pixel calibration:** `data/processed/{variety}/pixel2cm/pixel2cm.csv`
- **Measurement detail:** `data/processed/{variety}/pixel2cm/pixel2cm_detail.csv`


## Output Formats

### `landmarks.csv`
```
image,variedad,landmark_id,landmark,x,y,px_per_cm
syrah_Syrah 1.jpeg,syrah,0,seno_peciolar,573,814,59.0000
syrah_Syrah 1.jpeg,syrah,1,punta_L1,576,21,59.0000
...
```

Landmark IDs:
| ID | Name | Description |
|----|------|-------------|
| 0 | `seno_peciolar` | Petiolar sinus |
| 1 | `punta_L1` | Top lobe tip |
| 2 | `punta_L2_izq` | Left L2 lobe tip |
| 3 | `punta_L2_der` | Right L2 lobe tip |
| 4 | `punta_L3_izq` | Left L3 lobe tip |
| 5 | `punta_L3_der` | Right L3 lobe tip |
| 6 | `punta_L4_izq` | Left L4 lobe tip |
| 7 | `punta_L4_der` | Right L4 lobe tip |

### `pixel2cm.csv` (summary)
```
image,variedad,px_per_cm
syrah_Syrah 1.jpeg,syrah,59.0000
tempranillo_Temp 3.jpeg,tempranillo,57.0000
```

### `pixel2cm_detail.csv` (individual measurements)
```
image,variedad,direction,start_idx,end_idx,start_pos,end_pos,n_lines,distance_px,px_per_cm,is_outlier
syrah_Syrah 1.jpeg,syrah,H,0,10,1,86,10,85.00,85.0000,1
syrah_Syrah 1.jpeg,syrah,V,2,12,68,126,10,58.00,58.0000,0
...
```

Each row is one N-line span measurement. `is_outlier=1` marks measurements filtered by the robust MAD-based outlier rejection.


## Requirements

- Python >= 3.9
- Images should be taken on **squared paper** (1 cm grid with 1 mm subdivisions = 10x10 small squares per cm)
- Leaf should be well-lit, flat against the paper, with minimal shadow


## Algorithm Notes

**Crop:** HSV color segmentation `[0-100, 35-255, 20-255]` captures greens through browns; morphological open/close cleans noise and vein gaps; adaptive padding (`max(2% of size, 10px)`).

**PEC detection:** Hybrid approach -- uses contour convexity-defect depth to decide between visible U-notch (depth > 60px) or midrib skeleton endpoint (overlapping lobes case).

**Lobe tips:** Prominence-filtered peak detection on distance-from-PEC profile using `scipy.signal.find_peaks`.

**Pixel2cm:** Detects grid lines via 1D projections on a band around the leaf contour, clusters close peaks (line thickness), measures 10-line spans, filters outliers with MAD (Median Absolute Deviation).


## CLI Help

```bash
python leaf_pipeline.py --help
```
