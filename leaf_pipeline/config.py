from pathlib import Path
import numpy as np

# ======================================================
# RUTAS BASE
# ======================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# Sube 3 niveles: leaf_pipeline -> napari -> Projects -> Python
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  
DATA_ROOT = PROJECT_ROOT / "Data"
OUTPUT_ROOT = Path("data/processed")

ALL_DATASET_ALIAS = "all-dataset"
ALL_DATASET_NAME = "all_dataset"

# ======================================================
# UMBRALES DE COLOR (Para máscaras morfológicas)
# ======================================================
LOWER_LEAF = np.array([0, 35, 20])
UPPER_LEAF = np.array([100, 255, 255])

# ======================================================
# PARÁMETROS DE CROP
# ======================================================
PAD_RATIO = 0.02
PAD_MIN_PX = 10

# ======================================================
# LANDMARK CONFIGURATION
# ======================================================
LANDMARK_COLORS = [
    (0,   0,   255), (0,   100, 255), (0,   200, 255), (0,   255, 255),
    (255, 0,   255), (255, 0,   180), (200, 0,   200), (150, 0,   255),
    (100, 0,   255), (255, 0,   0),   (255, 100, 0),   (255, 200, 0),
    (255, 255, 0),   (0,   150, 200),
]

LANDMARK_NAMES = [
    "seno_peciolar", "punta_L1", "punta_L2_izq", "punta_L2_der",
    "punta_L3_izq", "punta_L3_der", "punta_L4_izq", "punta_L4_der",
]

LANDMARK_IDS = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}