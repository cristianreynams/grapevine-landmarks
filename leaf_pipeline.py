import cv2
import numpy as np
from pathlib import Path
import argparse
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

try:
    from skimage.morphology import skeletonize as sk_skeletonize
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "scikit-image", "-q"], check=True)
    from skimage.morphology import skeletonize as sk_skeletonize

from tqdm import tqdm


# ======================================================
# CLI
# ======================================================

parser = argparse.ArgumentParser(
    description=(
        "Grape leaf image processing pipeline. "
        "Supports crop, landmark detection, pixel-to-cm calibration, "
        "and full pipeline execution."
    ),
    epilog=(
        "Examples:\n"
        "  # Crop images from a single variety\n"
        "  python leaf_pipeline.py --mode crop --input tempranillo\n\n"
        "  # Detect landmarks on cropped images\n"
        "  python leaf_pipeline.py --mode landmark "
        "--input tempranillo --from-cropped\n\n"
        "  # Measure px/cm on cropped images\n"
        "  python leaf_pipeline.py --mode pixel2cm "
        "--input tempranillo --from-cropped\n\n"
        "  # Full pipeline: crop + landmark + pixel2cm for all varieties\n"
        "  python leaf_pipeline.py --mode all --input all-dataset\n"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument(
    "--mode",
    required=True,
    choices=[
        "crop",
        "landmark",
        "pixel2cm",
        "all",
    ],
    help=(
        "Pipeline stage. "
        "crop: detect leaf and create tight crop; "
        "landmark: detect 8 anatomical landmarks (PEC + L1-L4); "
        "pixel2cm: measure px/cm from grid paper background; "
        "all: run crop, then landmark, then pixel2cm in sequence."
    ),
)

parser.add_argument(
    "--input",
    default="",
    help=(
        "Folder name inside data/raw/ (or data/processed/ "
        "when using --from-cropped). "
        "Use 'all-dataset' to batch-process all subdirectories. "
        "Required for --mode crop, landmark, and pixel2cm."
    ),
)

parser.add_argument(
    "--from-cropped",
    action="store_true",
    help=(
        "Read input images from data/processed/{input}/cropped/ "
        "instead of data/raw/{input}/. "
        "Required for landmark and pixel2cm modes when processing "
        "images that were already cropped."
    ),
)

# args is parsed only when running as a script (see __main__ block)
# Functions that need args should handle the case where args is None
args = None  # placeholder, set in __main__


# ======================================================
# PATHS
# ======================================================

OUTPUT_ROOT = Path(
    "data/processed"
)


def get_input_dir():
    """
    Return the input directory based on CLI flags.

    Default:    data/raw/{input}
    --from-cropped: data/processed/{input}/cropped
    """
    global args

    if args is None:
        return Path("data/raw")

    if args.from_cropped:
        return (
            OUTPUT_ROOT /
            args.input /
            "cropped"
        )

    return (
        Path("data/raw") /
        args.input
    )


# Lazy initialization — INPUT_DIR is computed on first use, not at import time
INPUT_DIR = None


def _get_input_dir_lazy():
    """Return INPUT_DIR, initializing on first call."""
    global INPUT_DIR
    if INPUT_DIR is None:
        INPUT_DIR = get_input_dir()
    return INPUT_DIR


LANDMARK_CSV = Path(
    "output/debug_landmarks.csv"
)


# ======================================================
# COLOR THRESHOLDS
# ======================================================

# HSV range for leaf detection.
# Expanded from pure green to include yellows, browns,
# and reddish wilted tissue so the entire leaf is captured
# even when parts have dried or discolored.
#
# H: 0-100  covers greens through yellows/browns
#     (OpenCV hue wraps: 0 = red, 25 = yellow, 50 = green,
#      75 = cyan; browns are low-H low-V yellows)
# S: 35-255 saturation floor keeps out gray paper background
# V: 20-255 allows darker shadowed areas of the leaf

LOWER_LEAF = np.array([
    0,
    35,
    20
])

UPPER_LEAF = np.array([
    100,
    255,
    255
])


# ======================================================
# CROP PARAMETERS
# ======================================================

# Padding added around the detected leaf mask bbox.
# Computed as max(mask_dimension * PAD_RATIO, PAD_MIN_PX).
# 2% of leaf size with 10px minimum gives a tight but safe
# margin that preserves all leaf tips without over-cropping.
PAD_RATIO = 0.02
PAD_MIN_PX = 10


# ======================================================
# LANDMARK CONFIGURATION
# ======================================================

# 14 distinct high-contrast BGR colors for X marks.
# All avoid the green hue range of the leaf.
# Ordered: 7 sinuses first, then 7 lobe tips.

LANDMARK_COLORS = [
    (0,   0,   255),   # 0:  Red          -> seno_peciolar
    (0,   100, 255),   # 1:  Orange       -> seno_superior_izq
    (0,   200, 255),   # 2:  Gold         -> seno_superior_der
    (0,   255, 255),   # 3:  Yellow       -> seno_medio_izq
    (255, 0,   255),   # 4:  Magenta      -> seno_medio_der
    (255, 0,   180),   # 5:  Deep Pink    -> seno_inferior_izq
    (200, 0,   200),   # 6:  Purple       -> seno_inferior_der
    (150, 0,   255),   # 7:  Violet       -> punta_L1
    (100, 0,   255),   # 8:  Blue-Violet  -> punta_L2_izq
    (255, 0,   0),     # 9:  Blue         -> punta_L2_der
    (255, 100, 0),     # 10: Azure        -> punta_L3_izq
    (255, 200, 0),     # 11: Deep Sky     -> punta_L3_der
    (255, 255, 0),     # 12: Cyan         -> punta_L4_izq
    (0,   150, 200),   # 13: Coral        -> punta_L4_der
]

LANDMARK_NAMES = [
    "seno_peciolar",
    "punta_L1",
    "punta_L2_izq",
    "punta_L2_der",
    "punta_L3_izq",
    "punta_L3_der",
    "punta_L4_izq",
    "punta_L4_der",
]

# Map landmark name -> numeric ID (for CSV export)
LANDMARK_IDS = {
    name: idx for idx, name in enumerate(LANDMARK_NAMES)
}


# ======================================================
# ALL-DATASET CONFIGURATION
# ======================================================

# Alias used on CLI for processing all varieties at once
ALL_DATASET_ALIAS = "all-dataset"
ALL_DATASET_NAME = "all_dataset"


def is_all_dataset():
    """Check if the user requested processing all varieties."""
    global args
    if args is None:
        return False
    return args.input == ALL_DATASET_ALIAS


def get_raw_variety_dirs():
    """
    Return a list of variety subdirectories inside data/raw/.
    Each subfolder is treated as a different grape variety.
    """
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return []
    return sorted([
        p for p in raw_dir.iterdir()
        if p.is_dir() and p.name != ALL_DATASET_NAME
    ])


def all_dataset_cropped_dir():
    """Return the consolidated cropped output directory."""
    return OUTPUT_ROOT / ALL_DATASET_NAME / "cropped"


def all_dataset_landmarks_dir():
    """Return the consolidated landmarks output directory."""
    return OUTPUT_ROOT / ALL_DATASET_NAME / "landmarks"


def all_dataset_pixel2cm_dir():
    """Return the consolidated pixel2cm output directory."""
    return OUTPUT_ROOT / ALL_DATASET_NAME / "pixel2cm"


def all_dataset_csv_dir():
    """Return the consolidated CSV output directory."""
    return OUTPUT_ROOT / ALL_DATASET_NAME / "csv"


# ======================================================
# UTILS
# ======================================================

def ensure_dir(path):

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def get_image_files():
    return sorted([
        p for p in _get_input_dir_lazy().iterdir()
        if p.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]
    ])


# ======================================================
# MASK DETECTION
# ======================================================

def detect_leaf_mask(image_bgr):
    """
    Detect the leaf region using HSV color segmentation.

    Pipeline:
        1. HSV threshold with green range
        2. Morphological open to remove grid-paper noise
        3. Morphological close to fill vein-shadow gaps
        4. Keep largest contour (the leaf)
        5. Flood-fill to close any internal holes

    Args:
        image_bgr: Input image in BGR format.

    Returns:
        Binary mask (uint8) with the leaf region filled.
    """

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        LOWER_LEAF,
        UPPER_LEAF
    )

    # Step 2: Remove small noise from grid paper

    kernel_open = np.ones(
        (5, 5),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_open,
        iterations=1
    )

    # Step 3: Close gaps from dark vein shadows

    kernel_close = np.ones(
        (7, 7),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2
    )

    # Step 4: Keep only the largest contour (the leaf)

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return cleaned

    largest = max(
        contours,
        key=cv2.contourArea
    )

    mask_clean = np.zeros_like(mask)

    cv2.drawContours(
        mask_clean,
        [largest],
        -1,
        255,
        -1
    )

    # Step 5: Fill any internal holes

    h, w = mask_clean.shape

    flood = mask_clean.copy()

    mask_ff = np.zeros(
        (h + 2, w + 2),
        np.uint8
    )

    cv2.floodFill(
        flood,
        mask_ff,
        (0, 0),
        255
    )

    holes = cv2.bitwise_not(flood)

    # Exclude border regions from hole mask

    holes[:, 0] = 0
    holes[:, -1] = 0
    holes[0, :] = 0
    holes[-1, :] = 0

    mask_filled = cv2.bitwise_or(
        mask_clean,
        holes
    )

    return mask_filled


# ======================================================
# CROP
# ======================================================

def compute_crop_box(mask, img_h, img_w):
    """
    Compute a tight crop box from a binary leaf mask.

    The bbox is the mask's extent plus padding computed as:
        padding = max(max_dim * PAD_RATIO, PAD_MIN_PX)

    The result is clamped to image boundaries.

    Args:
        mask: Binary uint8 mask with leaf region.
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        (xmin, xmax, ymin, ymax, padding) crop box in pixel coords.
        If mask is empty, returns a centered fallback crop.
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        # Fallback: center crop at 80% of image

        margin_x = int(img_w * 0.1)
        margin_y = int(img_h * 0.1)

        return (
            margin_x,
            img_w - 1 - margin_x,
            margin_y,
            img_h - 1 - margin_y,
            0
        )

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    mask_w = x_max - x_min
    mask_h = y_max - y_min

    padding = max(
        int(max(mask_w, mask_h) * PAD_RATIO),
        PAD_MIN_PX
    )

    xmin = max(0, x_min - padding)
    xmax = min(img_w - 1, x_max + padding)
    ymin = max(0, y_min - padding)
    ymax = min(img_h - 1, y_max + padding)

    return (xmin, xmax, ymin, ymax, padding)


def _crop_and_save(image_path, output_dir, debug_dir,
                    name_prefix=""):
    """Crop a single image and save to output_dir."""

    print(
        f"\nCropping:\n{image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print("Could not load image.")
        return

    img_h, img_w = image.shape[:2]

    # ==================================================
    # DETECT LEAF MASK
    # ==================================================

    leaf_mask = detect_leaf_mask(image)

    mask_pixels = int(np.sum(leaf_mask > 0))

    print(
        f"  Leaf mask: "
        f"{mask_pixels} px"
    )

    # ==================================================
    # COMPUTE CROP BOX
    # ==================================================

    xmin, xmax, ymin, ymax, pad = compute_crop_box(
        leaf_mask,
        img_h,
        img_w
    )

    crop_w = xmax - xmin + 1
    crop_h = ymax - ymin + 1

    print(
        f"  Mask bbox:  "
        f"x[{xmin}-{xmax}] y[{ymin}-{ymax}]"
    )
    print(
        f"  Crop:       "
        f"{crop_w}x{crop_h}  "
        f"(padding={pad}px)"
    )

    # ==================================================
    # APPLY CROP
    # ==================================================

    cropped = image[ymin:ymax+1, xmin:xmax+1]

    # ==================================================
    # DEBUG VISUALIZATION
    # ==================================================

    debug = image.copy()

    # Leaf mask contour (green overlay)

    mask_contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        debug,
        mask_contours,
        -1,
        (0, 255, 0),
        2
    )

    # Crop rectangle (red)

    cv2.rectangle(
        debug,
        (xmin, ymin),
        (xmax, ymax),
        (0, 0, 255),
        3
    )

    # Padding info text

    label = (
        f"crop: {crop_w}x{crop_h}  "
        f"pad:{pad}px"
    )

    cv2.putText(
        debug,
        label,
        (xmin + 5, ymin - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # Ensure .png extension for saved crops
    stem = Path(image_path.name).stem
    png_name = f"{name_prefix}{stem}.png"

    ensure_dir(debug_dir)

    cv2.imwrite(
        str(debug_dir / png_name),
        debug
    )

    # ==================================================
    # SAVE CROPPED IMAGE (PNG, lossless)
    # ==================================================

    save_path = output_dir / png_name

    cv2.imwrite(str(save_path), cropped)

    reduction = (
        1.0 - (crop_w * crop_h) / (img_w * img_h)
    ) * 100.0

    print(
        f"  Saved:      {save_path}\n"
        f"  Reduction:  {reduction:.1f}%"
    )


def crop_images():
    """
    Crop images.  Supports both single-variety and batch (all-dataset).

    Single variety:
        --input my_variety     -> writes to data/processed/{input}/cropped/

    All varieties (batch):
        --input all-dataset    -> reads all subdirs in data/raw/
                                  writes ONLY to data/processed/all_dataset/cropped/
                                  (no per-variety duplication)
    """

    if is_all_dataset():
        # Batch mode: process every subdirectory in data/raw/
        variety_dirs = get_raw_variety_dirs()

        if not variety_dirs:
            print(
                "\nNo variety directories found in data/raw/"
            )
            return

        output_dir = all_dataset_cropped_dir()
        debug_dir = (
            OUTPUT_ROOT /
            ALL_DATASET_NAME /
            "cropped_debug"
        )

        ensure_dir(output_dir)

        print(
            f"\n=== BATCH CROP: {len(variety_dirs)} varieties ==="
        )
        print(
            f"Output: {output_dir}\n"
        )

        total_images = 0

        for variety_dir in variety_dirs:
            variety = variety_dir.name

            variety_input = Path("data/raw") / variety

            image_files = sorted([
                p for p in variety_input.iterdir()
                if p.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]
            ])

            if not image_files:
                continue

            print(
                f"  {variety}: {len(image_files)} images"
            )

            for image_path in tqdm(
                image_files,
                desc=f"  {variety}",
                unit="img",
                ncols=80,
                leave=True,
            ):
                prefix = f"{variety}_"
                _crop_and_save(
                    image_path,
                    output_dir,
                    debug_dir,
                    prefix
                )
                total_images += 1

        print(
            f"\n=== TOTAL: {total_images} images cropped ==="
        )
        print(
            f"Output: {output_dir}"
        )

    else:
        # Single-variety mode
        output_dir = (
            OUTPUT_ROOT /
            args.input /
            "cropped"
        )
        debug_dir = (
            OUTPUT_ROOT /
            args.input /
            "cropped_debug"
        )

        ensure_dir(output_dir)

        image_files = get_image_files()

        print(
            f"\nFound {len(image_files)} images"
        )

        for image_path in tqdm(
            image_files,
            desc=f"  {args.input}",
            unit="img",
            ncols=80,
            leave=True,
        ):
            _crop_and_save(
                image_path,
                output_dir,
                debug_dir
            )


# ======================================================
# PIXEL TO CM CONVERSION
# ======================================================

def measure_pixel2cm(image_bgr, leaf_mask=None, n_lines=10):
    """
    Measure pixel-to-cm ratio by counting grid lines on squared paper.

    Strategy:
        1. Create a band around the leaf contour where grid lines
           are visible on the background paper.
        2. Detect dark grid lines via 1D projections + peak detection.
        3. Cluster close peaks (same physical line has pixel thickness).
        4. Measure distances spanning N consecutive lines.
        5. Each N-line span represents 1 cm (10 subdivisions = 1 cm).
        6. Filter outliers using MAD and return robust median.

    Returns:
        (px_per_cm, measurements) where px_per_cm is the robust
        median px/cm and measurements is a list of dicts, one per
        individual measurement.
    """

    h_img, w_img = image_bgr.shape[:2]

    if leaf_mask is None:
        leaf_mask = detect_leaf_mask(image_bgr)

    contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, []

    contour = max(contours, key=cv2.contourArea)

    # ---- Band mask (area just outside leaf contour) ----

    contour_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    band_inner = 50
    band_outer = 250

    ki = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (band_inner * 2 + 1, band_inner * 2 + 1)
    )

    ko = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (band_outer * 2 + 1, band_outer * 2 + 1)
    )

    dilated_inner = cv2.dilate(contour_mask, ki, iterations=1)
    dilated_outer = cv2.dilate(contour_mask, ko, iterations=1)
    band_mask = cv2.subtract(dilated_outer, dilated_inner)

    # ---- Detect dark grid lines ----

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    dark_band = cv2.bitwise_and(dark, band_mask)

    h_proj = np.sum(dark_band, axis=1)
    v_proj = np.sum(dark_band, axis=0)

    h_peaks, _ = find_peaks(
        h_proj,
        distance=4,
        prominence=np.std(h_proj) * 0.3
    )

    v_peaks, _ = find_peaks(
        v_proj,
        distance=4,
        prominence=np.std(v_proj) * 0.3
    )

    # Cluster close peaks (grid lines have pixel-level thickness)

    def _cluster_peaks(peaks, max_gap=3):
        if len(peaks) == 0:
            return []
        peaks = sorted(peaks)
        clusters = []
        current = [peaks[0]]
        for p in peaks[1:]:
            if p - current[-1] <= max_gap:
                current.append(p)
            else:
                clusters.append(int(np.median(current)))
                current = [p]
        clusters.append(int(np.median(current)))
        return clusters

    h_lines = _cluster_peaks(h_peaks.tolist(), max_gap=3)
    v_lines = _cluster_peaks(v_peaks.tolist(), max_gap=3)

    # ---- Measure N-line spans ----

    def _measure_direction(lines, direction):
        if len(lines) < n_lines + 1:
            return []

        diffs = np.diff(lines)

        # Find continuous segments (no gaps > 20 px)
        segments = []
        start = 0
        for i, d in enumerate(diffs):
            if d > 20:
                if i - start >= n_lines:
                    segments.append((start, i))
                start = i + 1
        if len(lines) - 1 - start >= n_lines:
            segments.append((start, len(lines) - 1))

        measurements = []

        for seg_start, seg_end in segments:
            for i in range(seg_start, seg_end - n_lines + 1):
                segment_diffs = diffs[i:i + n_lines]
                if any(d > 20 for d in segment_diffs):
                    continue

                dist_px = lines[i + n_lines] - lines[i]

                measurements.append({
                    'direction': direction,
                    'start_idx': i,
                    'end_idx': i + n_lines,
                    'start_pos': lines[i],
                    'end_pos': lines[i + n_lines],
                    'n_lines': n_lines,
                    'distance_px': dist_px,
                    'px_per_cm': dist_px,
                })

        return measurements

    all_measurements = []
    all_measurements.extend(_measure_direction(h_lines, 'H'))
    all_measurements.extend(_measure_direction(v_lines, 'V'))

    if not all_measurements:
        return None, []

    # ---- Robust outlier filtering (MAD) ----

    all_px = [m['px_per_cm'] for m in all_measurements]
    median_px = float(np.median(all_px))
    mad = float(np.median(np.abs(np.array(all_px) - median_px)))

    for m in all_measurements:
        if mad == 0:
            m['is_outlier'] = False
        else:
            m['is_outlier'] = (
                abs(m['px_per_cm'] - median_px) / mad > 3.0
            )

    good_px = [
        m['px_per_cm'] for m in all_measurements
        if not m['is_outlier']
    ]

    if good_px:
        px_per_cm = float(np.median(good_px))
    else:
        px_per_cm = median_px

    return px_per_cm, all_measurements


# ======================================================
# LANDMARK DETECTION
# ======================================================

# ======================================================
# LANDMARK DETECTION — 5-Stage Pipeline
#
# Each landmark has its own function with isolated config.
# Adjusting one stage does NOT affect others.
# ======================================================

# ---- Stage 1: PECIOLE SINUS ----
# Tunable parameters for find_peciole_sinus()
PEC_DEFECT_DEPTH_THRESHOLD = 60    # px: U-notch vs skeleton decision
PEC_LOWER_Y_FACTOR = 0.6           # fraction of cy: lower half search
PEC_CENTER_BAND = 0.2              # fraction of w: horizontal search band
PEC_REFINE_STEPS = 25              # max refinement steps for v8
PEC_REFINE_STEP = 3                # px: refinement step size
PEC_SKELETON_BAND = 0.033          # fraction of w: midrib search band


def find_peciole_sinus(image_bgr, mask):
    """
    Stage 1: Find the petiolar sinus (PEC).

    Hybrid approach:
      - v8 (convexity defect): when U-notch is clearly visible
      - v9 (skeleton endpoint): when lower lobes overlap (e.g. Tempranillo)

    Parameters (module-level, tunable):
      PEC_DEFECT_DEPTH_THRESHOLD: px threshold for v8 vs v9 decision
      PEC_LOWER_Y_FACTOR: lower-half search region factor
      PEC_CENTER_BAND: horizontal search band around center
      PEC_REFINE_STEPS / PEC_REFINE_STEP: v8 refinement config
      PEC_SKELETON_BAND: midrib skeleton search band

    Returns: (x, y) of the PEC
    """
    h, w = image_bgr.shape[:2]
    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    # --- v8: deepest convexity defect in lower half ---
    best_defect = None
    best_depth = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            far_pt = tuple(contour[f][0])
            depth = d / 256.0
            if (far_pt[1] > cy * PEC_LOWER_Y_FACTOR and
                    abs(far_pt[0] - cx) < w * PEC_CENTER_BAND and
                    depth > best_depth):
                best_depth = depth
                best_defect = far_pt

    if best_defect is None:
        return (cx, int(cy + h * 0.25))

    # Measure defect depth at v8 position
    contour_arr = contour.reshape(-1, 1, 2).astype(np.int32)
    dp = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            fp = tuple(contour_arr[f][0])
            if (abs(fp[1] - best_defect[1]) +
                    abs(fp[0] - best_defect[0]) * 0.5 < 80):
                dp = max(dp, d / 256.0)

    # --- v9: midrib skeleton endpoint (fallback) ---
    from skimage.morphology import skeletonize as sk_skel
    skel = (sk_skel(mask > 0) * 255).astype(np.uint8)
    band = max(20, int(w * PEC_SKELETON_BAND))
    center_pixels = [(x, y)
                     for y in range(int(cy * 0.3), min(h, int(cy * 1.4)))
                     for x in range(max(0, cx - band), min(w, cx + band))
                     if skel[y, x] > 0]
    v9 = None
    if center_pixels:
        midrib_mask = np.zeros_like(skel)
        for x, y in center_pixels:
            midrib_mask[y, x] = 1
        k = np.ones((3, 3), np.uint8)
        mm = cv2.dilate(midrib_mask, k, iterations=1)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mm, connectivity=8)
        best_comp = -1
        best_e = 0
        for i in range(1, n):
            m = labels == i
            ys, xs = np.where(m)
            if len(ys) > 0:
                ye = ys.max() - ys.min()
                xc = np.mean(xs)
                if abs(xc - cx) < w * 0.15 and ye > best_e:
                    best_e = ye
                    best_comp = i
        if best_comp > -1:
            ys, xs = np.where(labels == best_comp)
            li = np.where(ys == ys.max())[0]
            if len(li) > 0:
                v9 = (int(np.median([xs[i] for i in li])), int(ys.max()))

    # --- Decision: v8 (deep notch) or v9 (overlapping lobes) ---
    if dp > PEC_DEFECT_DEPTH_THRESHOLD:
        # U-notch visible: use convexity defect with optional
        # refinement. For most leaves the defect point is already
        # close to the true sinus. Only refine when the defect
        # is very low (near the leaf bottom), suggesting a deep
        # notch where the true sinus is higher up.
        bx, by = best_defect

        # Only refine if defect is very low (below 85% of height)
        if by > h * 0.85:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            leaf_m = cv2.inRange(hsv, LOWER_LEAF, UPPER_LEAF)
            # Walk up in small steps, keep x fixed at defect
            for step in range(1, 8):
                sy = by - step * 4
                if sy < int(cy * 0.5):
                    break
                if 0 <= sy < h and leaf_m[sy, bx] > 0:
                    return (bx, sy)

        return best_defect
    else:
        # Overlapping: use v9 or fallback
        if v9:
            return (int(v9[0] * 0.5 + cx * 0.5), v9[1])
        return best_defect


# ---- Stage 2: L1 (Top Lobe Tip) ----
# Tunable parameters for find_L1()
L1_CENTER_BAND = 0.12   # fraction of w: horizontal search band around PEC


def find_L1(image_bgr, mask, pec):
    """
    Stage 2: Find L1 — the top lobe tip.

    Method: highest contour point within a horizontal band centered on PEC.

    Parameters:
      L1_CENTER_BAND: horizontal search band (fraction of image width)

    Returns: (x, y) of L1, or None if not found
    """
    h, w = image_bgr.shape[:2]
    pec_x = int(pec[0])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.reshape(-1, 2)

    band = w * L1_CENTER_BAND
    center_mask = np.abs(contour_pts[:, 0] - pec_x) < band
    top = contour_pts[center_mask]

    if len(top) > 0:
        return tuple(top[np.argmin(top[:, 1])])
    return None


# ---- Stage 3: L2 (Upper Lobe Tips) ----
# Tunable parameters for find_L2()
L2_SIDE_OFFSET = 30          # px: min horizontal distance from center
L2_UPPER_Y_FACTOR = 1.0      # fraction: upper region cutoff (y < cy * factor)
L2_PEAK_DISTANCE = 40        # px: min distance between peaks
L2_PROMINENCE_FRACTION = 0.25  # fraction of max prominence to keep


def find_L2(image_bgr, mask, pec, l1):
    """
    Stage 3: Find L2 left and right — upper lobe tips.

    Method: prominence-filtered peaks in distance-from-PEC profile
            for the upper half of each side of the contour.

    Parameters:
      L2_SIDE_OFFSET: min px from center to consider a point on a side
      L2_UPPER_Y_FACTOR: y cutoff for upper region (fraction of cy)
      L2_PEAK_DISTANCE: min px between peaks in find_peaks
      L2_PROMINENCE_FRACTION: fraction of max prominence to filter peaks

    Returns: dict with 'izq' and 'der' keys, each (x, y) or None
    """
    h, w = image_bgr.shape[:2]
    pec_x, pec_y = int(pec[0]), int(pec[1])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.reshape(-1, 2)

    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Distance from PEC along contour
    dists = np.sqrt((contour_pts[:, 0] - pec_x) ** 2 +
                    (contour_pts[:, 1] - pec_y) ** 2)
    dists_smooth = uniform_filter1d(dists, size=7, mode='wrap')

    result = {}

    for side in ['izq', 'der']:
        if side == 'izq':
            side_mask = contour_pts[:, 0] < cx - L2_SIDE_OFFSET
        else:
            side_mask = contour_pts[:, 0] > cx + L2_SIDE_OFFSET

        pts = contour_pts[side_mask]
        d_smooth = dists_smooth[side_mask]

        if len(pts) < 3:
            result[side] = None
            continue

        # Upper half region
        upper_mask = pts[:, 1] < cy * L2_UPPER_Y_FACTOR
        upper_pts = pts[upper_mask]
        upper_d = d_smooth[upper_mask]

        l2 = None
        if len(upper_pts) > 0:
            peaks, props = find_peaks(upper_d, distance=L2_PEAK_DISTANCE,
                                      prominence=0)
            if len(peaks) > 0:
                max_prom = np.max(props['prominences'])
                threshold = max_prom * L2_PROMINENCE_FRACTION
                filtered = [p for i, p in enumerate(peaks)
                            if props['prominences'][i] > threshold]
                if filtered:
                    peak_pts = [tuple(upper_pts[p]) for p in filtered]
                    # L2 is a lateral lobe tip: it should be both high
                    # (small y) AND far from center (lateral). Filter
                    # out points that are too low, then score the rest.
                    # L2 should be in upper 60% of the leaf (above cy*0.6)
                    y_threshold = cy * 0.65
                    valid_pts = [p for p in peak_pts if p[1] < y_threshold]
                    if not valid_pts:
                        valid_pts = peak_pts
                    if side == 'izq':
                        lateral_scores = [cx - p[0] for p in valid_pts]
                    else:
                        lateral_scores = [p[0] - cx for p in valid_pts]
                    height_scores = [cy - p[1] for p in valid_pts]
                    # Balance: laterality and height both matter
                    combined = [
                        lateral * 1.5 + height * 1.0
                        for lateral, height in zip(lateral_scores, height_scores)
                    ]
                    l2 = valid_pts[int(np.argmax(combined))]

        if l2 is None:
            # Fallback: point with max lateral distance in upper half
            upper = pts[pts[:, 1] < cy]
            if len(upper) > 0:
                if side == 'izq':
                    l2 = tuple(upper[np.argmin(upper[:, 0])])
                else:
                    l2 = tuple(upper[np.argmax(upper[:, 0])])
            else:
                l2 = tuple(pts[np.argmin(pts[:, 1])])

        result[side] = l2

    return result


# ---- Stage 4: L3 (Middle Lobe Tips) ----
# Tunable parameters for find_L3()
L3_BELOW_L2_OFFSET = 120  # px: min y distance below L2 (increased to avoid L2)
L3_BELOW_PEC_MARGIN = 30  # px: max y distance above PEC
L3_PEAK_DISTANCE = 30     # px: min distance between peaks
L3_BELOW_L2_MIN = 80      # px: fallback min distance below L2


def find_L3(image_bgr, mask, pec, l2_izq, l2_der):
    """
    Stage 4: Find L3 left and right — middle lobe tips.

    Method: prominence-filtered peaks in the region between L2 and L4.
            Uses dynamic search region based on L2 position and PEC.

    Parameters:
      L3_BELOW_L2_OFFSET: min px below L2 to start search
      L3_BELOW_PEC_MARGIN: max px above PEC to end search
      L3_PEAK_DISTANCE: min px between peaks in find_peaks
      L3_BELOW_L2_MIN: fallback min px below L2

    Returns: dict with 'izq' and 'der' keys, each (x, y) or None
    """
    h, w = image_bgr.shape[:2]
    pec_x, pec_y = int(pec[0]), int(pec[1])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.reshape(-1, 2)

    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    dists = np.sqrt((contour_pts[:, 0] - pec_x) ** 2 +
                    (contour_pts[:, 1] - pec_y) ** 2)
    dists_smooth = uniform_filter1d(dists, size=7, mode='wrap')

    result = {}

    for side, l2 in [('izq', l2_izq), ('der', l2_der)]:
        if l2 is None:
            result[side] = None
            continue

        if side == 'izq':
            side_mask = contour_pts[:, 0] < cx - 30
        else:
            side_mask = contour_pts[:, 0] > cx + 30

        pts = contour_pts[side_mask]
        d_smooth = dists_smooth[side_mask]

        if len(pts) < 3:
            result[side] = None
            continue

        # Search region: between L2 and PEC
        mid_mask = ((pts[:, 1] > l2[1] + L3_BELOW_L2_OFFSET) &
                    (pts[:, 1] < pec_y + L3_BELOW_PEC_MARGIN))
        mid_pts = pts[mid_mask]
        mid_d = d_smooth[mid_mask]

        l3 = None
        if len(mid_pts) > 0:
            peaks, props = find_peaks(mid_d, distance=L3_PEAK_DISTANCE,
                                      prominence=0)
            if len(peaks) > 0:
                # L3 is the "lower" lobe tip in the middle region.
                # Among prominent peaks, prefer the one that is:
                # 1. Low in the region (higher y = further from L2)
                # 2. Has good prominence
                peak_pts_list = [tuple(mid_pts[p]) for p in peaks]
                prominences = props['prominences']
                # Score: combine prominence with "low-ness"
                scores = [
                    prom * 1.0 + (pt[1] - l2[1]) * 0.5
                    for prom, pt in zip(prominences, peak_pts_list)
                ]
                best_idx = int(np.argmax(scores))
                l3 = peak_pts_list[best_idx]

        # Fallback: point with max lateral distance between L2 and PEC
        if l3 is None:
            mid = pts[(pts[:, 1] > l2[1] + L3_BELOW_L2_MIN) &
                      (pts[:, 1] < pec_y)]
            if len(mid) > 0:
                lateral = np.abs(mid[:, 0] - cx)
                l3 = tuple(mid[np.argmax(lateral)])
            else:
                l3 = l2

        result[side] = l3

    return result


# ---- Stage 5: L4 (Lower Lobe Tips) ----
# Tunable parameters for find_L4()
L4_LOWER_Y_OFFSET = 50    # px: min y distance below cy
L4_IDEAL_LATERAL = 0.22   # fraction of w: ideal lateral distance
L4_LATERAL_WEIGHT = 0.8   # weight for lateral distance in scoring


def _score_l4_point(point, pec_x, ideal_lateral):
    """Score a candidate L4 point: lowest with lateral distance close to ideal."""
    lateral = np.abs(point[0] - pec_x)
    # Score: high y (low position) rewarded, penalize deviation from ideal lateral
    return point[1] * 1.0 - np.abs(lateral - ideal_lateral) * L4_LATERAL_WEIGHT


def find_L4(image_bgr, mask, pec, l3_izq, l3_der):
    """
    Stage 5: Find L4 left and right — lower lobe tips.

    Method: lowest contour point below the center, weighted by
            proximity to an ideal lateral distance from PEC.

    Parameters:
      L4_LOWER_Y_OFFSET: min px below cy to consider
      L4_IDEAL_LATERAL: ideal lateral distance (fraction of w)
      L4_LATERAL_WEIGHT: weight for lateral distance penalty

    Returns: dict with 'izq' and 'der' keys, each (x, y) or None
    """
    h, w = image_bgr.shape[:2]
    pec_x = int(pec[0])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.reshape(-1, 2)

    M = cv2.moments(mask)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    result = {}

    for side in ['izq', 'der']:
        if side == 'izq':
            side_mask = contour_pts[:, 0] < cx - 30
        else:
            side_mask = contour_pts[:, 0] > cx + 30

        pts = contour_pts[side_mask]
        if len(pts) < 3:
            result[side] = None
            continue

        # Points below center + offset
        lower = pts[pts[:, 1] > cy + L4_LOWER_Y_OFFSET]
        ideal_lateral = w * L4_IDEAL_LATERAL
        if len(lower) > 0:
            scores = np.array([_score_l4_point(p, pec_x, ideal_lateral) for p in lower])
            result[side] = tuple(lower[np.argmax(scores)])
        else:
            result[side] = tuple(pts[np.argmax(pts[:, 1])])

    return result


# ======================================================
# ORCHESTRATOR: detect_landmarks
# ======================================================

def detect_landmarks(image_bgr, leaf_mask):
    """
    Orchestrator: runs the 5-stage landmark detection pipeline.

    Stage 1: find_peciole_sinus() → PEC
    Stage 2: find_L1() → L1 (top tip)
    Stage 3: find_L2() → L2_izq, L2_der (upper tips)
    Stage 4: find_L3() → L3_izq, L3_der (middle tips)
    Stage 5: find_L4() → L4_izq, L4_der (lower tips)

    Each stage is independent — tuning one does not affect others.
    """

    # Stage 1: PEC
    pec = find_peciole_sinus(image_bgr, leaf_mask)
    landmarks = {'seno_peciolar': pec}

    # Stage 2: L1
    l1 = find_L1(image_bgr, leaf_mask, pec)
    if l1:
        landmarks['punta_L1'] = l1

    # Stage 3: L2
    l2 = find_L2(image_bgr, leaf_mask, pec, l1)
    for side in ['izq', 'der']:
        if l2.get(side):
            landmarks[f'punta_L2_{side}'] = l2[side]

    # Stage 4: L3
    l3 = find_L3(image_bgr, leaf_mask, pec,
                 l2.get('izq'), l2.get('der'))
    for side in ['izq', 'der']:
        if l3.get(side):
            landmarks[f'punta_L3_{side}'] = l3[side]

    # Stage 5: L4
    l4 = find_L4(image_bgr, leaf_mask, pec,
                 l3.get('izq'), l3.get('der'))
    for side in ['izq', 'der']:
        if l4.get(side):
            landmarks[f'punta_L4_{side}'] = l4[side]

    return landmarks


def annotate_landmarks(image_bgr, landmarks):
    """
    Draw X-shaped colored landmarks and text labels on the image.

    Args:
        image_bgr: Input image in BGR format.
        landmarks: Dict from detect_landmarks().

    Returns:
        (annotated_image_bgr, legend_list)
    """

    annotated = image_bgr.copy()
    img_h, img_w = annotated.shape[:2]

    marker_size = max(
        12,
        int(min(img_w, img_h) * 0.018)
    )

    thickness = max(
        2,
        int(min(img_w, img_h) * 0.004)
    )

    legend = []

    for i, name in enumerate(LANDMARK_NAMES):

        if name not in landmarks:
            continue

        pt = landmarks[name]
        color = LANDMARK_COLORS[i]

        px = int(pt[0])
        py = int(pt[1])

        s = marker_size

        # Draw X mark (two diagonal lines)

        cv2.line(
            annotated,
            (px - s, py - s),
            (px + s, py + s),
            color,
            thickness
        )

        cv2.line(
            annotated,
            (px - s, py + s),
            (px + s, py - s),
            color,
            thickness
        )

        # Center dot

        cv2.circle(
            annotated,
            (px, py),
            3,
            color,
            -1
        )

        # Text label with adaptive offset

        label = name.replace(
            'seno_', 'S_'
        ).replace(
            'punta_', 'T_'
        )

        offset_x = s + 8
        offset_y = -s - 8
        text_x = px + offset_x
        text_y = py + offset_y

        if text_x + 70 > img_w:
            text_x = px - s - 70

        if text_y < 15:
            text_y = py + s + 20

        cv2.putText(
            annotated,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        legend.append((name, pt, color))

    return annotated, legend


# ======================================================
# LANDMARK MODE
# ======================================================

def _process_landmarks(image_path, output_dir, csv_lines):
    """Detect landmarks on a single image and save results."""

    print(
        f"\nProcessing landmarks:\n{image_path.name}"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print("Could not load image.")
        return

    # Detect leaf mask

    leaf_mask = detect_leaf_mask(image)

    mask_pixels = int(np.sum(leaf_mask > 0))

    print(
        f"  Leaf mask: {mask_pixels} px"
    )

    # Detect landmarks

    landmarks = detect_landmarks(
        image,
        leaf_mask
    )

    if not landmarks:
        print(
            "  Landmark detection failed. "
            "Skipping."
        )
        return

    print(
        f"  Detected {len(landmarks)} landmarks"
    )

    for name, pt in landmarks.items():
        print(
            f"    {name:25s}: ({pt[0]:4d}, {pt[1]:4d})"
        )

    # Draw X marks

    annotated, legend = annotate_landmarks(
        image,
        landmarks
    )

    # Save annotated image

    save_path = output_dir / image_path.name

    cv2.imwrite(
        str(save_path),
        annotated
    )

    print(f"  Saved: {save_path}")

    # ---- Measure px_per_cm for this image ----

    px_per_cm, _ = measure_pixel2cm(image, leaf_mask, n_lines=10)

    if px_per_cm is not None:
        print(f"  px_per_cm: {px_per_cm:.2f}")
    else:
        print("  px_per_cm: N/A")
        px_per_cm = 0.0

    # Append to CSV with columns:
    # image, variedad, landmark_id, landmark, x, y, px_per_cm

    filename = image_path.name

    if "_" in filename:
        parts = filename.split("_", 1)
        variedad = parts[0]
    else:
        variedad = args.input if not is_all_dataset() else ""

    for name, pt, color in legend:
        lm_id = LANDMARK_IDS.get(name, "")
        csv_lines.append(
            f"{filename},{variedad},{lm_id},{name},"
            f"{pt[0]},{pt[1]},{px_per_cm:.4f}"
        )


def landmark_images():
    """
    Detect landmarks.  Supports single-variety and batch (all-dataset).

    Single variety:
        --input my_variety [--from-cropped]

    All varieties (batch):
        --input all-dataset --from-cropped
        -> reads from data/processed/all_dataset/cropped/
           writes images to data/processed/all_dataset/landmarks/
           writes CSV  to data/processed/all_dataset/csv/
    """

    if is_all_dataset():
        # Batch mode: read from consolidated cropped dir
        cropped_dir = all_dataset_cropped_dir()

        if not cropped_dir.exists():
            print(
                f"\nCropped directory not found:\n{cropped_dir}"
            )
            print(
                "Run first:\n"
                "  python leaf_pipeline.py --mode crop --input all-dataset"
            )
            return

        image_files = sorted([
            p for p in cropped_dir.iterdir()
            if p.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png"
            ]
        ])

        if not image_files:
            print(
                f"\nNo images found in:\n{cropped_dir}"
            )
            return

        output_dir = all_dataset_landmarks_dir()
        ensure_dir(output_dir)

        csv_dir = all_dataset_csv_dir()
        ensure_dir(csv_dir)

        csv_path = csv_dir / "landmarks.csv"

        print(
            f"\n=== BATCH LANDMARKS ==="
        )
        print(
            f"Input:  {cropped_dir}"
        )
        print(
            f"Images: {output_dir}"
        )
        print(
            f"CSV:    {csv_dir}"
        )
        print(
            f"Images: {len(image_files)}\n"
        )

        csv_lines = [
            "image,variedad,landmark_id,landmark,x,y,px_per_cm"
        ]

        for image_path in tqdm(
            image_files,
            desc="Landmarks",
            unit="img",
            ncols=80,
            leave=True,
        ):
            _process_landmarks(
                image_path,
                output_dir,
                csv_lines
            )

        with open(csv_path, 'w') as f:
            f.write('\n'.join(csv_lines) + '\n')

        print(
            f"\n=== TOTAL: {len(image_files)} images processed ==="
        )
        print(
            f"CSV: {csv_path}"
        )

    else:
        # Single-variety mode
        output_dir = (
            OUTPUT_ROOT /
            args.input /
            "landmarks"
        )

        csv_dir = (
            OUTPUT_ROOT /
            args.input /
            "csv"
        )

        ensure_dir(output_dir)
        ensure_dir(csv_dir)

        csv_path = csv_dir / "landmarks.csv"

        image_files = get_image_files()

        source_label = (
            "(from cropped)"
            if args.from_cropped
            else "(from raw)"
        )

        print(
            f"\nFound {len(image_files)} images "
            f"for landmark detection {source_label}"
        )

        csv_lines = [
            "image,variedad,landmark_id,landmark,x,y,px_per_cm"
        ]

        for image_path in tqdm(
            image_files,
            desc="Landmarks",
            unit="img",
            ncols=80,
            leave=True,
        ):
            _process_landmarks(
                image_path,
                output_dir,
                csv_lines
            )

        with open(csv_path, 'w') as f:
            f.write('\n'.join(csv_lines) + '\n')

        print(
            f"\nLandmark CSV: {csv_path}"
        )
        print(
            f"Total entries: {len(csv_lines) - 1}"
        )





# ======================================================
# PIXEL2CM MODE
# ======================================================

def _process_pixel2cm(image_path, summary_csv_lines, detail_csv_lines):
    """Measure pixel2cm on a single image and append to both CSVs."""

    print(
        f"\nProcessing pixel2cm:\n{image_path.name}"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image.")
        return None

    leaf_mask = detect_leaf_mask(image)

    px_per_cm, measurements = measure_pixel2cm(
        image,
        leaf_mask,
        n_lines=10
    )

    if px_per_cm is None:
        print("  Grid detection failed. Skipping.")
        return None

    good = [m for m in measurements if not m['is_outlier']]

    print(
        f"  px_per_cm={px_per_cm:.2f}  "
        f"(measurements={len(measurements)}, "
        f"good={len(good)})"
    )

    # Extract variety from filename prefix
    filename = image_path.name

    if "_" in filename:
        parts = filename.split("_", 1)
        variedad = parts[0]
    else:
        variedad = (
            args.input
            if not is_all_dataset()
            else ""
        )

    # Summary line
    summary_csv_lines.append(
        f"{filename},{variedad},{px_per_cm:.4f}"
    )

    # Detail lines (one per measurement)
    for m in measurements:
        detail_csv_lines.append(
            f"{filename},{variedad},"
            f"{m['direction']},"
            f"{m['start_idx']},{m['end_idx']},"
            f"{m['start_pos']},{m['end_pos']},"
            f"{m['n_lines']},{m['distance_px']:.2f},"
            f"{m['px_per_cm']:.4f},"
            f"{'1' if m['is_outlier'] else '0'}"
        )

    return px_per_cm


def pixel2cm_images():
    """
    Measure pixel-to-cm ratio. Supports single-variety and
    batch (all-dataset).

    Generates two CSV files:
        - pixel2cm.csv: summary with one row per image
        - pixel2cm_detail.csv: one row per individual measurement

    Single variety:
        --input my_variety [--from-cropped]

    All varieties (batch):
        --input all-dataset --from-cropped
        -> reads from data/processed/all_dataset/cropped/
           writes CSVs to data/processed/all_dataset/csv/
    """

    if is_all_dataset():
        cropped_dir = all_dataset_cropped_dir()

        if not cropped_dir.exists():
            print(
                f"\nCropped directory not found:\n{cropped_dir}"
            )
            print(
                "Run first:\n"
                "  python leaf_pipeline.py --mode crop "
                "--input all-dataset"
            )
            return

        image_files = sorted([
            p for p in cropped_dir.iterdir()
            if p.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png"
            ]
        ])

        if not image_files:
            print(
                f"\nNo images found in:\n{cropped_dir}"
            )
            return

        output_dir = all_dataset_pixel2cm_dir()
        ensure_dir(output_dir)

        csv_dir = all_dataset_csv_dir()
        ensure_dir(csv_dir)

        csv_summary = csv_dir / "pixel2cm.csv"
        csv_detail = csv_dir / "pixel2cm_detail.csv"

        print(
            f"\n=== BATCH PIXEL2CM ==="
        )
        print(
            f"Input:  {cropped_dir}"
        )
        print(
            f"CSV:    {csv_dir}"
        )
        print(
            f"Images: {len(image_files)}\n"
        )

        summary_lines = [
            "image,variedad,px_per_cm"
        ]

        detail_lines = [
            "image,variedad,direction,"
            "start_idx,end_idx,"
            "start_pos,end_pos,"
            "n_lines,distance_px,"
            "px_per_cm,is_outlier"
        ]

        for image_path in tqdm(
            image_files,
            desc="Pixel2cm",
            unit="img",
            ncols=80,
            leave=True,
        ):
            _process_pixel2cm(
                image_path,
                summary_lines,
                detail_lines
            )

        with open(csv_summary, 'w') as f:
            f.write('\n'.join(summary_lines) + '\n')

        with open(csv_detail, 'w') as f:
            f.write('\n'.join(detail_lines) + '\n')

        print(
            f"\n=== TOTAL: {len(image_files)} "
            f"images processed ==="
        )
        print(
            f"Summary CSV: {csv_summary}"
        )
        print(
            f"Detail CSV:  {csv_detail}"
        )

    else:
        output_dir = (
            OUTPUT_ROOT /
            args.input /
            "pixel2cm"
        )

        csv_dir = (
            OUTPUT_ROOT /
            args.input /
            "csv"
        )

        ensure_dir(output_dir)
        ensure_dir(csv_dir)

        csv_summary = csv_dir / "pixel2cm.csv"
        csv_detail = csv_dir / "pixel2cm_detail.csv"

        image_files = get_image_files()

        source_label = (
            "(from cropped)"
            if args.from_cropped
            else "(from raw)"
        )

        print(
            f"\nFound {len(image_files)} images "
            f"for pixel2cm {source_label}"
        )

        summary_lines = [
            "image,variedad,px_per_cm"
        ]

        detail_lines = [
            "image,variedad,direction,"
            "start_idx,end_idx,"
            "start_pos,end_pos,"
            "n_lines,distance_px,"
            "px_per_cm,is_outlier"
        ]

        for image_path in tqdm(
            image_files,
            desc="Pixel2cm",
            unit="img",
            ncols=80,
            leave=True,
        ):
            _process_pixel2cm(
                image_path,
                summary_lines,
                detail_lines
            )

        with open(csv_summary, 'w') as f:
            f.write('\n'.join(summary_lines) + '\n')

        with open(csv_detail, 'w') as f:
            f.write('\n'.join(detail_lines) + '\n')

        print(
            f"\nSummary CSV: {csv_summary}"
        )
        print(
            f"Detail CSV:  {csv_detail}"
        )


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # Parse CLI args at runtime (not at import time)
    args = parser.parse_args()

    _modes_needing_input = [
        "crop",
        "landmark",
        "pixel2cm",
    ]

    if args.mode in _modes_needing_input and not args.input:
        parser.error(
            f"--input is required for --mode {args.mode}"
        )

    if args.mode == "crop":
        crop_images()
    elif args.mode == "landmark":
        landmark_images()
    elif args.mode == "pixel2cm":
        pixel2cm_images()
    elif args.mode == "all":
        crop_images()
        landmark_images()
        pixel2cm_images()