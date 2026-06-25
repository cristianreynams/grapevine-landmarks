import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.signal import find_peaks
from utils import ensure_dir, detect_leaf_mask

def measure_pixel2cm(image_bgr, leaf_mask=None, n_lines=10):
    h_img, w_img = image_bgr.shape[:2]
    if leaf_mask is None:
        leaf_mask = detect_leaf_mask(image_bgr)

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, []
    contour = max(contours, key=cv2.contourArea)

    contour_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (501, 501))
    dilated_inner = cv2.dilate(contour_mask, ki, iterations=1)
    dilated_outer = cv2.dilate(contour_mask, ko, iterations=1)
    band_mask = cv2.subtract(dilated_outer, dilated_inner)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    dark_band = cv2.bitwise_and(dark, band_mask)

    h_proj, v_proj = np.sum(dark_band, axis=1), np.sum(dark_band, axis=0)
    h_peaks, _ = find_peaks(h_proj, distance=4, prominence=np.std(h_proj) * 0.3)
    v_peaks, _ = find_peaks(v_proj, distance=4, prominence=np.std(v_proj) * 0.3)

    def _cluster_peaks(peaks, max_gap=3):
        if len(peaks) == 0: return []
        peaks = sorted(peaks)
        clusters, current = [], [peaks[0]]
        for p in peaks[1:]:
            if p - current[-1] <= max_gap: current.append(p)
            else: clusters.append(int(np.median(current))); current = [p]
        clusters.append(int(np.median(current)))
        return clusters

    h_lines, v_lines = _cluster_peaks(h_peaks.tolist(), max_gap=3), _cluster_peaks(v_peaks.tolist(), max_gap=3)

    def _measure_direction(lines, direction):
        if len(lines) < n_lines + 1: return []
        diffs = np.diff(lines)
        segments, start = [], 0
        for i, d in enumerate(diffs):
            if d > 20:
                if i - start >= n_lines: segments.append((start, i))
                start = i + 1
        if len(lines) - 1 - start >= n_lines: segments.append((start, len(lines) - 1))

        measurements = []
        for seg_start, seg_end in segments:
            for i in range(seg_start, seg_end - n_lines + 1):
                if any(d > 20 for d in diffs[i:i + n_lines]): continue
                dist_px = lines[i + n_lines] - lines[i]
                measurements.append({'direction': direction, 'start_idx': i, 'end_idx': i + n_lines, 'start_pos': lines[i], 'end_pos': lines[i + n_lines], 'n_lines': n_lines, 'distance_px': dist_px, 'px_per_cm': dist_px})
        return measurements

    all_measurements = _measure_direction(h_lines, 'H') + _measure_direction(v_lines, 'V')
    if not all_measurements: return None, []

    all_px = [m['px_per_cm'] for m in all_measurements]
    median_px = float(np.median(all_px))
    mad = float(np.median(np.abs(np.array(all_px) - median_px)))

    for m in all_measurements:
        m['is_outlier'] = False if mad == 0 else (abs(m['px_per_cm'] - median_px) / mad > 3.0)

    good_px = [m['px_per_cm'] for m in all_measurements if not m['is_outlier']]
    return float(np.median(good_px)) if good_px else median_px, all_measurements

def _process_pixel2cm(image_path, summary_lines, detail_lines, input_name):
    image = cv2.imread(str(image_path))
    if image is None: return

    px_per_cm, measurements = measure_pixel2cm(image, None, n_lines=10)
    if px_per_cm is None: return

    filename = image_path.name
    variedad = filename.split("_", 1)[0] if "_" in filename else input_name
    
    summary_lines.append(f"{filename},{variedad},{px_per_cm:.4f}")
    for m in measurements:
        detail_lines.append(f"{filename},{variedad},{m['direction']},{m['start_idx']},{m['end_idx']},{m['start_pos']},{m['end_pos']},{m['n_lines']},{m['distance_px']:.2f},{m['px_per_cm']:.4f},{'1' if m['is_outlier'] else '0'}")

def run_pixel2cm(input_dir, base_output_dir, input_name):
    if not input_dir.exists():
        print(f"\nError: Input directory not found: {input_dir}")
        return

    out_dir = base_output_dir / "pixel2cm"
    csv_dir = base_output_dir / "csv"
    ensure_dir(out_dir)
    ensure_dir(csv_dir)

    image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    
    summary_lines = ["image,variedad,px_per_cm"]
    detail_lines = ["image,variedad,direction,start_idx,end_idx,start_pos,end_pos,n_lines,distance_px,px_per_cm,is_outlier"]

    print(f"\n=== PIXEL2CM EXTRACTION ===")
    for image_path in tqdm(image_files, desc="Pixel2cm", unit="img", ncols=80, leave=True):
        _process_pixel2cm(image_path, summary_lines, detail_lines, input_name)

    with open(csv_dir / "pixel2cm.csv", 'w', newline='\n') as f:
        f.write('\n'.join(summary_lines) + '\n')
    with open(csv_dir / "pixel2cm_detail.csv", 'w', newline='\n') as f:
        f.write('\n'.join(detail_lines) + '\n')