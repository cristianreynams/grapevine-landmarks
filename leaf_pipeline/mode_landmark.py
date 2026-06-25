import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from utils import ensure_dir, detect_leaf_mask
from mode_pixel2cm import measure_pixel2cm
from config import LANDMARK_NAMES, LANDMARK_COLORS, LANDMARK_IDS, LOWER_LEAF, UPPER_LEAF

# --- PARÁMETROS DE LANDMARK (Puedes moverlos a config.py si deseas iterar sobre ellos) ---
PEC_DEFECT_DEPTH_THRESHOLD = 60
PEC_LOWER_Y_FACTOR = 0.6
PEC_CENTER_BAND = 0.2
PEC_SKELETON_BAND = 0.033
PEC_VEIN_CONVERGENCE_WEIGHT = 0.0
L1_CENTER_BAND = 0.12
L2_SIDE_OFFSET = 30
L2_UPPER_Y_FACTOR = 1.0
L2_PEAK_DISTANCE = 40
L2_PROMINENCE_FRACTION = 0.25
L3_BELOW_L2_OFFSET = 120
L3_BELOW_PEC_MARGIN = 30
L3_PEAK_DISTANCE = 30
L3_BELOW_L2_MIN = 80
L4_LOWER_Y_OFFSET = 50
L4_IDEAL_LATERAL = 0.22
L4_LATERAL_WEIGHT = 0.8

def refine_pec_from_vein_convergence(initial_pec, mask, l1, l2_izq, l2_der, l3_izq, l3_der, cx):
    lobes = [p for p in [l1, l2_izq, l2_der, l3_izq, l3_der] if p is not None]
    if len(lobes) < 3: return None
    h, w = mask.shape[:2]
    M = cv2.moments(mask)
    cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else h // 2
    line_pts, line_dirs = [], []
    for lx, ly in ((int(p[0]), int(p[1])) for p in lobes):
        if not (0 <= lx < w and 0 <= ly < h): continue
        dx, dy = cx - lx, cy - ly
        dnorm = np.hypot(dx, dy)
        if dnorm < 1e-6: continue
        line_pts.append(np.array([lx, ly], dtype=np.float64))
        line_dirs.append(np.array([dx / dnorm, dy / dnorm], dtype=np.float64))
    if len(line_pts) < 3: return None
    A, b_vec = np.zeros((2, 2), dtype=np.float64), np.zeros(2, dtype=np.float64)
    for pt, d in zip(line_pts, line_dirs):
        P = np.eye(2) - np.outer(d, d)
        A += P; b_vec += P @ pt
    try:
        conv = np.linalg.solve(A, b_vec)
        ix, iy = int(round(conv[0])), int(round(conv[1]))
        if (0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0 and abs(ix - cx) < w * 0.25 and h * 0.2 < iy < h * 0.95):
            return (ix, iy)
    except np.linalg.LinAlgError: pass
    return None

def find_peciole_sinus(image_bgr, mask):
    h, w = image_bgr.shape[:2]
    M = cv2.moments(mask)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    best_defect, best_depth = None, 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            far_pt, depth = tuple(contour[f][0]), d / 256.0
            if (far_pt[1] > cy * PEC_LOWER_Y_FACTOR and abs(far_pt[0] - cx) < w * PEC_CENTER_BAND and depth > best_depth):
                best_depth, best_defect = depth, far_pt
    if best_defect is None: return (cx, int(cy + h * 0.25))

    contour_arr, dp = contour.reshape(-1, 1, 2).astype(np.int32), 0
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            fp = tuple(contour_arr[f][0])
            if (abs(fp[1] - best_defect[1]) + abs(fp[0] - best_defect[0]) * 0.5 < 80): dp = max(dp, d / 256.0)

    from skimage.morphology import skeletonize as sk_skel
    skel = (sk_skel(mask > 0) * 255).astype(np.uint8)
    band = max(20, int(w * PEC_SKELETON_BAND))
    center_pixels = [(x, y) for y in range(int(cy * 0.3), min(h, int(cy * 1.4))) for x in range(max(0, cx - band), min(w, cx + band)) if skel[y, x] > 0]
    v9 = None
    if center_pixels:
        midrib_mask = np.zeros_like(skel)
        for x, y in center_pixels: midrib_mask[y, x] = 1
        mm = cv2.dilate(midrib_mask, np.ones((3, 3), np.uint8), iterations=1)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mm, connectivity=8)
        best_comp, best_e = -1, 0
        for i in range(1, n):
            ys, xs = np.where(labels == i)
            if len(ys) > 0 and abs(np.mean(xs) - cx) < w * 0.15 and (ys.max() - ys.min()) > best_e:
                best_e, best_comp = (ys.max() - ys.min()), i
        if best_comp > -1:
            ys, xs = np.where(labels == best_comp)
            li = np.where(ys == ys.max())[0]
            if len(li) > 0: v9 = (int(np.median([xs[i] for i in li])), int(ys.max()))

    if dp > PEC_DEFECT_DEPTH_THRESHOLD:
        bx, by = best_defect
        if by > h * 0.85:
            leaf_m = cv2.inRange(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV), LOWER_LEAF, UPPER_LEAF)
            for step in range(1, 8):
                sy = by - step * 4
                if sy < int(cy * 0.5): break
                if 0 <= sy < h and leaf_m[sy, bx] > 0: return (bx, sy)
        return best_defect
    else: return (int(v9[0] * 0.5 + cx * 0.5), v9[1]) if v9 else best_defect

def find_L1(image_bgr, mask, pec):
    h, w = image_bgr.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
    top = contour_pts[np.abs(contour_pts[:, 0] - int(pec[0])) < w * L1_CENTER_BAND]
    return tuple(top[np.argmin(top[:, 1])]) if len(top) > 0 else None

def find_L2(image_bgr, mask, pec, l1):
    h, w = image_bgr.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
    M = cv2.moments(mask)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    dists_smooth = uniform_filter1d(np.sqrt((contour_pts[:, 0] - int(pec[0])) ** 2 + (contour_pts[:, 1] - int(pec[1])) ** 2), size=7, mode='wrap')
    result = {}
    for side in ['izq', 'der']:
        side_mask = contour_pts[:, 0] < cx - L2_SIDE_OFFSET if side == 'izq' else contour_pts[:, 0] > cx + L2_SIDE_OFFSET
        pts, d_smooth = contour_pts[side_mask], dists_smooth[side_mask]
        if len(pts) < 3: result[side] = None; continue
        upper_mask = pts[:, 1] < cy * L2_UPPER_Y_FACTOR
        upper_pts, upper_d = pts[upper_mask], d_smooth[upper_mask]
        l2 = None
        if len(upper_pts) > 0:
            peaks, props = find_peaks(upper_d, distance=L2_PEAK_DISTANCE, prominence=0)
            if len(peaks) > 0:
                filtered = [p for i, p in enumerate(peaks) if props['prominences'][i] > np.max(props['prominences']) * L2_PROMINENCE_FRACTION]
                if filtered:
                    peak_pts = [tuple(upper_pts[p]) for p in filtered]
                    valid_pts = [p for p in peak_pts if p[1] < cy * 0.65] or peak_pts
                    combined = [(cx - p[0] if side == 'izq' else p[0] - cx) * 1.5 + (cy - p[1]) * 1.0 for p in valid_pts]
                    l2 = valid_pts[int(np.argmax(combined))]
        if l2 is None:
            upper = pts[pts[:, 1] < cy]
            l2 = tuple(upper[np.argmin(upper[:, 0]) if side == 'izq' else np.argmax(upper[:, 0])]) if len(upper) > 0 else tuple(pts[np.argmin(pts[:, 1])])
        result[side] = l2
    return result

def find_L3(image_bgr, mask, pec, l2_izq, l2_der):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
    M = cv2.moments(mask)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    dists_smooth = uniform_filter1d(np.sqrt((contour_pts[:, 0] - int(pec[0])) ** 2 + (contour_pts[:, 1] - int(pec[1])) ** 2), size=7, mode='wrap')
    result = {}
    for side, l2 in [('izq', l2_izq), ('der', l2_der)]:
        if l2 is None: result[side] = None; continue
        side_mask = contour_pts[:, 0] < cx - 30 if side == 'izq' else contour_pts[:, 0] > cx + 30
        pts, d_smooth = contour_pts[side_mask], dists_smooth[side_mask]
        if len(pts) < 3: result[side] = None; continue
        mid_mask = ((pts[:, 1] > l2[1] + L3_BELOW_L2_OFFSET) & (pts[:, 1] < int(pec[1]) + L3_BELOW_PEC_MARGIN))
        mid_pts, mid_d = pts[mid_mask], d_smooth[mid_mask]
        l3 = None
        if len(mid_pts) > 0:
            peaks, props = find_peaks(mid_d, distance=L3_PEAK_DISTANCE, prominence=0)
            if len(peaks) > 0:
                l3 = [tuple(mid_pts[p]) for p in peaks][int(np.argmax([prom * 1.0 + (pt[1] - l2[1]) * 0.5 for prom, pt in zip(props['prominences'], [tuple(mid_pts[p]) for p in peaks])]))]
        if l3 is None:
            mid = pts[(pts[:, 1] > l2[1] + L3_BELOW_L2_MIN) & (pts[:, 1] < int(pec[1]))]
            l3 = tuple(mid[np.argmax(np.abs(mid[:, 0] - cx))]) if len(mid) > 0 else l2
        result[side] = l3
    return result

def find_L4(image_bgr, mask, pec, l3_izq, l3_der):
    h, w = image_bgr.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
    M = cv2.moments(mask)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    result = {}
    for side in ['izq', 'der']:
        side_mask = contour_pts[:, 0] < cx - 30 if side == 'izq' else contour_pts[:, 0] > cx + 30
        pts = contour_pts[side_mask]
        if len(pts) < 3: result[side] = None; continue
        lower = pts[pts[:, 1] > cy + L4_LOWER_Y_OFFSET]
        result[side] = tuple(lower[np.argmax([p[1] * 1.0 - np.abs(np.abs(p[0] - int(pec[0])) - w * L4_IDEAL_LATERAL) * L4_LATERAL_WEIGHT for p in lower])]) if len(lower) > 0 else tuple(pts[np.argmax(pts[:, 1])])
    return result

def detect_landmarks(image_bgr, leaf_mask):
    pec = find_peciole_sinus(image_bgr, leaf_mask)
    l1 = find_L1(image_bgr, leaf_mask, pec)
    l2 = find_L2(image_bgr, leaf_mask, pec, l1)
    l3 = find_L3(image_bgr, leaf_mask, pec, l2.get('izq'), l2.get('der'))
    l4 = find_L4(image_bgr, leaf_mask, pec, l3.get('izq'), l3.get('der'))
    
    landmarks = {'seno_peciolar': pec}
    if l1: landmarks['punta_L1'] = l1
    for side in ['izq', 'der']:
        if l2.get(side): landmarks[f'punta_L2_{side}'] = l2[side]
        if l3.get(side): landmarks[f'punta_L3_{side}'] = l3[side]
        if l4.get(side): landmarks[f'punta_L4_{side}'] = l4[side]

    M2 = cv2.moments(leaf_mask)
    pec_refined = refine_pec_from_vein_convergence(pec, leaf_mask, l1, l2.get('izq'), l2.get('der'), l3.get('izq'), l3.get('der'), int(M2['m10'] / M2['m00']))
    if pec_refined is not None:
        landmarks['seno_peciolar'] = (int(round((1 - PEC_VEIN_CONVERGENCE_WEIGHT) * pec[0] + PEC_VEIN_CONVERGENCE_WEIGHT * pec_refined[0])), int(round((1 - PEC_VEIN_CONVERGENCE_WEIGHT) * pec[1] + PEC_VEIN_CONVERGENCE_WEIGHT * pec_refined[1])))
    return landmarks

def annotate_landmarks(image_bgr, landmarks):
    annotated = image_bgr.copy()
    img_h, img_w = annotated.shape[:2]
    s, thickness = max(12, int(min(img_w, img_h) * 0.018)), max(2, int(min(img_w, img_h) * 0.004))
    legend = []

    for i, name in enumerate(LANDMARK_NAMES):
        if name not in landmarks: continue
        px, py = landmarks[name]
        color = LANDMARK_COLORS[i]
        cv2.line(annotated, (px - s, py - s), (px + s, py + s), color, thickness)
        cv2.line(annotated, (px - s, py + s), (px + s, py - s), color, thickness)
        cv2.circle(annotated, (px, py), 3, color, -1)
        
        text_x, text_y = (px - s - 70 if px + s + 8 + 70 > img_w else px + s + 8), (py + s + 20 if py - s - 8 < 15 else py - s - 8)
        cv2.putText(annotated, name.replace('seno_', 'S_').replace('punta_', 'T_'), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        legend.append((name, (px, py), color))
    return annotated, legend

def _process_landmarks(image_path, output_dir, csv_lines, input_name):
    image = cv2.imread(str(image_path))
    if image is None: return
    
    leaf_mask = detect_leaf_mask(image)
    landmarks = detect_landmarks(image, leaf_mask)
    if not landmarks: return

    annotated, legend = annotate_landmarks(image, landmarks)
    cv2.imwrite(str(output_dir / image_path.name), annotated)

    px_per_cm, _ = measure_pixel2cm(image, leaf_mask, n_lines=10)
    px_per_cm = px_per_cm if px_per_cm is not None else 0.0

    filename = image_path.name
    variedad = filename.split("_", 1)[0] if "_" in filename else input_name
    for name, pt, color in legend:
        csv_lines.append(f"{filename},{variedad},{LANDMARK_IDS.get(name, '')},{name},{pt[0]},{pt[1]},{px_per_cm:.4f}")

def run_landmark(input_dir, base_output_dir, input_name):
    if not input_dir.exists():
        print(f"\nError: Input directory not found: {input_dir}")
        return

    out_dir = base_output_dir / "landmarks"
    csv_dir = base_output_dir / "csv"
    ensure_dir(out_dir)
    ensure_dir(csv_dir)

    image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    csv_lines = ["image,variedad,landmark_id,landmark,x,y,px_per_cm"]

    print(f"\n=== LANDMARK EXTRACTION ===")
    for image_path in tqdm(image_files, desc="Landmarks", unit="img", ncols=80, leave=True):
        _process_landmarks(image_path, out_dir, csv_lines, input_name)

    with open(csv_dir / "landmarks.csv", 'w', newline='\n') as f:
        f.write('\n'.join(csv_lines) + '\n')