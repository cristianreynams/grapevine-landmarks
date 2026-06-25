import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import ensure_dir, detect_leaf_mask
from config import PAD_RATIO, PAD_MIN_PX

def compute_crop_box(mask, img_h, img_w):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        margin_x = int(img_w * 0.1)
        margin_y = int(img_h * 0.1)
        return (margin_x, img_w - 1 - margin_x, margin_y, img_h - 1 - margin_y, 0)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    padding = max(int(max(x_max - x_min, y_max - y_min) * PAD_RATIO), PAD_MIN_PX)
    xmin = max(0, x_min - padding)
    xmax = min(img_w - 1, x_max + padding)
    ymin = max(0, y_min - padding)
    ymax = min(img_h - 1, y_max + padding)

    return (xmin, xmax, ymin, ymax, padding)

def _crop_and_save(image_path, output_dir, debug_dir, name_prefix=""):
    image = cv2.imread(str(image_path))
    if image is None: return

    img_h, img_w = image.shape[:2]
    leaf_mask = detect_leaf_mask(image)
    xmin, xmax, ymin, ymax, pad = compute_crop_box(leaf_mask, img_h, img_w)

    cropped = image[ymin:ymax+1, xmin:xmax+1]
    
    # Debug image
    debug = image.copy()
    mask_contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, mask_contours, -1, (0, 255, 0), 2)
    cv2.rectangle(debug, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    
    crop_w, crop_h = xmax - xmin + 1, ymax - ymin + 1
    cv2.putText(debug, f"crop: {crop_w}x{crop_h} pad:{pad}px", (xmin + 5, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    stem = Path(image_path.name).stem
    png_name = f"{name_prefix}{stem}.png"

    cv2.imwrite(str(debug_dir / png_name), debug)
    cv2.imwrite(str(output_dir / png_name), cropped)

def run_crop(input_dir, base_output_dir, is_all_dataset, data_root=None):
    out_crop = base_output_dir / "cropped"
    out_debug = base_output_dir / "cropped_debug"
    ensure_dir(out_crop)
    ensure_dir(out_debug)

    if is_all_dataset and data_root:
        dataset_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("raw-")])
        if not dataset_dirs:
            print("\nNo dataset directories found in Data/")
            return
            
        print(f"\n=== BATCH CROP ===")
        for dataset_dir in dataset_dirs:
            print(f"\nDataset: {dataset_dir.name}")
            variety_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
            for variety_dir in variety_dirs:
                variety = variety_dir.name
                image_files = sorted([p for p in variety_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                if not image_files: continue
                
                for image_path in tqdm(image_files, desc=variety, unit="img", ncols=80, leave=True):
                    _crop_and_save(image_path, out_crop, out_debug, f"{variety}_")
    else:
            if not input_dir.exists():
                print(f"Error: Input dir not found: {input_dir}")
                return
                
            print(f"\n=== CROP EXTRACTION ===")
            # 1. Identificar si hay subcarpetas de variedades
            variety_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
            
            if variety_dirs:
                # Extraer iterando sobre cada variedad e inyectando el prefijo
                for variety_dir in variety_dirs:
                    variety = variety_dir.name
                    image_files = sorted([p for p in variety_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                    if not image_files: continue
                    
                    for image_path in tqdm(image_files, desc=variety, unit="img", ncols=80, leave=True):
                        _crop_and_save(image_path, out_crop, out_debug, f"{variety}_")
            else:
                # Fallback: si el dataset es plano (no tiene carpetas de variedades)
                image_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                for image_path in tqdm(image_files, desc="Cropping", unit="img", ncols=80, leave=True):
                    _crop_and_save(image_path, out_crop, out_debug)