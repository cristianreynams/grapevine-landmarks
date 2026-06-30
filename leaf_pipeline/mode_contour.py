import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import ensure_dir

def remove_grid(image_bgr):
    """
    Elimina las líneas horizontales y verticales de la cuadrícula mediante
    morfología + inpainting.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Resaltar líneas oscuras
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )

    _, bw = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Detectar líneas horizontales
    horizontal = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    )

    # Detectar líneas verticales
    vertical = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 31))
    )

    grid = cv2.bitwise_or(horizontal, vertical)

    grid = cv2.dilate(
        grid,
        np.ones((3,3), np.uint8),
        iterations=1
    )

    repaired = cv2.inpaint(
        image_bgr,
        grid,
        3,
        cv2.INPAINT_TELEA
    )

    return repaired

def detect_leaf_contour_mask(image_bgr):
    """
    Extracción de contorno mediante Índice de Pigmentación Híbrido y Recorte de Varianza.
    Inmune a líneas de cuadrícula (acromáticas) y robusto ante necrosis marginal.
    """
    # 1. Espacio LAB (Cromaticidad)
    image_bgr = remove_grid(image_bgr)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    _, a, b = cv2.split(lab)
    chroma = np.sqrt(np.square(a - 128.0) + np.square(b - 128.0))
    chroma_norm = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX)

    # 2. Espacio HSV (Saturación: altamente sensible a marrones y negros biológicos)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    _, s, _ = cv2.split(hsv)
    s_norm = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)

    # 3. Índice de Pigmentación Híbrido
    # La cuadrícula/papel es acromática en ambos. La necrosis dispara la saturación.
    pigment_index = cv2.max(chroma_norm, s_norm).astype(np.uint8)

    # 4. Clipping (Control de Sesgo de Otsu)
    # Cualquier pigmento por encima de 60 se capa a 60. Esto evita que el verde 
    # brillante empuje el umbral estadístico hacia arriba y discrimine los bordes marrones.
    pigment_clipped = np.clip(pigment_index, 0, 60)
    
    # Normalizamos el rango recortado a 0-255 para maximizar el contraste local
    pigment_equalized = cv2.normalize(pigment_clipped, None, 0, 255, cv2.NORM_MINMAX)

    # 5. Umbralización Adaptativa
    _, mask = cv2.threshold(pigment_equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # 6. Extracción estricta (Preservación topológica CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours: 
        return mask
        
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.0008 * cv2.arcLength(largest, True)

    largest = cv2.approxPolyDP(
        largest,
        epsilon,
        True
    )
    mask_clean = np.zeros_like(mask)
    cv2.drawContours(mask_clean, [largest], -1, 255, -1)
    
    # 7. Relleno topológico interno (Flood fill invertido para sellar agujeros)
    h, w = mask_clean.shape
    flood = mask_clean.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    
    # Prevención de desbordamiento en los márgenes de la imagen
    holes[:, 0] = 0; holes[:, -1] = 0; holes[0, :] = 0; holes[-1, :] = 0
    
    return cv2.bitwise_or(mask_clean, holes)

def _process_contour(image_path, dirs):
    image = cv2.imread(str(image_path))
    if image is None: return

    mask = detect_leaf_contour_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return
        
    largest_contour = max(contours, key=cv2.contourArea)
    stem = Path(image_path.name).stem
    
    b, g, r = cv2.split(image)
    rgba = cv2.merge([b, g, r, mask])
    cv2.imwrite(str(dirs['leaf'] / f"{stem}.png"), rgba)
    cv2.imwrite(str(dirs['mask'] / f"{stem}.png"), mask)
    
    overlay = image.copy()
    cv2.drawContours(overlay, [largest_contour], -1, (255, 0, 255), 1)
    cv2.imwrite(str(dirs['overlay'] / f"{stem}.png"), overlay)
    
    contour_pts = largest_contour.reshape(-1, 2)
    csv_lines = ["id,x,y"]
    for idx, pt in enumerate(contour_pts):
        csv_lines.append(f"{idx},{pt[0]},{pt[1]}")
        
    with open(dirs['csv'] / f"{stem}.csv", 'w', newline='\n') as f:
        f.write('\n'.join(csv_lines) + '\n')

def run_contour(input_dir, base_output_dir):
    if not input_dir.exists():
        print(f"\nError: Input directory not found: {input_dir}")
        print("Asegúrate de ejecutar el modo 'crop' primero.")
        return

    image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    
    dirs = {
        'leaf': base_output_dir / "contour" / "leaf",
        'mask': base_output_dir / "contour" / "mask",
        'overlay': base_output_dir / "contour" / "overlay",
        'csv': base_output_dir / "contour" / "csv"
    }
    for d in dirs.values(): ensure_dir(d)

    print(f"\n=== CONTOUR EXTRACTION ===")
    for image_path in tqdm(image_files, desc="Contours", unit="img", ncols=80, leave=True):
        _process_contour(image_path, dirs)