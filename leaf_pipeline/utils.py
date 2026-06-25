import cv2
import numpy as np
from pathlib import Path
from config import LOWER_LEAF, UPPER_LEAF

def ensure_dir(path):
    """Crea un directorio si no existe."""
    Path(path).mkdir(parents=True, exist_ok=True)

def detect_leaf_mask(image_bgr):
    """
    Máscara morfológica base.
    Usada por Crop, Landmark y Pixel2cm. NO usar para Contour.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_LEAF, UPPER_LEAF)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    kernel_close = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned

    largest = max(contours, key=cv2.contourArea)
    mask_clean = np.zeros_like(mask)
    cv2.drawContours(mask_clean, [largest], -1, 255, -1)

    h, w = mask_clean.shape
    flood = mask_clean.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask_ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    
    holes[:, 0] = 0; holes[:, -1] = 0; holes[0, :] = 0; holes[-1, :] = 0
    return cv2.bitwise_or(mask_clean, holes)