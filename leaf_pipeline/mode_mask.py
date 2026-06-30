import cv2
from pathlib import Path
from tqdm import tqdm
from utils import ensure_dir
from mode_contour import detect_leaf_contour_mask

def _process_mask(image_path, out_dir):
    image = cv2.imread(str(image_path))
    if image is None: 
        return

    # Extraer la máscara usando el algoritmo de pigmentación híbrido robusto
    mask = detect_leaf_contour_mask(image)

    # Separar canales y fusionar la máscara como canal Alpha (Transparencia)
    b, g, r = cv2.split(image)
    rgba = cv2.merge([b, g, r, mask])

    # Forzar extensión .png para conservar el canal RGBA sin compresión destructiva
    stem = Path(image_path.name).stem
    cv2.imwrite(str(out_dir / f"{stem}.png"), rgba)

def run_mask(input_dir, base_output_dir):
    """
    Punto de entrada aislado para la eliminación de fondo.
    Asume que las imágenes de entrada ya han sido procesadas por 'crop'.
    """
    if not input_dir.exists():
        print(f"\nError: Input directory not found: {input_dir}")
        print("Asegúrate de ejecutar el modo 'crop' primero.")
        return

    out_dir = base_output_dir / "masked"
    ensure_dir(out_dir)

    image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    
    if not image_files:
        print(f"\nNo images found in:\n{input_dir}")
        return

    print(f"\n=== MASK EXTRACTION (BACKGROUND REMOVAL) ===")
    print(f"Input:  {input_dir}")
    print(f"Output: {out_dir}\n")

    for image_path in tqdm(image_files, desc="Masking", unit="img", ncols=80, leave=True):
        _process_mask(image_path, out_dir)
        
    print(f"\n=== TOTAL: {len(image_files)} backgrounds removed ===")