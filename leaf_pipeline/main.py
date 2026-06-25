import argparse
from config import DATA_ROOT, OUTPUT_ROOT, ALL_DATASET_NAME, ALL_DATASET_ALIAS
from mode_crop import run_crop
from mode_contour import run_contour
from mode_pixel2cm import run_pixel2cm
from mode_landmark import run_landmark

def build_paths(args):
    is_all = (args.input == ALL_DATASET_ALIAS)
    project_name = ALL_DATASET_NAME if is_all else args.output

    # Resolución estricta: contour, pixel2cm y landmark leen de cropped.
    # El crop lee del raw (DATA_ROOT).
    if args.mode in ["contour", "landmark", "pixel2cm"] or args.from_cropped:
        input_dir = OUTPUT_ROOT / project_name / "cropped"
    else:
        input_dir = DATA_ROOT / args.input if not is_all else DATA_ROOT

    base_output_dir = OUTPUT_ROOT / project_name
    return input_dir, base_output_dir, is_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grape leaf processing pipeline.")
    parser.add_argument("--mode", required=True, choices=["crop", "landmark", "pixel2cm", "contour", "all"])
    parser.add_argument("--input", required=True, help="Input dataset (e.g., raw-pasini or all-dataset)")
    parser.add_argument("--output", default="", help="Output project name")
    parser.add_argument("--from-cropped", action="store_true", help="Force read from cropped directory")
    
    args = parser.parse_args()
    
    # Auto-resolución de output si está vacío
    if not args.output:
        args.output = args.input.replace("raw-", "")

    input_dir, base_output_dir, is_all = build_paths(args)

    input_name = "" if is_all else args.input

    if args.mode == "crop":
        run_crop(input_dir, base_output_dir, is_all, data_root=DATA_ROOT)
    elif args.mode == "contour":
        run_contour(input_dir, base_output_dir)
    elif args.mode == "pixel2cm":
        run_pixel2cm(input_dir, base_output_dir, input_name)
    elif args.mode == "landmark":
        run_landmark(input_dir, base_output_dir, input_name)
    elif args.mode == "all":
        # Secuencia completa automatizada
        run_crop(DATA_ROOT if is_all else DATA_ROOT / args.input, base_output_dir, is_all, data_root=DATA_ROOT)
        
        # Después del crop, la entrada obligatoria para el resto es la carpeta cropped
        post_crop_input = OUTPUT_ROOT / (ALL_DATASET_NAME if is_all else args.output) / "cropped"
        
        run_contour(post_crop_input, base_output_dir)
        run_landmark(post_crop_input, base_output_dir, input_name)
        run_pixel2cm(post_crop_input, base_output_dir, input_name)