import argparse
import numpy as np
import pandas as pd

import napari
from pathlib import Path
from skimage.io import imread
from magicgui import magicgui

# ======================================================
# CONFIG (module level, no side effects)
# ======================================================

LANDMARK_NAMES = [
    "seno_peciolar",
    "punta_L4_izq",
    "seno_inf_izq",
    "punta_L3_izq",
    "seno_med_izq",
    "punta_L2_izq",
    "seno_sup_izq",
    "punta_L1",
    "seno_sup_der",
    "punta_L2_der",
    "seno_med_der",
    "punta_L3_der",
    "seno_inf_der",
    "punta_L4_der",
]

COLORS = [
    "red", "cyan", "magenta", "yellow",
    "orange", "blue", "white", "purple",
    "green", "pink", "brown", "lime",
    "gold", "turquoise",
]

MAX_POINTS = len(LANDMARK_NAMES)


# ======================================================
# CLI
# ======================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Manual landmark annotation tool for grape leaf images.",
        epilog=(
            "Examples:\n"
            "  python annotate.py --help\n"
            "  python annotate.py --input tempranillo --from-cropped\n"
            "  python annotate.py --input all_dataset --from-cropped\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Folder name (e.g. tempranillo, all_dataset). "
             "Images are read from data/processed/{input}/cropped/",
    )
    parser.add_argument(
        "--from-cropped",
        action="store_true",
        help="Read images from data/processed/{input}/cropped/ "
             "instead of data/raw/{input}/.",
    )
    return parser


# ======================================================
# WIDGET BUILDERS
# ======================================================

def build_landmark_progress_widget():
    from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
    container = QWidget()
    layout = QVBoxLayout()
    container.setLayout(layout)
    global _img_label, _lm_label, _lm_detail_label
    _img_label = QLabel("Images: 0/0")
    _lm_label = QLabel("Landmarks: Place first point")
    _lm_detail_label = QLabel("")
    layout.addWidget(_img_label)
    layout.addWidget(_lm_label)
    layout.addWidget(_lm_detail_label)
    return container


def build_legend_widget():
    from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QHBoxLayout
    container = QWidget()
    layout = QVBoxLayout()
    container.setLayout(layout)
    for i, (name, color) in enumerate(zip(LANDMARK_NAMES, COLORS)):
        row = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)
        row.setLayout(row_layout)
        dot = QLabel("  ")
        dot.setStyleSheet(f"background-color: {color}; border-radius: 6px;")
        dot.setFixedSize(12, 12)
        num = QLabel(f"{i+1:2d}.")
        lbl = QLabel(name)
        row_layout.addWidget(dot)
        row_layout.addWidget(num)
        row_layout.addWidget(lbl)
        row_layout.addStretch()
        layout.addWidget(row)
    return container


# ======================================================
# MAIN
# ======================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ---- Paths ----
    if args.from_cropped:
        image_dir = Path(f"data/processed/{args.input}/cropped")
    else:
        image_dir = Path(f"data/raw/{args.input}")

    csv_dir = Path(f"data/processed/{args.input}/csv")
    csv_dir.mkdir(parents=True, exist_ok=True)
    master_csv = csv_dir / "manual_landmarks.csv"

    # ---- Image list ----
    all_images = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png"))
    )

    if len(all_images) == 0:
        raise RuntimeError(f"No images found in:\n{image_dir}")

    total_images = len(all_images)

    # ---- Existing CSV (resume) ----
    if master_csv.exists():
        master_df = pd.read_csv(master_csv)
        completed_images = set(master_df["image_name"].unique())
    else:
        master_df = pd.DataFrame(
            columns=["folder_name", "image_name", "variedad", "landmark", "x", "y"]
        )
        completed_images = set()

    # ---- Pending ----
    pending_images = [p for p in all_images if p.name not in completed_images]

    if len(pending_images) == 0:
        raise RuntimeError(
            "All images already annotated.\n"
            f"CSV: {master_csv}"
        )

    current_image_index = 0

    # ---- Progress helpers ----
    def image_progress_str():
        done = len(completed_images) + current_image_index
        pct = (done / total_images) * 100
        return f"{done}/{total_images} img ({pct:.0f}%)"

    def extract_variedad(image_path):
        name = image_path.name
        if "_" in name:
            return name.split("_", 1)[0]
        return args.input

    # ---- Viewer setup ----
    viewer = napari.Viewer(title="Leaf Landmark Annotator")

    initial_path = pending_images[current_image_index]
    image_layer = viewer.add_image(imread(str(initial_path)), name="Leaf")

    points_layer = viewer.add_points(
        name="Landmarks",
        ndim=2,
        size=16,
        border_width=0.15,
        border_color="white",
    )

    # ---- Annotations ----
    def refresh_annotations():
        n = len(points_layer.data)
        if n == 0:
            return
        points_layer.face_color = COLORS[:n]
        points_layer.features = {"label": LANDMARK_NAMES[:n]}
        points_layer.text = {
            "string": "{label}",
            "size": 10,
            "color": "white",
            "translation": np.array([12, 0]),
        }

    def update_title():
        current = pending_images[current_image_index]
        n = len(points_layer.data)
        if n >= MAX_POINTS:
            status = "COMPLETE - Press Save + Next"
        else:
            status = f"Place: {LANDMARK_NAMES[n]}"
        viewer.title = (
            f"Leaf Annotator  |  "
            f"{image_progress_str()}  |  "
            f"{status}  |  {current.name}"
        )

    def update_landmark_progress_widget():
        if '_img_label' not in globals():
            return
        done_img = len(completed_images) + current_image_index
        pct_img = (done_img / total_images) * 100
        _img_label.setText(f"Images: {done_img}/{total_images} ({pct_img:.0f}%)")
        n = len(points_layer.data)
        if n >= MAX_POINTS:
            _lm_label.setText(f"All {MAX_POINTS} landmarks placed!")
            _lm_label.setStyleSheet("color: green; font-weight: bold;")
            _lm_detail_label.setText("Press 'Save + Next' to continue")
        else:
            next_lm = LANDMARK_NAMES[n]
            color = COLORS[n]
            _lm_label.setText(f"{n}/{MAX_POINTS} placed  |  Next: {next_lm}")
            _lm_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            _lm_detail_label.setText(f"({MAX_POINTS - n} remaining)")

    # ---- Point change callback ----
    @points_layer.events.data.connect
    def on_points_change(event):
        n = len(points_layer.data)
        if n > MAX_POINTS:
            points_layer.data = points_layer.data[:MAX_POINTS]
            print(f"  Max {MAX_POINTS} points reached.")
            return
        refresh_annotations()
        update_title()
        update_landmark_progress_widget()
        if 0 < n < MAX_POINTS:
            print(f"  {n}/{MAX_POINTS} placed  |  Next: {LANDMARK_NAMES[n]}  ({MAX_POINTS - n} left)")
        elif n == MAX_POINTS:
            print(f"  {n}/{MAX_POINTS} COMPLETE  |  Press 'Save + Next' or [s]")

    # ---- Load existing landmarks ----
    def load_landmarks_for_image(image_path):
        nonlocal master_df
        if master_df.empty:
            return False
        rows = master_df[master_df["image_name"] == image_path.name]
        if len(rows) == 0:
            return False
        master_df = master_df[master_df["image_name"] != image_path.name].reset_index(drop=True)
        completed_images.discard(image_path.name)
        rows = rows.copy()
        rows["_order"] = rows["landmark"].map({name: i for i, name in enumerate(LANDMARK_NAMES)})
        rows = rows.sort_values("_order").drop(columns=["_order"])
        pts = np.array([[row["y"], row["x"]] for _, row in rows.iterrows()])
        points_layer.data = pts
        refresh_annotations()
        update_title()
        update_landmark_progress_widget()
        print(f"  Loaded {len(pts)} landmarks from previous session.")
        return True

    # ---- Save ----
    def save_current_landmarks():
        nonlocal master_df
        n = len(points_layer.data)
        if n != MAX_POINTS:
            print(f"\n  ERROR: Expected {MAX_POINTS} points, got {n}.")
            return False
        path = pending_images[current_image_index]
        variedad = extract_variedad(path)
        rows = []
        for p in points_layer.data:
            y, x = p
            h, w = image_layer.data.shape[:2]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            rows.append({
                "folder_name": args.input,
                "image_name": path.name,
                "variedad": variedad,
                "landmark": LANDMARK_NAMES[len(rows)],
                "x": float(x),
                "y": float(y),
            })
        master_df = pd.concat([master_df, pd.DataFrame(rows)], ignore_index=True)
        master_df.to_csv(master_csv, index=False)
        print(f"\n  Saved: {master_csv}")
        return True

    # ---- Navigation ----
    @magicgui(call_button="Save + Next")
    def save_next():
        nonlocal current_image_index
        if not save_current_landmarks():
            return
        current_image_index += 1
        if current_image_index >= len(pending_images):
            print(f"\n{'='*50}\n  DATASET COMPLETE\n  {image_progress_str()}\n  CSV: {master_csv}\n{'='*50}")
            return
        nxt = pending_images[current_image_index]
        image_layer.data = imread(str(nxt))
        viewer.reset_view()
        loaded = load_landmarks_for_image(nxt)
        if not loaded:
            points_layer.data = np.empty((0, 2))
        update_title()
        update_landmark_progress_widget()
        print(f"\n  {image_progress_str()}  |  [{'loaded' if loaded else 'new'}] {nxt.name}")

    @magicgui(call_button="Previous Image")
    def go_previous():
        nonlocal current_image_index
        if current_image_index <= 0:
            print("\n  Already at first image.")
            return
        current_image_index -= 1
        prev = pending_images[current_image_index]
        image_layer.data = imread(str(prev))
        viewer.reset_view()
        loaded = load_landmarks_for_image(prev)
        if not loaded:
            points_layer.data = np.empty((0, 2))
        update_title()
        update_landmark_progress_widget()
        print(f"\n  {image_progress_str()}  |  [{'loaded' if loaded else 'new'}] {prev.name}")

    # ---- Hotkeys ----
    @viewer.bind_key("c")
    def clear_points(viewer):
        points_layer.data = np.empty((0, 2))

    @viewer.bind_key("z")
    def undo_last_point(viewer):
        n = len(points_layer.data)
        if n > 0:
            points_layer.data = points_layer.data[:-1]

    @viewer.bind_key("s")
    def hotkey_save_next(viewer):
        save_next()

    @viewer.bind_key("p")
    def hotkey_previous(viewer):
        go_previous()

    # ---- UI Layout ----
    viewer.window.add_dock_widget(
        build_landmark_progress_widget(), area="right", name="Landmark Progress"
    )
    viewer.window.add_dock_widget(
        build_legend_widget(), area="right", name="Landmark Order"
    )
    viewer.window.add_dock_widget(save_next, area="right", name="Navigation")
    viewer.window.add_dock_widget(go_previous, area="right", name="Navigation")

    # ---- Init ----
    loaded = load_landmarks_for_image(initial_path)
    update_title()
    update_landmark_progress_widget()

    print(
        f"\n{'='*50}"
        f"\n  Leaf Landmark Annotator"
        f"\n{'='*50}"
        f"\n  Input:   {image_dir}"
        f"\n  Output:  {master_csv}"
        f"\n  Images:  {total_images} total, {len(pending_images)} pending"
        f"\n  {image_progress_str()}"
        f"\n\n  Hotkeys: [c] Clear  [z] Undo  [s] Save+Next  [p] Previous"
        f"\n\n  Place points in this order:"
    )
    for i, name in enumerate(LANDMARK_NAMES):
        print(f"    {i+1:2d}. {name}")
    print(f"{'='*50}")

    napari.run()


if __name__ == "__main__":
    main()