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
            "  python annotate.py --input all_dataset --from-cropped --mode review\n"
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
             "instead of data/raw/{input}.",
    )
    parser.add_argument(
        "--mode",
        choices=["annotate", "review"],
        default="annotate",
        help=(
            "annotate (default): only show unannotated images, place points in order.\n"
            "review: show all images with existing landmarks editable. "
            "Move points to correct, then save."
        ),
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


def build_mode_indicator_widget(mode):
    from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
    container = QWidget()
    layout = QVBoxLayout()
    container.setLayout(layout)
    if mode == "review":
        lbl = QLabel("MODE: REVIEW\nMove any landmark to correct it.\nPress 'Save Changes' when done with this image.")
        lbl.setStyleSheet("color: orange; font-weight: bold; font-size: 12px;")
    else:
        lbl = QLabel("MODE: ANNOTATE\nPlace points in order.\nPress 'Save + Next' when complete.")
        lbl.setStyleSheet("color: green; font-weight: bold; font-size: 12px;")
    layout.addWidget(lbl)
    return container


# ======================================================
# SHARED HELPERS
# ======================================================

def get_rows_for_image(master_df, image_name):
    """Return DataFrame rows for a given image, ordered by LANDMARK_NAMES."""
    if master_df.empty:
        return None
    rows = master_df[master_df["image_name"] == image_name].copy()
    if len(rows) == 0:
        return None
    rows["_order"] = rows["landmark"].map({name: i for i, name in enumerate(LANDMARK_NAMES)})
    rows = rows.sort_values("_order").drop(columns=["_order"])
    return rows


def rows_to_points(rows):
    """Convert DataFrame rows to napari points array (y, x)."""
    return np.array([[row["y"], row["x"]] for _, row in rows.iterrows()])


def extract_variedad(image_path, input_name):
    name = image_path.name
    if "_" in name:
        return name.split("_", 1)[0]
    return input_name


# ======================================================
# ANNOTATE MODE (original behavior)
# ======================================================

def run_annotate_mode(args, image_dir, master_csv):
    csv_dir = master_csv.parent
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png"))
    )
    if len(all_images) == 0:
        raise RuntimeError(f"No images found in:\n{image_dir}")
    total_images = len(all_images)

    if master_csv.exists():
        master_df = pd.read_csv(master_csv)
        completed_images = set(master_df["image_name"].unique())
    else:
        master_df = pd.DataFrame(
            columns=["folder_name", "image_name", "variedad", "landmark", "x", "y"]
        )
        completed_images = set()

    pending_images = [p for p in all_images if p.name not in completed_images]
    if len(pending_images) == 0:
        raise RuntimeError("All images already annotated.\n" f"CSV: {master_csv}")

    current_image_index = 0
    _updating = False

    def image_progress_str():
        done = len(completed_images) + current_image_index
        pct = (done / total_images) * 100
        return f"{done}/{total_images} img ({pct:.0f}%)"

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

    @points_layer.events.data.connect
    def on_points_change(event):
        nonlocal _updating
        if _updating:
            return
        n = len(points_layer.data)
        if n > MAX_POINTS:
            _updating = True
            points_layer.data = points_layer.data[:MAX_POINTS]
            _updating = False
            refresh_annotations()
            update_title()
            update_landmark_progress_widget()
            print(f"  Max {MAX_POINTS} points reached.")
            return
        refresh_annotations()
        update_title()
        update_landmark_progress_widget()
        if 0 < n < MAX_POINTS:
            print(f"  {n}/{MAX_POINTS} placed  |  Next: {LANDMARK_NAMES[n]}  ({MAX_POINTS - n} left)")
        elif n == MAX_POINTS:
            print(f"  {n}/{MAX_POINTS} COMPLETE  |  Press 'Save + Next' or [s]")

    def load_landmarks_for_image(image_path):
        nonlocal master_df, _updating
        rows = get_rows_for_image(master_df, image_path.name)
        if rows is None:
            return False
        master_df = master_df[master_df["image_name"] != image_path.name].reset_index(drop=True)
        completed_images.discard(image_path.name)
        pts = rows_to_points(rows)
        _updating = True
        points_layer.data = pts
        _updating = False
        refresh_annotations()
        update_title()
        update_landmark_progress_widget()
        print(f"  Loaded {len(pts)} landmarks from previous session.")
        return True

    def save_current_landmarks():
        nonlocal master_df
        n = len(points_layer.data)
        if n != MAX_POINTS:
            print(f"\n  ERROR: Expected {MAX_POINTS} points, got {n}.")
            return False
        path = pending_images[current_image_index]
        variedad = extract_variedad(path, args.input)
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

    @magicgui(call_button="Save + Next")
    def save_next():
        nonlocal current_image_index, _updating
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
            _updating = True
            points_layer.data = np.empty((0, 2))
            _updating = False
        update_title()
        update_landmark_progress_widget()
        print(f"\n  {image_progress_str()}  |  [{'loaded' if loaded else 'new'}] {nxt.name}")

    @magicgui(call_button="Previous Image")
    def go_previous():
        nonlocal current_image_index, _updating
        if current_image_index <= 0:
            print("\n  Already at first image.")
            return
        current_image_index -= 1
        prev = pending_images[current_image_index]
        image_layer.data = imread(str(prev))
        viewer.reset_view()
        loaded = load_landmarks_for_image(prev)
        if not loaded:
            _updating = True
            points_layer.data = np.empty((0, 2))
            _updating = False
        update_title()
        update_landmark_progress_widget()
        print(f"\n  {image_progress_str()}  |  [{'loaded' if loaded else 'new'}] {prev.name}")

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

    viewer.window.add_dock_widget(
        build_landmark_progress_widget(), area="right", name="Landmark Progress"
    )
    viewer.window.add_dock_widget(
        build_legend_widget(), area="right", name="Landmark Order"
    )
    viewer.window.add_dock_widget(
        build_mode_indicator_widget("annotate"), area="right", name="Mode"
    )
    viewer.window.add_dock_widget(save_next, area="right", name="Navigation")
    viewer.window.add_dock_widget(go_previous, area="right", name="Navigation")

    loaded = load_landmarks_for_image(initial_path)
    update_title()
    update_landmark_progress_widget()

    print(
        f"\n{'='*50}"
        f"\n  Leaf Landmark Annotator  |  MODE: ANNOTATE"
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


# ======================================================
# REVIEW MODE (edit existing landmarks)
# ======================================================

def run_review_mode(args, image_dir, master_csv):
    """
    Review mode: load all images with their landmarks from CSV.
    Landmarks are editable -- drag points to correct them,
    then save changes back to CSV.
    """
    if not master_csv.exists():
        raise RuntimeError(
            f"No CSV found for review:\n  {master_csv}\n"
            f"Run annotate mode first to create landmarks."
        )

    master_df = pd.read_csv(master_csv)
    all_images = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png"))
    )
    if len(all_images) == 0:
        raise RuntimeError(f"No images found in:\n{image_dir}")

    # Only images that have landmarks in CSV
    annotated_names = set(master_df["image_name"].unique())
    review_images = [p for p in all_images if p.name in annotated_names]
    if len(review_images) == 0:
        raise RuntimeError("No annotated images found to review.")

    total_images = len(review_images)
    current_image_index = 0
    _updating = False
    dirty = False  # tracks unsaved changes

    def image_progress_str():
        return f"{current_image_index + 1}/{total_images}"

    viewer = napari.Viewer(title="Leaf Landmark Reviewer")
    initial_path = review_images[current_image_index]
    image_layer = viewer.add_image(imread(str(initial_path)), name="Leaf")

    points_layer = viewer.add_points(
        name="Landmarks",
        ndim=2,
        size=16,
        border_width=0.15,
        border_color="white",
    )
    # Enable point dragging in review mode
    points_layer.mode = "select"

    def refresh_annotations():
        n = len(points_layer.data)
        if n == 0:
            return
        # In review mode, always show all 14 landmarks with their names
        points_layer.face_color = COLORS[:min(n, MAX_POINTS)]
        labels = LANDMARK_NAMES[:min(n, MAX_POINTS)]
        points_layer.features = {"label": labels}
        points_layer.text = {
            "string": "{label}",
            "size": 10,
            "color": "white",
            "translation": np.array([12, 0]),
        }

    def update_title():
        current = review_images[current_image_index]
        mod = " *UNSAVED*" if dirty else ""
        viewer.title = (
            f"Leaf Reviewer{mod}  |  "
            f"{image_progress_str()}  |  "
            f"{current.name}"
        )

    def update_landmark_progress_widget():
        if '_img_label' not in globals():
            return
        _img_label.setText(f"Images: {image_progress_str()}")
        n = len(points_layer.data)
        if n > 0:
            _lm_label.setText(f"{n} landmarks loaded (drag to edit)")
            _lm_label.setStyleSheet("color: orange; font-weight: bold;")
            _lm_detail_label.setText("Move any point, then press 'Save Changes'")
        else:
            _lm_label.setText("No landmarks for this image")
            _lm_detail_label.setText("")

    @points_layer.events.data.connect
    def on_points_change(event):
        nonlocal _updating, dirty
        if _updating:
            return
        dirty = True
        refresh_annotations()
        update_title()

    def load_landmarks_for_image(image_path):
        nonlocal _updating, dirty
        rows = get_rows_for_image(master_df, image_path.name)
        if rows is None:
            return False
        pts = rows_to_points(rows)
        _updating = True
        points_layer.data = pts
        _updating = False
        refresh_annotations()
        dirty = False
        update_title()
        update_landmark_progress_widget()
        return True

    def save_current_changes():
        nonlocal master_df, dirty
        path = review_images[current_image_index]
        n = len(points_layer.data)
        if n == 0:
            print(f"\n  No points to save for {path.name}.")
            return False

        variedad = extract_variedad(path, args.input)
        # Remove existing rows for this image
        master_df = master_df[master_df["image_name"] != path.name].reset_index(drop=True)
        # Add new rows from current point positions
        rows = []
        for i, p in enumerate(points_layer.data):
            y, x = p
            h, w = image_layer.data.shape[:2]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            lm_name = LANDMARK_NAMES[i] if i < MAX_POINTS else f"extra_{i}"
            rows.append({
                "folder_name": args.input,
                "image_name": path.name,
                "variedad": variedad,
                "landmark": lm_name,
                "x": float(x),
                "y": float(y),
            })
        master_df = pd.concat([master_df, pd.DataFrame(rows)], ignore_index=True)
        master_df.to_csv(master_csv, index=False)
        dirty = False
        update_title()
        print(f"\n  Saved changes for: {path.name}")
        return True

    @magicgui(call_button="Save Changes")
    def save_changes():
        save_current_changes()

    @magicgui(call_button="Next Image")
    def go_next():
        nonlocal current_image_index, _updating
        if dirty:
            print("\n  WARNING: You have unsaved changes. Press 'Save Changes' first, or continue to discard.")
        current_image_index += 1
        if current_image_index >= total_images:
            current_image_index = total_images - 1
            print("\n  Already at last image.")
            return
        nxt = review_images[current_image_index]
        image_layer.data = imread(str(nxt))
        viewer.reset_view()
        load_landmarks_for_image(nxt)
        update_title()
        update_landmark_progress_widget()
        print(f"\n  [{image_progress_str()}] {nxt.name}")

    @magicgui(call_button="Previous Image")
    def go_previous():
        nonlocal current_image_index, _updating
        if dirty:
            print("\n  WARNING: You have unsaved changes. Press 'Save Changes' first, or continue to discard.")
        if current_image_index <= 0:
            print("\n  Already at first image.")
            return
        current_image_index -= 1
        prev = review_images[current_image_index]
        image_layer.data = imread(str(prev))
        viewer.reset_view()
        load_landmarks_for_image(prev)
        update_title()
        update_landmark_progress_widget()
        print(f"\n  [{image_progress_str()}] {prev.name}")

    # Hotkeys
    @viewer.bind_key("s")
    def hotkey_save(viewer):
        save_changes()

    @viewer.bind_key("n")
    def hotkey_next(viewer):
        go_next()

    @viewer.bind_key("p")
    def hotkey_previous(viewer):
        go_previous()

    @viewer.bind_key("r")
    def reload_landmarks(viewer):
        """Reload original landmarks from CSV (discard current edits)."""
        nonlocal master_df, dirty
        master_df = pd.read_csv(master_csv)
        path = review_images[current_image_index]
        load_landmarks_for_image(path)
        print(f"\n  Reloaded original landmarks for: {path.name}")

    # UI Layout
    viewer.window.add_dock_widget(
        build_landmark_progress_widget(), area="right", name="Landmark Progress"
    )
    viewer.window.add_dock_widget(
        build_legend_widget(), area="right", name="Landmark Order"
    )
    viewer.window.add_dock_widget(
        build_mode_indicator_widget("review"), area="right", name="Mode"
    )
    viewer.window.add_dock_widget(save_changes, area="right", name="Save")
    viewer.window.add_dock_widget(go_next, area="right", name="Navigation")
    viewer.window.add_dock_widget(go_previous, area="right", name="Navigation")

    # Init
    loaded = load_landmarks_for_image(initial_path)
    update_title()
    update_landmark_progress_widget()

    print(
        f"\n{'='*50}"
        f"\n  Leaf Landmark Reviewer  |  MODE: REVIEW"
        f"\n{'='*50}"
        f"\n  Input:   {image_dir}"
        f"\n  Output:  {master_csv}"
        f"\n  Images:  {total_images} with annotations to review"
        f"\n"
        f"\n  Hotkeys:"
        f"\n    [s]  Save changes to CSV"
        f"\n    [n]  Next image"
        f"\n    [p]  Previous image"
        f"\n    [r]  Reload original (discard edits)"
        f"\n"
        f"\n  Instructions:"
        f"\n    1. Drag any landmark to reposition it"
        f"\n    2. Press 'Save Changes' or [s] to write to CSV"
        f"\n    3. Use Next/Previous or [n]/[p] to navigate"
        f"\n    4. Press [r] to revert to original positions"
        f"\n{'='*50}"
    )

    napari.run()


# ======================================================
# MAIN
# ======================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.from_cropped:
        image_dir = Path(f"data/processed/{args.input}/cropped")
    else:
        image_dir = Path(f"data/raw/{args.input}")

    csv_dir = Path(f"data/processed/{args.input}/csv")
    csv_dir.mkdir(parents=True, exist_ok=True)
    master_csv = csv_dir / "manual_landmarks.csv"

    if args.mode == "review":
        run_review_mode(args, image_dir, master_csv)
    else:
        run_annotate_mode(args, image_dir, master_csv)


if __name__ == "__main__":
    main()