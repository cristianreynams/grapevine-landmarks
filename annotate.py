import napari
import numpy as np
import pandas as pd

from pathlib import Path
from skimage.io import imread
from magicgui import magicgui

# ======================================================
# CONFIG
# ======================================================

IMAGE_DIR = Path(
    "data/processed/debug/cropped"
)

OUTPUT_DIR = Path(
    "output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FOLDER_NAME = IMAGE_DIR.parent.name

MASTER_CSV = (
    OUTPUT_DIR /
    f"{FOLDER_NAME}_landmarks.csv"
)

LANDMARK_NAMES = [

    "PECIOLE_SINUS",

    "L4_LEFT",
    "LOWER_SINUS_LEFT",

    "L3_LEFT",
    "MEDIUM_SINUS_LEFT",

    "L2_LEFT",
    "UPPER_SINUS_LEFT",

    "L1",

    "UPPER_SINUS_RIGHT",
    "L2_RIGHT",

    "MEDIUM_SINUS_RIGHT",
    "L3_RIGHT",

    "LOWER_SINUS_RIGHT",
    "L4_RIGHT"
]

COLORS = [

    "red",
    "cyan",
    "magenta",
    "yellow",
    "orange",
    "blue",
    "white",
    "purple",
    "green",
    "pink",
    "brown",
    "lime",
    "gold",
    "turquoise"
]

MAX_POINTS = len(
    LANDMARK_NAMES
)

# ======================================================
# IMAGE LIST
# ======================================================

all_images = sorted(

    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

if len(all_images) == 0:

    raise RuntimeError(
        f"No images found in:\n{IMAGE_DIR}"
    )

# ======================================================
# EXISTING CSV
# ======================================================

if MASTER_CSV.exists():

    master_df = pd.read_csv(
        MASTER_CSV
    )

    completed_images = set(
        master_df["image_name"].unique()
    )

else:

    master_df = pd.DataFrame(
        columns=[
            "folder_name",
            "image_name",
            "landmark",
            "x",
            "y"
        ]
    )

    completed_images = set()

# ======================================================
# PENDING IMAGES
# ======================================================

pending_images = [

    p for p in all_images

    if p.name not in completed_images
]

if len(pending_images) == 0:

    raise RuntimeError(
        "All images already annotated."
    )

current_image_index = 0

# ======================================================
# INITIAL IMAGE
# ======================================================

initial_image_path = (
    pending_images[
        current_image_index
    ]
)

initial_image = imread(
    str(initial_image_path)
)

# ======================================================
# VIEWER
# ======================================================

viewer = napari.Viewer(
    title="Leaf Landmark Annotator"
)

image_layer = viewer.add_image(
    initial_image,
    name="Leaf"
)

# ======================================================
# POINTS LAYER
# ======================================================

points_layer = viewer.add_points(
    name="Landmarks",
    ndim=2,
    size=16,
    border_width=0.15,
    border_color="white",
)

# ======================================================
# REFRESH
# ======================================================

def refresh_annotations():

    n = len(points_layer.data)

    if n == 0:
        return

    points_layer.face_color = (
        COLORS[:n]
    )

    points_layer.features = {

        "label":
        LANDMARK_NAMES[:n]
    }

    points_layer.text = {

        "string": "{label}",

        "size": 10,

        "color": "white",

        "translation":
        np.array([12, 0]),
    }

# ======================================================
# CALLBACK
# ======================================================

@points_layer.events.data.connect
def on_points_change(event):

    n = len(points_layer.data)

    if n > MAX_POINTS:

        points_layer.data = (
            points_layer.data[:MAX_POINTS]
        )

        return

    refresh_annotations()

# ======================================================
# SAVE
# ======================================================

def save_current_landmarks():

    global master_df

    n = len(points_layer.data)

    if n != MAX_POINTS:

        print(
            f"\nERROR:"
            f"\nExpected {MAX_POINTS}"
            f"\nGot {n}"
        )

        return False

    image_path = pending_images[
        current_image_index
    ]

    rows = []

    for i, p in enumerate(
        points_layer.data
    ):

        y, x = p

        rows.append({

            "folder_name":
            image_path.parent.name,

            "image_name":
            image_path.name,

            "landmark":
            LANDMARK_NAMES[i],

            "x": float(x),

            "y": float(y)
        })

    current_df = pd.DataFrame(
        rows
    )

    master_df = pd.concat(
        [master_df, current_df],
        ignore_index=True
    )

    master_df.to_csv(
        MASTER_CSV,
        index=False
    )

    print(
        f"\nSaved:\n{MASTER_CSV}"
    )

    return True

# ======================================================
# SAVE NEXT
# ======================================================

@magicgui(call_button="Save + Next")
def save_next():

    global current_image_index

    success = save_current_landmarks()

    if not success:
        return

    current_image_index += 1

    if current_image_index >= len(
        pending_images
    ):

        print(
            "\nDATASET COMPLETE"
        )

        return

    next_image_path = pending_images[
        current_image_index
    ]

    next_image = imread(
        str(next_image_path)
    )

    image_layer.data = next_image

    points_layer.data = np.empty(
        (0, 2)
    )

    viewer.reset_view()

    print(
        f"\nImage:"
        f"\n{next_image_path.name}"
    )

# ======================================================
# HOTKEYS
# ======================================================

@viewer.bind_key("c")
def clear_points(viewer):

    points_layer.data = np.empty(
        (0, 2)
    )

# ======================================================
# UI
# ======================================================

viewer.window.add_dock_widget(
    save_next,
    area="right"
)

# ======================================================
# RUN
# ======================================================

napari.run()