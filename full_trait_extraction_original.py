"""
Full Daily-ROI Plant Trait Extraction Pipeline

For every image, this script:
1. Extracts the date and time from the filename.
2. Loads the matching daily ROI JSON.
3. Crops all 16 planting positions.
4. Segments green plant material.
5. Removes small noise and fills holes inside leaves.
6. Calculates:
   - segmented area
   - plant width
   - plant height
   - perimeter
   - convex-hull area
   - centroid
7. Saves all results into one CSV.

Expected image name:
image_20260617_130109.jpg

Expected ROI file:
20260617.json
"""

import csv
import cv2
import json
import re
from pathlib import Path

import numpy as np


# ============================================================
# FOLDERS
# ============================================================

IMAGE_FOLDER = Path(
    r"C:\Users\emman\Desktop\Day_photos"
)

ROI_FOLDER = Path(
    r"C:\Users\emman\Desktop\Daily_ROIs"
)

OUTPUT_FOLDER = Path(
    r"C:\Users\emman\Desktop\Plant_Trait_Results"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_FOLDER / "plant_traits_all_images.csv"


# ============================================================
# IMAGE DATE RANGE
# ============================================================

START_DATE = "20260608"
END_DATE = "20260622"


# ============================================================
# SEGMENTATION SETTINGS
# ============================================================

LOWER_GREEN = np.array(
    [35, 60, 40],
    dtype=np.uint8
)

UPPER_GREEN = np.array(
    [85, 255, 255],
    dtype=np.uint8
)

MORPHOLOGY_KERNEL_SIZE = 5

# Minimum disconnected green region to retain
MIN_COMPONENT_AREA = 80


# ============================================================
# SPATIAL CALIBRATION
# ============================================================

# Use a number only if the calibration applies to the ORIGINAL
# 4608 × 2592 images used by this script.
#
# Example:
# PIXELS_PER_CM = 38.46
#
# If calibration is not yet confirmed:
PIXELS_PER_CM = None


# ============================================================
# OPTIONAL OVERLAYS
# ============================================================

# False is recommended for the first complete run.
# Setting this to True will create 3,021 overview images.
SAVE_OVERLAYS = False

OVERLAY_FOLDER = OUTPUT_FOLDER / "overlays"

if SAVE_OVERLAYS:
    OVERLAY_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPECTED PLANT IDS
# ============================================================

EXPECTED_LABELS = [
    "T1_P1", "T1_P2", "T1_P3", "T1_P4",
    "T2_P1", "T2_P2", "T2_P3", "T2_P4",
    "T4_P1", "T4_P2", "T4_P3", "T4_P4",
    "T3_P1", "T3_P3", "T3_P2", "T3_P4",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_datetime(filename):
    """
    Extract YYYYMMDD and HHMMSS from an image filename.
    """

    match = re.search(
        r"(\d{8})_(\d{6})",
        filename
    )

    if not match:
        return None

    date_string = match.group(1)
    time_string = match.group(2)

    return date_string, time_string


def load_rois(json_path):
    """
    Load and validate one daily ROI JSON file.
    """

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:
        rois = json.load(file)

    if not isinstance(rois, list):
        raise ValueError(
            f"{json_path.name} does not contain an ROI list."
        )

    if len(rois) != 16:
        raise ValueError(
            f"{json_path.name} contains {len(rois)} ROIs, "
            "but 16 are required."
        )

    labels = [
        roi.get("Plant_ID")
        for roi in rois
    ]

    missing = [
        label
        for label in EXPECTED_LABELS
        if label not in labels
    ]

    if missing:
        raise ValueError(
            f"{json_path.name} is missing: {missing}"
        )

    return rois


def remove_small_components(mask, minimum_area):
    """
    Remove disconnected components smaller than minimum_area.
    """

    number_of_labels, labels, statistics, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(mask)

    for component_number in range(
        1,
        number_of_labels
    ):
        area = statistics[
            component_number,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_area:
            cleaned[
                labels == component_number
            ] = 255

    return cleaned


def fill_mask_holes(mask):
    """
    Fill enclosed dark holes inside segmented leaves.
    """

    padded = cv2.copyMakeBorder(
        mask,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0
    )

    flood_filled = padded.copy()

    flood_mask = np.zeros(
        (
            padded.shape[0] + 2,
            padded.shape[1] + 2
        ),
        dtype=np.uint8
    )

    cv2.floodFill(
        flood_filled,
        flood_mask,
        seedPoint=(0, 0),
        newVal=255
    )

    flood_filled = flood_filled[
        1:-1,
        1:-1
    ]

    holes = cv2.bitwise_not(
        flood_filled
    )

    filled_mask = cv2.bitwise_or(
        mask,
        holes
    )

    return filled_mask


def segment_plant(roi_image):
    """
    Segment green plant pixels inside one ROI.
    """

    hsv = cv2.cvtColor(
        roi_image,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv,
        LOWER_GREEN,
        UPPER_GREEN
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPHOLOGY_KERNEL_SIZE,
            MORPHOLOGY_KERNEL_SIZE
        )
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
        iterations=1
    )

    mask = remove_small_components(
        mask,
        MIN_COMPONENT_AREA
    )

    mask = fill_mask_holes(mask)

    return mask


def convert_length_to_cm(pixel_length):
    """
    Convert pixels to centimetres when calibration is available.
    """

    if PIXELS_PER_CM is None:
        return ""

    return pixel_length / PIXELS_PER_CM


def convert_area_to_cm2(pixel_area):
    """
    Convert pixel area to square centimetres.
    """

    if PIXELS_PER_CM is None:
        return ""

    return pixel_area / (
        PIXELS_PER_CM ** 2
    )


def calculate_traits(mask):
    """
    Calculate traits from the complete ROI mask.

    All retained disconnected leaf regions inside the ROI
    are treated as parts of the same plant.
    """

    area_pixels = int(
        cv2.countNonZero(mask)
    )

    if area_pixels == 0:
        return {
            "area_pixels": 0,
            "width_pixels": 0,
            "height_pixels": 0,
            "perimeter_pixels": 0.0,
            "convex_hull_area_pixels": 0.0,
            "centroid_x_roi": "",
            "centroid_y_roi": ""
        }

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    perimeter_pixels = sum(
        cv2.arcLength(
            contour,
            closed=True
        )
        for contour in contours
    )

    nonzero_points = cv2.findNonZero(mask)

    x, y, width, height = cv2.boundingRect(
        nonzero_points
    )

    all_contour_points = np.vstack(
        contours
    )

    hull = cv2.convexHull(
        all_contour_points
    )

    convex_hull_area = cv2.contourArea(
        hull
    )

    moments = cv2.moments(
        mask,
        binaryImage=True
    )

    if moments["m00"] > 0:
        centroid_x = (
            moments["m10"] /
            moments["m00"]
        )

        centroid_y = (
            moments["m01"] /
            moments["m00"]
        )

    else:
        centroid_x = ""
        centroid_y = ""

    return {
        "area_pixels": area_pixels,
        "width_pixels": int(width),
        "height_pixels": int(height),
        "perimeter_pixels": float(perimeter_pixels),
        "convex_hull_area_pixels": float(convex_hull_area),
        "centroid_x_roi": centroid_x,
        "centroid_y_roi": centroid_y
    }


def create_overlay(
    image,
    roi_data,
    mask
):
    """
    Add the segmented plant mask to the full image.
    """

    plant_id = roi_data["Plant_ID"]

    x1 = int(roi_data["x1"])
    y1 = int(roi_data["y1"])
    x2 = int(roi_data["x2"])
    y2 = int(roi_data["y2"])

    roi_overlay = image[
        y1:y2,
        x1:x2
    ].copy()

    green_layer = np.zeros_like(
        roi_overlay
    )

    green_layer[:] = (
        0,
        255,
        0
    )

    blended = cv2.addWeighted(
        roi_overlay,
        0.55,
        green_layer,
        0.45,
        0
    )

    roi_overlay[
        mask > 0
    ] = blended[
        mask > 0
    ]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        roi_overlay,
        contours,
        -1,
        (0, 0, 255),
        2
    )

    image[
        y1:y2,
        x1:x2
    ] = roi_overlay

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        image,
        plant_id,
        (x1, max(y1 - 5, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# CHECK FOLDERS
# ============================================================

if not IMAGE_FOLDER.exists():
    raise FileNotFoundError(
        f"Image folder was not found:\n{IMAGE_FOLDER}"
    )

if not ROI_FOLDER.exists():
    raise FileNotFoundError(
        f"ROI folder was not found:\n{ROI_FOLDER}"
    )


# ============================================================
# COLLECT IMAGE FILES
# ============================================================

image_files = []

for image_path in sorted(
    IMAGE_FOLDER.glob("*.jpg")
):
    extracted = extract_datetime(
        image_path.name
    )

    if extracted is None:
        continue

    image_date, _ = extracted

    if START_DATE <= image_date <= END_DATE:
        image_files.append(image_path)


if not image_files:
    raise RuntimeError(
        "No images were found in the selected date range."
    )


print("\n" + "=" * 68)
print("FULL PLANT TRAIT EXTRACTION")
print("=" * 68)
print(f"Images found: {len(image_files)}")
print(f"Output CSV: {CSV_PATH}")

if PIXELS_PER_CM is None:
    print(
        "Calibration: not set — centimetre columns "
        "will remain blank."
    )
else:
    print(
        f"Calibration: {PIXELS_PER_CM} pixels/cm"
    )

print("=" * 68)


# ============================================================
# CSV HEADINGS
# ============================================================

fieldnames = [
    "Image_Name",
    "Date",
    "Time",
    "DateTime",
    "Plant_ID",

    "ROI_x1",
    "ROI_y1",
    "ROI_x2",
    "ROI_y2",

    "Area_pixels",
    "Leaf_Area_cm2",

    "Width_pixels",
    "Plant_Width_cm",

    "Height_pixels",
    "Plant_Height_cm",

    "Perimeter_pixels",
    "Perimeter_cm",

    "Convex_Hull_Area_pixels",
    "Convex_Hull_Area_cm2",

    "Centroid_x_ROI",
    "Centroid_y_ROI",
    "Centroid_x_Full_Image",
    "Centroid_y_Full_Image"
]


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

roi_cache = {}

processed_images = 0
failed_images = 0
total_rows = 0


with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for image_number, image_path in enumerate(
        image_files,
        start=1
    ):

        extracted = extract_datetime(
            image_path.name
        )

        image_date, image_time = extracted

        roi_json_path = (
            ROI_FOLDER /
            f"{image_date}.json"
        )

        if not roi_json_path.exists():
            print(
                f"\nSkipped {image_path.name}: "
                f"{roi_json_path.name} not found."
            )

            failed_images += 1
            continue

        try:
            if image_date not in roi_cache:
                roi_cache[image_date] = load_rois(
                    roi_json_path
                )

            rois = roi_cache[image_date]

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                raise RuntimeError(
                    "OpenCV could not read the image."
                )

            image_height, image_width = (
                image.shape[:2]
            )

            if SAVE_OVERLAYS:
                full_overlay = image.copy()

            for roi_data in rois:

                plant_id = roi_data[
                    "Plant_ID"
                ]

                x1 = int(roi_data["x1"])
                y1 = int(roi_data["y1"])
                x2 = int(roi_data["x2"])
                y2 = int(roi_data["y2"])

                x1 = max(
                    0,
                    min(x1, image_width - 1)
                )

                y1 = max(
                    0,
                    min(y1, image_height - 1)
                )

                x2 = max(
                    x1 + 1,
                    min(x2, image_width)
                )

                y2 = max(
                    y1 + 1,
                    min(y2, image_height)
                )

                roi_image = image[
                    y1:y2,
                    x1:x2
                ].copy()

                mask = segment_plant(
                    roi_image
                )

                traits = calculate_traits(
                    mask
                )

                centroid_x_roi = traits[
                    "centroid_x_roi"
                ]

                centroid_y_roi = traits[
                    "centroid_y_roi"
                ]

                if centroid_x_roi == "":
                    centroid_x_full = ""
                    centroid_y_full = ""

                else:
                    centroid_x_full = (
                        x1 + centroid_x_roi
                    )

                    centroid_y_full = (
                        y1 + centroid_y_roi
                    )

                writer.writerow({
                    "Image_Name":
                        image_path.name,

                    "Date":
                        image_date,

                    "Time":
                        image_time,

                    "DateTime":
                        (
                            f"{image_date[:4]}-"
                            f"{image_date[4:6]}-"
                            f"{image_date[6:8]} "
                            f"{image_time[:2]}:"
                            f"{image_time[2:4]}:"
                            f"{image_time[4:6]}"
                        ),

                    "Plant_ID":
                        plant_id,

                    "ROI_x1": x1,
                    "ROI_y1": y1,
                    "ROI_x2": x2,
                    "ROI_y2": y2,

                    "Area_pixels":
                        traits["area_pixels"],

                    "Leaf_Area_cm2":
                        convert_area_to_cm2(
                            traits["area_pixels"]
                        ),

                    "Width_pixels":
                        traits["width_pixels"],

                    "Plant_Width_cm":
                        convert_length_to_cm(
                            traits["width_pixels"]
                        ),

                    "Height_pixels":
                        traits["height_pixels"],

                    "Plant_Height_cm":
                        convert_length_to_cm(
                            traits["height_pixels"]
                        ),

                    "Perimeter_pixels":
                        traits["perimeter_pixels"],

                    "Perimeter_cm":
                        convert_length_to_cm(
                            traits["perimeter_pixels"]
                        ),

                    "Convex_Hull_Area_pixels":
                        traits[
                            "convex_hull_area_pixels"
                        ],

                    "Convex_Hull_Area_cm2":
                        convert_area_to_cm2(
                            traits[
                                "convex_hull_area_pixels"
                            ]
                        ),

                    "Centroid_x_ROI":
                        centroid_x_roi,

                    "Centroid_y_ROI":
                        centroid_y_roi,

                    "Centroid_x_Full_Image":
                        centroid_x_full,

                    "Centroid_y_Full_Image":
                        centroid_y_full
                })

                total_rows += 1

                if SAVE_OVERLAYS:
                    create_overlay(
                        full_overlay,
                        roi_data,
                        mask
                    )

            if SAVE_OVERLAYS:
                overlay_path = (
                    OVERLAY_FOLDER /
                    f"{image_path.stem}_overlay.jpg"
                )

                cv2.imwrite(
                    str(overlay_path),
                    full_overlay
                )

            processed_images += 1

        except Exception as error:
            failed_images += 1

            print(
                f"\nFailed: {image_path.name}\n"
                f"Reason: {error}"
            )

        if (
            image_number == 1
            or image_number % 50 == 0
            or image_number == len(image_files)
        ):
            print(
                f"Progress: {image_number}/"
                f"{len(image_files)} images"
            )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 68)
print("EXTRACTION COMPLETED")
print("=" * 68)
print(f"Successfully processed: {processed_images}")
print(f"Failed or skipped:       {failed_images}")
print(f"CSV rows written:        {total_rows}")
print(f"Results saved to:        {CSV_PATH}")
print("=" * 68)