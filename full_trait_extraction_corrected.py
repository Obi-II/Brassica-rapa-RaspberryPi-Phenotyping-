"""Auditable daily-ROI plant trait extraction (16 positions per image).

Requires: pip install opencv-python numpy
Run: python full_trait_extraction_corrected.py

IMPORTANT: Measurements in cm/cm² require independently validated calibration
at the actual image resolution. Image vertical extent is NOT true plant height.
ROI label swap before 2026-06-17 16:00:09 follows the supplied correction;
verify against a manually annotated frame before analysis.
"""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------- USER SETTINGS ----------------------
IMAGE_FOLDER = Path(r"C:\Users\emman\Desktop\Day_photos")
ROI_FOLDER = Path(r"C:\Users\emman\Desktop\Daily_ROIs")
OUTPUT_FOLDER = Path(r"C:\Users\emman\Desktop\Plant_Trait_Results")
START_DATE, END_DATE = "20260608", "20260622"
LOWER_GREEN = np.array([35, 60, 40], dtype=np.uint8)
UPPER_GREEN = np.array([85, 255, 255], dtype=np.uint8)
MORPHOLOGY_KERNEL_SIZE = 5
MIN_COMPONENT_AREA = 80
# Set ONLY after validating a physical reference in the ORIGINAL images.
PIXELS_PER_CM = 38.46
# Optionally enable sample visual checks (every Nth image, and flagged images).
SAVE_QC_OVERLAYS = True
OVERLAY_EVERY_N_IMAGES = 100
# If source ROI labels were corrected already, set to False to avoid double swap.
APPLY_HISTORICAL_TRAY_SWAP = True
LABEL_CORRECTION_CUTOFF = datetime(2026, 6, 17, 16, 0, 9)
EXPECTED_LABELS = {f"T{t}_P{p}" for t in range(1, 5) for p in range(1, 5)}
# QA flags are REVIEW prompts, not grounds for automatic deletion.
EDGE_MARGIN_PX = 2
MIN_MASK_PIXELS_FLAG = 100
MAX_MASK_FRACTION_FLAG = 0.80

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OVERLAY_FOLDER = OUTPUT_FOLDER / "qc_overlays"
if SAVE_QC_OVERLAYS:
    OVERLAY_FOLDER.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_FOLDER / "plant_traits_all_images_corrected.csv"
QC_PATH = OUTPUT_FOLDER / "extraction_quality_report.csv"


def image_datetime(name):
    match = re.search(r"(\d{8})_(\d{6})", name)
    if not match:
        return None
    try:
        return datetime.strptime(''.join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def corrected_label(label, timestamp):
    if APPLY_HISTORICAL_TRAY_SWAP and timestamp < LABEL_CORRECTION_CUTOFF:
        if label.startswith("T3_"):
            return "T4_" + label[3:]
        if label.startswith("T4_"):
            return "T3_" + label[3:]
    return label


def load_and_validate_rois(path, timestamp, width, height):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 16:
        raise ValueError("ROI JSON must be a list of exactly 16 entries")
    result = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"ROI {index}: expected object")
        source_id = entry.get("Plant_ID")
        if not isinstance(source_id, str):
            raise ValueError(f"ROI {index}: missing Plant_ID")
        plant_id = corrected_label(source_id, timestamp)
        try:
            raw = [entry[k] for k in ("x1", "y1", "x2", "y2")]
            coords = [int(v) for v in raw]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"ROI {source_id}: invalid coordinates") from exc
        x1, y1, x2, y2 = coords
        # Reject rather than silently clip invalid coordinates.
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"ROI {source_id}: outside image bounds: {coords}")
        result.append((source_id, plant_id, x1, y1, x2, y2))
    labels = [r[1] for r in result]
    if len(set(labels)) != 16 or set(labels) != EXPECTED_LABELS:
        raise ValueError(f"Missing/duplicate/unexpected plant IDs: {labels}")
    return result


def segment_plant(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MORPHOLOGY_KERNEL_SIZE,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean[labels == i] = 255
    # Fill only enclosed holes; preserve disconnected green regions.
    padded = cv2.copyMakeBorder(clean, 1, 1, 1, 1,
                                cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2),
                          dtype=np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flooded[1:-1, 1:-1])
    return cv2.bitwise_or(clean, holes)


def traits_and_flags(mask):
    area = int(cv2.countNonZero(mask))
    h, w = mask.shape
    flags = []
    if area == 0:
        return dict(area=0, width=0, vertical_extent=0, perimeter=0.,
                    hull_area=0., cx="", cy=""), ["EMPTY_MASK"]
    if area < MIN_MASK_PIXELS_FLAG:
        flags.append("SMALL_MASK")
    if area / (h * w) > MAX_MASK_FRACTION_FLAG:
        flags.append("LARGE_MASK_FRACTION")
    edge = EDGE_MARGIN_PX
    if (np.any(mask[:edge, :]) or np.any(mask[-edge:, :]) or
            np.any(mask[:, :edge]) or np.any(mask[:, -edge:])):
        flags.append("TOUCHES_ROI_EDGE")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    pts = cv2.findNonZero(mask)
    _, _, bw, bh = cv2.boundingRect(pts)
    hull = cv2.convexHull(np.vstack(contours))
    m = cv2.moments(mask, binaryImage=True)
    return dict(area=area, width=bw, vertical_extent=bh,
                perimeter=sum(cv2.arcLength(c, True) for c in contours),
                hull_area=cv2.contourArea(hull),
                cx=m['m10'] / m['m00'], cy=m['m01'] / m['m00']), flags


def length_cm(pixels):
    return "" if PIXELS_PER_CM is None else pixels / PIXELS_PER_CM


def area_cm2(pixels):
    return "" if PIXELS_PER_CM is None else pixels / PIXELS_PER_CM ** 2


def save_overlay(image, items, path):
    output = image.copy()
    for roi, mask, flags in items:
        _, plant_id, x1, y1, x2, y2 = roi
        patch = output[y1:y2, x1:x2]
        color = (0, 0, 255) if flags else (0, 255, 0)
        tint = np.zeros_like(patch)
        tint[:] = color
        blended = cv2.addWeighted(patch, .6, tint, .4, 0)
        patch[mask > 0] = blended[mask > 0]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, plant_id, (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    cv2.imwrite(str(path), output)


def main():
    if PIXELS_PER_CM is not None and PIXELS_PER_CM <= 0:
        raise ValueError("PIXELS_PER_CM must be positive")
    if MORPHOLOGY_KERNEL_SIZE < 1 or MORPHOLOGY_KERNEL_SIZE % 2 == 0:
        raise ValueError("MORPHOLOGY_KERNEL_SIZE must be positive and odd")
    for folder in (IMAGE_FOLDER, ROI_FOLDER):
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
    images = sorted((p for p in IMAGE_FOLDER.iterdir()
                     if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
                     and image_datetime(p.name) is not None
                     and START_DATE <= image_datetime(p.name).strftime('%Y%m%d') <= END_DATE),
                    key=lambda p: (image_datetime(p.name), p.name))
    if not images:
        raise RuntimeError("No images found in selected date range")
    columns = ["Image_Name", "Date", "Time", "DateTime", "Plant_ID",
               "Original_ROI_Label", "ROI_x1", "ROI_y1", "ROI_x2", "ROI_y2",
               "Area_pixels", "Leaf_Area_cm2", "Width_pixels", "Plant_Width_cm",
               "Vertical_Extent_pixels", "Vertical_Extent_cm",
               "Perimeter_pixels", "Perimeter_cm", "Convex_Hull_Area_pixels",
               "Convex_Hull_Area_cm2", "Centroid_x_ROI", "Centroid_y_ROI",
               "Centroid_x_Full_Image", "Centroid_y_Full_Image", "QC_Flags"]
    qc_columns = ["Image_Name", "Plant_ID", "Severity", "Issue"]
    successful, failed, written = 0, 0, 0
    with CSV_PATH.open('w', newline='', encoding='utf-8') as output, \
         QC_PATH.open('w', newline='', encoding='utf-8') as qc_file:
        writer = csv.DictWriter(output, fieldnames=columns)
        qc_writer = csv.DictWriter(qc_file, fieldnames=qc_columns)
        writer.writeheader()
        qc_writer.writeheader()
        for index, path in enumerate(images, 1):
            timestamp = image_datetime(path.name)
            try:
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError("Image cannot be read")
                image_h, image_w = img.shape[:2]
                roi_path = ROI_FOLDER / (timestamp.strftime('%Y%m%d') + '.json')
                if not roi_path.exists():
                    raise FileNotFoundError(f"Missing daily ROI: {roi_path.name}")
                rois = load_and_validate_rois(roi_path, timestamp, image_w, image_h)
                image_rows, qc_rows, overlay_items = [], [], []
                for roi in rois:
                    original, label, x1, y1, x2, y2 = roi
                    mask = segment_plant(img[y1:y2, x1:x2])
                    t, flags = traits_and_flags(mask)
                    image_rows.append({
                        "Image_Name": path.name, "Date": timestamp.strftime('%Y%m%d'),
                        "Time": timestamp.strftime('%H%M%S'),
                        "DateTime": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        "Plant_ID": label, "Original_ROI_Label": original,
                        "ROI_x1": x1, "ROI_y1": y1, "ROI_x2": x2, "ROI_y2": y2,
                        "Area_pixels": t['area'], "Leaf_Area_cm2": area_cm2(t['area']),
                        "Width_pixels": t['width'], "Plant_Width_cm": length_cm(t['width']),
                        "Vertical_Extent_pixels": t['vertical_extent'],
                        "Vertical_Extent_cm": length_cm(t['vertical_extent']),
                        "Perimeter_pixels": t['perimeter'],
                        "Perimeter_cm": length_cm(t['perimeter']),
                        "Convex_Hull_Area_pixels": t['hull_area'],
                        "Convex_Hull_Area_cm2": area_cm2(t['hull_area']),
                        "Centroid_x_ROI": t['cx'], "Centroid_y_ROI": t['cy'],
                        "Centroid_x_Full_Image": x1 + t['cx'] if t['cx'] != '' else '',
                        "Centroid_y_Full_Image": y1 + t['cy'] if t['cy'] != '' else '',
                        "QC_Flags": ';'.join(flags)})
                    for flag in flags:
                        qc_rows.append({"Image_Name": path.name, "Plant_ID": label,
                                        "Severity": "REVIEW", "Issue": flag})
                    overlay_items.append((roi, mask, flags))
                # Commit only after all 16 ROIs succeed.
                writer.writerows(image_rows)
                qc_writer.writerows(qc_rows)
                written += len(image_rows)
                successful += 1
                if SAVE_QC_OVERLAYS and (index == 1 or index % OVERLAY_EVERY_N_IMAGES == 0 or qc_rows):
                    save_overlay(img, overlay_items,
                                 OVERLAY_FOLDER / f"{path.stem}_qc.jpg")
            except Exception as exc:
                failed += 1
                qc_writer.writerow({"Image_Name": path.name, "Plant_ID": "",
                                    "Severity": "ERROR", "Issue": str(exc)})
                print(f"SKIPPED {path.name}: {exc}")
            if index == 1 or index % 50 == 0 or index == len(images):
                print(f"Progress {index}/{len(images)} | complete {successful} | failed {failed}")
    print(f"CSV: {CSV_PATH}\nQuality report: {QC_PATH}")
    print(f"Processed images: {successful}; skipped: {failed}; rows: {written}")
    print("WARNING: Validate calibration, labels, and masks against reference images/Fiji.")


if __name__ == '__main__':
    main()
