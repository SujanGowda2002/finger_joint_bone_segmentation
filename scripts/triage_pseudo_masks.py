import csv
import argparse
from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_mask_files(input_dir: Path):
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and p.name.endswith("_mask.png"):
            files.append(p)
    return files


def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def connected_component_stats(mask: np.ndarray):
    """
    Returns:
    - num_components
    - largest_area
    - second_largest_area
    - total_white_pixels
    - border_touching_component_count
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = mask.shape
    component_areas = []
    border_touching_count = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        component_areas.append(area)

        touches_border = (x <= 0) or (y <= 0) or (x + ww >= w) or (y + hh >= h)
        if touches_border:
            border_touching_count += 1

    component_areas.sort(reverse=True)

    largest_area = component_areas[0] if len(component_areas) >= 1 else 0
    second_largest_area = component_areas[1] if len(component_areas) >= 2 else 0
    total_white_pixels = int(np.sum(mask > 0))

    return {
        "num_components": max(0, num_labels - 1),
        "largest_area": largest_area,
        "second_largest_area": second_largest_area,
        "total_white_pixels": total_white_pixels,
        "border_touching_component_count": border_touching_count,
    }


def bounding_box_stats(mask: np.ndarray):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return {
            "bbox_xmin": "",
            "bbox_ymin": "",
            "bbox_xmax": "",
            "bbox_ymax": "",
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_area": 0,
            "bbox_area_ratio": 0.0,
            "bbox_center_x_ratio": "",
            "bbox_center_y_ratio": "",
        }

    h, w = mask.shape
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1
    bbox_area = bbox_width * bbox_height
    bbox_area_ratio = bbox_area / (h * w)

    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    return {
        "bbox_xmin": xmin,
        "bbox_ymin": ymin,
        "bbox_xmax": xmax,
        "bbox_ymax": ymax,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "bbox_area_ratio": round(bbox_area_ratio, 6),
        "bbox_center_x_ratio": round(center_x / w, 6),
        "bbox_center_y_ratio": round(center_y / h, 6),
    }


def probability_stats(prob_map: np.ndarray):
    """
    prob_map is uint8 grayscale image saved from probabilities.
    Convert back to [0,1] float-ish scale for summary stats.
    """
    p = prob_map.astype(np.float32) / 255.0

    return {
        "mean_probability": round(float(np.mean(p)), 6),
        "max_probability": round(float(np.max(p)), 6),
        "std_probability": round(float(np.std(p)), 6),
        "p90_probability": round(float(np.percentile(p, 90)), 6),
        "p95_probability": round(float(np.percentile(p, 95)), 6),
        "p99_probability": round(float(np.percentile(p, 99)), 6),
    }


def heuristic_triage(row):
    """
    Conservative triage rules.
    These are not ground truth labels.
    They only help prioritize review.
    """

    white_ratio = row["white_ratio"]
    num_components = row["num_components"]
    largest_area = row["largest_area"]
    mean_prob = row["mean_probability"]
    max_prob = row["max_probability"]
    bbox_area_ratio = row["bbox_area_ratio"]

    reasons = []

    if white_ratio == 0:
        reasons.append("empty_mask")
    if white_ratio < 0.01:
        reasons.append("very_low_mask_area")
    if white_ratio > 0.75:
        reasons.append("very_high_mask_area")
    if num_components > 10:
        reasons.append("many_components")
    if largest_area < 50 and white_ratio > 0:
        reasons.append("tiny_largest_component")
    if mean_prob < 0.10:
        reasons.append("low_mean_probability")
    if max_prob < 0.40:
        reasons.append("low_peak_probability")
    if isinstance(bbox_area_ratio, float) and bbox_area_ratio > 0.90:
        reasons.append("bbox_almost_full_image")

    if "empty_mask" in reasons:
        triage = "bad"
    elif len(reasons) >= 3:
        triage = "bad"
    elif len(reasons) >= 1:
        triage = "suspicious"
    else:
        triage = "likely_usable"

    return triage, ";".join(reasons)


def process_one(mask_path: Path, prob_dir: Path):
    mask = load_gray(mask_path)
    mask = binarize_mask(mask)

    stem = mask_path.name.replace("_mask.png", "")
    prob_path = prob_dir / f"{stem}_prob.png"

    if not prob_path.exists():
        raise FileNotFoundError(f"Missing probability map for {mask_path.name}: {prob_path}")

    prob_map = load_gray(prob_path)

    h, w = mask.shape
    white_pixels = int(np.sum(mask > 0))
    white_ratio = white_pixels / float(h * w)

    cc = connected_component_stats(mask)
    bb = bounding_box_stats(mask)
    ps = probability_stats(prob_map)

    row = {
        "image_stem": stem,
        "mask_name": mask_path.name,
        "prob_map_name": prob_path.name,
        "height": h,
        "width": w,
        "white_pixels": white_pixels,
        "white_ratio": round(white_ratio, 6),
        **cc,
        **bb,
        **ps,
    }

    triage_label, triage_reasons = heuristic_triage(row)
    row["triage_label"] = triage_label
    row["triage_reasons"] = triage_reasons

    return row


def main():
    parser = argparse.ArgumentParser(description="Non-destructive triage for pseudo masks.")
    parser.add_argument(
        "--mask_input_dir",
        type=str,
        default="outputs/model_pseudo_masks_pip_dip_only/masks", 
        help="Directory containing raw masks"
    )
    parser.add_argument(
        "--prob_input_dir",
        type=str,
        default="outputs/model_pseudo_masks_pip_dip_only/prob_maps",
        help="Directory containing probability maps"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/model_pseudo_masks/triage_summary.csv",
        help="CSV file to save triage results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Process only the first N masks"
    )
    args = parser.parse_args()

    mask_input_dir = Path(args.mask_input_dir)
    prob_input_dir = Path(args.prob_input_dir)
    output_csv = Path(args.output_csv)

    ensure_dir(output_csv.parent)

    if not mask_input_dir.exists():
        raise FileNotFoundError(f"Mask input directory does not exist: {mask_input_dir}")

    if not prob_input_dir.exists():
        raise FileNotFoundError(f"Probability map directory does not exist: {prob_input_dir}")

    all_masks = list_mask_files(mask_input_dir)
    total_masks = len(all_masks)

    print(f"Found {total_masks} masks in: {mask_input_dir}")
    if total_masks == 0:
        raise ValueError("No valid mask files found.")

    selected_masks = all_masks[:args.limit]
    print(f"Processing first {len(selected_masks)} masks only...")

    results = []
    success_count = 0
    fail_count = 0

    for idx, mask_path in enumerate(selected_masks, start=1):
        try:
            row = process_one(mask_path, prob_input_dir)
            results.append(row)
            success_count += 1
            print(f"[{idx}/{len(selected_masks)}] OK  - {mask_path.name} -> {row['triage_label']}")
        except Exception as e:
            fail_count += 1
            results.append({
                "image_stem": mask_path.name.replace("_mask.png", ""),
                "mask_name": mask_path.name,
                "prob_map_name": "",
                "height": "",
                "width": "",
                "white_pixels": "",
                "white_ratio": "",
                "num_components": "",
                "largest_area": "",
                "second_largest_area": "",
                "total_white_pixels": "",
                "border_touching_component_count": "",
                "bbox_xmin": "",
                "bbox_ymin": "",
                "bbox_xmax": "",
                "bbox_ymax": "",
                "bbox_width": "",
                "bbox_height": "",
                "bbox_area": "",
                "bbox_area_ratio": "",
                "bbox_center_x_ratio": "",
                "bbox_center_y_ratio": "",
                "mean_probability": "",
                "max_probability": "",
                "std_probability": "",
                "p90_probability": "",
                "p95_probability": "",
                "p99_probability": "",
                "triage_label": "error",
                "triage_reasons": str(e),
            })
            print(f"[{idx}/{len(selected_masks)}] FAIL - {mask_path.name} -> {e}")

    fieldnames = [
        "image_stem",
        "mask_name",
        "prob_map_name",
        "height",
        "width",
        "white_pixels",
        "white_ratio",
        "num_components",
        "largest_area",
        "second_largest_area",
        "total_white_pixels",
        "border_touching_component_count",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "bbox_area_ratio",
        "bbox_center_x_ratio",
        "bbox_center_y_ratio",
        "mean_probability",
        "max_probability",
        "std_probability",
        "p90_probability",
        "p95_probability",
        "p99_probability",
        "triage_label",
        "triage_reasons",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nDone.")
    print(f"Processed: {len(selected_masks)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Triage CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()