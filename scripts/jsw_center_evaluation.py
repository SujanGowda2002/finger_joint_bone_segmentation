import os
import csv
import numpy as np
from PIL import Image


UPPER_BONE = 1
LOWER_BONE = 2

# Fraction of the valid JSW columns to keep from the center region.
# Example:
# 0.50 -> keep the center 50% of valid columns
# 0.30 -> keep the center 30% of valid columns
#Choose any percentage of the fraction of the center needed.
CENTER_FRACTION = 0.50

MODEL_DIRS = {
    "unet": "outputs/segmentation_inference_unet_multiclass_all_joints",
    "attention_unet": "outputs/segmentation_inference_attention_unet_multiclass_all_joints",
    "deeplabv3": "outputs/segmentation_inference_deeplabv3_multiclass_all_joints",
}

OUTPUT_DIR = "outputs/jsw_results_test_center"


def load_mask(path):
    """
    Converts grayscale saved masks into class IDs.

    Expected mask colors:
    black/dark  = background
    gray        = upper bone
    white       = lower bone
    """
    mask = np.array(Image.open(path).convert("L"))

    class_mask = np.zeros(mask.shape, dtype=np.uint8)
    class_mask[(mask >= 80) & (mask < 200)] = UPPER_BONE
    class_mask[mask >= 200] = LOWER_BONE

    return class_mask


def compute_jsw_center(mask, min_gap=1, max_gap=80, center_fraction=0.50):
    """
    Computes JSW using only the center part of the valid joint-gap columns.

    Steps:
    1. Scan every image column
    2. Keep columns where both upper and lower bone exist
    3. Compute vertical gap for each valid column
    4. Keep only the center fraction of those valid columns
    """
    upper = mask == UPPER_BONE
    lower = mask == LOWER_BONE

    valid_entries = []

    for x in range(mask.shape[1]):
        upper_y = np.where(upper[:, x])[0]
        lower_y = np.where(lower[:, x])[0]

        if len(upper_y) == 0 or len(lower_y) == 0:
            continue

        upper_bottom = upper_y.max()
        lower_top = lower_y.min()
        gap = lower_top - upper_bottom - 1

        if min_gap <= gap <= max_gap:
            valid_entries.append((x, gap))

    if len(valid_entries) == 0:
        return None

    valid_entries = sorted(valid_entries, key=lambda t: t[0])

    n_valid = len(valid_entries)
    n_center = max(1, int(round(n_valid * center_fraction)))

    start = (n_valid - n_center) // 2
    end = start + n_center

    center_entries = valid_entries[start:end]
    center_columns = [x for x, _ in center_entries]
    center_gaps = np.array([gap for _, gap in center_entries], dtype=np.float32)

    return {
        "mean": float(np.mean(center_gaps)),
        "median": float(np.median(center_gaps)),
        "min": float(np.min(center_gaps)),
        "max": float(np.max(center_gaps)),
        "std": float(np.std(center_gaps)),
        "valid_columns_total": int(n_valid),
        "valid_columns_center_used": int(len(center_gaps)),
        "center_fraction_used": float(center_fraction),
        "center_x_min": int(min(center_columns)),
        "center_x_max": int(max(center_columns)),
    }


def evaluate_model(model_name, inference_dir):
    rows = []

    gt_files = sorted([
        f for f in os.listdir(inference_dir)
        if f.endswith("_gt_mask.png")
    ])

    for gt_file in gt_files:
        base_name = gt_file.replace("_gt_mask.png", "")
        pred_file = base_name + "_pred_mask.png"

        gt_path = os.path.join(inference_dir, gt_file)
        pred_path = os.path.join(inference_dir, pred_file)

        if not os.path.exists(pred_path):
            print(f"[{model_name}] Missing prediction: {pred_file}")
            continue

        gt_mask = load_mask(gt_path)
        pred_mask = load_mask(pred_path)

        gt_jsw = compute_jsw_center(
            gt_mask,
            min_gap=1,
            max_gap=80,
            center_fraction=CENTER_FRACTION
        )
        pred_jsw = compute_jsw_center(
            pred_mask,
            min_gap=1,
            max_gap=80,
            center_fraction=CENTER_FRACTION
        )

        if gt_jsw is None or pred_jsw is None:
            print(f"[{model_name}] Could not compute center JSW for: {base_name}")
            continue

        abs_error = abs(gt_jsw["mean"] - pred_jsw["mean"])
        percent_error = (
            abs_error / gt_jsw["mean"] * 100
            if gt_jsw["mean"] != 0 else 0
        )

        rows.append({
            "model": model_name,
            "sample": base_name,
            "gt_mean_jsw_px": gt_jsw["mean"],
            "pred_mean_jsw_px": pred_jsw["mean"],
            "gt_median_jsw_px": gt_jsw["median"],
            "pred_median_jsw_px": pred_jsw["median"],
            "gt_min_jsw_px": gt_jsw["min"],
            "pred_min_jsw_px": pred_jsw["min"],
            "gt_max_jsw_px": gt_jsw["max"],
            "pred_max_jsw_px": pred_jsw["max"],
            "gt_std_jsw_px": gt_jsw["std"],
            "pred_std_jsw_px": pred_jsw["std"],
            "gt_valid_columns_total": gt_jsw["valid_columns_total"],
            "pred_valid_columns_total": pred_jsw["valid_columns_total"],
            "gt_center_columns_used": gt_jsw["valid_columns_center_used"],
            "pred_center_columns_used": pred_jsw["valid_columns_center_used"],
            "gt_center_x_min": gt_jsw["center_x_min"],
            "gt_center_x_max": gt_jsw["center_x_max"],
            "pred_center_x_min": pred_jsw["center_x_min"],
            "pred_center_x_max": pred_jsw["center_x_max"],
            "center_fraction_used": gt_jsw["center_fraction_used"],
            "abs_error_px": abs_error,
            "percent_error": percent_error,
        })

    return rows


def save_csv(path, rows):
    if not rows:
        print(f"No rows to save for: {path}")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_rows = []

    for model_name, inference_dir in MODEL_DIRS.items():
        if not os.path.exists(inference_dir):
            print(f"Folder not found for {model_name}: {inference_dir}")
            continue

        rows = evaluate_model(model_name, inference_dir)

        output_csv = os.path.join(OUTPUT_DIR, f"{model_name}_jsw_center_test.csv")
        save_csv(output_csv, rows)

        if rows:
            mean_abs_error = np.mean([r["abs_error_px"] for r in rows])
            mean_percent_error = np.mean([r["percent_error"] for r in rows])

            summary_rows.append({
                "model": model_name,
                "samples_evaluated": len(rows),
                "center_fraction_used": CENTER_FRACTION,
                "mean_abs_error_px": mean_abs_error,
                "mean_percent_error": mean_percent_error,
            })

            print(f"\n{model_name}")
            print(f"Saved: {output_csv}")
            print(f"Samples evaluated: {len(rows)}")
            print(f"Center fraction used: {CENTER_FRACTION:.2f}")
            print(f"Mean absolute center-JSW error: {mean_abs_error:.4f} px")
            print(f"Mean percentage center-JSW error: {mean_percent_error:.2f}%")

    summary_csv = os.path.join(OUTPUT_DIR, "jsw_model_summary_center_test.csv")
    save_csv(summary_csv, summary_rows)

    print(f"\nCombined summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()