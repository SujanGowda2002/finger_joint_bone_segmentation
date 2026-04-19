import os
import csv
import numpy as np
from PIL import Image


UPPER_BONE = 1
LOWER_BONE = 2


MODEL_DIRS = {
    "unet": "outputs/segmentation_inference_unet_multiclass_all_joints",
    "attention_unet": "outputs/segmentation_inference_attention_unet_multiclass_all_joints",
    "deeplabv3": "outputs/segmentation_inference_deeplabv3_multiclass_all_joints",
}

OUTPUT_DIR = "outputs/jsw_results_expanded"


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


def compute_jsw(mask, min_gap=1, max_gap=80):
    """
    Computes vertical Joint Space Width in pixels.

    For each column:
    - lowest pixel of upper bone
    - highest pixel of lower bone
    - gap between them
    """
    upper = mask == UPPER_BONE
    lower = mask == LOWER_BONE

    jsw_values = []

    for x in range(mask.shape[1]):
        upper_y = np.where(upper[:, x])[0]
        lower_y = np.where(lower[:, x])[0]

        if len(upper_y) == 0 or len(lower_y) == 0:
            continue

        upper_bottom = upper_y.max()
        lower_top = lower_y.min()

        gap = lower_top - upper_bottom - 1

        if min_gap <= gap <= max_gap:
            jsw_values.append(gap)

    if len(jsw_values) == 0:
        return None

    jsw_values = np.array(jsw_values)

    return {
        "mean": float(np.mean(jsw_values)),
        "median": float(np.median(jsw_values)),
        "min": float(np.min(jsw_values)),
        "max": float(np.max(jsw_values)),
        "std": float(np.std(jsw_values)),
        "valid_columns": int(len(jsw_values)),
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

        gt_jsw = compute_jsw(gt_mask)
        pred_jsw = compute_jsw(pred_mask)

        if gt_jsw is None or pred_jsw is None:
            print(f"[{model_name}] Could not compute JSW for: {base_name}")
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
            "gt_valid_columns": gt_jsw["valid_columns"],
            "pred_valid_columns": pred_jsw["valid_columns"],
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

        output_csv = os.path.join(OUTPUT_DIR, f"{model_name}_jsw.csv")
        save_csv(output_csv, rows)

        if rows:
            mean_abs_error = np.mean([r["abs_error_px"] for r in rows])
            mean_percent_error = np.mean([r["percent_error"] for r in rows])

            summary_rows.append({
                "model": model_name,
                "samples_evaluated": len(rows),
                "mean_abs_error_px": mean_abs_error,
                "mean_percent_error": mean_percent_error,
            })

            print(f"\n{model_name}")
            print(f"Saved: {output_csv}")
            print(f"Samples evaluated: {len(rows)}")
            print(f"Mean absolute JSW error: {mean_abs_error:.4f} px")
            print(f"Mean percentage JSW error: {mean_percent_error:.2f}%")

    summary_csv = os.path.join(OUTPUT_DIR, "jsw_model_summary.csv")
    save_csv(summary_csv, summary_rows)

    print(f"\nCombined summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()