import os
import csv
import json
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#This will be evaluating the Dice and  IoU values based on the KL types.

MODEL_CONFIGS = {
    "unet": {
        "inference_dir": "outputs/segmentation_inference_unet_multiclass_all_joints_test",
        "split_manifest": "outputs/data_splits/unet_multiclass_all_joints_split.json",
    },
    "attention_unet": {
        "inference_dir": "outputs/segmentation_inference_attention_unet_multiclass_all_joints_test",
        "split_manifest": "outputs/data_splits/attention_unet_multiclass_all_joints_split.json",
    },
    "deeplabv3": {
        "inference_dir": "outputs/segmentation_inference_deeplabv3_multiclass_all_joints_test",
        "split_manifest": "outputs/data_splits/deeplabv3_multiclass_all_joints_split.json",
    },
}

OUTPUT_DIR = "outputs/segmentation_eval_by_kl"

BACKGROUND = 0
UPPER_BONE = 1
LOWER_BONE = 2


def load_saved_mask_as_class_ids(path):
    """
    Reads saved grayscale mask PNG and converts it back to class IDs.
    Expected values after save_image(class_mask_to_grayscale):
      0   -> background
      ~127 -> class 1
      255 -> class 2
    """
    mask = np.array(Image.open(path).convert("L"))

    class_mask = np.zeros(mask.shape, dtype=np.uint8)
    class_mask[(mask >= 64) & (mask < 192)] = UPPER_BONE
    class_mask[mask >= 192] = LOWER_BONE

    return class_mask


def multiclass_dice(gt_mask, pred_mask, num_classes=3, include_background=False, smooth=1e-6):
    dices = []
    class_range = range(num_classes) if include_background else range(1, num_classes)

    for c in class_range:
        gt_c = (gt_mask == c).astype(np.float32)
        pred_c = (pred_mask == c).astype(np.float32)

        intersection = np.sum(gt_c * pred_c)
        union = np.sum(gt_c) + np.sum(pred_c)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dices.append(dice)

    return float(np.mean(dices))


def multiclass_iou(gt_mask, pred_mask, num_classes=3, include_background=False, smooth=1e-6):
    ious = []
    class_range = range(num_classes) if include_background else range(1, num_classes)

    for c in class_range:
        gt_c = (gt_mask == c).astype(np.float32)
        pred_c = (pred_mask == c).astype(np.float32)

        intersection = np.sum(gt_c * pred_c)
        union = np.sum(gt_c) + np.sum(pred_c) - intersection

        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return float(np.mean(ious))


def load_test_kl_map(split_manifest_path):
    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_entries = manifest["test"]

    kl_map = {}
    for entry in test_entries:
        image_name = entry["image_name"]
        base_name = os.path.splitext(image_name)[0]
        kl_map[base_name] = int(entry["kl"])

    return kl_map


def evaluate_model(model_name, inference_dir, split_manifest_path):
    if not os.path.exists(inference_dir):
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    kl_map = load_test_kl_map(split_manifest_path)

    gt_files = sorted([
        f for f in os.listdir(inference_dir)
        if f.endswith("_gt_mask.png")
    ])

    per_image_rows = []

    for gt_file in gt_files:
        base_name = gt_file.replace("_gt_mask.png", "")
        pred_file = f"{base_name}_pred_mask.png"

        gt_path = os.path.join(inference_dir, gt_file)
        pred_path = os.path.join(inference_dir, pred_file)

        if not os.path.exists(pred_path):
            print(f"[{model_name}] Missing prediction for: {base_name}")
            continue

        if base_name not in kl_map:
            print(f"[{model_name}] KL label missing for: {base_name}")
            continue

        gt_mask = load_saved_mask_as_class_ids(gt_path)
        pred_mask = load_saved_mask_as_class_ids(pred_path)

        dice = multiclass_dice(gt_mask, pred_mask, num_classes=3, include_background=False)
        iou = multiclass_iou(gt_mask, pred_mask, num_classes=3, include_background=False)
        kl = kl_map[base_name]

        per_image_rows.append({
            "model": model_name,
            "sample": base_name,
            "kl": kl,
            "dice": round(dice, 6),
            "iou": round(iou, 6),
        })

    return per_image_rows


def summarize_by_kl(model_name, per_image_rows):
    grouped = defaultdict(list)
    for row in per_image_rows:
        grouped[row["kl"]].append(row)

    summary_rows = []
    for kl in sorted(grouped.keys()):
        rows = grouped[kl]
        dice_vals = [r["dice"] for r in rows]
        iou_vals = [r["iou"] for r in rows]

        summary_rows.append({
            "model": model_name,
            "kl": kl,
            "num_samples": len(rows),
            "mean_dice": round(float(np.mean(dice_vals)), 6),
            "mean_iou": round(float(np.mean(iou_vals)), 6),
            "min_dice": round(float(np.min(dice_vals)), 6),
            "max_dice": round(float(np.max(dice_vals)), 6),
            "min_iou": round(float(np.min(iou_vals)), 6),
            "max_iou": round(float(np.max(iou_vals)), 6),
        })

    return summary_rows


def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not rows:
        print(f"No rows to save for: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_metric(summary_rows, metric_key, ylabel, title, save_path):
    if not summary_rows:
        print(f"No summary rows available for plot: {save_path}")
        return

    models = ["unet", "attention_unet", "deeplabv3"]
    all_kls = sorted(set(row["kl"] for row in summary_rows))

    metric_map = {
        row["model"]: {row["kl"]: row[metric_key] for row in summary_rows if row["model"] == row["model"]}
        for row in summary_rows
    }

    x = np.arange(len(all_kls))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for idx, model in enumerate(models):
        model_rows = [row for row in summary_rows if row["model"] == model]
        model_map = {row["kl"]: row[metric_key] for row in model_rows}
        values = [model_map.get(kl, 0.0) for kl in all_kls]
        plt.bar(x + (idx - 1) * width, values, width=width, label=model)

    plt.xticks(x, [f"KL{kl}" for kl in all_kls])
    plt.xlabel("KL Type")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sample_counts(summary_rows, save_path):
    if not summary_rows:
        print(f"No summary rows available for plot: {save_path}")
        return

    models = ["unet", "attention_unet", "deeplabv3"]
    all_kls = sorted(set(row["kl"] for row in summary_rows))

    x = np.arange(len(all_kls))
    width = 0.25

    plt.figure(figsize=(10, 6))

    for idx, model in enumerate(models):
        model_rows = [row for row in summary_rows if row["model"] == model]
        model_map = {row["kl"]: row["num_samples"] for row in model_rows}
        values = [model_map.get(kl, 0) for kl in all_kls]
        plt.bar(x + (idx - 1) * width, values, width=width, label=model)

    plt.xticks(x, [f"KL{kl}" for kl in all_kls])
    plt.xlabel("KL Type")
    plt.ylabel("Number of Test Samples")
    plt.title("Test Sample Count by KL Type")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    combined_summary_rows = []

    for model_name, cfg in MODEL_CONFIGS.items():
        inference_dir = cfg["inference_dir"]
        split_manifest_path = cfg["split_manifest"]

        if not os.path.exists(inference_dir):
            print(f"Skipping {model_name}: inference folder not found -> {inference_dir}")
            continue

        if not os.path.exists(split_manifest_path):
            print(f"Skipping {model_name}: split manifest not found -> {split_manifest_path}")
            continue

        per_image_rows = evaluate_model(
            model_name=model_name,
            inference_dir=inference_dir,
            split_manifest_path=split_manifest_path,
        )

        summary_rows = summarize_by_kl(model_name, per_image_rows)

        per_image_csv = os.path.join(OUTPUT_DIR, f"{model_name}_per_image_metrics_by_kl.csv")
        summary_csv = os.path.join(OUTPUT_DIR, f"{model_name}_summary_by_kl.csv")

        save_csv(per_image_csv, per_image_rows)
        save_csv(summary_csv, summary_rows)

        combined_summary_rows.extend(summary_rows)

        print(f"\n{model_name}")
        print(f"Per-image results saved to: {per_image_csv}")
        print(f"KL-wise summary saved to: {summary_csv}")

        if summary_rows:
            for row in summary_rows:
                print(
                    f"  KL{row['kl']} | n={row['num_samples']} | "
                    f"Dice={row['mean_dice']:.4f} | IoU={row['mean_iou']:.4f}"
                )

    combined_summary_csv = os.path.join(OUTPUT_DIR, "all_models_summary_by_kl.csv")
    save_csv(combined_summary_csv, combined_summary_rows)

    dice_plot_path = os.path.join(OUTPUT_DIR, "mean_dice_by_kl.png")
    iou_plot_path = os.path.join(OUTPUT_DIR, "mean_iou_by_kl.png")
    count_plot_path = os.path.join(OUTPUT_DIR, "sample_count_by_kl.png")

    plot_grouped_metric(
        summary_rows=combined_summary_rows,
        metric_key="mean_dice",
        ylabel="Mean Dice",
        title="Mean Dice by KL Type",
        save_path=dice_plot_path,
    )

    plot_grouped_metric(
        summary_rows=combined_summary_rows,
        metric_key="mean_iou",
        ylabel="Mean IoU",
        title="Mean IoU by KL Type",
        save_path=iou_plot_path,
    )

    plot_sample_counts(
        summary_rows=combined_summary_rows,
        save_path=count_plot_path,
    )

    print(f"\nCombined summary saved to: {combined_summary_csv}")
    print(f"Dice plot saved to: {dice_plot_path}")
    print(f"IoU plot saved to: {iou_plot_path}")
    print(f"Sample count plot saved to: {count_plot_path}")


if __name__ == "__main__":
    main()