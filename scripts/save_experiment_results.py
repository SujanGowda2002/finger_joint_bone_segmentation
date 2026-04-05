import os
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "experiment_results.csv")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    rows = [
        {
            "experiment_name": "unet_pip2_dice_only",
            "joints_used": "pip2",
            "num_samples": 9,
            "loss_function": "DiceLoss",
            "input_size": "176x176",
            "batch_size": 4,
            "epochs": 30,
            "best_val_dice": 0.9488,
            "best_val_iou": 0.9027,
            "eval_dice": 0.9488,
            "eval_iou": 0.9027,
            "notes": "Initial seed-set proof of concept on pip2 only"
        },
        {
            "experiment_name": "unet_pip2_dice_bce",
            "joints_used": "pip2",
            "num_samples": 9,
            "loss_function": "DiceBCELoss",
            "input_size": "176x176",
            "batch_size": 4,
            "epochs": 30,
            "best_val_dice": 0.9559,
            "best_val_iou": 0.9156,
            "eval_dice": 0.9559,
            "eval_iou": 0.9156,
            "notes": "Slight improvement over Dice-only, but training was unstable due to tiny dataset"
        },
        {
            "experiment_name": "unet_pip2_dip2_dice_bce",
            "joints_used": "pip2+dip2",
            "num_samples": 14,
            "loss_function": "DiceBCELoss",
            "input_size": "176x176",
            "batch_size": 4,
            "epochs": 30,
            "best_val_dice": 0.9292,
            "best_val_iou": 0.8677,
            "eval_dice": 0.9382,
            "eval_iou": 0.8842,
            "notes": "Mixed-joint experiment with slightly lower but still strong performance"
        },
    ]

    fieldnames = [
        "experiment_name",
        "joints_used",
        "num_samples",
        "loss_function",
        "input_size",
        "batch_size",
        "epochs",
        "best_val_dice",
        "best_val_iou",
        "eval_dice",
        "eval_iou",
        "notes",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved experiment results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()