import os
import sys
import csv
import json
import random

import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.segmentation_dataset import FingerSegmentationDataset
from src.models.unet import UNet
from src.training.losses import CrossEntropyDiceLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multiclass_dice_score(logits, targets, num_classes=3, include_background=False, smooth=1e-6):
    preds = torch.argmax(logits, dim=1)
    dices = []

    class_range = range(num_classes) if include_background else range(1, num_classes)

    for c in class_range:
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dices.append(dice.mean().item())

    return sum(dices) / len(dices)


def multiclass_iou_score(logits, targets, num_classes=3, include_background=False, smooth=1e-6):
    preds = torch.argmax(logits, dim=1)
    ious = []

    class_range = range(num_classes) if include_background else range(1, num_classes)

    for c in class_range:
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum(dim=(1, 2))
        union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection

        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou.mean().item())

    return sum(ious) / len(ious)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            total_dice += multiclass_dice_score(logits, masks)
            total_iou += multiclass_iou_score(logits, masks)
            count += 1

    if count == 0:
        return 0.0, 0.0, 0.0

    return total_loss / count, total_dice / count, total_iou / count


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    if count == 0:
        return 0.0

    return total_loss / count


def save_history_csv(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_dice", "val_iou"]
        )
        writer.writeheader()
        writer.writerows(history)


def plot_training_history(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_dice = [row["val_dice"] for row in history]
    val_iou = [row["val_iou"] for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.plot(epochs, val_dice, label="Validation Dice", linewidth=2)
    plt.plot(epochs, val_iou, label="Validation IoU", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("U-Net Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_split_manifest(dataset, train_dataset, val_dataset, test_dataset, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    manifest = {
        "train": [dataset.samples[i][0] for i in train_dataset.indices],
        "val": [dataset.samples[i][0] for i in val_dataset.indices],
        "test": [dataset.samples[i][0] for i in test_dataset.indices],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved split manifest to: {save_path}")


def main():
    seed = 42

    joints = [
        "pip2", "dip2",
        "pip3", "dip3",
        "pip4", "dip4",
        "pip5", "dip5",
        "mcp2", "mcp3", "mcp4", "mcp5",
    ]

    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 100

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    image_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "masks")

    experiment_name = "unet_multiclass_all_joints"
    checkpoint_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    logs_dir = os.path.join(PROJECT_ROOT, "outputs", "training_logs")
    split_dir = os.path.join(PROJECT_ROOT, "outputs", "data_splits")

    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_{experiment_name}.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, f"last_{experiment_name}.pth")
    history_csv_path = os.path.join(logs_dir, f"{experiment_name}_history.csv")
    plot_path = os.path.join(logs_dir, f"{experiment_name}_history_plot.png")
    split_manifest_path = os.path.join(split_dir, f"{experiment_name}_split.json")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    print("Using device:", device)
    print("Experiment:", experiment_name)
    print("Joints used:", joints)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    dataset = FingerSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        joints=joints,
    )

    print(f"Total samples for joints {joints}: {len(dataset)}")
    print("First 10 pairs:")
    for sample in dataset.samples[:10]:
        print(sample)

    if len(dataset) < 3:
        raise ValueError(f"Need at least 3 samples, found {len(dataset)}")

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    if test_size <= 0:
        raise ValueError("Test split size became zero or negative. Adjust split ratios.")

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    save_split_manifest(dataset, train_dataset, val_dataset, test_dataset, split_manifest_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    model = UNet(in_channels=1, out_channels=3).to(device)

    criterion = CrossEntropyDiceLoss(
        num_classes=3,
        ce_weight=0.5,
        dice_weight=0.5,
        include_background_in_dice=False,
        class_weights=[1.0, 2.0, 2.0],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_dice = 0.0
    history = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_dice": round(val_dice, 6),
            "val_iou": round(val_iou, 6),
        })

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), last_checkpoint_path)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Saved best model to: {best_checkpoint_path}")

    save_history_csv(history, history_csv_path)
    plot_training_history(history, plot_path)

    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Saved training history to: {history_csv_path}")
    print(f"Saved training plot to: {plot_path}")
    print(f"Saved last checkpoint to: {last_checkpoint_path}")
    print("Training complete. Test set has been held out for final evaluation.")


if __name__ == "__main__":
    main()