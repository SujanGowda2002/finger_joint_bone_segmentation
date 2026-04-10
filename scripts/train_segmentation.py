import os
import sys
import csv
import random

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.segmentation_dataset import FingerSegmentationDataset
from src.models.unet import UNet
from src.training.losses import DiceBCELoss
from src.evaluation.metrics import dice_score, iou_score


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
            total_iou += iou_score(outputs, masks)
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
        outputs = model(images)
        loss = criterion(outputs, masks)
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


def main():
    # -----------------------------
    # Configuration
    # -----------------------------
    seed = 42
    joints = ["pip2", "pip3", "pip4", "pip5", "dip2", "dip3", "dip4", "dip5"]
    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 100
    train_ratio = 0.8

    image_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "masks")

    experiment_name = f"unet_{'_'.join(joints)}_dice_bce"
    checkpoint_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    logs_dir = os.path.join(PROJECT_ROOT, "outputs", "training_logs")

    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_{experiment_name}.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, f"last_{experiment_name}.pth")
    history_csv_path = os.path.join(logs_dir, f"{experiment_name}_history.csv")

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

    total_samples = len(dataset)
    print(f"Total samples for joints {joints}: {total_samples}")

    if total_samples < 2:
        raise ValueError(
            f"Need at least 2 image-mask pairs for train/validation split, found {total_samples}."
        )

    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size

    if train_size == 0:
        train_size = 1
        val_size = total_samples - 1

    if val_size == 0:
        val_size = 1
        train_size = total_samples - 1

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
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

    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Saved training history to: {history_csv_path}")
    print(f"Saved last checkpoint to: {last_checkpoint_path}")


if __name__ == "__main__":
    main()