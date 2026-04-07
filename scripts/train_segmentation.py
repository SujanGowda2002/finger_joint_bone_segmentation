import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.segmentation_dataset import FingerSegmentationDataset
from src.models.unet import UNet
from src.training.losses import DiceBCELoss
from src.evaluation.metrics import dice_score, iou_score


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

    return total_loss / count, total_dice / count, total_iou / count


def main():
    image_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "masks")
    checkpoint_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_unet_pip2_dip2.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FingerSegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    joints=["pip2", "dip2"]
    )

    print("Total pip2 samples:", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_dice = 0.0
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to: {checkpoint_path}")

    print(f"Best validation Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()