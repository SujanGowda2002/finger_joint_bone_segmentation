import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# =========================================================
# CONFIG
# =========================================================
PACKAGE_DIR = "SeedSet27_Training_Only_Package"
MANIFEST_PATH = os.path.join(PACKAGE_DIR, "manifest.csv")

OUTPUT_DIR = "seedset27_training_outputs"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3
IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_THRESHOLD = 0.5


# =========================================================
# UTILS
# =========================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_path(base_dir, p):
    p = str(p).strip()
    candidate = Path(p)

    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    candidate = Path(base_dir) / p
    if candidate.exists():
        return str(candidate)

    name_only = Path(p).name
    matches = list(Path(base_dir).rglob(name_only))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(f"Could not resolve path: {p}")


# =========================================================
# DATASET
# =========================================================
class SeedSegmentationDataset(Dataset):
    def __init__(self, package_dir, manifest_path, image_size=128):
        self.package_dir = package_dir
        self.df = pd.read_csv(manifest_path)

        # auto-detect columns
        self.image_col = None
        self.mask_col = None
        self.joint_col = None

        for c in self.df.columns:
            c_low = c.lower()
            if self.image_col is None and "image" in c_low:
                self.image_col = c
            if self.mask_col is None and "mask" in c_low:
                self.mask_col = c
            if self.joint_col is None and "joint" in c_low:
                self.joint_col = c

        if self.image_col is None or self.mask_col is None:
            raise ValueError(f"Could not detect image/mask columns: {list(self.df.columns)}")

        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = resolve_path(self.package_dir, row[self.image_col])
        mask_path = resolve_path(self.package_dir, row[self.mask_col])
        joint = str(row[self.joint_col]) if self.joint_col is not None else "unknown"

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)  # [1,H,W], float [0,1]
        mask = self.mask_transform(mask)      # [1,H,W], float [0,1]
        mask = (mask > 0.1).float()

        return {
            "image": image,
            "mask": mask,
            "joint": joint,
            "image_path": image_path,
            "mask_path": mask_path,
        }


# =========================================================
# MODEL
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(32, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(32, 16)

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))

        x = self.up2(xb)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        return self.out_conv(x)


# =========================================================
# LOSS + METRIC
# =========================================================
def dice_coefficient(pred_probs, target, eps=1e-6):
    pred_bin = (pred_probs > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


# =========================================================
# SAVE PREDICTIONS
# =========================================================
def save_prediction_examples(model, loader, device, save_dir, max_examples=6):
    ensure_dir(save_dir)
    model.eval()

    saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            joints = batch["joint"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > MASK_THRESHOLD).float()

            for i in range(images.size(0)):
                img = images[i, 0].cpu().numpy()
                gt = masks[i, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(img, cmap="gray")
                plt.title("Input ROI")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(gt, cmap="gray")
                plt.title("Pseudo Mask")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(img, cmap="gray")
                plt.imshow(pred, cmap="Reds", alpha=0.35)
                plt.title(f"Prediction | {joints[i]}")
                plt.axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"pred_{saved:02d}.png"), dpi=150, bbox_inches="tight")
                plt.close()

                saved += 1
                if saved >= max_examples:
                    return


# =========================================================
# MAIN TRAIN
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PRED_DIR)
    ensure_dir(CHECKPOINT_DIR)

    dataset = SeedSegmentationDataset(
        package_dir=PACKAGE_DIR,
        manifest_path=MANIFEST_PATH,
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNetSmall().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()

    history = {
        "epoch": [],
        "loss": [],
        "dice": []
    }

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Start training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        epoch_dices = []

        for batch in loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss_bce = bce_loss(logits, masks)
            loss_dice = dice_loss_from_logits(logits, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                dice = dice_coefficient(probs, masks)

            epoch_losses.append(loss.item())
            epoch_dices.append(dice)

        avg_loss = float(np.mean(epoch_losses))
        avg_dice = float(np.mean(epoch_dices))

        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["dice"].append(avg_dice)

        print(f"[Epoch {epoch:02d}] loss={avg_loss:.4f} | dice={avg_dice:.4f}")

    # save checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, "unet_seedset27.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint: {ckpt_path}")

    # save history csv
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(OUTPUT_DIR, "training_history.csv")
    hist_df.to_csv(hist_csv, index=False)

    # save loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("SeedSet27 Sanity-Check Training Loss")
    plt.grid(True)
    plt.tight_layout()
    loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    # save dice curve
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["dice"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Dice")
    plt.title("SeedSet27 Sanity-Check Training Dice")
    plt.grid(True)
    plt.tight_layout()
    dice_curve_path = os.path.join(OUTPUT_DIR, "dice_curve.png")
    plt.savefig(dice_curve_path, dpi=150)
    plt.close()

    # save prediction examples
    save_prediction_examples(model, loader, DEVICE, PRED_DIR, max_examples=6)

    summary = {
        "dataset_size": len(dataset),
        "epochs": EPOCHS,
        "final_loss": history["loss"][-1],
        "final_dice": history["dice"][-1],
        "device": DEVICE,
        "notes": "This is a sanity-check run on pseudo-mask data, not a final segmentation evaluation."
    }

    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()