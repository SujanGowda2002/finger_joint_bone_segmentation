import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# =========================================================
# CONFIG
# =========================================================
IMAGE_DIR = "Batch-1 Dataset/Batch-1 Images"
MASK_DIR  = "Batch-1 Dataset/Batch-1 Fiji Masks"

OUTPUT_DIR = "batch1_training_outputs"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

BATCH_SIZE = 4
EPOCHS = 100
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

class SimpleSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(valid_exts)
        ])
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(valid_exts)
        ])

        # stem -> filename
        image_map = {os.path.splitext(f)[0].lower(): f for f in image_files}
        mask_map = {os.path.splitext(f)[0].lower(): f for f in mask_files}

        common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))
        image_only = sorted(set(image_map.keys()) - set(mask_map.keys()))
        mask_only = sorted(set(mask_map.keys()) - set(image_map.keys()))

        self.pairs = [
            {
                "image_path": os.path.join(image_dir, image_map[k]),
                "mask_path": os.path.join(mask_dir, mask_map[k]),
                "filename": image_map[k],
                "key": k,
            }
            for k in common_keys
        ]

        print(f"[INFO] Found {len(image_files)} images")
        print(f"[INFO] Found {len(mask_files)} masks")
        print(f"[INFO] Matched {len(self.pairs)} image-mask pairs")

        if image_only:
            print(f"[WARN] {len(image_only)} images have no matching mask")
            print("[WARN] Example unmatched images:", image_only[:10])

        if mask_only:
            print(f"[WARN] {len(mask_only)} masks have no matching image")
            print("[WARN] Example unmatched masks:", mask_only[:10])

        if len(self.pairs) == 0:
            raise ValueError("No matched image-mask pairs found. Check naming in both folders.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample = self.pairs[idx]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        filename = sample["filename"]

        image = Image.open(image_path).convert("L")
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)

        # binary mask
        mask = (mask > 127).astype(np.float32)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "filename": filename,
            "image_path": image_path,
            "mask_path": mask_path,
            "index": idx,
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
# LOSS + METRICS
# =========================================================
def dice_coefficient(pred_probs, target, eps=1e-6):
    pred_bin = (pred_probs > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    dice = torch.where(union == 0, torch.zeros_like(dice), dice)
    return dice.mean().item()


def iou_score(pred_probs, target, eps=1e-6):
    pred_bin = (pred_probs > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    iou = torch.where(union == 0, torch.zeros_like(iou), iou)
    return iou.mean().item()


def dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


# =========================================================
# VIS HELPERS
# =========================================================
def save_gray(arr, save_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(img, mask, save_path, title=None):
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    plt.imshow(mask, cmap="Reds", alpha=0.35)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_prediction_examples(model, loader, device, save_dir, max_examples=8):
    ensure_dir(save_dir)
    model.eval()

    saved = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            filenames = batch["filename"]
            indices = batch["index"]

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > MASK_THRESHOLD).float()

            for i in range(images.size(0)):
                img = images[i, 0].cpu().numpy()
                gt = masks[i, 0].cpu().numpy()
                pred = preds[i, 0].cpu().numpy()

                stem = os.path.splitext(str(filenames[i]))[0]
                base = f"sample_{int(indices[i]):02d}_{stem}"

                save_gray(img, os.path.join(save_dir, f"{base}_image.png"))
                save_gray(gt, os.path.join(save_dir, f"{base}_gt_mask.png"))
                save_gray(pred, os.path.join(save_dir, f"{base}_pred_mask.png"))
                save_overlay(img, gt, os.path.join(save_dir, f"{base}_gt_overlay.png"), title="GT Overlay")
                save_overlay(img, pred, os.path.join(save_dir, f"{base}_pred_overlay.png"), title="Pred Overlay")

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

    from torch.utils.data import random_split

    dataset = SimpleSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        image_size=IMAGE_SIZE
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNetSmall().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce_loss = nn.BCEWithLogitsLoss()

    history = {
        "epoch": [],
        "loss": [],
        "dice": [],
        "iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Start training...")

    for epoch in range(1, EPOCHS + 1):

        # ======================
        # TRAIN
        # ======================
        model.train()
        epoch_losses = []
        epoch_dices = []
        epoch_ious = []

        for batch in train_loader:
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
                iou = iou_score(probs, masks)

            epoch_losses.append(loss.item())
            epoch_dices.append(dice)
            epoch_ious.append(iou)

        avg_loss = float(np.mean(epoch_losses))
        avg_dice = float(np.mean(epoch_dices))
        avg_iou = float(np.mean(epoch_ious))

        # ======================
        # VALIDATION
        # ======================
        model.eval()

        val_losses = []
        val_dices = []
        val_ious = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                logits = model(images)

                loss_bce = bce_loss(logits, masks)
                loss_dice = dice_loss_from_logits(logits, masks)
                val_loss = loss_bce + loss_dice

                probs = torch.sigmoid(logits)

                val_dice = dice_coefficient(probs, masks)
                val_iou = iou_score(probs, masks)

                val_losses.append(val_loss.item())
                val_dices.append(val_dice)
                val_ious.append(val_iou)

        avg_val_loss = float(np.mean(val_losses))
        avg_val_dice = float(np.mean(val_dices))
        avg_val_iou = float(np.mean(val_ious))

        # ======================
        # SAVE HISTORY
        # ======================
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["dice"].append(avg_dice)
        history["iou"].append(avg_iou)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)
        history["val_iou"].append(avg_val_iou)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={avg_loss:.4f} | "
            f"train_dice={avg_dice:.4f} | "
            f"train_iou={avg_iou:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_dice={avg_val_dice:.4f} | "
            f"val_iou={avg_val_iou:.4f}"
        )

    ckpt_path = os.path.join(CHECKPOINT_DIR, "unet_batch1.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint: {ckpt_path}")

    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(OUTPUT_DIR, "training_history.csv")
    hist_df.to_csv(hist_csv, index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Batch-1 Sanity-Check Training Loss")
    plt.grid(True)
    plt.tight_layout()
    loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["dice"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Dice")
    plt.title("Batch-1 Sanity-Check Training Dice")
    plt.grid(True)
    plt.tight_layout()
    dice_curve_path = os.path.join(OUTPUT_DIR, "dice_curve.png")
    plt.savefig(dice_curve_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["iou"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training IoU")
    plt.title("Batch-1 Sanity-Check Training IoU")
    plt.grid(True)
    plt.tight_layout()
    iou_curve_path = os.path.join(OUTPUT_DIR, "iou_curve.png")
    plt.savefig(iou_curve_path, dpi=150)
    plt.close()

    save_prediction_examples(model, val_loader, DEVICE, PRED_DIR, max_examples=8)

    summary = {
        "dataset_name": "Batch-1 Dataset",
        "dataset_size": len(dataset),
        "epochs": EPOCHS,
        "final_loss": history["loss"][-1],
        "final_dice": history["dice"][-1],
        "final_iou": history["iou"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_dice": history["val_dice"][-1],
        "final_val_iou": history["val_iou"][-1],
        "device": DEVICE,
        "notes": "Sanity-check run on Batch-1 Fiji pseudo-mask subset."
    }

    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()