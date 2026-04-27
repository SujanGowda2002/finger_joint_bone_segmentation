import os
import sys
import csv
import json
import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.segmentation_dataset import FingerSegmentationDataset
from src.models.deeplabv3_model import DeepLabV3Model
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
    plt.title("DeepLabV3 Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def parse_filename_to_id_and_joint(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")

    image_id = parts[0]
    joint = parts[1].lower()
    return image_id, joint


def joint_to_kl_column(joint):
    joint = joint.lower()
    if joint.startswith("dip"):
        return f"v00DIP{joint[-1]}_KL"
    if joint.startswith("pip"):
        return f"v00PIP{joint[-1]}_KL"
    if joint.startswith("mcp"):
        return f"v00MCP{joint[-1]}_KL"
    raise ValueError(f"Unsupported joint token: {joint}")


def build_kl_lookup(hand_csv_path):
    if not os.path.exists(hand_csv_path):
        raise FileNotFoundError(f"Hand.csv not found: {hand_csv_path}")

    df = pd.read_csv(hand_csv_path)

    if "id" not in df.columns:
        raise ValueError("Hand.csv must contain an 'id' column.")

    df["id"] = df["id"].astype(str)

    lookup = {}
    for _, row in df.iterrows():
        lookup[row["id"]] = row

    return lookup


def attach_kl_labels_to_dataset(dataset, hand_csv_path):
    kl_lookup = build_kl_lookup(hand_csv_path)

    sample_infos = []

    for idx, (image_name, mask_name) in enumerate(dataset.samples):
        image_id, joint = parse_filename_to_id_and_joint(image_name)

        if image_id not in kl_lookup:
            raise ValueError(f"Image id {image_id} from {image_name} not found in Hand.csv")

        row = kl_lookup[image_id]
        kl_column = joint_to_kl_column(joint)

        if kl_column not in row.index:
            raise ValueError(f"KL column {kl_column} not found in Hand.csv")

        kl_value = row[kl_column]

        if pd.isna(kl_value):
            raise ValueError(f"Missing KL value for {image_name} using column {kl_column}")

        kl_value = int(kl_value)

        sample_infos.append({
            "dataset_index": idx,
            "image_name": image_name,
            "mask_name": mask_name,
            "image_id": image_id,
            "joint": joint,
            "kl": kl_value,
        })

    return sample_infos


def make_kl_aware_split(sample_infos, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = random.Random(seed)

    by_kl = defaultdict(list)
    for info in sample_infos:
        by_kl[info["kl"]].append(info)

    train_infos = []
    val_infos = []
    test_infos = []

    for _, infos in sorted(by_kl.items()):
        infos = infos.copy()
        rng.shuffle(infos)
        n = len(infos)

        if n == 1:
            test_infos.extend(infos)
            continue

        if n == 2:
            train_infos.append(infos[0])
            test_infos.append(infos[1])
            continue

        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_test = n - n_train - n_val

        if n_train == 0:
            n_train = 1
        if n_val == 0:
            n_val = 1

        n_test = n - n_train - n_val

        if n_test <= 0:
            n_test = 1
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            n_test = n - n_train - n_val

        train_infos.extend(infos[:n_train])
        val_infos.extend(infos[n_train:n_train + n_val])
        test_infos.extend(infos[n_train + n_val:])

    rng.shuffle(train_infos)
    rng.shuffle(val_infos)
    rng.shuffle(test_infos)

    return train_infos, val_infos, test_infos


def save_split_manifest(train_infos, val_infos, test_infos, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    manifest = {
        "train": [
            {
                "image_name": info["image_name"],
                "image_id": info["image_id"],
                "joint": info["joint"],
                "kl": info["kl"],
            }
            for info in train_infos
        ],
        "val": [
            {
                "image_name": info["image_name"],
                "image_id": info["image_id"],
                "joint": info["joint"],
                "kl": info["kl"],
            }
            for info in val_infos
        ],
        "test": [
            {
                "image_name": info["image_name"],
                "image_id": info["image_id"],
                "joint": info["joint"],
                "kl": info["kl"],
            }
            for info in test_infos
        ],
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
    learning_rate = 1e-4
    num_epochs = 100

    image_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "masks")
    hand_csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "Hand.csv")

    experiment_name = "deeplabv3_multiclass_all_joints"
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

    sample_infos = attach_kl_labels_to_dataset(dataset, hand_csv_path)

    train_infos, val_infos, test_infos = make_kl_aware_split(
        sample_infos=sample_infos,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )

    print(f"Train samples: {len(train_infos)}")
    print(f"Validation samples: {len(val_infos)}")
    print(f"Test samples: {len(test_infos)}")

    save_split_manifest(train_infos, val_infos, test_infos, split_manifest_path)

    train_indices = [info["dataset_index"] for info in train_infos]
    val_indices = [info["dataset_index"] for info in val_infos]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

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

    model = DeepLabV3Model(num_classes=3, pretrained_backbone=False).to(device)

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