import os
import sys
import math
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_loader import HandOADataset
from src.data.subject_split import create_subject_splits
from src.data.augmentation import get_train_transforms, get_eval_transforms
from src.models.kl_classifier import KLClassifierCNN


def filter_valid_indices(dataset):
    valid_indices = []

    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample["label"].item()

        if not math.isnan(label):
            valid_indices.append(i)

    return valid_indices


def compute_class_weights_from_subset(subset):
    label_counter = Counter()

    for i in range(len(subset)):
        label = subset[i]["label"].item()
        if not math.isnan(label):
            label_counter[int(label)] += 1

    num_classes = 5
    total_samples = sum(label_counter.values())

    class_weights = []
    for cls in range(num_classes):
        cls_count = label_counter.get(cls, 1)
        weight = total_samples / (num_classes * cls_count)
        class_weights.append(weight)

    return torch.tensor(class_weights, dtype=torch.float32), label_counter


def create_weighted_sampler(subset):
    labels = []

    for i in range(len(subset)):
        label = subset[i]["label"].item()
        labels.append(int(label))

    label_counts = Counter(labels)

    sample_weights = []
    for label in labels:
        sample_weights.append(1.0 / label_counts[label])

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def create_dataloaders(image_dir, metadata_path, batch_size=32):
    splits = create_subject_splits(metadata_path)

    train_dataset = HandOADataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=get_train_transforms(),
        allowed_subjects=splits["train"]
    )

    val_dataset = HandOADataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=get_eval_transforms(),
        allowed_subjects=splits["val"]
    )

    test_dataset = HandOADataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=get_eval_transforms(),
        allowed_subjects=splits["test"]
    )

    train_indices = filter_valid_indices(train_dataset)
    val_indices = filter_valid_indices(val_dataset)
    test_indices = filter_valid_indices(test_dataset)

    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)

    train_sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, train_dataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_labels, all_preds


def save_checkpoint(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def main():
    image_dir = os.path.join(PROJECT_ROOT, "data", "raw", "images")
    metadata_path = os.path.join(PROJECT_ROOT, "data", "raw", "Hand.csv")
    checkpoint_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_kl_classifier_sampler_aug.pth")

    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3
    num_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, train_dataset_for_weights = create_dataloaders(
        image_dir=image_dir,
        metadata_path=metadata_path,
        batch_size=batch_size
    )

    class_weights, label_counter = compute_class_weights_from_subset(train_dataset_for_weights)
    class_weights = class_weights.to(device)

    print("\nTraining label distribution:")
    for k, v in sorted(label_counter.items()):
        print(f"KL={k}: {v}")

    print("\nClass weights:")
    print(class_weights)

    model = KLClassifierCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    print("\nStarting training with augmentation + WeightedRandomSampler...\n")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, checkpoint_path)
            print(f"Saved best model to: {checkpoint_path}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()