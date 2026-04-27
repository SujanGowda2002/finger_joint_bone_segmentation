import os
import sys
import math
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset_loader import HandOADataset
from src.data.preprocessing import get_transforms
from src.data.subject_split import create_subject_splits
from src.models.kl_classifier import KLClassifierCNN


def filter_valid_indices(dataset):
    valid_indices = []

    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample["label"].item()

        if not math.isnan(label):
            valid_indices.append(i)

    return valid_indices


def create_test_loader(image_dir, metadata_path, batch_size=32):
    splits = create_subject_splits(metadata_path)
    transform = get_transforms()

    test_dataset = HandOADataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=transform,
        allowed_subjects=splits["test"]
    )

    test_indices = filter_valid_indices(test_dataset)
    test_dataset = Subset(test_dataset, test_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return test_loader


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / total
    return avg_loss, all_labels, all_preds


def main():
    image_dir = os.path.join(PROJECT_ROOT, "data", "raw", "images")
    metadata_path = os.path.join(PROJECT_ROOT, "data", "raw", "Hand.csv")
    checkpoint_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_kl_classifier_macro_f1.pth")
    batch_size = 32
    num_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader = create_test_loader(
        image_dir=image_dir,
        metadata_path=metadata_path,
        batch_size=batch_size
    )

    model = KLClassifierCNN(num_classes=num_classes).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from: {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()

    test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    within_1_acc = sum(abs(t - p) <= 1 for t, p in zip(y_true, y_pred)) / len(y_true)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== TEST RESULTS =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Within-1 Accuracy: {within_1_acc:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=["KL0", "KL1", "KL2", "KL3", "KL4"],
            digits=4,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()