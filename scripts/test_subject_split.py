import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.subject_split import create_subject_splits
from src.data.dataset_loader import HandOADataset
from src.data.preprocessing import get_transforms

splits = create_subject_splits("data/raw/Hand.csv")

train_dataset = HandOADataset(
    image_dir="data/raw/images",
    metadata_path="data/raw/Hand.csv",
    transform=get_transforms(),
    allowed_subjects=splits["train"]
)

val_dataset = HandOADataset(
    image_dir="data/raw/images",
    metadata_path="data/raw/Hand.csv",
    transform=get_transforms(),
    allowed_subjects=splits["val"]
)

test_dataset = HandOADataset(
    image_dir="data/raw/images",
    metadata_path="data/raw/Hand.csv",
    transform=get_transforms(),
    allowed_subjects=splits["test"]
)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))



#CHECKING FOR DATA LEAKAGE

train_ids = splits["train"]
val_ids = splits["val"]
test_ids = splits["test"]

print("Train-Val overlap:", len(train_ids & val_ids))
print("Train-Test overlap:", len(train_ids & test_ids))
print("Val-Test overlap:", len(val_ids & test_ids))