import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader
from src.data.dataset_loader import HandOADataset
from src.data.preprocessing import get_transforms

dataset = HandOADataset(
    image_dir="data/raw/images",
    metadata_path="data/raw/Hand.csv",
    transform=get_transforms()
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

sample = next(iter(loader))

print("Image shape:", sample["image"].shape)
print("Labels:", sample["label"])
print("Subject IDs:", sample["subject_id"])
print("Joints:", sample["joint"])