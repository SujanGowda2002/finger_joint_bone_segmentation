import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class HandOADataset(Dataset):
    def __init__(self, image_dir, metadata_path, transform=None, allowed_subjects=None):
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform

        self.subject_lookup = {
            row["id"]: row for _, row in self.metadata.iterrows()
        }

        self.image_files = sorted(os.listdir(image_dir))

        if allowed_subjects is not None:
            filtered_files = []
            for image_name in self.image_files:
                subject_id = int(image_name.split("_")[0])
                if subject_id in allowed_subjects:
                    filtered_files.append(image_name)
            self.image_files = filtered_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        subject_id = int(image_name.split("_")[0])
        joint = image_name.split("_")[1].split(".")[0].upper()

        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        row = self.subject_lookup.get(subject_id)
        if row is None:
            raise ValueError(f"Metadata not found for subject {subject_id}")

        label_column = f"v00{joint}_KL"

        if label_column not in row.index:
            raise KeyError(f"Column {label_column} not found for image {image_name}")

        label = row[label_column]
        label = torch.tensor(label, dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "subject_id": subject_id,
            "joint": joint
        }