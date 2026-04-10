import os
from typing import List, Tuple, Union

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


class FingerSegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, joints: Union[str, List[str]] = "pip2"):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        if isinstance(joints, str):
            self.joints = [joints.lower()]
        else:
            self.joints = [j.lower() for j in joints]

        self.image_transform = transforms.Compose([
            transforms.Resize((176, 176), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((176, 176), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.samples = self._collect_pairs()

        if len(self.samples) == 0:
            raise ValueError(f"No matching image-mask pairs found for joints={self.joints}")

    def _is_valid_image_file(self, filename: str) -> bool:
        return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))

    def _normalize_image_key(self, filename: str) -> str:
        stem = os.path.splitext(filename)[0].lower()
        if stem.endswith("_image"):
            stem = stem[:-6]
        return stem

    def _normalize_mask_key(self, filename: str) -> str:
        stem = os.path.splitext(filename)[0].lower()
        if stem.endswith("_mask"):
            stem = stem[:-5]
        return stem

    def _contains_target_joint(self, key: str) -> bool:
        return any(f"_{joint}" in key for joint in self.joints)

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        image_files = sorted(
            [f for f in os.listdir(self.image_dir) if self._is_valid_image_file(f)]
        )
        mask_files = sorted(
            [f for f in os.listdir(self.mask_dir) if self._is_valid_image_file(f)]
        )

        mask_lookup = {}
        for mask_name in mask_files:
            key = self._normalize_mask_key(mask_name)
            mask_lookup[key] = mask_name

        pairs = []

        for image_name in image_files:
            image_key = self._normalize_image_key(image_name)

            if not self._contains_target_joint(image_key):
                continue

            if image_key in mask_lookup:
                pairs.append((image_name, mask_lookup[image_key]))

        return pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "image_name": image_name,
            "mask_name": mask_name
        }