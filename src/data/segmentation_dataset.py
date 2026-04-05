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

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        image_files = sorted(os.listdir(self.image_dir))
        mask_files = set(os.listdir(self.mask_dir))

        pairs = []

        for image_name in image_files:
            if not image_name.endswith("_image.png"):
                continue

            matched_joint = None
            for joint in self.joints:
                if f"_{joint}_image.png" in image_name:
                    matched_joint = joint
                    break

            if matched_joint is None:
                continue

            base_key = image_name.replace("_image.png", "")
            mask_name = f"{base_key}_mask.png"

            if mask_name in mask_files:
                pairs.append((image_name, mask_name))

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