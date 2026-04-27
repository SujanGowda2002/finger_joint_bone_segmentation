import os
from typing import List, Tuple, Union

import numpy as np
from PIL import Image
import torch
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

        self.mask_resize = transforms.Resize(
            (176, 176), interpolation=InterpolationMode.NEAREST
        )

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

    def _binary_mask_to_multiclass(self, mask_np: np.ndarray) -> np.ndarray:
        """
        Input:
            binary mask with values 0/1, shape [H, W]
        Output:
            multiclass mask with values:
                0 = background
                1 = upper bone
                2 = lower bone
        Logic:
            - find connected components
            - keep 2 largest
            - sort by centroid y
        """
        h, w = mask_np.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        def bfs(start_y, start_x):
            queue = [(start_y, start_x)]
            visited[start_y, start_x] = True
            pixels = []

            while queue:
                y, x = queue.pop(0)
                pixels.append((y, x))

                for ny, nx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
                    if 0 <= ny < h and 0 <= nx < w:
                        if not visited[ny, nx] and mask_np[ny, nx] > 0:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            return pixels

        for y in range(h):
            for x in range(w):
                if mask_np[y, x] > 0 and not visited[y, x]:
                    comp = bfs(y, x)
                    components.append(comp)

        multiclass = np.zeros((h, w), dtype=np.uint8)

        if len(components) == 0:
            return multiclass

        components = sorted(components, key=lambda c: len(c), reverse=True)[:2]

        if len(components) == 1:
            comp = components[0]
            ys = [p[0] for p in comp]
            centroid_y = np.mean(ys)
            cls = 1 if centroid_y < (h / 2) else 2
            for y, x in comp:
                multiclass[y, x] = cls
            return multiclass

        comp_info = []
        for comp in components:
            ys = [p[0] for p in comp]
            centroid_y = np.mean(ys)
            comp_info.append((centroid_y, comp))

        comp_info.sort(key=lambda x: x[0])  # top first

        for y, x in comp_info[0][1]:
            multiclass[y, x] = 1

        for y, x in comp_info[1][1]:
            multiclass[y, x] = 2

        return multiclass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, mask_name = self.samples[idx]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)

        mask = self.mask_resize(mask)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 127).astype(np.uint8)

        multiclass_mask = self._binary_mask_to_multiclass(mask_np)
        multiclass_mask = torch.from_numpy(multiclass_mask).long()

        return {
            "image": image,                    # [1, H, W], float
            "mask": multiclass_mask,          # [H, W], long, values {0,1,2}
            "image_name": image_name,
            "mask_name": mask_name
        }