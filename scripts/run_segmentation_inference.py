import os
import sys
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.segmentation_dataset import FingerSegmentationDataset
from src.models.unet import UNet


def split_upper_lower_components(mask_tensor):
    """
    mask_tensor: [1, H, W], binary {0,1}
    returns:
        upper_mask: [1, H, W]
        lower_mask: [1, H, W]
    """
    mask_np = mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8)

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
                pixels = bfs(y, x)
                components.append(pixels)

    if len(components) == 0:
        empty = torch.zeros_like(mask_tensor)
        return empty, empty

    components = sorted(components, key=lambda c: len(c), reverse=True)[:2]

    if len(components) == 1:
        comp = components[0]
        ys = [p[0] for p in comp]
        centroid_y = np.mean(ys)

        upper = torch.zeros_like(mask_tensor)
        lower = torch.zeros_like(mask_tensor)

        target = upper if centroid_y < (h / 2) else lower
        for y, x in comp:
            target[0, y, x] = 1.0

        return upper, lower

    comp_info = []
    for comp in components:
        ys = [p[0] for p in comp]
        centroid_y = np.mean(ys)
        comp_info.append((centroid_y, comp))

    comp_info.sort(key=lambda x: x[0])

    upper = torch.zeros_like(mask_tensor)
    lower = torch.zeros_like(mask_tensor)

    for y, x in comp_info[0][1]:
        upper[0, y, x] = 1.0

    for y, x in comp_info[1][1]:
        lower[0, y, x] = 1.0

    return upper, lower


def make_two_color_overlay(image_tensor, mask_tensor):
    """
    image_tensor: [1, H, W]
    mask_tensor: [1, H, W], binary
    returns: [3, H, W]

    upper bone = red
    lower bone = green
    """
    overlay = image_tensor.repeat(3, 1, 1).clone()

    upper_mask, lower_mask = split_upper_lower_components(mask_tensor)

    # Upper bone in red
    overlay[0] = torch.maximum(overlay[0], upper_mask[0])
    overlay[1][upper_mask[0] > 0] *= 0.4
    overlay[2][upper_mask[0] > 0] *= 0.4

    # Lower bone in green
    overlay[1] = torch.maximum(overlay[1], lower_mask[0])
    overlay[0][lower_mask[0] > 0] *= 0.4
    overlay[2][lower_mask[0] > 0] *= 0.4

    return overlay.clamp(0, 1)


def prepare_output_dir(output_dir):
    """
    Replace old outputs with a fresh directory.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


@torch.no_grad()
def run_inference(model, loader, device, output_dir, max_examples=None):
    model.eval()
    prepare_output_dir(output_dir)

    saved = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        image_names = batch["image_name"]

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        for i in range(images.size(0)):
            image_name = image_names[i]
            base_name = os.path.splitext(image_name)[0]
            if base_name.endswith("_image"):
                base_name = base_name[:-6]

            image_path = os.path.join(output_dir, f"{base_name}_image.png")
            gt_mask_path = os.path.join(output_dir, f"{base_name}_gt_mask.png")
            pred_mask_path = os.path.join(output_dir, f"{base_name}_pred_mask.png")
            gt_overlay_path = os.path.join(output_dir, f"{base_name}_gt_overlay.png")
            pred_overlay_path = os.path.join(output_dir, f"{base_name}_pred_overlay.png")

            save_image(images[i].cpu(), image_path)
            save_image(masks[i].cpu(), gt_mask_path)
            save_image(preds[i].cpu(), pred_mask_path)

            gt_overlay = make_two_color_overlay(images[i].cpu(), masks[i].cpu())
            pred_overlay = make_two_color_overlay(images[i].cpu(), preds[i].cpu())

            save_image(gt_overlay, gt_overlay_path)
            save_image(pred_overlay, pred_overlay_path)

            saved += 1
            if max_examples is not None and saved >= max_examples:
                return


def main():
    image_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "masks")
    checkpoint_path = os.path.join(
        PROJECT_ROOT, "outputs", "checkpoints", "best_unet_pip2_dip2.pth"
    )
    output_dir = os.path.join(
        PROJECT_ROOT, "outputs", "segmentation_inference_pip2_dip2"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    joints = ["pip2", "pip3", "pip4", "pip5", "dip2", "dip3", "dip4", "dip5"]

    dataset = FingerSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        joints=joints
    )

    print(f"Total samples for joints {joints}: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from: {checkpoint_path}")

    run_inference(
        model=model,
        loader=loader,
        device=device,
        output_dir=output_dir,
        max_examples=None
    )

    print(f"Replaced old outputs and saved new inference outputs to: {output_dir}")


if __name__ == "__main__":
    main()