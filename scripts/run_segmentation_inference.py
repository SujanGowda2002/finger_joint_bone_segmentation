import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.segmentation_dataset import FingerSegmentationDataset
from src.models.unet import UNet


def make_overlay(image_tensor, pred_mask_tensor):
    """
    image_tensor: [1, H, W]
    pred_mask_tensor: [1, H, W]
    returns: [3, H, W]
    """
    overlay = image_tensor.repeat(3, 1, 1).clone()

    # highlight predicted mask in red
    overlay[0] = torch.maximum(overlay[0], pred_mask_tensor[0])

    return overlay


@torch.no_grad()
def run_inference(model, loader, device, output_dir, max_examples=None):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    saved = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        image_names = batch["image_name"]

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        for i in range(images.size(0)):
            base_name = image_names[i].replace("_image.png", "")

            image_path = os.path.join(output_dir, f"{base_name}_image.png")
            gt_mask_path = os.path.join(output_dir, f"{base_name}_gt_mask.png")
            pred_mask_path = os.path.join(output_dir, f"{base_name}_pred_mask.png")
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")

            save_image(images[i], image_path)
            save_image(masks[i], gt_mask_path)
            save_image(preds[i], pred_mask_path)

            overlay = make_overlay(images[i].cpu(), preds[i].cpu())
            save_image(overlay, overlay_path)

            saved += 1
            if max_examples is not None and saved >= max_examples:
                return


def main():
    image_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "masks")
    checkpoint_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_unet_pip2_dip2.pth")
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "segmentation_inference_pip2_dip2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FingerSegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    joints=["pip2", "dip2"]
    )

    print("Total pip2 samples:", len(dataset))

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

    print(f"Saved inference outputs to: {output_dir}")


if __name__ == "__main__":
    main()