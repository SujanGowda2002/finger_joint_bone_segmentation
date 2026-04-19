import os
import sys
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.segmentation_dataset import FingerSegmentationDataset
from src.models.deeplabv3_model import DeepLabV3Model


def prepare_output_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


def class_mask_to_grayscale(mask_tensor):
    mask_float = mask_tensor.float() / 2.0
    return mask_float.unsqueeze(0)


def make_multiclass_overlay(image_tensor, class_mask_tensor):
    overlay = image_tensor.repeat(3, 1, 1).clone()

    upper = (class_mask_tensor == 1)
    lower = (class_mask_tensor == 2)

    overlay[0][upper] = 1.0
    overlay[1][upper] *= 0.35
    overlay[2][upper] *= 0.35

    overlay[1][lower] = 1.0
    overlay[0][lower] *= 0.35
    overlay[2][lower] *= 0.35

    return overlay.clamp(0, 1)


@torch.no_grad()
def run_inference(model, loader, device, output_dir, max_examples=None):
    model.eval()
    prepare_output_dir(output_dir)

    saved = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        image_names = batch["image_name"]

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

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

            image_cpu = images[i].cpu()
            gt_mask_cpu = masks[i].cpu()
            pred_mask_cpu = preds[i].cpu()

            save_image(image_cpu, image_path)
            save_image(class_mask_to_grayscale(gt_mask_cpu), gt_mask_path)
            save_image(class_mask_to_grayscale(pred_mask_cpu), pred_mask_path)

            gt_overlay = make_multiclass_overlay(image_cpu, gt_mask_cpu)
            pred_overlay = make_multiclass_overlay(image_cpu, pred_mask_cpu)

            save_image(gt_overlay, gt_overlay_path)
            save_image(pred_overlay, pred_overlay_path)

            saved += 1
            if max_examples is not None and saved >= max_examples:
                return


def main():
    image_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "masks")

    joints = [
        "pip2", "dip2",
        "pip3", "dip3",
        "pip4", "dip4",
        "pip5", "dip5",
        "mcp2", "mcp3", "mcp4", "mcp5",
    ]

    experiment_name = "deeplabv3_multiclass_all_joints"

    checkpoint_path = os.path.join(
        PROJECT_ROOT, "outputs", "checkpoints", f"best_{experiment_name}.pth"
    )
    output_dir = os.path.join(
        PROJECT_ROOT, "outputs", f"segmentation_inference_{experiment_name}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FingerSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        joints=joints
    )

    print(f"Total samples for joints {joints}: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = DeepLabV3Model(num_classes=3, pretrained_backbone=False).to(device)

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