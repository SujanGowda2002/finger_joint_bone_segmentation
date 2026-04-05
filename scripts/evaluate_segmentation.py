import os
import sys

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.segmentation_dataset import FingerSegmentationDataset
from src.models.unet import UNet
from src.training.losses import DiceLoss
from src.evaluation.metrics import dice_score, iou_score


def evaluate(model, loader, criterion, device, save_dir=None):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    count = 0

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            image_names = batch["image_name"]

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
            total_iou += iou_score(outputs, masks)
            count += 1

            if save_dir is not None:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                for i in range(images.size(0)):
                    base_name = os.path.splitext(image_names[i])[0]

                    image_path = os.path.join(save_dir, f"{base_name}_image.png")
                    mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
                    pred_path = os.path.join(save_dir, f"{base_name}_pred.png")
                    overlay_path = os.path.join(save_dir, f"{base_name}_overlay.png")

                    save_image(images[i], image_path)
                    save_image(masks[i], mask_path)
                    save_image(preds[i], pred_path)

                    # Simple overlay: original image + predicted mask highlight
                    overlay = images[i].repeat(3, 1, 1).clone()
                    overlay[0] = torch.maximum(overlay[0], preds[i, 0])  # red channel boost
                    save_image(overlay, overlay_path)

    avg_loss = total_loss / count
    avg_dice = total_dice / count
    avg_iou = total_iou / count

    return avg_loss, avg_dice, avg_iou


def main():
    image_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "images")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "segmentation_seed", "masks")
    checkpoint_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_unet_pip2_dip2.pth")
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "segmentation_eval_pip2_dip2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FingerSegmentationDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    joints=["pip2", "dip2"]
    )

    print("Total pip2 samples:", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from: {checkpoint_path}")

    criterion = DiceLoss()

    val_loss, val_dice, val_iou = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        save_dir=output_dir
    )

    print("\n===== SEGMENTATION EVALUATION RESULTS =====")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Dice: {val_dice:.4f}")
    print(f"Validation IoU: {val_iou:.4f}")
    print(f"Saved predictions to: {output_dir}")


if __name__ == "__main__":
    main()