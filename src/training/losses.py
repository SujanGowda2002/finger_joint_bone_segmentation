import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-6, include_background=False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        """
        logits:  [B, C, H, W]
        targets: [B, H, W] with values in {0,1,2}
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.include_background:
            class_indices = range(self.num_classes)
        else:
            class_indices = range(1, self.num_classes)

        dice_losses = []

        for c in class_indices:
            pred_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append(1.0 - dice.mean())

        return sum(dice_losses) / len(dice_losses)


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes=3,
        ce_weight=0.5,
        dice_weight=0.5,
        include_background_in_dice=False,
        class_weights=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.dice = MultiClassDiceLoss(
            num_classes=num_classes,
            include_background=include_background_in_dice
        )

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss