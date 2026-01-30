# Advanced Loss Functions

import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss for binary classification"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        targets_one_hot = targets_one_hot.permute(0, 1)

        # Calculate Dice coefficient
        intersection = (probs[:, 1] * targets_one_hot[:, 1]).sum()
        dice = (2. * intersection + self.smooth) / (probs[:, 1].sum() + targets_one_hot[:, 1].sum() + self.smooth)

        return 1 - dice

class CombinedLoss(nn.Module):
    """Combines multiple loss functions (CE + Focal + Dice)"""
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            self.focal_loss = FocalLoss(alpha=class_weights, gamma=2)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            self.focal_loss = FocalLoss(gamma=2)

        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)

        return self.alpha * ce + self.beta * focal + self.gamma * dice

print("Loss function classes defined:")
print("  - FocalLoss: Addresses class imbalance with focusing parameter")
print("  - DiceLoss: Optimizes overlap between prediction and ground truth")
print("  - CombinedLoss: Weighted combination of CE + Focal + Dice")
print("\nNote: Class weights will be calculated separately for each fold")
print("      during the 10-fold cross-validation training loop.")
