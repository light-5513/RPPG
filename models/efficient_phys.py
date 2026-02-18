import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution - efficient for CPU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AttentionBlock(nn.Module):
    """Temporal attention block for focusing on relevant frames."""

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class EfficientPhys(nn.Module):
    """
    Lightweight rPPG model optimized for CPU inference.

    Input: Difference frames (T-1, 3, H, W) - temporal difference of consecutive frames
    Output: rPPG signal (T-1,)
    """

    def __init__(self, img_size=72, in_channels=3):
        super().__init__()

        # Spatial feature extractor
        self.spatial = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            AttentionBlock(64),
            DepthwiseSeparableConv(64, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(64, 128, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )

        # Temporal processing
        self.temporal = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of video clips

        Returns:
            rppg_signal: (B, T) - predicted rPPG signal
        """
        B, T, C, H, W = x.shape

        # Compute difference frames
        diff = x[:, 1:] - x[:, :-1]  # (B, T-1, C, H, W)
        T_diff = T - 1

        # Reshape for spatial processing
        diff = diff.reshape(B * T_diff, C, H, W)

        # Spatial features
        features = self.spatial(diff)  # (B*T_diff, 128, 1, 1)
        features = features.view(B * T_diff, 128)

        # Temporal prediction
        out = self.temporal(features)  # (B*T_diff, 1)
        out = out.view(B, T_diff)

        return out


class NegPearsonLoss(nn.Module):
    """Negative Pearson Correlation Loss - standard for rPPG training."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, T) predicted signal
            target: (B, T) ground truth signal
        """
        # Ensure same length
        min_len = min(pred.shape[1], target.shape[1])
        pred = pred[:, :min_len]
        target = target[:, :min_len]

        # Center signals
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        # Pearson correlation
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt(
            (pred_centered ** 2).sum(dim=1) * (target_centered ** 2).sum(dim=1) + 1e-8
        )

        pearson = numerator / denominator

        # Negative pearson (we want to maximize correlation = minimize negative)
        loss = 1 - pearson.mean()

        return loss


def get_model(img_size=72, in_channels=3):
    """Factory function to create model."""
    model = EfficientPhys(img_size=img_size, in_channels=in_channels)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EfficientPhys")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB\n")

    return model