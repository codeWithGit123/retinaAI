import torch
import torch.nn as nn
import timm


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg = self.fc(self.avg_pool(x).view(b, c))
        max_ = self.fc(self.max_pool(x).view(b, c))

        out = avg + max_
        out = self.sigmoid(out).view(b, c, 1, 1)

        return x * out


class DRModel(nn.Module):
    """EfficientNetV2-S + CBAM Attention for Diabetic Retinopathy Classification"""

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=False,
            num_classes=0
        )

        self.attention = CBAM(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = self.head(x)
        return x