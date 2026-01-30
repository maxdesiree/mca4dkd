import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Attention Module
# =============================================================================
class AttentionModule(nn.Module):
    """Self-attention module for feature enhancement"""
    def __init__(self, in_features):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ChannelAttention(nn.Module):
    """Channel attention for CNN features"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)

# =============================================================================
# 2. Model
# =============================================================================
class EnhancedMultimodalModel(nn.Module):
    def __init__(self, num_tabular_features, num_classes=2, dropout_rate=0.5):
        super(EnhancedMultimodalModel, self).__init__()

        # IMAGE ENCODER
        self.image_encoder = models.resnet50(pretrained=False)  # False for fast testing
        self.channel_attention = ChannelAttention(2048)
        self.image_encoder.fc = nn.Identity()

        # IMAGE PROJECTION
        self.image_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            AttentionModule(512),
            nn.Linear(512, 256)
        )

        # TABULAR ENCODER
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            AttentionModule(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # FIXED CROSS-MODAL ATTENTION
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=dropout_rate * 0.5,
            batch_first=True
        )

        # PROJECTION LAYERS
        self.h1_proj = nn.Linear(256, 64)
        self.h2_proj = nn.Linear(64, 64)

        # GATING + FUSION (128-dim)
        self.gate = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # ENSEMBLE
        self.classifier1 = nn.Linear(128, num_classes)
        self.classifier2 = nn.Linear(128, num_classes)
        self.classifier3 = nn.Linear(128, num_classes)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, images, tabular, return_features=False):
        # IMAGE FEATURES
        x = self.image_encoder.conv1(images)
        x = self.image_encoder.bn1(x)
        x = self.image_encoder.relu(x)
        x = self.image_encoder.maxpool(x)
        x = self.image_encoder.layer1(x)
        x = self.image_encoder.layer2(x)
        x = self.image_encoder.layer3(x)
        x = self.image_encoder.layer4(x)
        x = self.channel_attention(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        h1 = self.image_projection(x)  # (N, 256)

        # TABULAR FEATURES
        h2 = self.tabular_encoder(tabular)  # (N, 64)

        # BIDIRECTIONAL CROSS-MODAL
        h1_proj = self.h1_proj(h1).unsqueeze(1)  # (N, 1, 64)
        h2_proj = self.h2_proj(h2).unsqueeze(1)  # (N, 1, 64)

        # h2' = Attention(h2, h1)
        h2_attended, _ = self.cross_attention(h2_proj, h1_proj, h1_proj)
        h2_prime = h2_attended.squeeze(1)

        # h1' = Attention(h1, h2)
        h1_attended, _ = self.cross_attention(h1_proj, h2_proj, h2_proj)
        h1_prime = h1_attended.squeeze(1)

        # CONCATENATE
        combined = torch.cat([h1_prime, h2_prime], dim=1)  # (N, 128)

        # GATING + FUSION
        gate = self.gate(combined)
        combined = combined * gate
        fused = self.fusion(combined)

        if return_features:
            return fused

        # ENSEMBLE
        out1 = self.classifier1(fused)
        out2 = self.classifier2(fused)
        out3 = self.classifier3(fused)
        weights = F.softmax(self.ensemble_weights, dim=0)
        output = weights[0] * out1 + weights[1] * out2 + weights[2] * out3

        return output