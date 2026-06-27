import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEncoder(nn.Module):
    def __init__(self, in_channels=1, feat_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(8, feat_channels),
            nn.SiLU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.GroupNorm(8, feat_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class SpatialAttentionFusion(nn.Module):
    def __init__(self, feat_channels=32, num_slices=3):
        super().__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(feat_channels * num_slices, feat_channels, 3, padding=1),
            nn.GroupNorm(8, feat_channels),
            nn.SiLU(),
            nn.Conv2d(feat_channels, num_slices, 1),
        )
        self.num_slices = num_slices

    def forward(self, feat_list):
        concat = torch.cat(feat_list, dim=1)
        weights = self.attn_conv(concat)
        weights = torch.softmax(weights, dim=1)

        merged = torch.zeros_like(feat_list[0])
        for i in range(self.num_slices):
            merged = merged + weights[:, i:i+1, :, :] * feat_list[i]

        return merged


class FeatureAdaptiveFusion(nn.Module):
    def __init__(self, in_channels=1, feat_channels=32, num_slices=3):
        super().__init__()
        self.shared_encoder = SharedEncoder(in_channels, feat_channels)
        self.spatial_attn = SpatialAttentionFusion(feat_channels, num_slices)
        self.feat_channels = feat_channels

    def forward(self, slice_list):
        feat_list = [self.shared_encoder(s) for s in slice_list]
        merged = self.spatial_attn(feat_list)
        return merged
