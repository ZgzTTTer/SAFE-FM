import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        residual = self.residual_conv(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        time_emb = F.silu(self.time_mlp(t))
        h = h + time_emb[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + residual


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W).permute(0, 2, 1)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        scale = 1.0 / math.sqrt(C)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.proj(out)
        return out + residual


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionUNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        attn_resolutions=(2, 3),
    ):
        super().__init__()
        self.num_levels = len(channel_mults)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels_list = [base_channels]
        now_ch = base_channels

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = level in attn_resolutions
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResBlock(now_ch, out_ch, time_emb_dim))
                if use_attn:
                    block.append(AttentionBlock(out_ch))
                now_ch = out_ch
                channels_list.append(now_ch)
            self.down_blocks.append(block)
            if level != self.num_levels - 1:
                self.down_samples.append(Downsample(now_ch))
                channels_list.append(now_ch)
            else:
                self.down_samples.append(nn.Identity())

        self.mid_block1 = ResBlock(now_ch, now_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(now_ch)
        self.mid_block2 = ResBlock(now_ch, now_ch, time_emb_dim)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            use_attn = level in attn_resolutions
            block = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = channels_list.pop()
                block.append(ResBlock(now_ch + skip_ch, out_ch, time_emb_dim))
                if use_attn:
                    block.append(AttentionBlock(out_ch))
                now_ch = out_ch
            self.up_blocks.append(block)
            if level != 0:
                self.up_samples.append(Upsample(now_ch))
            else:
                self.up_samples.append(nn.Identity())

        self.final_norm = nn.GroupNorm(8, now_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(now_ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        skips = [x]

        for level in range(self.num_levels):
            for layer in self.down_blocks[level]:
                if isinstance(layer, ResBlock):
                    x = layer(x, t_emb)
                    skips.append(x)
                else:
                    x = layer(x)
            if level != self.num_levels - 1:
                x = self.down_samples[level](x)
                skips.append(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for idx, level in enumerate(reversed(range(self.num_levels))):
            for layer in self.up_blocks[idx]:
                if isinstance(layer, ResBlock):
                    x = torch.cat([x, skips.pop()], dim=1)
                    x = layer(x, t_emb)
                else:
                    x = layer(x)
            if level != 0:
                x = self.up_samples[idx](x)

        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)
        return x
