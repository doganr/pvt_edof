"""Transformer building blocks of the PVT-EDOF encoder.

Patch embedding, window partitioning, window-based multi-head self-attention
(W-MSA) and the Pyramid Vision Transformer (PVT) block described in Section 3
of the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

try:  # timm >= 0.9
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:  # timm < 0.9 (deprecated import path)
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = [
    "Mlp",
    "PatchEmbed",
    "WindowAttention",
    "PyramidBlock",
    "window_partition",
    "window_reverse",
]


class Mlp(nn.Module):
    """Two-layer feed-forward network used inside each PVT block."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """Patch-encoding layer.

    The paper uses non-overlapping 1x1 patches, so this is a 1x1 convolution
    that lifts the single luminance channel to ``embed_dim`` (t = 96) feature
    channels without changing the spatial resolution.
    """

    def __init__(self, img_size=224, patch_size=1, in_chans=1, embed_dim=32,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, height, width = x.shape
        if (height, width) != (self.img_size[0], self.img_size[1]):
            raise ValueError(
                f"Input image size ({height}x{width}) does not match the model "
                f"({self.img_size[0]}x{self.img_size[1]}). PVT-EDOF operates on "
                f"square subimages of the training size."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, Ph*Pw, C
        if self.norm is not None:
            x = self.norm(x)
        return x


def window_partition(x, window_size):
    """(B, H, W, C) -> (B*nW, window_size, window_size, C)."""
    return rearrange(x, "b (h ws1) (w ws2) c -> (b h w) ws1 ws2 c",
                     ws1=window_size, ws2=window_size)


def window_reverse(windows, window_size, height, width):
    """Inverse of :func:`window_partition`."""
    batch = int(windows.shape[0] / ((height // window_size) * (width // window_size)))
    return rearrange(windows, "(b h w) ws1 ws2 c -> b (h ws1) (w ws2) c",
                     b=batch, h=height // window_size, w=width // window_size,
                     ws1=window_size, ws2=window_size)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with a relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table, one entry per relative offset and head.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Pair-wise relative position index for each token inside a window.
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (t h d) -> t b h n d", t=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1],
               self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = rearrange(relative_position_bias, "w1 w2 h -> h w1 w2")
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


class PyramidBlock(nn.Module):
    """One W-MSA + MLP stage of a PVT block (Figure 4 of the paper).

    Order of operations: LayerNorm -> W-MSA -> residual -> LayerNorm -> MLP ->
    residual. Input and output are token sequences of shape (B, H*W, C).
    """

    def __init__(self, dim=32, img_size=224, num_heads=8, win_size=8,
                 mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.input_resolution = (img_size, img_size)
        self.num_heads = num_heads
        self.window_size = win_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.window_size = min(self.input_resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, window_size=to_2tuple(win_size),
                                    num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU, drop=0)

    def forward(self, x):
        height, width = self.input_resolution
        batch, length, channels = x.shape
        if length != height * width:
            raise ValueError(
                f"Token count {length} does not match the block resolution "
                f"{height}x{width}."
            )

        shortcut = x
        x = self.norm1(x)
        x = x.view(batch, height, width, channels)

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        x = window_reverse(attn_windows, self.window_size, height, width)
        x = x.view(batch, height * width, channels)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
