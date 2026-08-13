"""The PVT-EDOF encoder-decoder model.

Module 1 of the technique (Section 3.1 of the paper): a Pyramid Vision
Transformer encoder trained self-supervised to reconstruct its own grayscale
input. At inference time only the encoder is used - its deep feature maps feed
the Spatial Frequency focus measure in :mod:`pvtedof.focus_measures`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from .blocks import PatchEmbed, PyramidBlock

try:  # timm >= 0.9
    from timm.layers import trunc_normal_
except ImportError:  # timm < 0.9
    from timm.models.layers import trunc_normal_

__all__ = ["ModelConfig", "StageModule", "PVTEDOF"]


@dataclass
class ModelConfig:
    """Configuration of the PVT-EDOF encoder-decoder.

    The defaults reproduce the architecture of the published model exactly.
    """

    image_size: int = 128
    """Side length of the square subimage the encoder consumes (128 in the paper)."""

    embed_dim: int = 96
    """Number of deep feature maps, called ``t`` in the paper."""

    ape: bool = False
    """Add a learnable absolute position embedding. Off in the published model."""

    depths: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    """Blocks per PVT stage. Four stages of three blocks in the published model."""

    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16])
    """Attention heads of the three blocks inside each stage."""

    window_sizes: List[int] = field(default_factory=lambda: [2, 4, 8])
    """W-MSA window sizes of the three blocks inside each stage (w, 2w, 4w)."""

    drop_path_rate: float = 0.2
    """Stochastic-depth rate used to build the per-block decay schedule."""

    num_active_stages: int = 3
    """Stages actually applied in :meth:`PVTEDOF.encoder`.

    ``len(depths)`` stages are instantiated - and are therefore all present in
    a trained checkpoint - but the encoder applies only the first
    ``num_active_stages`` of them. The published model was trained and evaluated
    with three active stages, so this default reproduces every published
    result. At ``t = 96`` the fourth stage accounts for 339,548 of the
    1,358,865 stored parameters, leaving 1,019,317 in the executed model.
    """

    def __post_init__(self):
        if self.num_active_stages > len(self.depths):
            raise ValueError(
                f"num_active_stages={self.num_active_stages} exceeds the "
                f"{len(self.depths)} instantiated stages."
            )


class StageModule(nn.Module):
    """One PVT stage: ``depth`` consecutive :class:`PyramidBlock` layers."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_sizes,
                 drop_path=0.0, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            PyramidBlock(dim=dim, img_size=input_resolution,
                         num_heads=num_heads[i], win_size=window_sizes[i])
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"depth={self.depth}")


class PVTEDOF(nn.Module):
    """Pyramid Vision Transformer autoencoder for extended depth of focus."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.img_size = cfg.image_size
        self.ape = cfg.ape
        self.in_channels = 1
        self.embed = cfg.embed_dim

        self.patch_embed = PatchEmbed(
            img_size=cfg.image_size, patch_size=1, in_chans=1,
            embed_dim=self.embed, norm_layer=nn.LayerNorm)
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=0.0)

        # Stochastic-depth decay rule (kept for reference; the published model
        # was trained with drop_path disabled inside the blocks).
        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(cfg.depths))]

        self.stage_layers = nn.ModuleList([
            StageModule(dim=self.embed,
                        input_resolution=self.img_size,
                        depth=cfg.depths[i],
                        num_heads=cfg.num_heads,
                        window_sizes=cfg.window_sizes,
                        drop_path=dpr[sum(cfg.depths[:i]):sum(cfg.depths[:i + 1])])
            for i in range(len(cfg.depths))
        ])

        self.norm = nn.LayerNorm(self.embed)
        self.conv = nn.Conv2d(in_channels=self.embed, out_channels=1,
                              kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ init
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    @property
    def device(self):
        return next(self.parameters()).device

    # --------------------------------------------------------------- forward
    def encoder(self, x):
        """Grayscale subimages ``(B, 1, S, S)`` -> deep features ``(B, t, S, S)``."""
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i in range(self.config.num_active_stages):
            x = self.stage_layers[i](x) + x

        x = self.norm(x)
        side = int(x.shape[1] ** 0.5)
        return rearrange(x, "b (h w) c -> b c h w", h=side, w=side)

    def decoder(self, x):
        """Deep features -> reconstructed grayscale subimage."""
        return self.tanh(self.conv(x))

    def autoencoder(self, x):
        """Full reconstruction path, used only during self-supervised training."""
        return self.decoder(self.encoder(x))

    forward = autoencoder

    # ------------------------------------------------------------ diagnostics
    def parameter_counts(self):
        """Return ``(stored, executed)`` parameter counts.

        ``stored`` counts every instantiated parameter, which is what a
        checkpoint contains. ``executed`` excludes the stages beyond
        ``num_active_stages``, which are never applied.
        """
        stored = sum(p.numel() for p in self.parameters())
        inactive = sum(
            p.numel()
            for i in range(self.config.num_active_stages, len(self.stage_layers))
            for p in self.stage_layers[i].parameters()
        )
        return stored, stored - inactive
