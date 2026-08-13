"""Loading PVT-EDOF weights.

No trained weights ship with this repository; produce one with
``python -m pvtedof.train`` (the README has the procedure). Checkpoints are
plain ``state_dict`` files (an ``OrderedDict`` of tensors), so they load
cleanly under the ``weights_only=True`` default that PyTorch 2.6 introduced.
Nothing here needs ``weights_only=False``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from .model import ModelConfig, PVTEDOF

__all__ = ["load_state_dict", "infer_embed_dim", "build_model", "load_model"]

# Conventional location of the checkpoint in use. Training writes descriptive
# per-run names; copy or symlink the one you settled on here to avoid passing
# --checkpoint to every command. The feature dimension is read off the file, so
# the name carries no meaning the loader depends on.
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "pvtedof.pt")


def load_state_dict(path, map_location="cpu"):
    """Read a checkpoint into a ``state_dict``, tolerating older PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location=map_location)


def infer_embed_dim(state_dict, default=96):
    """Read ``t`` (the feature dimension) off the decoder convolution."""
    weight = state_dict.get("conv.weight")
    return int(weight.shape[1]) if weight is not None else default


def build_model(config: Optional[ModelConfig] = None, device="cpu") -> PVTEDOF:
    """Instantiate an untrained model, randomly initialised."""
    model = PVTEDOF(config or ModelConfig())
    return model.eval().to(device)


def load_model(path, device="cpu", config: Optional[ModelConfig] = None,
               strict=True, verbose=True) -> PVTEDOF:
    """Build a model and load ``path`` into it.

    The feature dimension and the presence of an absolute position embedding
    are read from the checkpoint unless ``config`` fixes them, so a checkpoint
    never has to be paired with a matching command line.

    One thing a ``state_dict`` does not record is the resolution it was trained
    at: no parameter shape depends on ``image_size`` while ``ape`` is off, so a
    model trained with a non-default ``--image-size`` loads without complaint
    and is then fed tiles of the wrong size. If you retrain at another
    resolution, say so explicitly here and in :func:`pvtedof.fuse`::

        model = load_model(path, config=ModelConfig(image_size=64))
        fused, dm = fuse(model, rgb, gray, tile=64)

    ``strict=True`` is deliberate. Loading a checkpoint from a different
    architecture with ``strict=False`` silently leaves most of the encoder
    randomly initialised, which produces plausible-looking but meaningless
    output. Pass ``strict=False`` only when you intend a partial load, and read
    the reported key counts.
    """
    state = load_state_dict(path, map_location="cpu")

    if config is None:
        config = ModelConfig(embed_dim=infer_embed_dim(state),
                             ape="absolute_pos_embed" in state)

    model = PVTEDOF(config)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if (missing or unexpected) and verbose:
        print(f"[pvtedof] partial load from {os.path.basename(path)}: "
              f"{len(missing)} missing and {len(unexpected)} unexpected keys. "
              f"The encoder is only partly trained; results will not be meaningful.")

    model.eval().to(device)
    if verbose:
        stored, executed = model.parameter_counts()
        print(f"[pvtedof] loaded {os.path.basename(path)} "
              f"(t={config.embed_dim}, {stored:,} stored / {executed:,} executed "
              f"parameters, device={device})")
    return model


def resolve_device(preference="auto"):
    """Pick a torch device. ``auto`` prefers CUDA, then Apple MPS, then CPU."""
    if preference not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"unknown device preference {preference!r}")
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if preference in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference in ("cuda", "mps"):
        print(f"[pvtedof] {preference} is not available, falling back to CPU.")
    return torch.device("cpu")
