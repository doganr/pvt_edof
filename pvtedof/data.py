"""Loading focal stacks from disk.

A *focal stack* is a directory of images of the same scene captured at
different focal distances, in acquisition order. All planes must share one
resolution.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image

__all__ = ["IMAGE_EXTENSIONS", "stack_paths", "load_stack", "to_luminance", "InputScale"]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class InputScale:
    """How pixel values are scaled before entering the encoder.

    ``UNIT`` divides by 255, matching ``torchvision.transforms.ToTensor`` used
    during training. This is the convention behind the published focus-map
    figures and the runtime table, and is the default everywhere in this
    repository.

    ``RAW`` feeds 0-255 values unchanged. Some of the original experiment
    scripts did this; the option exists so those runs can be reproduced.
    """

    UNIT = "unit"
    RAW = "raw"
    CHOICES = (UNIT, RAW)

    @staticmethod
    def apply(gray, scale):
        if scale == InputScale.UNIT:
            return gray / 255.0
        if scale == InputScale.RAW:
            return gray
        raise ValueError(f"unknown input scale {scale!r}, expected one of {InputScale.CHOICES}")


def stack_paths(folder):
    """Sorted image paths inside ``folder``, filtered by extension."""
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"no images found in {folder!r} "
            f"(looked for {', '.join(IMAGE_EXTENSIONS)})")
    return paths


def load_stack(folder, scale=InputScale.UNIT):
    """Load a focal stack.

    Returns ``(rgb, gray)`` where ``rgb`` is ``(n, H, W, 3)`` uint8 - the
    original pixels the fused image is assembled from - and ``gray`` is
    ``(n, H, W)`` float32 luminance scaled according to ``scale``, which is
    what the encoder sees.
    """
    paths = stack_paths(folder)
    images = [np.array(Image.open(p).convert("RGB")) for p in paths]

    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise ValueError(
            f"{folder!r} mixes resolutions {sorted(shapes)}; every focal plane "
            f"of a stack must have the same size.")

    rgb = np.stack(images)
    gray = to_luminance(rgb, scale)
    return rgb, gray


def to_luminance(rgb, scale=InputScale.UNIT):
    """``(n, H, W, 3)`` uint8 -> ``(n, H, W)`` float32 luminance."""
    gray = np.stack([
        np.array(Image.fromarray(im).convert("L")) for im in rgb
    ]).astype(np.float32)
    return InputScale.apply(gray, scale)
