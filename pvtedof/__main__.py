"""Fuse one focal stack into an all-in-focus image.

    python -m pvtedof --stack data/synthetic_demo --out results/demo

A thin wrapper around :func:`pvtedof.fusion.fuse`, which handles the subimage
seams as part of the fusion itself: subimages overlap, the outer border is
reflection-padded, and every output pixel comes from the subimage it is most
central in, so the 128 px tile grid never appears in the result.

Three files are written to ``--out``:

``<name>_fused.png``
    All-in-focus image at the input resolution. Every pixel is copied
    unchanged from one of the input planes.
``<name>_focusmap.png``
    Selected focal-plane index per pixel, colour-coded with viridis, which is
    a relative depth map.
``<name>_focusmap.npy``
    The raw integer plane indices, for further analysis.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from PIL import Image

from .checkpoints import DEFAULT_CHECKPOINT, load_model, resolve_device
from .colormap import COLORMAPS, colorize
from .data import InputScale, load_stack
from .fusion import DEFAULT_PAD, DEFAULT_STRIDE, fuse


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m pvtedof",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True,
                        help="directory holding the focal planes of one stack")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default="results",
                        help="output directory")
    parser.add_argument("--name", default=None,
                        help="output file prefix (default: the stack folder name)")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="subimage step; below 128 so subimages overlap, "
                             "which is what keeps the tile grid out of the output")
    parser.add_argument("--pad", type=int, default=DEFAULT_PAD,
                        help="reflection padding before tiling, which removes "
                             "the outer-border seam")
    parser.add_argument("--subimage-batch", type=int, default=64,
                        help="subimages per encoder call; 0 selects the "
                             "unbatched reference implementation")
    parser.add_argument("--input-scale", default=InputScale.UNIT,
                        choices=InputScale.CHOICES,
                        help="'unit' divides by 255 to match training (default); "
                             "'raw' feeds 0-255 through unchanged")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--cmap", default="viridis", choices=sorted(COLORMAPS),
                        help="colour table for the focus map PNG; the raw "
                             "indices in the .npy are unaffected")
    return parser


def main():
    args = build_parser().parse_args()

    device = resolve_device(args.device)
    name = args.name or os.path.basename(os.path.normpath(args.stack))
    os.makedirs(args.out, exist_ok=True)

    rgb, gray = load_stack(args.stack, scale=args.input_scale)
    n_planes, height, width = gray.shape
    print(f"[{name}] {n_planes} planes, {width}x{height}, "
          f"stride={args.stride}, pad={args.pad}")

    model = load_model(args.checkpoint, device=device)

    started = time.perf_counter()
    fused, dm = fuse(model, rgb, gray, device=device, stride=args.stride,
                     pad=args.pad, subimage_batch=args.subimage_batch)
    elapsed = time.perf_counter() - started

    print(f"  fused in {elapsed:.2f} s "
          f"({elapsed / n_planes * 1000:.1f} ms per focal plane)")
    print(f"  selected planes span {dm.min()}..{dm.max()} of 0..{n_planes - 1}")

    prefix = os.path.join(args.out, name)
    Image.fromarray(fused).save(f"{prefix}_fused.png")
    np.save(f"{prefix}_focusmap.npy", dm)

    # Colour-coded purely so the map is legible; the .npy above keeps the
    # actual plane indices.
    Image.fromarray(colorize(dm, args.cmap)).save(f"{prefix}_focusmap.png")

    print(f"  wrote {prefix}_fused.png, {prefix}_focusmap.png, "
          f"{prefix}_focusmap.npy")


if __name__ == "__main__":
    main()
