"""Module 2 of the technique: optimally focused imaging.

The image is covered with square subimages of the training size. Every focal
plane of a subimage is encoded, the Spatial Frequency measure turns the deep
features into a per-pixel focal-plane index, and the all-in-focus image is
assembled by copying, at each coordinate, the pixel of the selected plane. No
pixel value is ever computed, averaged or interpolated - the output is a mosaic
of original acquired pixels (Eq. 9 of the paper).

:func:`fuse` is the entry point: it takes a focal stack and returns the
all-in-focus image together with the focus map.

The tiling handles the seams itself. Subimages are laid down at
``stride < tile`` so they overlap, the outer border is reflection-padded, and
every output pixel is taken from the subimage in which it lies most centrally.
A pixel is therefore never read from a raw tile edge, which is what would
otherwise imprint the 128 px tile grid onto the result. No filtering pass is
involved, so there is nothing to tune.

:func:`decision_map_grid` is the non-overlapping tiling, kept only as the
baseline the boundary-consistency experiment of Table 3 compares against. It is
not part of the technique and should not be used to produce results.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import torch

from .focus_measures import spatial_frequency

__all__ = [
    "TILE",
    "DEFAULT_STRIDE",
    "DEFAULT_PAD",
    "fuse",
    "tile_starts",
    "grid_origins",
    "centrality_weights",
    "decision_map",
    "decision_map_batched",
    "decision_map_grid",
    "select_pixels",
    "throughput_sweep",
]

TILE = 128
"""Subimage side length; must equal the size the encoder was trained on."""

DEFAULT_STRIDE = 64
"""Subimage step. Below ``TILE``, so consecutive subimages overlap."""

DEFAULT_PAD = 64
"""Reflection padding applied before tiling, so border pixels are never at a raw tile edge."""


def tile_starts(total, tile=TILE, stride=TILE):
    """Start offsets covering ``[0, total)``, last tile flush with the border."""
    if total < tile:
        raise ValueError(
            f"image side {total} is smaller than the {tile}px subimage; pad the "
            f"image or use a model trained at a smaller size.")
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def grid_origins(shape, tile=TILE):
    """Non-overlapping tile origins for ``shape=(H, W)``."""
    height, width = shape
    return [(y, x) for y in tile_starts(height, tile, tile)
            for x in tile_starts(width, tile, tile)]


def centrality_weights(tile=TILE):
    """Distance of each tile pixel to the nearest tile edge (higher = central)."""
    axis = np.arange(tile)
    return np.minimum(
        np.minimum(axis[:, None], (tile - 1 - axis)[:, None]),
        np.minimum(axis[None, :], (tile - 1 - axis)[None, :]),
    ).astype(np.int32)


def _encode_tile(model, gray, y, x, device, tile):
    """Encode every plane of one tile; return its ``(tile, tile)`` decision map."""
    features = []
    for plane in gray:
        patch = plane[y:y + tile, x:x + tile][None, None].astype(np.float32)
        tensor = torch.from_numpy(patch).to(device)
        with torch.no_grad():
            features.append(model.encoder(tensor))
    _, dm = spatial_frequency(torch.cat(features, dim=0), device)
    return dm.astype(np.int16)


def decision_map(model, gray, stride=64, pad=64, device="cpu", tile=TILE):
    """Per-pixel focal-plane index, one encoder call per (plane, subimage).

    This is the slow path, kept because it is easy to follow and uses little
    memory. Use :func:`decision_map_batched` for anything large.

    Parameters
    ----------
    gray : np.ndarray
        ``(n, H, W)`` float32 luminance, already scaled (see
        :class:`pvtedof.data.InputScale`).
    stride : int
        Subimage step. ``stride == tile`` gives the non-overlapping grid.
    pad : int
        Reflection padding applied before tiling, removing the outer-border
        seam. ``0`` disables it.
    """
    device = torch.device(device)
    _, height, width = gray.shape
    padded = np.pad(gray, ((0, 0), (pad, pad), (pad, pad)), mode="reflect") if pad > 0 else gray
    padded_h, padded_w = padded.shape[1], padded.shape[2]

    dm = np.zeros((padded_h, padded_w), np.int16)
    best = np.full((padded_h, padded_w), -1, np.int32)
    weights = centrality_weights(tile)

    for y in tile_starts(padded_h, tile, stride):
        for x in tile_starts(padded_w, tile, stride):
            local = _encode_tile(model, padded, y, x, device, tile)
            best_region = best[y:y + tile, x:x + tile]
            better = weights > best_region
            dm[y:y + tile, x:x + tile][better] = local[better]
            best_region[better] = weights[better]

    return dm[pad:pad + height, pad:pad + width] if pad > 0 else dm


def decision_map_batched(model, gray, stride=64, pad=64, device="cpu",
                         subimage_batch=64, tile=TILE):
    """Same result as :func:`decision_map`, with batched encoder calls.

    Every encoder forward processes about ``subimage_batch`` subimages at once,
    which is the data-parallel form that actually saturates a GPU. All planes
    of a tile always stay together so the Spatial Frequency ``argmax`` is exact.
    ``subimage_batch`` is the throughput knob swept by
    :func:`throughput_sweep`.
    """
    device = torch.device(device)
    n_planes, height, width = gray.shape
    padded = np.pad(gray, ((0, 0), (pad, pad), (pad, pad)), mode="reflect") if pad > 0 else gray
    padded_h, padded_w = padded.shape[1], padded.shape[2]

    tiles = [(y, x) for y in tile_starts(padded_h, tile, stride)
             for x in tile_starts(padded_w, tile, stride)]
    dm = np.zeros((padded_h, padded_w), np.int16)
    best = np.full((padded_h, padded_w), -1, np.int32)
    weights = centrality_weights(tile)
    tiles_per_call = max(1, subimage_batch // n_planes)

    for i in range(0, len(tiles), tiles_per_call):
        batch = tiles[i:i + tiles_per_call]
        arrays = np.stack([padded[:, y:y + tile, x:x + tile] for (y, x) in batch])
        count = arrays.shape[0]
        tensor = torch.from_numpy(
            arrays.reshape(count * n_planes, 1, tile, tile).astype(np.float32)).to(device)
        with torch.no_grad():
            features = model.encoder(tensor)
        features = features.view(count, n_planes, *features.shape[1:])

        for j, (y, x) in enumerate(batch):
            _, local = spatial_frequency(features[j], device)
            best_region = best[y:y + tile, x:x + tile]
            better = weights > best_region
            dm[y:y + tile, x:x + tile][better] = local.astype(np.int16)[better]
            best_region[better] = weights[better]
        del features, tensor

    return dm[pad:pad + height, pad:pad + width] if pad > 0 else dm


def decision_map_grid(model, gray, device="cpu", tile=TILE):
    """Non-overlapping tiling, last tile in each row/column slid to the border.

    **Not part of the technique.** This is the seam-prone baseline the overlap
    scheme replaced: tiles do not overlap, the border is not padded, and later
    tiles simply overwrite earlier ones in the strip where they meet, so the
    128 px grid can show up in the output. It exists to quantify what the
    overlap scheme buys (Table 3). Use :func:`fuse` to produce results.
    """
    device = torch.device(device)
    _, height, width = gray.shape
    dm = np.zeros((height, width), np.int16)
    for (y, x) in grid_origins((height, width), tile):
        dm[y:y + tile, x:x + tile] = _encode_tile(model, gray, y, x, device, tile)
    return dm


def select_pixels(stack, dm):
    """Assemble the all-in-focus image (Eq. 9).

    ``stack`` is ``(n, H, W, ...)`` and ``dm`` is an ``(H, W)`` array of plane
    indices; the result takes ``stack[dm[y, x], y, x]`` at every coordinate.
    """
    height, width = dm.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return stack[dm, yy, xx]


def fuse(model, rgb, gray, device="cpu", stride=DEFAULT_STRIDE, pad=DEFAULT_PAD,
         subimage_batch=64, tile=TILE):
    """Focal stack in, all-in-focus image and focus map out.

    This is the whole of Module 2 in one call, and the entry point you should
    normally use. The seam handling is part of it, not an extra step layered on
    afterwards: subimages overlap, the outer border is reflection-padded, and
    every output pixel is taken from the subimage in which it lies most
    centrally. A pixel is therefore never read from a raw tile edge, which is
    what would otherwise imprint the tile grid onto the result. Table 3 of the
    paper is the quantitative check.

    Parameters
    ----------
    model : PVTEDOF
        A loaded encoder, in ``eval()`` mode.
    rgb : np.ndarray
        ``(n, H, W, ...)`` original planes. The fused image is assembled from
        these, so whatever they hold - colour, extra channels - is preserved
        exactly.
    gray : np.ndarray
        ``(n, H, W)`` float32 luminance, scaled as during training. Used only
        to score focus. :func:`pvtedof.data.load_stack` returns both arrays.
    device : str or torch.device
    stride : int
        Subimage step. Must stay below ``tile`` for the overlap to exist.
    pad : int
        Reflection padding before tiling. ``0`` reintroduces the outer-border
        seam.
    subimage_batch : int
        Subimages per encoder call. Speed and memory only, not results.
        ``0`` selects the unbatched reference implementation.

    Returns
    -------
    tuple
        ``(all_in_focus, focus_map)``. ``all_in_focus`` has the dtype and
        trailing shape of ``rgb``; ``focus_map`` is an ``(H, W)`` int array of
        selected plane indices, which doubles as a relative depth map.

    Examples
    --------
    >>> rgb, gray = load_stack("data/synthetic_demo")
    >>> all_in_focus, focus_map = fuse(model, rgb, gray, device="cuda")
    """
    if stride >= tile or pad <= 0:
        warnings.warn(
            f"fuse() called with stride={stride}, pad={pad}: with stride >= {tile} "
            f"or pad = 0 the subimages no longer overlap everywhere, so tile "
            f"borders can become visible in the output. The seam-free defaults "
            f"are stride={DEFAULT_STRIDE}, pad={DEFAULT_PAD}.",
            RuntimeWarning, stacklevel=2)

    if subimage_batch > 0:
        focus_map = decision_map_batched(model, gray, stride, pad, device,
                                         subimage_batch, tile)
    else:
        focus_map = decision_map(model, gray, stride, pad, device, tile)

    return select_pixels(rgb, focus_map), focus_map


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def throughput_sweep(model, gray, stride=64, pad=64, device="cpu",
                     batches=(8, 16, 32, 64, 128), repeats=3, tile=TILE):
    """Measure throughput and peak GPU memory against ``subimage_batch``.

    Returns one dict per batch size with keys ``subimage_batch``,
    ``seconds_per_stack``, ``stacks_per_second``, ``subimages_per_second``,
    ``peak_gpu_gb`` and ``oom``.
    """
    device = torch.device(device)
    n_planes, height, width = gray.shape
    n_subimages = (len(tile_starts(height + 2 * pad, tile, stride))
                   * len(tile_starts(width + 2 * pad, tile, stride))
                   * n_planes)

    results = []
    for batch in batches:
        try:
            decision_map_batched(model, gray, stride, pad, device, batch, tile)  # warm-up
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append(dict(subimage_batch=batch, seconds_per_stack=float("nan"),
                                stacks_per_second=float("nan"),
                                subimages_per_second=float("nan"),
                                peak_gpu_gb=float("nan"), oom=True))
            continue

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        timings = []
        for _ in range(repeats):
            _synchronize(device)
            started = time.perf_counter()
            decision_map_batched(model, gray, stride, pad, device, batch, tile)
            _synchronize(device)
            timings.append(time.perf_counter() - started)

        seconds = float(np.median(timings))
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        results.append(dict(subimage_batch=batch, seconds_per_stack=seconds,
                            stacks_per_second=1.0 / seconds,
                            subimages_per_second=n_subimages / seconds,
                            peak_gpu_gb=peak, oom=False))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results
