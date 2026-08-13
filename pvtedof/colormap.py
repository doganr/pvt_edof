"""Viridis as a lookup table, so a focus map can be coloured without matplotlib.

These are matplotlib's 256 viridis entries, the colormap the paper's focus maps
were rendered with. They are written out here rather than imported so that
matplotlib stays out of this package's dependencies. :func:`colorize` also
copies matplotlib's indexing rule, so the output is pixel for pixel what
``plt.imsave(..., cmap="viridis")`` used to produce.
"""

from __future__ import annotations

import numpy as np

__all__ = ["VIRIDIS", "GRAY", "COLORMAPS", "colorize"]

_VIRIDIS_HEX = """
    440154 440256 450457 450559 46075a 46085c 460a5d 460b5e
    470d60 470e61 471063 471164 471365 481467 481668 481769
    48186a 481a6c 481b6d 481c6e 481d6f 481f70 482071 482173
    482374 482475 482576 482677 482878 482979 472a7a 472c7a
    472d7b 472e7c 472f7d 46307e 46327e 46337f 463480 453581
    453781 453882 443983 443a83 443b84 433d84 433e85 423f85
    424086 424186 414287 414487 404588 404688 3f4788 3f4889
    3e4989 3e4a89 3e4c8a 3d4d8a 3d4e8a 3c4f8a 3c508b 3b518b
    3b528b 3a538b 3a548c 39558c 39568c 38588c 38598c 375a8c
    375b8d 365c8d 365d8d 355e8d 355f8d 34608d 34618d 33628d
    33638d 32648e 32658e 31668e 31678e 31688e 30698e 306a8e
    2f6b8e 2f6c8e 2e6d8e 2e6e8e 2e6f8e 2d708e 2d718e 2c718e
    2c728e 2c738e 2b748e 2b758e 2a768e 2a778e 2a788e 29798e
    297a8e 297b8e 287c8e 287d8e 277e8e 277f8e 27808e 26818e
    26828e 26828e 25838e 25848e 25858e 24868e 24878e 23888e
    23898e 238a8d 228b8d 228c8d 228d8d 218e8d 218f8d 21908d
    21918c 20928c 20928c 20938c 1f948c 1f958b 1f968b 1f978b
    1f988b 1f998a 1f9a8a 1e9b8a 1e9c89 1e9d89 1f9e89 1f9f88
    1fa088 1fa188 1fa187 1fa287 20a386 20a486 21a585 21a685
    22a785 22a884 23a983 24aa83 25ab82 25ac82 26ad81 27ad81
    28ae80 29af7f 2ab07f 2cb17e 2db27d 2eb37c 2fb47c 31b57b
    32b67a 34b679 35b779 37b878 38b977 3aba76 3bbb75 3dbc74
    3fbc73 40bd72 42be71 44bf70 46c06f 48c16e 4ac16d 4cc26c
    4ec36b 50c46a 52c569 54c568 56c667 58c765 5ac864 5cc863
    5ec962 60ca60 63cb5f 65cb5e 67cc5c 69cd5b 6ccd5a 6ece58
    70cf57 73d056 75d054 77d153 7ad151 7cd250 7fd34e 81d34d
    84d44b 86d549 89d548 8bd646 8ed645 90d743 93d741 95d840
    98d83e 9bd93c 9dd93b a0da39 a2da37 a5db36 a8db34 aadc32
    addc30 b0dd2f b2dd2d b5de2b b8de29 bade28 bddf26 c0df25
    c2df23 c5e021 c8e020 cae11f cde11d d0e11c d2e21b d5e21a
    d8e219 dae319 dde318 dfe318 e2e418 e5e419 e7e419 eae51a
    ece51b efe51c f1e51d f4e61e f6e620 f8e621 fbe723 fde725
"""

VIRIDIS = np.array(
    [[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in _VIRIDIS_HEX.split()],
    dtype=np.uint8)
"""``(256, 3)`` uint8 RGB table, dark blue at 0 through yellow at 255."""

GRAY = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
"""``(256, 3)`` uint8 identity table, for a plain grayscale rendering."""

COLORMAPS = {"viridis": VIRIDIS, "gray": GRAY}


def colorize(focus_map, colormap="viridis"):
    """Turn an integer focus map into an RGB image for viewing.

    The plane indices actually present are stretched across the whole table, so
    the colours show relative depth within one stack. Two stacks with different
    plane counts are not comparable by colour; use the raw indices for that.

    Parameters
    ----------
    focus_map : np.ndarray
        ``(H, W)`` integer array of selected focal-plane indices, as returned
        by :func:`pvtedof.fuse`.
    colormap : str or np.ndarray
        A name from :data:`COLORMAPS`, or your own ``(N, 3)`` uint8 table.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` uint8, ready for ``PIL.Image.fromarray``.
    """
    lut = COLORMAPS[colormap] if isinstance(colormap, str) else np.asarray(colormap)
    size = len(lut)
    fm = np.asarray(focus_map)
    low = int(fm.min())
    span = max(int(fm.max()) - low, 1)

    # matplotlib scales the normalised value by N and truncates, mapping the
    # single value 1.0 down into the last bin rather than past the end.
    index = ((fm - low) * (size / span)).astype(np.intp)
    np.clip(index, 0, size - 1, out=index)
    return lut[index]
