"""PVT-EDOF: Pyramid Vision Transformer based extended depth of focus.

Reference implementation accompanying

    R. O. Dogan, H. Dogan, A. Yilmaz and S. F. Sezen,
    "An effective pyramid vision transformer-based extended depth of focus
    technique for optimally focused microscopic imaging",
    The Journal of Supercomputing 82, 634 (2026).
    https://doi.org/10.1007/s11227-026-08761-6

Typical use::

    from pvtedof import fuse, load_model, load_stack

    model = load_model("checkpoints/pvtedof.pt", device="cuda")
    rgb, gray = load_stack("data/synthetic_demo")
    all_in_focus, focus_map = fuse(model, rgb, gray, device="cuda")

``fuse`` runs the whole of Module 2, seam handling included; see
:mod:`pvtedof.fusion` for why the subimage grid leaves no trace.

Two command-line entry points cover the same ground::

    python -m pvtedof.train --dataset /path/to/coco/train2017   # pretraining
    python -m pvtedof --stack data/synthetic_demo --out results # fusion
"""

from .checkpoints import build_model, load_model, load_state_dict, resolve_device
from .colormap import VIRIDIS, colorize
from .data import InputScale, load_stack, stack_paths, to_luminance
from .focus_measures import MEASURES, spatial_frequency
from .fusion import (
    DEFAULT_PAD,
    DEFAULT_STRIDE,
    TILE,
    decision_map,
    decision_map_batched,
    decision_map_grid,
    fuse,
    select_pixels,
    throughput_sweep,
)
from .model import ModelConfig, PVTEDOF

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "ModelConfig",
    "PVTEDOF",
    "build_model",
    "load_model",
    "load_state_dict",
    "resolve_device",
    "InputScale",
    "load_stack",
    "stack_paths",
    "to_luminance",
    "MEASURES",
    "spatial_frequency",
    "VIRIDIS",
    "colorize",
    "TILE",
    "DEFAULT_STRIDE",
    "DEFAULT_PAD",
    "fuse",
    "decision_map",
    "decision_map_batched",
    "decision_map_grid",
    "select_pixels",
    "throughput_sweep",
]
