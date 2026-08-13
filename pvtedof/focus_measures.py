"""Focus measures evaluated on the PVT deep feature maps.

Every function takes a feature tensor ``fs`` of shape ``(n, t, H, W)`` - the
encoder output for the ``n`` focal planes of one subimage - and returns a
per-pixel decision map of shape ``(H, W)`` whose entries are the index of the
selected focal plane, i.e. ``argmax`` over the ``n`` planes.

:func:`spatial_frequency` is the measure used by PVT-EDOF (Eq. 7-9 of the
paper). The others are the alternatives of the focus-measure ablation
(Table 5): the same selection rule applied to a different measure, on raw
images as well as deep features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "spatial_frequency",
    "tenengrad",
    "energy_of_laplacian",
    "wavelet",
    "variance",
    "haar_energy",
    "MEASURES",
]


def spatial_frequency(fs, device, kernel_radius=5):
    """Spatial Frequency focus measure (Eq. 7-8), the measure used by PVT-EDOF.

    Row and column first-order differences are squared, summed over an
    ``(2*kernel_radius+1)`` square neighbourhood - 11x11 by default, matching
    the paper - and accumulated over the ``t`` feature channels.

    Parameters
    ----------
    fs : torch.Tensor
        Feature maps of shape ``(n, t, H, W)``.
    device : torch.device
        Device on which to run the convolutions.
    kernel_radius : int
        Half-width of the aggregation window ``Omega(x, y)``.

    Returns
    -------
    tuple
        ``(focal_value_matrix, decision_map)`` where ``focal_value_matrix`` has
        shape ``(n, H, W)`` and ``decision_map`` is an ``(H, W)`` int array of
        selected plane indices.
    """
    fs = fs.to(device)
    _, channels, _, _ = fs.shape

    r_shift_kernel = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
    b_shift_kernel = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                                  dtype=torch.float32, device=device).view(1, 1, 3, 3)
    r_shift_kernel = r_shift_kernel.repeat(channels, 1, 1, 1)
    b_shift_kernel = b_shift_kernel.repeat(channels, 1, 1, 1)

    fs_r_shift = F.conv2d(fs, r_shift_kernel, padding=1, groups=channels)
    fs_b_shift = F.conv2d(fs, b_shift_kernel, padding=1, groups=channels)

    fs_grad = torch.pow(fs_r_shift - fs, 2) + torch.pow(fs_b_shift - fs, 2)

    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((channels, 1, kernel_size, kernel_size),
                            dtype=torch.float32, device=device)
    fs_sf = torch.sum(
        F.conv2d(fs_grad, add_kernel, padding=kernel_size // 2, groups=channels), dim=1)

    decision_map = torch.argmax(fs_sf, dim=0).squeeze().cpu().numpy().astype(int)
    return fs_sf, decision_map


def tenengrad(fs, device):
    """Tenengrad (Sobel gradient magnitude) focus measure."""
    fs = fs.to(device)
    _, channels, _, _ = fs.shape

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(channels, 1, 1, 1)

    grad_x = F.conv2d(fs, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(fs, sobel_y, padding=1, groups=channels)
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    scores = grad_magnitude.mean(dim=1)
    return torch.argmax(scores, dim=0).cpu().numpy().astype(int)


def energy_of_laplacian(fs, device):
    """Energy of Laplacian focus measure."""
    fs = fs.to(device)
    _, channels, _, _ = fs.shape

    laplacian_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)
    laplacian_filter = laplacian_filter.repeat(channels, 1, 1, 1)

    laplacian = F.conv2d(fs, laplacian_filter, padding=1, groups=channels)
    energy = torch.sum(torch.abs(laplacian), dim=1)
    return torch.argmax(energy, dim=0).cpu().numpy().astype(int)


def haar_energy(tensor, device):
    """Energy of four Haar-like 3x3 high-pass responses, summed over channels.

    The kernels are 3x3 so that the response keeps the input resolution, which
    is what the per-pixel selection rule needs. Returns an ``(n, H, W)`` energy
    tensor.
    """
    kernels = [
        torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32, device=device) / 3,
        torch.tensor([[1, 1, 1], [-1, -1, -1], [1, 1, 1]], dtype=torch.float32, device=device) / 3,
        torch.tensor([[1, -1, 1], [1, -1, 1], [1, -1, 1]], dtype=torch.float32, device=device) / 3,
        torch.tensor([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=torch.float32, device=device) / 3,
    ]

    n, channels, height, width = tensor.shape
    energy = torch.zeros((n, height, width), dtype=torch.float32, device=device)
    for kernel in kernels:
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        filtered = F.conv2d(tensor, kernel, stride=1, padding=1, groups=channels)
        energy += torch.sum(filtered ** 2, dim=1)
    return energy


def wavelet(fs, device):
    """Wavelet-coefficient-energy focus measure (see :func:`haar_energy`)."""
    fs = fs.to(device)
    return torch.argmax(haar_energy(fs, device), dim=0).cpu().numpy().astype(int)


def variance(fs, device):
    """Across-channel variance focus measure.

    With a single input channel (the raw-image arm of the ablation) the
    variance is identically zero, so the plane index is degenerate; the ablation
    therefore reports this measure on deep features only.
    """
    fs = fs.to(device)
    variances = torch.var(fs, dim=1, keepdim=False)
    return torch.argmax(variances, dim=0).cpu().numpy().astype(int)


def _spatial_frequency_map(fs, device):
    return spatial_frequency(fs, device)[1]


MEASURES = {
    "spatial_frequency": _spatial_frequency_map,
    "tenengrad": tenengrad,
    "energy_of_laplacian": energy_of_laplacian,
    "wavelet": wavelet,
    "variance": variance,
}
"""Name -> ``f(features, device) -> decision_map``, for selecting a measure by name."""
