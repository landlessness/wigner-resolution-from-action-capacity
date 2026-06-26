"""Matplotlib ellipse patches for phase-space regions.

`Ellipse` takes width and height as full extents (not semi-axes) and an
angle in degrees, so we convert from the semi-axes/radians convention used
throughout the rest of the package.

All drawn lines in the data figures — overlays, cross-section
traces, zero baselines — share a single line-width constant so a
tuning change flows through the whole figure consistently. Tune
``LINEWIDTH`` here.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Ellipse

from ..geometry import Capacity, QuantumBlob, Resolution


# Single line-width constant governing every drawn line in the data
# figures: overlay ellipse strokes, cross-section data traces, and
# zero baselines. At typical PRA panel sizes 0.3-0.5 pt reads as a
# present-but-non-competing line weight. Lower if the strokes dominate
# the data; raise if lines vanish in print.
LINEWIDTH: float = 0.4


def capacity_patch(
    capacity: Capacity,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = LINEWIDTH,
    **kwargs,
) -> Ellipse:
    """Matplotlib ellipse patch for a capacity A (axis-aligned)."""
    return Ellipse(
        xy=capacity.center,
        width=2 * capacity.Delta_x,
        height=2 * capacity.Delta_p,
        angle=0.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )


def quantum_blob_patch(
    blob: QuantumBlob,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = LINEWIDTH,
    **kwargs,
) -> Ellipse:
    """Matplotlib ellipse patch for a quantum blob β_θ rotated by θ."""
    return Ellipse(
        xy=blob.center,
        width=2 * blob.r_parallel,
        height=2 * blob.r_perp,
        angle=np.degrees(blob.theta),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )


def resolution_patch(
    resolution: Resolution,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = LINEWIDTH,
    **kwargs,
) -> Ellipse:
    """Matplotlib ellipse patch for the resolution a.

    Axis-aligned ellipse with semi-axes (δx, δp). The common interior
    of the quantum blob family inscribed in the capacity.
    """
    return Ellipse(
        xy=resolution.center,
        width=2 * resolution.delta_x,
        height=2 * resolution.delta_p,
        angle=0.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )
