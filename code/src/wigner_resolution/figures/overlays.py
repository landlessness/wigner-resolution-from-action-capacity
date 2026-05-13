"""Matplotlib ellipse patches for phase-space cells.

`Ellipse` takes width and height as full extents (not semi-axes) and an
angle in degrees, so we convert from the semi-axes/radians convention used
throughout the rest of the package.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Ellipse

from ..cells import ExtendedCell, SqueezedCell


def extended_cell_patch(
    cell: ExtendedCell,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = 0.6,
    **kwargs,
) -> Ellipse:
    """Matplotlib ellipse patch for an extended cell (axis-aligned)."""
    return Ellipse(
        xy=cell.center,
        width=2 * cell.Delta_x,
        height=2 * cell.Delta_p,
        angle=0.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )


def squeezed_cell_patch(
    cell: SqueezedCell,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = 0.6,
    **kwargs,
) -> Ellipse:
    """Matplotlib ellipse patch for a squeezed cell rotated by θ."""
    return Ellipse(
        xy=cell.center,
        width=2 * cell.delta_parallel,
        height=2 * cell.delta_perp,
        angle=np.degrees(cell.theta),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )
