"""P_θ(x, p) = (W * K_θ)(x, p) — the convolved phase-space density.

Convolving a Wigner function with a Gaussian of action ≥ h/2 yields a
non-negative function (Cartwright, Physica A 83, 210 (1976); see also
de Gosson 2005, Cordero et al. 2018). The bitangent kernel K_θ has
action h/2 exactly, so P_θ ≥ 0 everywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve


def convolve_W_with_K(
    W: np.ndarray,
    K: np.ndarray,
    dx: float,
    dp: float,
) -> np.ndarray:
    """2D convolution of W with kernel K on a uniform grid.

    Both arrays must share the same shape. The kernel K must be centered at
    its array midpoint (the multivariate_normal builder in kernels.py with
    center=(0,0) produces this). The output has the same shape as W; the
    `dx * dp` factor converts the discrete sum into a Riemann integral.
    """
    if W.shape != K.shape:
        raise ValueError(f"W shape {W.shape} != K shape {K.shape}")
    return fftconvolve(W, K, mode="same") * dx * dp


def cross_section_at_p0(
    M: np.ndarray,
    x_grid: np.ndarray,
    p_grid: np.ndarray,
    x_display: np.ndarray,
) -> np.ndarray:
    """Interpolate a 2D phase-space density at p=0, then resample on x_display.

    Used to extract the W(x, 0) and P_θ(x, 0) cross-sections shown in the
    center and right columns of Figs. 3 and 4.
    """
    # First: find or interpolate to p = 0.
    p0_idx = int(np.argmin(np.abs(p_grid)))
    if np.isclose(p_grid[p0_idx], 0.0):
        row = M[:, p0_idx]
    else:
        # Linear interp between the two p_grid samples bracketing 0.
        f_p = interp1d(
            p_grid, M, axis=1, kind="linear", bounds_error=False, fill_value=0.0
        )
        row = f_p(0.0)
    # Then: resample row from x_grid onto x_display.
    f_x = interp1d(
        x_grid, row, kind="linear", bounds_error=False, fill_value=0.0
    )
    return f_x(x_display)
