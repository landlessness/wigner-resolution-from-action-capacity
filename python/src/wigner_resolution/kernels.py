"""The bitangent kernel family K_θ.

    K_θ(x, p) = (1/πℏ) exp(-x_θ²/r_∥² - p_θ²/r_⊥²)

with rotated coordinates

    x_θ =  x cosθ + p sinθ
    p_θ = -x sinθ + p cosθ.

This is the Wigner function of the bitangent blob a_θ at quadrature
angle θ. Its action is h/2 by construction; its non-negativity follows
from being the Wigner function of a pure Gaussian state.

Implementation: a rotated 2D Gaussian with diagonal covariance
Σ₀ = diag(r_∥²/2, r_⊥²/2) in the rotated frame, then Σ_θ = R(θ) Σ₀ R(θ)ᵀ in
the unrotated phase-space frame. Evaluated via scipy.stats.multivariate_normal.

The prefactor 1/(πℏ) in the displayed kernel comes out of
scipy.stats.multivariate_normal *only* under the reciprocal-axes condition
r_∥ r_⊥ = ℏ. The assertion at the top of K_theta_mesh enforces this.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import multivariate_normal

from .cells import BitangentBlob, bitangent_blob_at, HeisenbergCell

# Tolerance for the reciprocal-axes check. The blob-building routines compute
# r_⊥ = ℏ/r_∥ from the blob's r_∥, so the product should be exact to floating-
# point — a slack tolerance still catches construction bugs without false
# positives.
_RECIPROCAL_TOL = 1e-10


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def K_theta_mesh(
    blob: BitangentBlob,
    xx: np.ndarray,
    pp: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Evaluate K_θ on a 2D mesh (xx, pp).

    `xx` and `pp` are 2D arrays of identical shape from ``np.meshgrid``.
    Returns an array of the same shape.

    Asserts reciprocal axes r_∥ r_⊥ = ℏ. The scipy.stats.multivariate_normal
    prefactor 1/(2π √det Σ) equals 1/(π r_∥ r_⊥), so the manuscript's
    1/(πℏ) prefactor (Eq. 6) is recovered only when this product equals ℏ.
    Violating it would silently produce a kernel that integrates to 1 but
    no longer matches the displayed equation.
    """
    product = blob.r_parallel * blob.r_perp
    if abs(product - hbar) > _RECIPROCAL_TOL:
        raise ValueError(
            f"Reciprocal axes violated: r_∥·r_⊥ = {product:.6e} ≠ ℏ = {hbar:.6e}. "
            "Build the blob via bitangent_blob_at() to ensure r_⊥ = ℏ/r_∥."
        )

    # Build Σ_θ in the unrotated phase-space frame.
    R = _rotation_matrix(blob.theta)
    Sigma_0 = np.diag([blob.r_parallel ** 2 / 2, blob.r_perp ** 2 / 2])
    Sigma_theta = R @ Sigma_0 @ R.T

    pos = np.dstack([xx - blob.center[0], pp - blob.center[1]])
    rv = multivariate_normal(mean=[0.0, 0.0], cov=Sigma_theta)
    return rv.pdf(pos)


def K_theta_from_heisenberg(
    theta: float,
    heisenberg: HeisenbergCell,
    xx: np.ndarray,
    pp: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """Convenience: build the bitangent blob a_θ inscribed in `heisenberg`,
    then evaluate K_θ on (xx, pp)."""
    blob = bitangent_blob_at(theta, heisenberg, hbar=hbar)
    return K_theta_mesh(blob, xx, pp, hbar=hbar)
