"""Pure phase-space portraits: W̃_θ = W * K_θ and the family integral W̃.

This module factors the convolved-portrait computation out of
``figures/panels.py`` so it can be consumed without importing matplotlib.
``tilde_W_heatmap`` in panels.py is a thin plotting wrapper around
``tilde_W_portrait`` here; the two return bit-identical arrays because the
kernel construction (centered at the integration-grid midpoint for
``fftconvolve(mode="same")`` alignment) and the θ sampling
(``linspace(0, π, n_theta, endpoint=False)``) are shared.

Nothing in this module imports matplotlib, so the similarity battery in
``metrics.py`` can build W̃ with numpy/scipy alone.
"""

from __future__ import annotations

import numpy as np

from .geometry import Capacity, quantum_blob_at
from .convolve import convolve_W_with_K
from .kernels import K_theta_mesh
from .state import State


def _capacity_at_grid_center(state: State) -> Capacity:
    """capacity anchored at the integration-grid midpoint.

    ``fftconvolve(mode="same")`` aligns the kernel's array center with W's
    array center, so the kernel must be built centered on the grid
    midpoint rather than on ``state.overlay_center_x``. This matches the
    convention in ``panels.P_theta_cross_section`` and
    ``panels.tilde_W_heatmap`` exactly.
    """
    assert state.x_int is not None and state.p_int is not None
    x_mid = 0.5 * (state.x_int[0] + state.x_int[-1])
    p_mid = 0.5 * (state.p_int[0] + state.p_int[-1])
    return Capacity(
        Delta_x=state.rs.Delta_x,
        Delta_p=state.rs.Delta_p,
        center=(x_mid, p_mid),
    )


def tilde_W_theta(state: State, theta: float) -> np.ndarray:
    """The single-angle convolved portrait W̃_θ = W * K_θ on the full grid."""
    assert state.W is not None and state.x_int is not None and state.p_int is not None
    capacity = _capacity_at_grid_center(state)
    xx, pp = np.meshgrid(state.x_int, state.p_int, indexing="ij")
    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])
    blob = quantum_blob_at(theta, capacity, hbar=state.hbar)
    K = K_theta_mesh(blob, xx, pp, hbar=state.hbar)
    return convolve_W_with_K(state.W, K, dx, dp)


def tilde_W_portrait(state: State, n_theta: int = 360) -> np.ndarray:
    """The convolved portrait W̃ = (1/N_θ) Σ_θ (W * K_θ), full grid.

    Bit-identical to the array returned by
    ``figures.panels.tilde_W_heatmap`` for the same ``n_theta``: same
    grid-centered kernels, same ``θ ∈ [0, π)`` sampling with the endpoint
    excluded (K_θ has the Z₂ symmetry K_θ = K_{θ+π}).
    """
    assert state.W is not None and state.x_int is not None and state.p_int is not None
    capacity = _capacity_at_grid_center(state)
    xx, pp = np.meshgrid(state.x_int, state.p_int, indexing="ij")
    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])

    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    tilde_W = np.zeros_like(state.W)
    for theta in thetas:
        blob = quantum_blob_at(theta, capacity, hbar=state.hbar)
        K = K_theta_mesh(blob, xx, pp, hbar=state.hbar)
        tilde_W += convolve_W_with_K(state.W, K, dx, dp)
    tilde_W /= n_theta
    return tilde_W


def family_averaged_kernel(state: State, n_theta: int = 360) -> np.ndarray:
    """The angle-averaged kernel K̄ = (1/N_θ) Σ_θ K_θ on the full grid.

    Because convolution is linear, W̃ = W * K̄ exactly, so K̄ is the single
    effective smoothing kernel of the portrait. Its (symplectic) Fourier
    transform is the transfer function relating W and W̃; see
    ``metrics.transfer_function``.
    """
    assert state.x_int is not None and state.p_int is not None
    capacity = _capacity_at_grid_center(state)
    xx, pp = np.meshgrid(state.x_int, state.p_int, indexing="ij")
    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    K_bar = np.zeros_like(xx)
    for theta in thetas:
        blob = quantum_blob_at(theta, capacity, hbar=state.hbar)
        K_bar += K_theta_mesh(blob, xx, pp, hbar=state.hbar)
    K_bar /= n_theta
    return K_bar
