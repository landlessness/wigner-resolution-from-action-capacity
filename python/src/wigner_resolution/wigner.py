"""Wigner phase-space density W(x, p) from a sampled ψ(x).

    W(x, p) = (1/πℏ) ∫ ψ*(x + y) ψ(x - y) exp(2ipy/ℏ) dy

FFT method (Leonhardt 1997, Ch. 5): build the autocorrelation matrix
ρ(x, k dx) = ψ(x + k dx) ψ*(x - k dx), then FFT along k.

For real ψ the result is even in p; for complex ψ (cat states) the full
two-sided spectrum is interpolated.

Direct port of the R reference (wigner_density.R). QuTiP's qutip.wigner uses
essentially the same algorithm but expects Fock-basis input; keeping our
own implementation makes it trivial to feed position-basis ψ from any source
(analytic, FD solver, sum of Gaussians) without conversion.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def wigner_fft(
    psi: np.ndarray,
    x_grid: np.ndarray,
    p_grid: np.ndarray,
    hbar: float = 1.0,
) -> np.ndarray:
    """W(x, p) via FFT of the autocorrelation along the shift dimension.

    Parameters
    ----------
    psi
        Wavefunction sampled on ``x_grid`` (complex or real). Re-normalized
        internally; the input scale doesn't matter.
    x_grid
        Uniform position grid.
    p_grid
        Momentum grid at which W is evaluated.
    hbar
        Planck constant in chosen units.

    Returns
    -------
    W of shape ``(len(x_grid), len(p_grid))``.
    """
    nx = len(x_grid)
    np_ = len(p_grid)
    dx = float(x_grid[1] - x_grid[0])

    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm <= 0:
        raise ValueError("ψ has zero norm.")
    psi_n = psi / norm
    psi_c = np.asarray(psi_n, dtype=complex)

    # Build ρ[ix, k] = ψ(x_ix + k dx) ψ*(x_ix - k dx) for k = -k_max .. k_max.
    # Indices outside the support of x_grid are treated as zero.
    k_max = nx - 1
    k_vals = np.arange(-k_max, k_max + 1)
    nk = len(k_vals)

    ix = np.arange(nx)[:, None]               # (nx, 1)
    k = k_vals[None, :]                        # (1, nk)
    ip_idx = ix + k
    im_idx = ix - k
    valid = (ip_idx >= 0) & (ip_idx < nx) & (im_idx >= 0) & (im_idx < nx)
    safe_ip = np.clip(ip_idx, 0, nx - 1)
    safe_im = np.clip(im_idx, 0, nx - 1)

    rho = psi_c[safe_ip] * np.conj(psi_c[safe_im])
    rho[~valid] = 0.0

    # Pad ρ to length M (power of 2) for fine p sampling, with the standard
    # FFT layout: k=0 at index 0, positive k in ascending slots, negative k
    # wrapped to the tail.
    M = int(2 ** np.ceil(np.log2(max(2 * nx, 4 * np_))))
    rho_padded = np.zeros((nx, M), dtype=complex)
    # Positive k including 0: positions k=0..k_max go into indices 0..k_max
    rho_padded[:, : k_max + 1] = rho[:, k_max : nk]
    # Negative k: positions k=-k_max..-1 go into indices M-k_max..M-1
    rho_padded[:, M - k_max :] = rho[:, : k_max]

    fft_full = np.fft.fft(rho_padded, axis=1)

    # The discrete Wigner is W(x, p) = (dx/π) Σ_k ρ(x, k dx) e^{-2 i p k dx}.
    # NumPy's FFT uses e^{-2π i j k / M}, so matching 2 p dx = 2π j / M gives
    # p_j = π j / (M dx).
    half = M // 2 + 1
    # Use the full bilateral spectrum so it works for complex ψ (cat states).
    W_full = (dx / np.pi) * np.real(fft_full)
    p_pos = np.pi * np.arange(half) / (M * dx)
    p_neg = np.pi * (np.arange(half, M) - M) / (M * dx)
    p_full = np.concatenate([p_pos, p_neg])
    order = np.argsort(p_full)
    p_sorted = p_full[order]
    W_sorted = W_full[:, order]

    # Interpolate each row onto the requested p_grid, zeroing outside.
    W_out = np.zeros((nx, np_))
    for ix_row in range(nx):
        f = interp1d(
            p_sorted,
            W_sorted[ix_row],
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )
        W_out[ix_row] = f(p_grid)

    return W_out


def wigner_norm(W: np.ndarray, x_grid: np.ndarray, p_grid: np.ndarray) -> float:
    """Integrated norm of W. Should be ~1 for a normalized ψ."""
    dx = float(x_grid[1] - x_grid[0])
    dp = float(p_grid[1] - p_grid[0])
    return float(np.sum(W) * dx * dp)
