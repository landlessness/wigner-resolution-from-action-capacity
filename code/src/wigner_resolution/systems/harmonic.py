"""Harmonic oscillator eigenstates: Hermite-Gaussian wavefunctions.

In natural units (ℏ = m = ω = 1):

    ψ_n(x) = (2ⁿ n! √π)^(-1/2) H_n(x) exp(-x²/2)

with H_n the physicists' Hermite polynomial. Energies E_n = n + 1/2.
For state n: Δx = Δp = √(2n+1), so A/(h/2) = 2n+1.

Implementation uses scipy.special.eval_hermite plus log-space normalization
to handle high n without n! overflow.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_hermite, gammaln

from ..state import DisplayWindow, State, build_state


def harmonic_psi(n: int, x: np.ndarray) -> np.ndarray:
    """Normalized harmonic oscillator eigenstate ψ_n(x).

    log(2ⁿ n! √π) = n log 2 + log Γ(n+1) + (1/2) log π.
    """
    log_norm = 0.5 * (n * np.log(2) + gammaln(n + 1) + 0.5 * np.log(np.pi))
    Hn = eval_hermite(n, x)
    return np.exp(-x ** 2 / 2 - log_norm) * Hn


def harmonic_state(
    n: int = 1,
    *,
    name: str | None = None,
    hbar: float = 1.0,
) -> State:
    """Build the n-th harmonic oscillator eigenstate."""
    if n < 0:
        raise ValueError("n must be non-negative")

    # Sampling grid: wide enough to capture the classical turning points at
    # ±√(2n+1) plus a generous tail.
    classical_turning = np.sqrt(2 * n + 1)
    x_max = max(5.0 * classical_turning, 15.0)
    # Resolution: ~50 samples per node for n=100 (R reference uses dx=0.02
    # there). For lower n, dx=0.01 is plenty.
    dx = 0.02 if n > 50 else 0.01
    n_pts = int(2 * x_max / dx) + 1
    x_psi = np.linspace(-x_max, x_max, n_pts)
    psi = harmonic_psi(n, x_psi)

    # Display window: build_state expands to fit the extended cell, picks
    # round-number ticks around the cell center.
    window = DisplayWindow(x_lim=0.0, p_lim=0.0)

    return build_state(
        name=name or f"harmonic_n{n}",
        psi=psi,
        x_grid_psi=x_psi,
        window=window,
        hbar=hbar,
    )
