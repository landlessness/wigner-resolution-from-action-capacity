"""Squeezed vacuum state, row 1 of Fig. 3.

In natural units (ℏ = m = ω = 1), the position-squeezed vacuum at squeezing
parameter r is

    ψ_sq(x) = (π σ_x²)^(-1/4) exp(-x² / (2 σ_x²))

with σ_x = e^(-r) / √2. This gives Δx = e^(-r)/√2, Δp = √2 e^r — Heisenberg-
saturated for every r.
"""

from __future__ import annotations

import numpy as np

from ..state import DisplayWindow, State, build_state


def squeezed_vacuum_state(
    r: float = 0.5,
    *,
    name: str | None = None,
    hbar: float = 1.0,
) -> State:
    """Build a squeezed vacuum state at squeezing parameter r."""
    sigma_x = np.exp(-r) / np.sqrt(2)

    # Wavefunction sampling grid: dense enough for FFT-derived <p²> to be
    # accurate, wide enough that ψ has decayed to floating-point noise at
    # the edges.
    x_psi = np.linspace(-15.0, 15.0, 6001)
    psi = (1 / (np.pi * sigma_x ** 2)) ** 0.25 * np.exp(-x_psi ** 2 / (2 * sigma_x ** 2))

    # Display window: minimal limits + empty ticks. build_state expands
    # the window to fit the extended cell and chooses round-number ticks.
    window = DisplayWindow(x_lim=0.0, p_lim=0.0)

    return build_state(
        name=name or f"squeezed_vacuum_r{r:g}",
        psi=psi,
        x_grid_psi=x_psi,
        window=window,
        hbar=hbar,
    )
