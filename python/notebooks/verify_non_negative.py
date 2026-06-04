"""Verify that the convolved portrait W̃ is non-negative for every state.

The manuscript's central numerical claim is that the convolved portrait

    W̃(x, p) = (1/N_θ) Σ_θ (W * K_θ)(x, p),   θ ∈ [0, π),

is non-negative everywhere. Each quantum kernel K_θ is the Wigner function
of a de Gosson quantum blob of action h/2, so each convolution W * K_θ is
non-negative (Cartwright, Physica A 83, 210 (1976); de Gosson 2005; Cordero
et al. 2018), and a mean of non-negative functions is non-negative. This
script confirms that property numerically across the twelve states of the
manuscript, on the same integration grid and via the same convolution path
the figures use (figures/panels.tilde_W_heatmap).

For each state it recomputes W̃ at the publication angular sampling
(N_θ = 360) and reports

    W̃_min            the most negative value of W̃ anywhere on the grid,
    W̃_min / peak     that value relative to the portrait's peak,
    norm             ∫ W̃ dx dp, which the construction preserves at 1.

A state passes when W̃_min / peak ≥ −REL_TOL, i.e. any negativity is at
the floating-point floor rather than physical. The script is fail-fast:
it exits non-zero if any state's portrait dips below that floor.

Run:
    cd python
    uv run python notebooks/verify_non_negative.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

from wigner_resolution.cells import HeisenbergCell, quantum_blob_at
from wigner_resolution.convolve import convolve_W_with_K
from wigner_resolution.kernels import K_theta_mesh
from wigner_resolution.systems.cat import cat_state
from wigner_resolution.systems.double_well import double_well_state
from wigner_resolution.systems.harmonic import harmonic_state
from wigner_resolution.systems.kerr import kerr_state
from wigner_resolution.systems.morse import morse_state
from wigner_resolution.systems.squeezed_vacuum import squeezed_vacuum_state

# Publication angular sampling of the family integral (matches the render
# scripts' tilde_W_n_theta=360).
N_THETA = 360

# A portrait passes when its deepest value, relative to its peak, sits no
# lower than this. The construction is non-negative exactly; realized
# negativity is floating-point convolution noise, ~1e-16 relative. 1e-9
# is far below any physical negativity yet far above fp noise, so it
# catches a broken construction without flagging roundoff.
REL_TOL = 1e-9

# The twelve states, in the order of the three data figures
# (render_harmonic, render_eigen, render_cat). Each entry is a no-argument
# thunk so construction happens inside the warnings-suppressed block.
STATE_BUILDERS = [
    (lambda: harmonic_state(n=0), "QHO n=0"),
    (lambda: harmonic_state(n=1), "QHO n=1"),
    (lambda: harmonic_state(n=2), "QHO n=2"),
    (lambda: harmonic_state(n=20), "QHO n=20"),
    (lambda: squeezed_vacuum_state(r=0.5), "squeezed vacuum r=0.5"),
    (lambda: morse_state(n=8), "Morse n=8"),
    (lambda: double_well_state(n=5), "double-well n=5"),
    (lambda: kerr_state(), "Kerr crescent"),
    (lambda: cat_state(2), "2-cat"),
    (lambda: cat_state(3), "3-cat"),
    (lambda: cat_state(4, variant="diag"), "4-cat"),
    (lambda: cat_state(4, variant="axis"), "Zurek compass"),
]


def compute_tilde_W(state, n_theta: int = N_THETA) -> np.ndarray:
    """Recompute the convolved portrait W̃ on the state's integration grid.

    Identical to figures/panels.tilde_W_heatmap but headless: the kernel
    is centered at the integration-grid midpoint so fftconvolve(mode="same")
    aligns it with W, the family is summed over θ ∈ [0, π) at evenly spaced
    angles, and the mean is taken. No display window, no overlays.
    """
    assert state.W is not None and state.x_int is not None and state.p_int is not None

    x_mid = 0.5 * (state.x_int[0] + state.x_int[-1])
    p_mid = 0.5 * (state.p_int[0] + state.p_int[-1])
    heisenberg = HeisenbergCell(
        Delta_x=state.rs.Delta_x,
        Delta_p=state.rs.Delta_p,
        center=(x_mid, p_mid),
    )

    xx, pp = np.meshgrid(state.x_int, state.p_int, indexing="ij")
    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])

    thetas = np.linspace(0.0, np.pi, n_theta, endpoint=False)
    tilde_W = np.zeros_like(state.W)
    for theta in thetas:
        blob = quantum_blob_at(theta, heisenberg, hbar=state.hbar)
        K = K_theta_mesh(blob, xx, pp, hbar=state.hbar)
        tilde_W += convolve_W_with_K(state.W, K, dx, dp)
    tilde_W /= n_theta
    return tilde_W


def check_state(build, label: str) -> bool:
    state = build()
    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])

    tilde_W = compute_tilde_W(state, N_THETA)
    tW_min = float(tilde_W.min())
    tW_peak = float(tilde_W.max())
    rel = tW_min / tW_peak if tW_peak > 0 else float("nan")
    norm = float(np.sum(tilde_W) * dx * dp)

    ok = rel >= -REL_TOL
    print(f"{label:<22}{state.rs.A_over_h_half:>9.3f}"
          f"{state.W.min():>10.4f}{tW_min:>12.2e}{rel:>12.2e}"
          f"{norm:>9.4f}   {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Verifying the convolved portrait W̃ is non-negative across the "
          "twelve states.")
    print(f"angular sampling: N_theta = {N_THETA} in [0, pi);  "
          f"pass tolerance: W̃_min/peak >= {-REL_TOL:g}\n")
    hdr = (f"{'state':<22}{'A/(h/2)':>9}{'W_min':>10}{'W̃_min':>12}"
           f"{'W̃_min/pk':>12}{'norm':>9}   {'verdict'}")
    print(hdr)
    print("-" * (len(hdr) + 2))

    # State construction emits Wigner-norm UserWarnings for the wide-grid
    # FFT states (heavy tails, asymmetric support); those concern grid
    # coverage, not the non-negativity under test, so we silence them.
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for build, label in STATE_BUILDERS:
            results.append(check_state(build, label))

    print("\n" + "=" * 64)
    if all(results):
        print("ALL CHECKS PASS.")
        print("The convolved portrait W̃ is non-negative to floating-point "
              "precision for every state, on the publication grid and "
              "convolution path.")
        return 0
    print("VERIFICATION FAILED. See messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())