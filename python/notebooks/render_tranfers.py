"""Render the Supplement transfer-function validation figure.

The analytic resolution section (Sec. S2) predicts that the portrait is a
single convolution W~ = W * Kbar whose chord-domain multiplier is the
parameter-free, isotropic transfer function

    Kbar_hat(|xi|) = exp[-(|xi|^2/8)(1 + (a~/(h/2))^2)]
                       * I0[(|xi|^2/8)(1 - (a~/(h/2))^2)],

with the single ratio a~/(h/2) = (h/2)/A fixed by the state. This script
measures that transfer on the twelve states and overlays the prediction
with no fitting, in the SCALED chord where Sec. S2 predicts isotropy,

    xi = hypot(k_x * Delta_x, k_p * Delta_p).

Two checks per state:
  kernel-level   |FFT(Kbar)| vs the exact I0 form, unbinned 2D  -> the math
  portrait-level Re<W^* W~>/<|W|^2> per chord bin, on real states

Output: tex/figures/transfer_validation.pdf
The printed table lets the Supplement numbers be cross-checked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import i0e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wigner_resolution.plotstyle import use_prl_style, DOUBLE_WIDTH
from wigner_resolution.portrait import tilde_W_portrait, family_averaged_kernel
from wigner_resolution.systems.cat import cat_state
from wigner_resolution.systems.double_well import double_well_state
from wigner_resolution.systems.harmonic import harmonic_state
from wigner_resolution.systems.kerr import kerr_state
from wigner_resolution.systems.morse import morse_state
from wigner_resolution.systems.squeezed_vacuum import squeezed_vacuum_state

HERE = Path(__file__).resolve().parent
FIG = HERE.parent.parent / "tex" / "figures"

use_prl_style(use_tex=True)


# ----------------------------------------------------------------------
# analytic prediction (Eq. S-Kbar), normalized to 1 at zero chord
# ----------------------------------------------------------------------
def kbar_analytic(xi: np.ndarray, rt: float) -> np.ndarray:
    """exp[-(xi^2/8)(1+rt^2)] I0[(xi^2/8)(1-rt^2)], rt = a~/(h/2) = (h/2)/A.

    Written through i0e(a)=exp(-a)I0(a) to stay finite at large chord:
        = exp(-2 rt^2 z) * i0e((1-rt^2) z),  z = xi^2/8.
    rt=1 -> exp(-xi^2/4) (vacuum Gaussian); rt->0 -> i0e(z) (soft edge).
    """
    z = xi ** 2 / 8.0
    return np.exp(-2.0 * rt ** 2 * z) * i0e((1.0 - rt ** 2) * z)


def scaled_chord(state, n_bins: int = 90):
    """Scaled-chord magnitude grid and radial-bin index."""
    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])
    kx = np.fft.fftshift(np.fft.fftfreq(len(state.x_int), d=dx)) * 2 * np.pi
    kp = np.fft.fftshift(np.fft.fftfreq(len(state.p_int), d=dp)) * 2 * np.pi
    KX, KP = np.meshgrid(kx, kp, indexing="ij")
    Dx, Dp = state.rs.Delta_x, state.rs.Delta_p
    XI = np.hypot(KX * Dx, KP * Dp)
    xi_max = float(np.min([kx.max() * Dx, kp.max() * Dp]))
    edges = np.linspace(0.0, xi_max, n_bins + 1)
    idx = np.digitize(XI.ravel(), edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    return XI, idx, centers, n_bins


def transfer_portrait(state, Wt, idx, n_bins):
    """Power-weighted real transfer Re<W^* W~>/<|W|^2> per chord bin.

    W~ = W * Kbar with Kbar real-symmetric, so W^* W~ = Kbar_hat |W|^2 and
    the power-weighted ratio returns Kbar_hat, robust where |W| is small.
    """
    cW = np.fft.fftshift(np.fft.fft2(state.W)).ravel()
    cT = np.fft.fftshift(np.fft.fft2(Wt)).ravel()
    num = np.real(np.conj(cW) * cT)
    den = np.abs(cW) ** 2
    T = np.full(n_bins, np.nan)
    P = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            d = den[m].sum()
            T[b] = num[m].sum() / d if d > 0 else np.nan
            P[b] = d
    return T, P


def build_states():
    states = [
        harmonic_state(n=0), harmonic_state(n=1), harmonic_state(n=2),
        harmonic_state(n=20), squeezed_vacuum_state(r=0.5), morse_state(n=8),
        double_well_state(n=5), kerr_state(), cat_state(2), cat_state(3),
        cat_state(4, variant="diag"), cat_state(4, variant="axis"),
    ]
    labels = ["QHO $n=0$", "QHO $n=1$", "QHO $n=2$", "QHO $n=20$", "Squeezed",
              "Morse", "Double-well", "Kerr", "2-cat", "3-cat",
              "4-cat diag", "Compass"]
    return states, labels


def main(n_theta: int = 180) -> None:
    states, labels = build_states()
    fig, axes = plt.subplots(3, 4, figsize=(DOUBLE_WIDTH, DOUBLE_WIDTH * 0.72))

    print(f"{'state':12s} {'A/(h/2)':>8s} {'a~/(h/2)':>9s} "
          f"{'kernel 2D':>11s} {'portrait rms':>13s}")
    krn, prt = [], []
    for ax, lab, s in zip(axes.ravel(), labels, states):
        rt = float(s.hbar / (s.rs.Delta_x * s.rs.Delta_p))   # a~/(h/2) = (h/2)/A
        Wt = tilde_W_portrait(s, n_theta=n_theta)
        Kbar = family_averaged_kernel(s, n_theta=n_theta)
        XI, idx, xi, n_bins = scaled_chord(s)

        T2, P = transfer_portrait(s, Wt, idx, n_bins)
        pred = kbar_analytic(xi, rt)
        core = P > 1e-2 * np.nanmax(P)
        xi_core = float(np.nanmax(xi[core]))

        H = np.abs(np.fft.fftshift(np.fft.fft2(Kbar))); H = H / H.max()
        disk = (XI < xi_core) & np.isfinite(H)
        kernel_2d = float(np.max(np.abs(H[disk] - kbar_analytic(XI, rt)[disk])))
        rms = float(np.sqrt(np.nanmean(((T2 - pred)[core]) ** 2)))
        krn.append(kernel_2d); prt.append(rms)
        print(f"{lab:12s} {s.rs.A_over_h_half:8.2f} {rt:9.4f} "
              f"{kernel_2d:11.1e} {rms:13.2e}")

        xx = np.linspace(0, xi_core * 1.05, 400)
        ax.plot(xx, kbar_analytic(xx, rt), "k-", lw=1.2, zorder=1)
        ax.plot(xi[core], T2[core], "o", ms=3.0, color="crimson",
                mfc="none", mew=0.8, zorder=3)
        ax.axhline(0, color="0.8", lw=0.4)
        ax.set_title(rf"{lab}, $A/(h/2)={s.rs.A_over_h_half:.1f}$", fontsize=7)
        ax.set_xlim(0, max(5.0, xi_core * 1.05))
        ax.set_ylim(-0.04, 1.05)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"scaled chord $|\xi|$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"transfer $\widehat{\bar K}(|\xi|)$")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "transfer_validation.pdf")
    print(f"\nwrote {FIG / 'transfer_validation.pdf'}")
    print(f"kernel-level worst (unbinned 2D): {max(krn):.1e}")
    print(f"portrait-level rms: median {np.median(prt):.2e}, worst {max(prt):.2e}")


if __name__ == "__main__":
    main()