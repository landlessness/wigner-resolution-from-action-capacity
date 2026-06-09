"""Numerical-validation instruments and figures for the Supplement.

Position-based comparison of the bare Wigner function ``W`` and the
non-negative portrait ``W-tilde``, every instrument run identically on both.

Instruments
  coherence_phase   cross-spectral coherence and phase   (positions coincide)
  colocation        positive-maxima matching, offset in delta
  radial_spectrum   radially averaged chord power         (shared support)

Figure builders (each returns a Figure, saved by figures.grid.save_grid)
  build_composite        feature radius in W vs in W-tilde, all states
  build_grid             W-tilde heat map with W's positive maxima, all states
  build_shared_support   chord power of W and W-tilde for one state
  build_ring_colocation  radial profiles of W and W-tilde for one state

The amplitude-weighted inverse-rms bandwidth is intentionally absent. It reads
the deep structure of W but the bright envelope of W-tilde, so it cannot run as
one instrument on both. Position is what the portrait preserves, so position is
what every instrument here reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks

from wigner_resolution.cells import HeisenbergCell, quantum_blob_at, quorum_cell
from wigner_resolution.convolve import convolve_W_with_K
from wigner_resolution.kernels import K_theta_mesh
from wigner_resolution.plotstyle import COLUMN_WIDTH, DOUBLE_WIDTH
from wigner_resolution.state import State


# ----------------------------------------------------------------------
# portrait
# ----------------------------------------------------------------------
def tilde_W_portrait(state: State, n_theta: int = 360) -> np.ndarray:
    """The convolved portrait W-tilde = (1/N) sum_theta (W * K_theta).

    Built from the same low-level pieces as figures.panels.tilde_W_heatmap,
    so it returns the array that panel plots, without importing matplotlib.
    The kernel is centered on the integration-grid midpoint to match
    fftconvolve(mode="same") alignment, and theta runs over [0, pi) with the
    endpoint excluded since K_theta has the Z2 symmetry K_theta = K_{theta+pi}.
    """
    assert state.W is not None and state.x_int is not None and state.p_int is not None
    x_mid = 0.5 * (state.x_int[0] + state.x_int[-1])
    p_mid = 0.5 * (state.p_int[0] + state.p_int[-1])
    heisenberg = HeisenbergCell(
        Delta_x=state.rs.Delta_x, Delta_p=state.rs.Delta_p, center=(x_mid, p_mid),
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


# ----------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------
def _quorum_semiaxis(state: State) -> float:
    """Geometric-mean quorum semi-axis delta of the state's quorum cell."""
    h = HeisenbergCell(Delta_x=state.rs.Delta_x, Delta_p=state.rs.Delta_p)
    q = quorum_cell(h)
    return float(np.sqrt(q.delta_x * q.delta_p))


def coherence_phase(W: np.ndarray, Wt: np.ndarray, state: State,
                    *, n_bins: int = 80) -> tuple[float, float]:
    """In-band magnitude-squared coherence and max phase of the W -> W~ map."""
    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])
    cF = np.fft.fftshift(np.fft.fft2(W))
    cG = np.fft.fftshift(np.fft.fft2(Wt))
    kx = np.fft.fftshift(np.fft.fftfreq(len(state.x_int), d=dx)) * 2 * np.pi
    kp = np.fft.fftshift(np.fft.fftfreq(len(state.p_int), d=dp)) * 2 * np.pi
    KX, KP = np.meshgrid(kx, kp, indexing="ij")
    KR = np.hypot(KX, KP)
    edges = np.linspace(0, min(kx.max(), kp.max()), n_bins + 1)
    idx = np.digitize(KR.ravel(), edges) - 1
    cross = (np.conj(cF) * cG).ravel()
    pF = (np.abs(cF) ** 2).ravel()
    pG = (np.abs(cG) ** 2).ravel()
    g2, ph, wt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        X = cross[m].sum()
        a, c = pF[m].sum(), pG[m].sum()
        g2.append(abs(X) ** 2 / (a * c))
        ph.append(np.angle(X))
        wt.append(a)
    g2, ph, wt = np.array(g2), np.array(ph), np.array(wt)
    band = wt > 1e-2 * wt.max()
    return float(np.average(g2[band], weights=wt[band])), float(np.max(np.abs(ph[band])))


def positive_maxima(F: np.ndarray, x: np.ndarray, p: np.ndarray,
                    *, thresh: float = 0.05) -> np.ndarray:
    """Phase-space coordinates of local maxima above ``thresh`` of the peak."""
    loc = (F == maximum_filter(F, size=9)) & (F > thresh * F.max())
    ii, jj = np.where(loc)
    return np.column_stack([x[ii], p[jj]])


def colocation(W: np.ndarray, Wt: np.ndarray, state: State,
               ) -> tuple[int, int, int, float, tuple[list[float], list[float]]]:
    """Counts in W and W~, number matched, median offset in delta, radii pairs."""
    x, p = state.x_int, state.p_int
    d = _quorum_semiaxis(state)
    mW = positive_maxima(W, x, p)
    mT = positive_maxima(Wt, x, p)
    offs: list[float] = []
    rW: list[float] = []
    rT: list[float] = []
    for z in mW:
        if len(mT) == 0:
            break
        j = int(np.argmin(np.sum((mT - z) ** 2, axis=1)))
        off = float(np.hypot(*(mT[j] - z)))
        if off < 3 * d:
            offs.append(off)
            rW.append(float(np.hypot(*z)))
            rT.append(float(np.hypot(*mT[j])))
    med = float(np.median(offs) / d) if offs else float("nan")
    return len(mW), len(mT), len(offs), med, (rW, rT)


def radial_spectrum(field: np.ndarray, state: State,
                    *, n_bins: int = 120) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged chord power of ``field`` as a function of chord."""
    dx = float(state.x_int[1] - state.x_int[0])
    dp = float(state.p_int[1] - state.p_int[0])
    kx = np.fft.fftshift(np.fft.fftfreq(len(state.x_int), d=dx)) * 2 * np.pi
    kp = np.fft.fftshift(np.fft.fftfreq(len(state.p_int), d=dp)) * 2 * np.pi
    KX, KP = np.meshgrid(kx, kp, indexing="ij")
    KR = np.hypot(KX, KP)
    edges = np.linspace(0, min(kx.max(), kp.max()), n_bins + 1)
    idx = np.digitize(KR.ravel(), edges) - 1
    f = field.ravel()
    prof = np.array([f[idx == b].mean() if np.any(idx == b) else np.nan
                     for b in range(n_bins)])
    return 0.5 * (edges[:-1] + edges[1:]), prof


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
@dataclass
class ValidationRow:
    """One row of the three validation tables for a single state."""
    name: str
    A_over_h_half: float
    gamma2: float
    phase: float
    n_W: int
    n_Wt: int
    matched: int
    offset_delta: float
    min_W: float
    min_Wt: float


def run_validation(states: list[State], labels: list[str], *, n_theta: int = 180,
                   ) -> tuple[list[ValidationRow], dict[str, tuple[list[float], list[float]]],
                              dict[str, np.ndarray]]:
    """Compute every portrait once and return the table rows, the radius
    scatter per state, and the portraits keyed by label."""
    rows: list[ValidationRow] = []
    scatter: dict[str, tuple[list[float], list[float]]] = {}
    portraits: dict[str, np.ndarray] = {}
    for lab, s in zip(labels, states):
        Wt = tilde_W_portrait(s, n_theta=n_theta)
        portraits[lab] = Wt
        g2, ph = coherence_phase(s.W, Wt, s)
        nW, nT, nm, off, (rW, rT) = colocation(s.W, Wt, s)
        scatter[lab] = (rW, rT)
        rows.append(ValidationRow(
            name=lab, A_over_h_half=s.rs.A_over_h_half, gamma2=g2, phase=ph,
            n_W=nW, n_Wt=nT, matched=nm, offset_delta=off,
            min_W=float(s.W.min() / s.W.max()), min_Wt=float(Wt.min() / Wt.max()),
        ))
    return rows, scatter, portraits


def print_tables(rows: list[ValidationRow]) -> None:
    """Print Tables A, B, C to stdout for cross-checking the LaTeX values."""
    print("TABLE A  coherence and phase")
    for r in rows:
        print(f"  {r.name:12s} A={r.A_over_h_half:6.2f}  g2={r.gamma2:.3f}  "
              f"max|phase|={r.phase:.0e}")
    print("\nTABLE B  co-location  N_W / N_~W / matched / offset(delta)")
    for r in rows:
        print(f"  {r.name:12s} A={r.A_over_h_half:6.2f}  {r.n_W:3d} {r.n_Wt:3d} "
              f"{r.matched:3d}  {r.offset_delta:.2f}")
    print("\nTABLE C  non-negativity  minW/maxW  min~W/max~W")
    for r in rows:
        print(f"  {r.name:12s} A={r.A_over_h_half:6.2f}  {r.min_W:7.3f}  {r.min_Wt:.0e}")


# ----------------------------------------------------------------------
# figure builders
# ----------------------------------------------------------------------
_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p"]


def build_composite(scatter: dict[str, tuple[list[float], list[float]]],
                    labels: list[str]) -> Figure:
    """Feature radius in W against radius in W-tilde, all states on the diagonal."""
    colors = plt.cm.turbo(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH * 0.6, DOUBLE_WIDTH * 0.6))
    rmax = 0.0
    for lab, mk, c in zip(labels, _MARKERS, colors):
        rW, rT = scatter[lab]
        if rW:
            ax.scatter(rW, rT, marker=mk, s=22, facecolor="none",
                       edgecolor=c, linewidths=0.9, label=lab)
            rmax = max(rmax, max(rW + rT))
    ax.plot([0, rmax * 1.05], [0, rmax * 1.05], "k-", lw=0.8, zorder=0)
    ax.set_xlabel(r"feature radius in $W$")
    ax.set_ylabel(r"feature radius in $\widetilde W$")
    ax.legend(fontsize=5.5, ncol=2, loc="upper left", framealpha=0.9)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def build_grid(states: list[State], labels: list[str],
               portraits: dict[str, np.ndarray]) -> Figure:
    """W-tilde heat map per state with the positive maxima of W marked."""
    fig, axes = plt.subplots(3, 4, figsize=(DOUBLE_WIDTH, DOUBLE_WIDTH * 0.78))
    for ax, lab, s in zip(axes.ravel(), labels, states):
        Wt = portraits[lab]
        x, p = s.x_int, s.p_int
        ax.imshow(Wt.T, origin="lower", extent=[x[0], x[-1], p[0], p[-1]],
                  cmap="magma", aspect="equal")
        m = positive_maxima(s.W, x, p)
        if len(m):
            ax.scatter(m[:, 0], m[:, 1], s=2, c="cyan", linewidths=0)
        ax.set_title(lab, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def build_shared_support(state: State, portrait: np.ndarray) -> Figure:
    """Radially averaged chord power of W and W-tilde for one state."""
    PW = np.abs(np.fft.fftshift(np.fft.fft2(state.W))) ** 2
    PT = np.abs(np.fft.fftshift(np.fft.fft2(portrait))) ** 2
    a, pw = radial_spectrum(PW, state)
    _, pt = radial_spectrum(PT, state)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH / 1.3))
    ax.semilogy(a, pw / np.nanmax(pw), "k", lw=1.0, label=r"$P_W$")
    ax.semilogy(a, pt / np.nanmax(pt), "crimson", lw=1.0, label=r"$P_{\widetilde W}$")
    ax.set_xlim(0, a.max())
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel(r"chord $|a|$")
    ax.set_ylabel("structure factor (norm.)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def build_ring_colocation(state: State, portrait: np.ndarray,
                          *, r_max: float = 9.0, n_bins: int = 300) -> Figure:
    """Radial profiles of W and W-tilde with their positive rings marked."""
    x, p = state.x_int, state.p_int
    XX, PP = np.meshgrid(x, p, indexing="ij")
    R = np.hypot(XX, PP)
    edges = np.linspace(0, r_max, n_bins)
    ri = np.digitize(R.ravel(), edges)

    def _profile(F: np.ndarray) -> np.ndarray:
        f = F.ravel()
        return np.array([f[ri == k].mean() if np.any(ri == k) else np.nan
                         for k in range(1, len(edges))])

    rc = 0.5 * (edges[:-1] + edges[1:])
    Wr = _profile(state.W)
    Tr = _profile(portrait)
    pkW, _ = find_peaks(Wr, height=0.02 * np.nanmax(Wr))
    pkT, _ = find_peaks(Tr, height=0.02 * np.nanmax(Tr))
    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH * 0.62, DOUBLE_WIDTH / 3.2))
    ax.axhline(0, color="0.7", lw=0.5)
    ax.plot(rc, Wr / np.nanmax(Wr), "k", lw=0.9, label=r"$W$")
    ax.plot(rc, Tr / np.nanmax(Tr), "crimson", lw=0.9, label=r"$\widetilde W$")
    ax.plot(rc[pkW], (Wr / np.nanmax(Wr))[pkW], "k.", ms=5)
    ax.plot(rc[pkT], (Tr / np.nanmax(Tr))[pkT], "r.", ms=5)
    ax.set_xlabel("phase-space radius")
    ax.set_ylabel("radial profile (norm.)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig