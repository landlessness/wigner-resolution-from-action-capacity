"""Cluster 4: states that test the construction's assumptions.

Each of the four states here isolates one premise the construction
leans on, so that a result holds it up rather than a reader's doubt
tearing it down.

  1. Finite-energy cubic-phase state. Its Wigner structure is sheared
     into a cubic, not a quadratic, profile, so it cannot be brought to
     principal frame by any symplectic map. This is the sharpest test of
     the premise that the covariance ellipse describes the state's
     geometry: the elliptical cell A is necessarily a coarse proxy.
  2. Thermal state. A featureless Gaussian W with large covariance and
     no sub-structure. Tests whether the quorum cell a-tilde marks the
     state's finest structure or is merely an algebraic dual that
     happens to coincide with structure for pure states.
  3. Asymmetric cat. Three collinear lobes of unequal weight. The
     covariance is dominated by lobe separation, decoupling it from the
     fringe spacing that the symmetric cats lock together by symmetry.
  4. Heavy-tailed state. A cusped wavefunction whose momentum second
     moment is large but finite, straining the covariance ellipse that
     the whole apparatus rests on.

States 1 and 4 are built in the position basis (their natural home) and
carried through the FFT Wigner path. States 2 and 3 are built in the
number basis via QuTiP.

Orientation. Each state is built so that its bare Wigner negativity
crosses the p = 0 axis, so the column-2 cross-section reads the
negativity and the slice convention stays consistent with the other
figures. The cubic-phase state is rotated by a rigid pi/2 phase-space
rotation (numerically the Fourier transform of its wavefunction) to
bring the cubic structure onto the x-axis; the asymmetric cat is built
with its lobes along p so the interference fringes lie on the x-axis.
Both rotations are symplectic maps, leaving the states physically
identical, and the theta-integral of column 5 is rotation-invariant, so
the portrait is unaffected by orientation.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from ..state import (
    DisplayWindow,
    State,
    build_state_from_psi,
    build_state_from_qobj,
)


# ===========================================================================
# 1. Finite-energy cubic-phase state
# ===========================================================================

# Cubicity gamma of the phase exp(i gamma x^3). Chosen together with the
# envelope below to bend the state into a visibly cubic profile with
# genuine Wigner negativity, while the second moments stay finite. The
# off-page robustness sweep dials this; the figure shows one value.
CUBIC_GAMMA_DEFAULT = 0.25

# Envelope squeezing r of the Gaussian that makes the state finite-
# energy. A negative r (anti-squeezed, wide in x) gives the cubic phase
# room to act over the state's support; a narrow envelope would leave
# x^3 doing almost nothing and the state near the vacuum. Without any
# envelope every moment of exp(i gamma x^3)|0> diverges and no
# covariance ellipse exists, so the envelope is what makes A well-defined.
CUBIC_SQUEEZE_DEFAULT = -0.5

CUBIC_N_DEFAULT = 160


def cubic_phase_state(
    *,
    gamma: float = CUBIC_GAMMA_DEFAULT,
    squeeze_r: float = CUBIC_SQUEEZE_DEFAULT,
    name: str | None = None,
    hbar: float = 1.0,
    N: int = CUBIC_N_DEFAULT,
) -> State:
    """Finite-energy cubic-phase state V(gamma)|sq>.

    Built as the cubic-phase unitary exp(i gamma x^3) acting on a
    squeezed vacuum, the standard finite-energy realization (Gottesman,
    Kitaev & Preskill, PRA 64, 012310 (2001); Gu et al., PRA 79, 062318
    (2009)). The squeezed envelope keeps all moments finite so the
    Heisenberg cell A exists; the cubic phase shears the state into a
    profile no symplectic map can straighten.

    Built in the position basis and carried through the FFT Wigner path,
    since the cubic phase is diagonal in x.

    Orientation. The cubic phase exp(i gamma x^3) places the boomerang's
    concave (negative) side along the momentum direction, so the bare
    negativity sits off the x-axis and the p = 0 cross-section (column 2)
    misses it. We apply a rigid pi/2 phase-space rotation -- numerically
    the Fourier transform of the wavefunction, the Fourier transform being
    the metaplectic representative of the pi/2 rotation -- so the cubic
    structure lies along x and the negativity crosses p = 0. The rotation
    is a symplectic map, so the state is identical and the portrait
    W-tilde (rotation-invariant) is unchanged; only the cross-section
    slice now reads the negativity, keeping the p = 0 convention
    consistent with the other figures.
    """
    if hbar != 1.0:
        raise NotImplementedError("Only hbar=1 is supported.")

    # Squeezed-vacuum envelope, evaluated analytically. In QuTiP's
    # convention x = (a + a^dag)/sqrt(2), the squeezed vacuum at
    # parameter r is a real Gaussian with sigma_x = e^{-r}/sqrt(2),
    # matching systems/squeezed_vacuum.py. The envelope is what makes
    # the cubic-phase state finite-energy and gives it a covariance.
    sigma_x = np.exp(-squeeze_r) / np.sqrt(2.0)
    x_uniform = np.linspace(-12.0, 12.0, 1601)
    psi_u = np.exp(-(x_uniform**2) / (4.0 * sigma_x**2)).astype(complex)

    # Apply the cubic phase exp(i gamma x^3).
    psi_u = psi_u * np.exp(1j * gamma * x_uniform**3)

    # Rigid pi/2 phase-space rotation via the Fourier transform of psi.
    # With hbar = 1 the transform kernel is e^{-i x y}; the prefactor and
    # grid spacing are chosen so the rotated state is sampled on the
    # conjugate grid y (which becomes the new position coordinate). This
    # swaps the roles of x and p, bringing the cubic structure onto the
    # x-axis so the p = 0 cross-section catches the negativity.
    n = len(x_uniform)
    dx = float(x_uniform[1] - x_uniform[0])
    psi_rot = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_u)))
    psi_rot = psi_rot * dx / np.sqrt(2.0 * np.pi)
    y_grid = np.fft.fftshift(np.fft.fftfreq(n, d=dx)) * 2.0 * np.pi

    # Normalize on the conjugate grid.
    dy = float(y_grid[1] - y_grid[0])
    psi_rot /= np.sqrt(np.sum(np.abs(psi_rot) ** 2) * dy)

    return build_state_from_psi(
        name=name or f"cubic_gamma{gamma:g}_r{squeeze_r:g}",
        psi=psi_rot,
        x_grid_psi=y_grid,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )


# ===========================================================================
# 2. Thermal state
# ===========================================================================

# Mean photon number of the thermal state. n_th = 3 gives a covariance
# several times the vacuum (A/(h/2) = 2 n_th + 1 = 7) with a perfectly
# featureless Gaussian W, the clean test of a-tilde against a state that
# has no fine structure to resolve.
THERMAL_NBAR_DEFAULT = 3.0

THERMAL_N_DEFAULT = 120


def thermal_state(
    *,
    nbar: float = THERMAL_NBAR_DEFAULT,
    name: str | None = None,
    hbar: float = 1.0,
    N: int = THERMAL_N_DEFAULT,
) -> State:
    """Thermal (chaotic) state with mean photon number ``nbar``.

    A maximally mixed state at fixed energy: a featureless isotropic
    Gaussian in phase space with covariance A/(h/2) = 2 nbar + 1 and no
    sub-structure whatsoever. The portrait W-tilde is the same Gaussian;
    the quorum cell a-tilde is fine, but there is nothing at that scale
    to resolve, so a-tilde reads here as an upper bound on attainable
    resolution rather than a marker of present structure.

    Built as a density matrix via qutip.thermal_dm and carried through
    the QuTiP Wigner path. The covariance is isotropic, so no principal-
    frame alignment is needed.
    """
    rho = qt.thermal_dm(N, nbar)
    return build_state_from_qobj(
        name=name or f"thermal_nbar{nbar:g}",
        qobj=rho,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )


# ===========================================================================
# 3. Asymmetric cat: collinear, unequal-weight lobes
# ===========================================================================

ASYM_CAT_N_DEFAULT = 100


def asymmetric_cat_state(
    *,
    separation: float = 4.0,
    weights: tuple[float, float, float] = (1.0, 0.55, 0.30),
    name: str | None = None,
    hbar: float = 1.0,
    N: int = ASYM_CAT_N_DEFAULT,
) -> State:
    """Three collinear coherent lobes of unequal weight.

    The lobes sit at p = -separation, 0, +separation along the momentum
    axis, with amplitudes set by ``weights``. Because the lobes are
    unequally weighted and strung along one axis, the covariance is
    dominated by the lobe separation while the interference fringes
    between adjacent lobes set a finer scale. The symmetric cats of
    Cluster 3 lock these two scales together; this state separates them,
    testing whether a-tilde tracks the fringe spacing or the separation.

    Orientation. The lobes are placed along p (not q) so that the
    interference fringes, which run perpendicular to the line of lobes,
    lie along the x-axis and cross p = 0. The bare negativity then
    appears in the p = 0 cross-section (column 2), keeping the slice
    convention consistent with the other figures. With the lobes on a
    coordinate axis and symmetric in p about the origin, the covariance
    cross-term vanishes by parity, so no principal-frame alignment is
    needed and the axis-aligned cell overlays are exact.
    """
    ps = np.array([-separation, 0.0, separation])
    qs = np.zeros_like(ps)
    alphas = (qs + 1j * ps) / np.sqrt(2)
    w = np.asarray(weights, dtype=float)

    psi = sum(wi * qt.coherent(N, a) for wi, a in zip(w, alphas))
    psi = psi.unit()

    return build_state_from_qobj(
        name=name or "asymmetric_cat",
        qobj=psi,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )


# ===========================================================================
# 4. Heavy-tailed state: cusped wavefunction, large finite <p^2>
# ===========================================================================

# Decay length of the symmetric exponential (cusp) wavefunction. A
# smaller lambda sharpens the cusp and raises <p^2>, straining the
# covariance ellipse. lambda = 1.0 keeps <p^2> large but finite and the
# RS product comfortably above (hbar/2)^2; the off-page sweep lowers it
# toward the strain limit.
HEAVY_LAMBDA_DEFAULT = 1.0


def heavy_tailed_state(
    *,
    lam: float = HEAVY_LAMBDA_DEFAULT,
    name: str | None = None,
    hbar: float = 1.0,
    n_grid: int = 2001,
    x_half: float = 14.0,
) -> State:
    """Symmetric exponential (cusped) wavefunction psi(x) ~ exp(-|x|/lam).

    The cusp at the origin puts weight in the tails of the momentum
    distribution, making <p^2> large. For a pure exponential <p^2> would
    be exactly 1/lam^2 (finite but with a slowly decaying momentum tail),
    so the Heisenberg cell A is well-defined yet the covariance ellipse
    is a poor proxy for the genuinely non-Gaussian support. This stress-
    tests A = pi Delta_x Delta_p as a sensible finite object.

    Built directly in the position basis (the cusp has no compact
    number-basis form) and carried through the FFT Wigner path. The
    wavefunction is real and even, so <p> = 0 and the covariance is
    diagonal; no principal-frame alignment is needed.
    """
    if hbar != 1.0:
        raise NotImplementedError("Only hbar=1 is supported.")

    x = np.linspace(-x_half, x_half, n_grid)
    psi = np.exp(-np.abs(x) / lam).astype(complex)
    dx = float(x[1] - x[0])
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

    return build_state_from_psi(
        name=name or f"heavy_tailed_lam{lam:g}",
        psi=psi,
        x_grid_psi=x,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )
