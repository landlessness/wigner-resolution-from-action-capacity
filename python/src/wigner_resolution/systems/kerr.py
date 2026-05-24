"""Kerr-evolved coherent state: the crescent (banana) state.

A coherent state |alpha> evolved under the Kerr Hamiltonian

    H = chi * n_hat^2          (n_hat = a-dagger a)

for a time t shears in phase space. For small chi*t the coherent disk
bends into a crescent; the same Hamiltonian, run to chi*t = pi, produces
a two-component cat (Yurke-Stoler), so the crescent is the dynamical
bridge from a single coherent state to a cat.

We parameterize by the dimensionless shear phi = chi * t. The evolution
exp(-i phi n^2) is diagonal in the Fock basis, so it is applied exactly
as a phase on each number-basis amplitude (no ODE integration).

Frame alignment. A sheared crescent has a covariance ellipse whose
principal axes are tilted off the (x, p) coordinate axes, i.e. a
nonzero <xp> cross-term. The manuscript's cell overlays (Heisenberg
cell A, quorum cell a-tilde) are axis-aligned ellipses, valid for
states with diagonal covariance. The convolved portrait (the theta
integral) is rotation-invariant and so is unaffected by orientation,
but the cross-sections and the A / a-tilde overlays read cleanest when
a principal axis is aligned with a coordinate axis.

We therefore rotate the state rigidly in phase space by exp(-i psi n)
so that the major principal axis of its covariance lands on a
coordinate axis. A phase-space rotation commutes with the Kerr shape
(it only spins the crescent, preserving it exactly), and afterward the
covariance cross-term vanishes, so the axis-aligned overlay machinery
is exact rather than approximate. This is the "rotated ellipsoid"
case the manuscript notes is reachable through the symplectic
covariance of the family; here we simply choose the frame in which the
ellipse is non-rotated.

Reference for Kerr cat dynamics: Yurke and Stoler, Phys. Rev. Lett. 57,
13 (1986); Kirchmair et al., Nature 495, 205 (2013) for observed
phase-space crescents.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from ..state import DisplayWindow, State, build_state_from_qobj

# Coherent amplitude (real, placed on +q). |alpha|^2 = 4 photons.
KERR_ALPHA_DEFAULT = 2.0

# Shear phi = chi*t. phi=0 is the bare coherent state; phi=pi is a 2-cat.
# phi ~ 0.3 gives a pronounced crescent with a CLEAR major axis (the
# rotated covariance is anisotropic, sigma_par/sigma_perp ~ 1.5); by
# phi ~ 0.5 the second moments go isotropic and there is no longer a
# well-defined "furthest extent" axis to align. 0.3 is the crescent
# sweet spot: visibly bent, modest Wigner negativity, alignable axis.
KERR_PHI_DEFAULT = 0.3

KERR_N_DEFAULT = 80


def _full_covariance(psi: qt.Qobj, N: int) -> tuple[np.ndarray, float, float]:
    """Full 2x2 (x, p) covariance of psi, plus the means <x>, <p>.

    Includes the symmetrized cross-term sigma_xp = <xp + px>/2 - <x><p>,
    which is nonzero for a sheared state.
    """
    x = qt.position(N)
    p = qt.momentum(N)
    xm = float(qt.expect(x, psi))
    pm = float(qt.expect(p, psi))
    sxx = float(qt.expect(x * x, psi)) - xm ** 2
    spp = float(qt.expect(p * p, psi)) - pm ** 2
    sxp = 0.5 * float(qt.expect(x * p + p * x, psi)) - xm * pm
    return np.array([[sxx, sxp], [sxp, spp]]), xm, pm


def _principal_angle(cov: np.ndarray) -> float:
    """Angle (radians) of the major principal axis of a 2x2 covariance."""
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    return float(np.arctan2(major[1], major[0]))


def kerr_state(
    *,
    alpha: float = KERR_ALPHA_DEFAULT,
    phi: float = KERR_PHI_DEFAULT,
    align_major_axis: bool = True,
    name: str | None = None,
    hbar: float = 1.0,
    N: int = KERR_N_DEFAULT,
) -> State:
    """Build a Kerr-sheared coherent state (crescent / banana).

    ``alpha``  coherent amplitude (real); |alpha|^2 is the mean photon
               number. Placed on the +q axis before evolution.
    ``phi``    dimensionless shear chi*t. phi=0 is the bare coherent
               state; phi=pi is a 2-cat. The default gives a crescent.
    ``align_major_axis``  if True (default), rigidly rotate the state in
               phase space so the major principal axis of its covariance
               lands on a coordinate axis, making the covariance diagonal
               and the axis-aligned cell overlays exact.
    ``N``      Fock-basis truncation.
    """
    if N <= 0:
        raise ValueError("N must be positive.")

    # Coherent state on +q axis, then exact diagonal Kerr phase.
    psi0 = qt.coherent(N, alpha)
    n_idx = np.arange(N)
    amps = psi0.full().ravel() * np.exp(-1j * phi * n_idx.astype(float) ** 2)
    psi = qt.Qobj(amps.reshape(N, 1)).unit()

    if align_major_axis:
        cov, _, _ = _full_covariance(psi, N)
        psi_ang = _principal_angle(cov)
        # Rigid phase-space rotation by -psi_ang: exp(+i psi_ang n).
        R = (1j * psi_ang * qt.num(N)).expm()
        psi = (R * psi).unit()

    return build_state_from_qobj(
        name=name or f"kerr_alpha{alpha:g}_phi{phi:g}",
        qobj=psi,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )
