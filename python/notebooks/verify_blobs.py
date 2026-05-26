"""Verify that every bitangent blob a_theta is a genuine de Gosson quantum blob.

The manuscript calls the family {a_theta} a family of de Gosson quantum blobs
of action h/2, bitangent to the Heisenberg cell A from within and to the
quorum cell a-tilde from without. "Quantum blob" is a *symplectic* statement:
a_theta = S(B(sqrt(hbar))) for some S in Sp(1), equivalently its symplectic
capacity equals pi*hbar = h/2, equivalently it is self-dual under the
symplectic polar duality of de Gosson & de Gosson, Symmetry 14, 1890 (2022).

In one degree of freedom the symplectic capacity of a planar ellipse equals
its Euclidean area, so the reciprocal-axes identity r_par * r_perp = hbar
enforced in cells.py is exactly the blob condition. This script verifies that
identity at the *symplectic* level directly, across a dense theta sweep and
across the capacity range A/(h/2) in [1, 70], so the manuscript's symplectic
language is checked rather than assumed.

Checks (all fail-fast; the script exits non-zero on any failure):
  1. Symplectic capacity of a_theta equals pi*hbar for every theta.
  2. The symplectic eigenvalue of the covariance equals hbar/2 for every theta
     (the self-duality / quantum-blob condition).
  3. Capacity is theta-independent (constant action across the family).
  4. The kernel K_theta (kernels.py) and the cell a_theta (cells.py) describe
     the same ellipse: same covariance to numerical tolerance.
  5. a_theta is bitangent to A and to a-tilde in the literal sense: exactly
     two antipodal contact points with each, at every theta (inscribed in A,
     circumscribed about a-tilde).

Run:
    cd python
    uv run python notebooks/verify_blobs.py
"""

from __future__ import annotations

import sys

import numpy as np

from wigner_resolution.cells import (
    HeisenbergCell,
    bitangent_blob_at,
    quorum_cell,
)
from wigner_resolution.kernels import _rotation_matrix

# Symplectic form (1 DOF) and the blob targets in hbar = 1 units.
J = np.array([[0.0, 1.0], [-1.0, 0.0]])
HBAR = 1.0
BLOB_CAPACITY = np.pi * HBAR      # = h/2
BLOB_LAMBDA = HBAR / 2.0          # symplectic eigenvalue of a blob's covariance

ATOL = 1e-9          # tight: construction should be exact to fp noise
N_THETA = 181        # dense sweep over [0, pi)

# Capacity regimes to span: (Delta_x, Delta_p, label). A/(h/2) = Dx*Dp/hbar.
REGIMES = [
    (1.0, 1.0, "minimum uncertainty, A = h/2"),
    (2.0, 0.5, "squeezed, A = h/2"),
    (np.sqrt(3.0), np.sqrt(3.0), "harmonic n=1, A = 3 h/2"),
    (4.0, 2.0, "asymmetric, A = 8 h/2"),
    (10.0, 7.0, "large, A = 70 h/2"),
]


def _M_orig(theta: float, Dx: float, Dp: float, hbar: float = HBAR) -> np.ndarray:
    """Inverse-covariance matrix of a_theta in the (x, p) frame, rebuilt
    exactly as cells.bitangent_blob_at constructs it. The ellipse is
    {z : M z.z <= hbar}."""
    r_tilde = hbar / (Dx * Dp)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    M_disk = R @ np.diag([1.0, 1.0 / r_tilde**2]) @ R.T
    D_inv = np.diag([1.0 / Dx, 1.0 / Dp])
    return D_inv @ M_disk @ D_inv


def _symplectic_capacity(M: np.ndarray, hbar: float = HBAR) -> float:
    """Symplectic capacity of {z : M z.z <= hbar}. In 1 DOF this is
    pi / sqrt(det(M/hbar))."""
    A = M / hbar
    return float(np.pi / np.sqrt(np.linalg.det(A)))


def _symplectic_eigenvalue(M: np.ndarray, hbar: float = HBAR) -> float:
    """Symplectic eigenvalue of the covariance Sigma = (hbar/2) M^{-1}.
    Blob <=> this equals hbar/2."""
    Sigma = (hbar / 2.0) * np.linalg.inv(M)
    w = np.linalg.eigvals(J @ Sigma)
    return float(np.abs(w[0]))


def _blob_covariance_from_cells(theta, heis, hbar=HBAR):
    """Covariance of a_theta as the rest of the package would read it off
    the BitangentBlob dataclass (principal angle + semi-axes), matching
    kernels.K_theta_mesh."""
    blob = bitangent_blob_at(theta, heis, hbar=hbar)
    R = _rotation_matrix(blob.principal_angle)
    Sigma_0 = np.diag([blob.r_parallel**2 / 2.0, blob.r_perp**2 / 2.0])
    return R @ Sigma_0 @ R.T


def _fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


# Number of boundary samples for the contact-point count. Dense enough that
# the discrete extremum of the normalized radius sits within CONTACT_TOL of
# its true value (1.0) even for highly anisotropic blobs, where the contact
# region is sharp. The discrete-extremum error falls as 1/N^2; at N = 40000
# it is below 1e-5 across the capacity range tested.
N_BOUNDARY = 40000

# A local extremum of the normalized radius (x/sx)^2 + (p/sp)^2 counts as a
# contact when its value lands within this band of 1. The band only needs to
# absorb the 1/N^2 discrete-sampling overshoot of a genuine tangency (~1e-5
# at N_BOUNDARY), so 1e-3 leaves a comfortable margin while staying far tighter
# than any spurious near-approach a non-tangent blob would produce.
CONTACT_TOL = 1e-3


def _blob_boundary(blob, n: int = N_BOUNDARY) -> tuple[np.ndarray, np.ndarray]:
    """Sample the a_theta ellipse boundary in the original (x, p) frame.

    The ellipse has principal semi-axes (r_parallel, r_perp) oriented at
    blob.principal_angle, centered at the origin (cells.py builds it
    centered; the kernel center used for convolution is a separate display
    concern).
    """
    a = blob.principal_angle
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cp, sp = np.cos(a), np.sin(a)
    ex = blob.r_parallel * np.cos(t) * cp - blob.r_perp * np.sin(t) * sp
    ep = blob.r_parallel * np.cos(t) * sp + blob.r_perp * np.sin(t) * cp
    return ex, ep


def _count_antipodal_contacts(
    ex: np.ndarray,
    ep: np.ndarray,
    sx: float,
    sp: float,
    tol: float = CONTACT_TOL,
) -> tuple[int, bool]:
    """Count contact clusters of the boundary (ex, ep) with the centered
    ellipse of semi-axes (sx, sp), and report whether they are antipodal.

    A contact is where the normalized radius r = (x/sx)^2 + (p/sp)^2 equals
    1: for a_theta inscribed in A the blob's boundary reaches out to r = 1
    at the two points nearest A (local maxima of r); for a_theta
    circumscribing a-tilde it reaches in to r = 1 at the two points nearest
    a-tilde (local minima of r). Rather than count samples that land in a
    fixed band of 1 -- which is fragile when one curve is tiny and the
    contact region is sharp -- we locate the local extrema of r directly
    (sampling-independent) and accept those whose value is within tol of 1.

    Returns (n_contacts, antipodal): the number of accepted extremal
    contacts, and whether exactly two of them sit antipodally through the
    origin.
    """
    r = (ex / sx) ** 2 + (ep / sp) ** 2

    # Local minima and maxima on the circular boundary. A tangency from
    # inside (a-tilde) is a local min at r ~ 1; from outside (A) a local
    # max at r ~ 1. We gather both kinds and keep those at r ~ 1.
    rprev = np.roll(r, 1)
    rnext = np.roll(r, -1)
    is_min = (r <= rprev) & (r <= rnext)
    is_max = (r >= rprev) & (r >= rnext)
    extremal = np.where((is_min | is_max) & np.isclose(r, 1.0, atol=tol))[0]

    if extremal.size == 0:
        return 0, False

    # Merge extrema that are adjacent samples (a flat extremum spans a few
    # points) into single contacts.
    gaps = np.diff(extremal)
    n_contacts = int(np.sum(gaps > 1) + 1)
    if extremal[0] == 0 and extremal[-1] == len(r) - 1:
        n_contacts -= 1  # wrap-around merge

    antipodal = bool(
        n_contacts == 2
        and np.allclose(
            [ex[extremal].mean(), ep[extremal].mean()], 0.0, atol=1e-2
        )
    )
    return n_contacts, antipodal


def check_regime(Dx: float, Dp: float, label: str) -> bool:
    print(f"\n[{label}]  Dx={Dx:.4f} Dp={Dp:.4f}  A/(h/2)={Dx*Dp/HBAR:.3f}")
    thetas = np.linspace(0.0, np.pi, N_THETA, endpoint=False)
    caps = np.empty_like(thetas)
    lams = np.empty_like(thetas)
    cov_mismatch = 0.0
    ok = True

    heis = HeisenbergCell(Delta_x=Dx, Delta_p=Dp)
    qc = quorum_cell(heis, hbar=HBAR)

    # Bitangency accumulators (Check 5): worst-case (fewest) contact counts
    # and whether antipodality held at every theta. A genuine bitangency is
    # exactly two antipodal contacts with each of A and a-tilde.
    min_contacts_A = 99
    min_contacts_q = 99
    antipodal_A_all = True
    antipodal_q_all = True

    for i, th in enumerate(thetas):
        M = _M_orig(th, Dx, Dp)
        caps[i] = _symplectic_capacity(M)
        lams[i] = _symplectic_eigenvalue(M)

        # Check 4: kernel covariance == cell covariance (= (hbar/2) M^{-1}).
        Sigma_cell = (HBAR / 2.0) * np.linalg.inv(M)
        Sigma_kernel = _blob_covariance_from_cells(th, heis)
        cov_mismatch = max(cov_mismatch,
                           float(np.max(np.abs(Sigma_cell - Sigma_kernel))))

        # Check 5: count contact points of a_theta with A and with a-tilde,
        # in the original (x, p) frame. A is the centered ellipse with
        # semi-axes (Dx, Dp); a-tilde has semi-axes (hbar/Dp, hbar/Dx).
        blob = bitangent_blob_at(th, heis, hbar=HBAR)
        ex, ep = _blob_boundary(blob)
        nA, apA = _count_antipodal_contacts(ex, ep, Dx, Dp)
        nq, apq = _count_antipodal_contacts(ex, ep, qc.delta_x, qc.delta_p)
        min_contacts_A = min(min_contacts_A, nA)
        min_contacts_q = min(min_contacts_q, nq)
        antipodal_A_all = antipodal_A_all and apA
        antipodal_q_all = antipodal_q_all and apq

    # Check 1: capacity == h/2 everywhere.
    if not np.allclose(caps, BLOB_CAPACITY, atol=ATOL):
        _fail(f"capacity != pi*hbar; range "
              f"[{caps.min():.3e}, {caps.max():.3e}]")
        ok = False

    # Check 2: symplectic eigenvalue == hbar/2 everywhere.
    if not np.allclose(lams, BLOB_LAMBDA, atol=ATOL):
        _fail(f"symplectic eigenvalue != hbar/2; range "
              f"[{lams.min():.3e}, {lams.max():.3e}]")
        ok = False

    # Check 3: capacity theta-independent.
    if (caps.max() - caps.min()) > ATOL:
        _fail(f"capacity varies with theta; spread "
              f"{caps.max() - caps.min():.3e}")
        ok = False

    # Check 4 verdict.
    if cov_mismatch > 1e-9:
        _fail(f"kernel/cell covariance mismatch {cov_mismatch:.3e}")
        ok = False

    # Check 5: bitangency, the literal sense. a_theta must have exactly two
    # antipodal contact points with A (inscribed, touching from within) and
    # exactly two with a-tilde (circumscribed, touching from without), at
    # every theta. Two concentric ellipses that touch do so at two
    # antipodal points by central symmetry; a count other than two, or a
    # broken antipodality, would mean the blob crosses or stands off a curve.
    #
    # Degenerate case A = h/2: here A, a_theta, and a-tilde all coincide
    # (Dx Dp = hbar, so r_tilde = 1 and the blob fills A exactly). The two
    # curves are identical rather than tangent, so the two-point contact
    # count does not apply; bitangency is trivially the whole-boundary
    # coincidence. We detect this and report it rather than fail.
    cells_coincide = np.isclose(Dx * Dp, HBAR, atol=1e-9)

    if cells_coincide:
        bitangency_note = ("A = a_theta = a-tilde coincide (A = h/2); "
                           "bitangency is whole-boundary coincidence")
    else:
        if min_contacts_A != 2 or not antipodal_A_all:
            _fail(f"a_theta not bitangent to A: min contacts "
                  f"{min_contacts_A}, antipodal {antipodal_A_all}")
            ok = False
        if min_contacts_q != 2 or not antipodal_q_all:
            _fail(f"a_theta not bitangent to a-tilde: min contacts "
                  f"{min_contacts_q}, antipodal {antipodal_q_all}")
            ok = False
        bitangency_note = (f"{min_contacts_A} antipodal contacts with A, "
                           f"{min_contacts_q} with a-tilde, at every theta "
                           f"(inscribed in A, circumscribed about a-tilde)")

    print(f"  capacity   : {caps.mean():.10f}  (target {BLOB_CAPACITY:.10f})  "
          f"spread {caps.max()-caps.min():.2e}")
    print(f"  sympl. eig : {lams.mean():.10f}  (target {BLOB_LAMBDA:.10f})  "
          f"spread {lams.max()-lams.min():.2e}")
    print(f"  kernel/cell cov max|diff| : {cov_mismatch:.2e}")
    print(f"  bitangency : {bitangency_note}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Verifying a_theta are symplectic de Gosson quantum blobs "
          "(capacity = h/2) for all theta.")
    print(f"theta sweep: {N_THETA} points in [0, pi);  tolerance: {ATOL:g}")
    results = [check_regime(Dx, Dp, label) for Dx, Dp, label in REGIMES]
    print("\n" + "=" * 64)
    if all(results):
        print("ALL CHECKS PASS.")
        print("Every a_theta has symplectic capacity h/2 and is self-dual: a "
              "genuine de Gosson quantum blob, bitangent to A and a-tilde, at "
              "constant action across theta.")
        return 0
    print("VERIFICATION FAILED. See messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
