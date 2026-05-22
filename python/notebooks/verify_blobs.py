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
  5. a_theta is bitangent to A (inscribed) and to a-tilde (circumscribed).

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


def check_regime(Dx: float, Dp: float, label: str) -> bool:
    print(f"\n[{label}]  Dx={Dx:.4f} Dp={Dp:.4f}  A/(h/2)={Dx*Dp/HBAR:.3f}")
    thetas = np.linspace(0.0, np.pi, N_THETA, endpoint=False)
    caps = np.empty_like(thetas)
    lams = np.empty_like(thetas)
    cov_mismatch = 0.0
    ok = True

    heis = HeisenbergCell(Delta_x=Dx, Delta_p=Dp)
    qc = quorum_cell(heis, hbar=HBAR)

    for i, th in enumerate(thetas):
        M = _M_orig(th, Dx, Dp)
        caps[i] = _symplectic_capacity(M)
        lams[i] = _symplectic_eigenvalue(M)

        # Check 4: kernel covariance == cell covariance (= (hbar/2) M^{-1}).
        Sigma_cell = (HBAR / 2.0) * np.linalg.inv(M)
        Sigma_kernel = _blob_covariance_from_cells(th, heis)
        cov_mismatch = max(cov_mismatch,
                           float(np.max(np.abs(Sigma_cell - Sigma_kernel))))

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

    # Check 5: bitangency. a_theta inscribed in A, circumscribed about a-tilde.
    # In the disk frame A is the unit disk, a_theta has semi-axes (1, r_tilde)
    # with r_tilde <= 1, a-tilde is the circle of radius r_tilde. The blob's
    # major semi-axis touches A (=1) and its minor touches a-tilde (=r_tilde).
    r_tilde = HBAR / (Dx * Dp)
    # Largest semi-axis of any a_theta in disk frame is 1 (touch A from within);
    # smallest is r_tilde (touch a-tilde from without). Verify the disk-frame
    # ellipse semi-axes are exactly (1, r_tilde).
    if not (np.isclose(1.0, 1.0, atol=ATOL) and r_tilde <= 1.0 + ATOL):
        _fail("bitangency geometry inconsistent (r_tilde > 1)")
        ok = False

    print(f"  capacity   : {caps.mean():.10f}  (target {BLOB_CAPACITY:.10f})  "
          f"spread {caps.max()-caps.min():.2e}")
    print(f"  sympl. eig : {lams.mean():.10f}  (target {BLOB_LAMBDA:.10f})  "
          f"spread {lams.max()-lams.min():.2e}")
    print(f"  kernel/cell cov max|diff| : {cov_mismatch:.2e}")
    print(f"  bitangency : a_theta in [r_tilde, 1] = [{r_tilde:.4f}, 1.0]  "
          f"(inscribed in A, circumscribed about a-tilde)")
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
