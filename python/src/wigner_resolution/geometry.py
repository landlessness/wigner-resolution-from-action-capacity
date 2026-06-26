"""Phase-space geometry: the capacity A, the quantum blobs β_θ,
and the resolution a.

A state's covariance defines its capacity

    A = π Δx Δp  ≥  h/2,

the outer ellipse with capacity semi-axes (Δx, Δp), floored at the
quantum of action by the uncertainty principle.

The symplectic polar dual of the capacity is the resolution

    a = π δx δp  ≤  h/2,

an axis-aligned ellipse with semi-axes

    (δx, δp) = (ℏ/Δp, ℏ/Δx),

reciprocal to the capacity at action h/2: A · a = (h/2)².
Reference: M. de Gosson and C. de Gosson, Symmetry 14, 1890 (2022).

Nested between A and a, circumscribed by the capacity A and
circumscribing the resolution a, is a one-parameter family of de
Gosson quantum blobs β_θ of fixed action h/2, parameterized by the
family-orientation angle θ. The semi-axes
r_∥(θ), r_⊥(θ) of β_θ are reciprocal in ℏ,

    r_∥(θ) r_⊥(θ) = ℏ.

The family is constructed in the affine-normalized frame where A
becomes the unit disk and a becomes the inscribed concentric circle
of radius

    r_a = ℏ/(Δx Δp).

In that frame, β_θ is the rigid rotation by θ of a single ellipse
with semi-axes (1, r_a), circumscribed by the unit disk and
circumscribing the inscribed circle. Pulled back to the original
(x, p) frame via the inverse of T(x, p) = (x/Δx, p/Δp), the family
deforms across θ at constant action h/2 while remaining circumscribed
by A and circumscribing a.

At the principal angles θ = 0 and θ = π/2, the blob's principal axes
align with the (x, p) coordinate axes, and the semi-axes recover the
Zurek scales δx = ℏ/Δp and δp = ℏ/Δx. At intermediate angles, the
blob's principal axes are tilted off Euclidean θ; the orientation
parameter θ continuously interpolates between the two principal
endpoints. Reference for Zurek's scales: W. H. Zurek, Nature 412,
712 (2001).

A note on "quantum blob" as a symplectic statement. Calling β_θ a de
Gosson quantum blob means its symplectic capacity equals πℏ = h/2 and
it is self-dual under symplectic polar duality (de Gosson & de Gosson
2022), not merely that its Euclidean area is h/2. In one degree of
freedom the symplectic capacity of a planar ellipse equals its area,
so the reciprocal-axes identity r_∥ r_⊥ = ℏ enforced below is exactly
the quantum-blob condition. The QuantumBlob.__post_init__ assertion
is phrased as the symplectic capacity the manuscript claims; the full
symplectic verification across θ and across the capacity range is in
notebooks/verify_blobs.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Capacity:
    """The state's outer extent A in phase space.

    Capacity semi-axes Δx, Δp along the principal axes
    (Robertson-Schrödinger cross-term assumed zero, which holds for
    the symmetric states in the manuscript).
    """

    Delta_x: float
    Delta_p: float
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def area(self) -> float:
        """A = π Δx Δp."""
        return np.pi * self.Delta_x * self.Delta_p

    def __post_init__(self) -> None:
        if self.Delta_x <= 0 or self.Delta_p <= 0:
            raise ValueError("Semi-axis widths must be positive.")


@dataclass(frozen=True)
class QuantumBlob:
    """One member β_θ of the quantum blob family.

    A de Gosson quantum blob of action h/2, circumscribed by the
    capacity A and circumscribing the resolution a. Labeled
    by the family-orientation angle θ.

    Geometric attributes in the original (x, p) frame:
      r_parallel: longer principal semi-axis (the "parallel" axis).
      r_perp:     shorter principal semi-axis (the "perpendicular" axis).
      principal_angle: Euclidean angle (radians) of the r_parallel axis
                       from the +x-axis, in [0, π).

    The semi-axes are reciprocal in ℏ: r_∥ · r_⊥ = ℏ. At θ ∈ {0, π/2},
    principal_angle equals θ exactly. At intermediate angles, the affine
    pullback from the disk frame tilts the principal axes off θ.
    """

    theta: float
    r_parallel: float
    r_perp: float
    principal_angle: float
    center: tuple[float, float] = (0.0, 0.0)
    hbar: float = 1.0

    @property
    def area(self) -> float:
        """β_θ = π r_∥ r_⊥ = h/2 (always)."""
        return np.pi * self.r_parallel * self.r_perp

    @property
    def symplectic_capacity(self) -> float:
        """Symplectic capacity of the blob, c(β_θ).

        For a planar ellipse with principal semi-axes (r_∥, r_⊥) the
        symplectic capacity equals the Euclidean area π r_∥ r_⊥; in one
        degree of freedom symplectic capacity and area coincide. A de
        Gosson quantum blob has c = πℏ = h/2.
        """
        return np.pi * self.r_parallel * self.r_perp

    def __post_init__(self) -> None:
        # Symplectic blob condition: c(β_θ) = πℏ = h/2. In 1 DOF this
        # equals the Euclidean area, so enforcing it is equivalent to the
        # reciprocal-axes identity r_∥ r_⊥ = ℏ, but stated as the symplectic
        # capacity the manuscript actually claims. The full symplectic
        # verification across θ and capacity is in notebooks/verify_blobs.py.
        capacity = np.pi * self.r_parallel * self.r_perp
        if not np.isclose(capacity, np.pi * self.hbar, rtol=1e-10):
            raise ValueError(
                f"Symplectic capacity violated: "
                f"c(β_θ) = π·r_∥·r_⊥ = {capacity:.6g} "
                f"≠ πℏ = h/2 = {np.pi * self.hbar:.6g}"
            )


@dataclass(frozen=True)
class Resolution:
    """The resolution a with semi-axes (δx, δp) = (ℏ/Δp, ℏ/Δx).

    Symplectic polar dual of the capacity, circumscribed by every
    blob of the family. An axis-aligned ellipse with no
    rotation parameter, not itself a member of the quantum blob family.
    Its semi-axes are Zurek's interference scales.

    Reference for polar duality: M. de Gosson and C. de Gosson,
    Symmetry 14, 1890 (2022). Reference for Zurek's scales:
    W. H. Zurek, Nature 412, 712 (2001).
    """

    delta_x: float
    delta_p: float
    center: tuple[float, float] = (0.0, 0.0)
    hbar: float = 1.0

    @property
    def area(self) -> float:
        """a = π δx δp = π ℏ² / (Δx Δp)."""
        return np.pi * self.delta_x * self.delta_p

    def __post_init__(self) -> None:
        if self.delta_x <= 0 or self.delta_p <= 0:
            raise ValueError("Resolution semi-axes must be positive.")


def quantum_blob_at(
    theta: float,
    capacity: Capacity,
    hbar: float = 1.0,
) -> QuantumBlob:
    """The quantum blob β_θ at family-orientation angle θ.

    Construction (affine pullback):
      1. In the disk frame where A is the unit disk and a is the
         inscribed circle of radius r_a = ℏ/(Δx Δp), the blob β_θ is
         the rigid rotation by θ of an ellipse with semi-axes (1, r_a).
      2. Pulled back to the original (x, p) frame, the blob has
         inverse-covariance matrix M_θ = D⁻¹ R_θ diag(1, 1/r_a²) R_θᵀ D⁻¹
         with D = diag(Δx, Δp).
      3. Principal-axis lengths and orientation are recovered from the
         eigendecomposition of M_θ.

    At principal angles θ ∈ {0, π/2}, the principal axes coincide with
    the coordinate axes and r_parallel reduces to Δx or Δp respectively.
    """
    Dx, Dp = capacity.Delta_x, capacity.Delta_p
    r_a = hbar / (Dx * Dp)

    # Disk-frame inverse-covariance matrix, rotated by theta
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    M_disk = R @ np.diag([1.0, 1.0 / r_a**2]) @ R.T

    # Pull back to original (x, p) frame
    D_inv = np.diag([1.0 / Dx, 1.0 / Dp])
    M_orig = D_inv @ M_disk @ D_inv

    # Eigendecomposition: smaller eigenvalue ↔ larger semi-axis
    eigs, vecs = np.linalg.eigh(M_orig)
    r_par = 1.0 / np.sqrt(eigs[0])
    r_perp = 1.0 / np.sqrt(eigs[1])

    # Principal angle: direction of the major-axis eigenvector, in [0, π)
    major_dir = vecs[:, 0]
    alpha = np.arctan2(major_dir[1], major_dir[0])
    if alpha < 0:
        alpha += np.pi
    if alpha >= np.pi:
        alpha -= np.pi

    return QuantumBlob(
        theta=theta,
        r_parallel=r_par,
        r_perp=r_perp,
        principal_angle=alpha,
        center=capacity.center,
        hbar=hbar,
    )


def blob_beta_pi_half(capacity: Capacity, hbar: float = 1.0) -> QuantumBlob:
    """The quantum blob β_{π/2}: the family member at θ = π/2.

    Semi-axes (r_∥, r_⊥) = (Δp, δx) = (Δp, ℏ/Δp), principal angle π/2.
    """
    return quantum_blob_at(np.pi / 2, capacity, hbar)


def blob_beta_zero(capacity: Capacity, hbar: float = 1.0) -> QuantumBlob:
    """The quantum blob β_{θ=0}: the family member at θ = 0.

    Semi-axes (r_∥, r_⊥) = (Δx, δp) = (Δx, ℏ/Δx), principal angle 0.
    """
    return quantum_blob_at(0.0, capacity, hbar)


def resolution(capacity: Capacity, hbar: float = 1.0) -> Resolution:
    """The resolution a with semi-axes (ℏ/Δp, ℏ/Δx).

    The symplectic polar dual of `capacity`, circumscribed by every
    blob in the quantum blob family. Area a = π ℏ²/(Δx Δp), the
    resolution of the convolved portrait W̃.
    """
    return Resolution(
        delta_x=hbar / capacity.Delta_p,
        delta_p=hbar / capacity.Delta_x,
        center=capacity.center,
        hbar=hbar,
    )