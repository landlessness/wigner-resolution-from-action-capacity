"""Phase-space cells: the Heisenberg cell A, the bitangent blobs a_θ,
and the quorum cell ã.

A state's covariance defines its Heisenberg cell

    A = π Δx Δp  ≥  h/2,

the outer ellipse with capacity semi-axes (Δx, Δp), floored at the
quantum of action by the uncertainty principle.

Inscribed within A and bitangent to it is a one-parameter family of
de Gosson quantum blobs a_θ of fixed action h/2, each parameterized by
the quadrature angle θ. The semi-axes are reciprocal in ℏ,
r_∥ r_⊥ = ℏ, with

    r_∥(θ) = sqrt(Δx² cos²θ + Δp² sin²θ),
    r_⊥(θ) = ℏ / r_∥(θ).

r_∥ runs along the quadrature direction at angle θ from the x-axis.
r_⊥ runs perpendicular. Neither is the "major" axis in general; which
is larger depends on the aspect ratio of the Heisenberg cell.

Two members of the family align with the principal axes:
a_{θ=0} at θ = 0 (capacity Δx along x, resolution δp along p) and
a_{π/2} at θ = π/2 (resolution δx along x, capacity Δp along p).
At those principal angles r_⊥ reduces to the Zurek scales

    δx = ℏ/Δp,  δp = ℏ/Δx.

The quorum cell ã is the common interior of the bitangent family,
an axis-aligned ellipse with semi-axes

    (δx, δp) = (ℏ/Δp, ℏ/Δx),

with area

    π δx δp = π ℏ² / (Δx Δp),

the resolution of the convolved portrait W̃. The quorum cell is not a
member of the bitangent family; it is its own object, marking the inner
envelope of the family integral. Zurek's interference scales
δx = ℏ/Δp and δp = ℏ/Δx, derived in W. H. Zurek, Nature 412, 712 (2001),
are recovered here as the resolutions along the principal axes of the
quorum cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeisenbergCell:
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
            raise ValueError("Cell widths must be positive.")


@dataclass(frozen=True)
class BitangentBlob:
    """One member a_θ of the bitangent family inscribed in a Heisenberg cell.

    A de Gosson quantum blob of fixed action h/2, bitangent to the
    Heisenberg cell. Oriented at quadrature angle θ from the x-axis.
    r_∥ is along θ; r_⊥ is perpendicular. The semi-axes are reciprocal
    in ℏ: r_∥ r_⊥ = ℏ.
    """

    theta: float
    r_parallel: float
    r_perp: float
    center: tuple[float, float] = (0.0, 0.0)
    hbar: float = 1.0

    @property
    def area(self) -> float:
        """a_θ = π r_∥ r_⊥ = h/2 (always)."""
        return np.pi * self.r_parallel * self.r_perp

    def __post_init__(self) -> None:
        product = self.r_parallel * self.r_perp
        if not np.isclose(product, self.hbar, rtol=1e-10):
            raise ValueError(
                f"Reciprocal-axes identity violated: r_∥·r_⊥ = {product:.6g} ≠ ℏ = {self.hbar}"
            )


@dataclass(frozen=True)
class QuorumCell:
    """The quorum cell ã with semi-axes (δx, δp) = (ℏ/Δp, ℏ/Δx).

    Common interior of the bitangent family, an axis-aligned ellipse
    with no rotation parameter. Not itself a member of the bitangent
    family. Its semi-axes are Zurek's interference scales.

    Reference for Zurek's scales: W. H. Zurek, Nature 412, 712 (2001).
    """

    delta_x: float
    delta_p: float
    center: tuple[float, float] = (0.0, 0.0)
    hbar: float = 1.0

    @property
    def area(self) -> float:
        """ã = π δx δp = π ℏ² / (Δx Δp)."""
        return np.pi * self.delta_x * self.delta_p

    def __post_init__(self) -> None:
        if self.delta_x <= 0 or self.delta_p <= 0:
            raise ValueError("Quorum-cell semi-axes must be positive.")


def bitangent_blob_at(
    theta: float,
    heisenberg: HeisenbergCell,
    hbar: float = 1.0,
) -> BitangentBlob:
    """The unique bitangent blob a_θ of action h/2 inscribed in `heisenberg`
    at quadrature angle θ.

    r_∥(θ) = sqrt(Δx² cos²θ + Δp² sin²θ).
    r_⊥(θ) = ℏ / r_∥(θ).
    """
    r_par = np.sqrt(
        (heisenberg.Delta_x * np.cos(theta)) ** 2
        + (heisenberg.Delta_p * np.sin(theta)) ** 2
    )
    r_perp = hbar / r_par
    return BitangentBlob(
        theta=theta,
        r_parallel=r_par,
        r_perp=r_perp,
        center=heisenberg.center,
        hbar=hbar,
    )


def blob_a_pi_half(heisenberg: HeisenbergCell, hbar: float = 1.0) -> BitangentBlob:
    """The bitangent blob a_{π/2}: the family member at θ = π/2.

    r_∥ = Δp along p, r_⊥ = δx = ℏ/Δp along x. Semi-axes (δx, Δp).
    """
    return bitangent_blob_at(np.pi / 2, heisenberg, hbar)


def blob_a_zero(heisenberg: HeisenbergCell, hbar: float = 1.0) -> BitangentBlob:
    """The bitangent blob a_{θ=0}: the family member at θ = 0.

    r_∥ = Δx along x, r_⊥ = δp = ℏ/Δx along p. Semi-axes (Δx, δp).
    """
    return bitangent_blob_at(0.0, heisenberg, hbar)


def quorum_cell(heisenberg: HeisenbergCell, hbar: float = 1.0) -> QuorumCell:
    """The quorum cell ã with semi-axes (ℏ/Δp, ℏ/Δx).

    The common interior of the bitangent family inscribed in `heisenberg`.
    Area ã = π ℏ²/(Δx Δp), the resolution of the convolved portrait W̃.
    """
    return QuorumCell(
        delta_x=hbar / heisenberg.Delta_p,
        delta_p=hbar / heisenberg.Delta_x,
        center=heisenberg.center,
        hbar=hbar,
    )
