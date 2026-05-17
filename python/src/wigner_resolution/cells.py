"""Phase-space cells: the Heisenberg cell A, the squeezed family a_θ,
and the quorum cell a.

A state's covariance defines its Heisenberg cell

    A = π Δx Δp  ≥  h/2,

the outer ellipse with anti-squeezed semi-axes (Δx, Δp), floored at the
quantum of action by the uncertainty principle.

Inscribed within A is a one-parameter family of squeezed cells a_θ of
fixed action h/2, each parameterized by orientation θ. The semi-axes
obey the polar-dual identity δ‖ δ⊥ = ℏ, with

    δ‖(θ) = sqrt(Δx² cos²θ + Δp² sin²θ),
    δ⊥(θ) = ℏ / δ‖(θ).

δ‖ is along direction θ from the x-axis. δ⊥ is perpendicular to δ‖.
Neither is the "major" axis in general; which is larger depends on the
aspect ratio of the Heisenberg cell.

Two members of the family align with the principal axes: a_{δp} at
θ = 0 and a_{δx} at θ = π/2. At those orientations δ⊥ reduces to the
reciprocal scales

    δx = ℏ/Δp,  δp = ℏ/Δx.

The quorum cell a is the axis-aligned ellipse with squeezed semi-axes

    (δx, δp) = (ℏ/Δp, ℏ/Δx),

the polar dual of A. It is *not* a member of the inscribed family; it
is its own object, marking the inner envelope of sub-Planck structure
that the rotation-averaged kernel can no longer resolve. Its area is

    π δx δp = π ℏ² / (Δx Δp) = (h/2)² / A.

The quorum cell was identified by Zurek, Nature 412, 712 (2001).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeisenbergCell:
    """The state's outer extent A in phase space.

    Anti-squeezed semi-axes Δx, Δp along the principal axes
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
class SqueezedCell:
    """One member a_θ of the polar-dual family inscribed in a Heisenberg cell.

    Of fixed action h/2. Oriented at angle θ from the x-axis. δ‖ is along
    θ; δ⊥ is perpendicular. δ‖ δ⊥ = ℏ.
    """

    theta: float
    delta_parallel: float
    delta_perp: float
    center: tuple[float, float] = (0.0, 0.0)
    hbar: float = 1.0

    @property
    def area(self) -> float:
        """a_θ = π δ‖ δ⊥ = h/2 (always)."""
        return np.pi * self.delta_parallel * self.delta_perp

    def __post_init__(self) -> None:
        product = self.delta_parallel * self.delta_perp
        if not np.isclose(product, self.hbar, rtol=1e-10):
            raise ValueError(
                f"Polar-dual identity violated: δ‖·δ⊥ = {product:.6g} ≠ ℏ = {self.hbar}"
            )


@dataclass(frozen=True)
class QuorumCell:
    """The quorum cell a with squeezed semi-axes (δx, δp) = (ℏ/Δp, ℏ/Δx).

    Polar dual of the Heisenberg cell: A is Δx-wide and Δp-tall, a is
    δx-wide (= ℏ/Δp, narrow when Δp is large) and δp-tall (= ℏ/Δx).
    Axis-aligned with no rotation parameter; not a member of the
    inscribed squeezed family.

    Reference: Zurek, Nature 412, 712 (2001).
    """

    delta_x: float
    delta_p: float
    center: tuple[float, float] = (0.0, 0.0)
    hbar: float = 1.0

    @property
    def area(self) -> float:
        """π δx δp = π ℏ² / (Δx Δp) = (h/2)² / A."""
        return np.pi * self.delta_x * self.delta_p

    def __post_init__(self) -> None:
        if self.delta_x <= 0 or self.delta_p <= 0:
            raise ValueError("Quorum-cell semi-axes must be positive.")


def squeezed_cell_at(
    theta: float,
    heisenberg: HeisenbergCell,
    hbar: float = 1.0,
) -> SqueezedCell:
    """The unique squeezed cell a_θ of action h/2 inscribed in `heisenberg`
    at angle θ.

    δ‖(θ) = sqrt(Δx² cos²θ + Δp² sin²θ).
    δ⊥(θ) = ℏ / δ‖(θ).
    """
    d_par = np.sqrt(
        (heisenberg.Delta_x * np.cos(theta)) ** 2
        + (heisenberg.Delta_p * np.sin(theta)) ** 2
    )
    d_perp = hbar / d_par
    return SqueezedCell(
        theta=theta,
        delta_parallel=d_par,
        delta_perp=d_perp,
        center=heisenberg.center,
        hbar=hbar,
    )


def cell_a_delta_x(heisenberg: HeisenbergCell, hbar: float = 1.0) -> SqueezedCell:
    """The squeezed-family member a_{δx}: the cell at θ = π/2.

    δ‖ = Δp along p, δ⊥ = δx = ℏ/Δp along x.
    """
    return squeezed_cell_at(np.pi / 2, heisenberg, hbar)


def cell_a_delta_p(heisenberg: HeisenbergCell, hbar: float = 1.0) -> SqueezedCell:
    """The squeezed-family member a_{δp}: the cell at θ = 0.

    δ‖ = Δx along x, δ⊥ = δp = ℏ/Δx along p.
    """
    return squeezed_cell_at(0.0, heisenberg, hbar)


def quorum_cell(heisenberg: HeisenbergCell, hbar: float = 1.0) -> QuorumCell:
    """The quorum cell a with semi-axes (ℏ/Δp, ℏ/Δx).

    Axis-aligned ellipse with semi-axes (δx, δp) = (ℏ/Δp, ℏ/Δx). The
    polar dual of `heisenberg` in the symplectic-area sense:
    a · A = (h/2)² · (π² / π²) — see QuorumCell.area for details.
    """
    return QuorumCell(
        delta_x=hbar / heisenberg.Delta_p,
        delta_p=hbar / heisenberg.Delta_x,
        center=heisenberg.center,
        hbar=hbar,
    )
