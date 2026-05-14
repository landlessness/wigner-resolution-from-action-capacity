"""Phase-space cells of the polar-dual family.

The state's covariance defines two phase-space cells (Mulloy 2026, Fig. 1).

The extended cell:

    A = π Δx Δp  ≥  h/2

is the state's outer extent, floored at the quantum of action by Heisenberg.

Inscribed within it is a one-parameter family of squeezed cells of fixed
action

    a = π δ‖ δ⊥  =  h/2

each parameterized by orientation θ. The semi-axes obey the polar-dual
identity δ‖ δ⊥ = ℏ, with

    δ‖(θ) = sqrt(Δx² cos²θ + Δp² sin²θ)
    δ⊥(θ) = ℏ / δ‖(θ).

δ‖ is along direction θ from the x-axis. δ⊥ is perpendicular to δ‖.
Neither is the "major" axis in general; which is larger depends on the
aspect ratio of the extended cell.

At θ = 0 and θ = π/2 (Fig. 2) the squeezed cell aligns with the principal
axes and δ⊥ reduces to Zurek's reciprocal scales

    δx = ℏ/Δp,  δp = ℏ/Δx.

The Zurek cell is the axis-aligned ellipse with semi-axes (δx, δp) ---
the polar dual of the extended cell. It is *not* a member of the
inscribed family; it is its own object, marking the inner envelope of
sub-Planck structure that the rotation-averaged kernel can no longer
resolve. Its area is

    π δx δp = π ℏ² / (Δx Δp) = (h/2)² / A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExtendedCell:
    """The state's outer extent in phase space.

    Semi-axes Δx, Δp along the principal axes (Robertson-Schrödinger
    cross-term assumed zero, which holds for the symmetric states in
    the manuscript).
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
    """One member of the polar-dual family inscribed in an extended cell.

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
        """a = π δ‖ δ⊥ = h/2 (always)."""
        return np.pi * self.delta_parallel * self.delta_perp

    def __post_init__(self) -> None:
        product = self.delta_parallel * self.delta_perp
        if not np.isclose(product, self.hbar, rtol=1e-10):
            raise ValueError(
                f"Polar-dual identity violated: δ‖·δ⊥ = {product:.6g} ≠ ℏ = {self.hbar}"
            )


@dataclass(frozen=True)
class ZurekCell:
    """The sub-Planck cell with semi-axes (δx, δp) = (ℏ/Δp, ℏ/Δx).

    Polar dual of the extended cell: the outer ellipse is Δx-wide and
    Δp-tall, the Zurek cell is δx-wide (= ℏ/Δp, narrow when Δp is large)
    and δp-tall (= ℏ/Δx). Axis-aligned with no rotation parameter; not
    a member of the inscribed squeezed family.

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
            raise ValueError("Zurek-cell semi-axes must be positive.")


def squeezed_cell_at(
    theta: float,
    extended: ExtendedCell,
    hbar: float = 1.0,
) -> SqueezedCell:
    """The unique squeezed cell of action h/2 inscribed in `extended` at angle θ.

    δ‖(θ) = sqrt(Δx² cos²θ + Δp² sin²θ).
    δ⊥(θ) = ℏ / δ‖(θ).
    """
    d_par = np.sqrt(
        (extended.Delta_x * np.cos(theta)) ** 2
        + (extended.Delta_p * np.sin(theta)) ** 2
    )
    d_perp = hbar / d_par
    return SqueezedCell(
        theta=theta,
        delta_parallel=d_par,
        delta_perp=d_perp,
        center=extended.center,
        hbar=hbar,
    )


def cell_a_delta_x(extended: ExtendedCell, hbar: float = 1.0) -> SqueezedCell:
    """Zurek's δx cell as a member of the squeezed family: squeezed cell
    at θ = π/2.

    δ‖ = Δp along p, δ⊥ = δx = ℏ/Δp along x.
    """
    return squeezed_cell_at(np.pi / 2, extended, hbar)


def cell_a_delta_p(extended: ExtendedCell, hbar: float = 1.0) -> SqueezedCell:
    """Zurek's δp cell as a member of the squeezed family: squeezed cell
    at θ = 0.

    δ‖ = Δx along x, δ⊥ = δp = ℏ/Δx along p.
    """
    return squeezed_cell_at(0.0, extended, hbar)


def zurek_cell(extended: ExtendedCell, hbar: float = 1.0) -> ZurekCell:
    """The Zurek sub-Planck cell with semi-axes (ℏ/Δp, ℏ/Δx).

    Axis-aligned ellipse with semi-axes (δx, δp) = (ℏ/Δp, ℏ/Δx). The
    polar dual of `extended` in the symplectic-area sense: their areas
    multiply to (h/2)² · ... — see ZurekCell.area for details.
    """
    return ZurekCell(
        delta_x=hbar / extended.Delta_p,
        delta_p=hbar / extended.Delta_x,
        center=extended.center,
        hbar=hbar,
    )
