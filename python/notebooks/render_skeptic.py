"""Render the Cluster 4 figure: states that test the assumptions.

Five columns by four rows, layout identical to the other data figures.
Each row isolates one premise the construction rests on, so that a
result holds the claim up rather than a reader's doubt undercutting it.

Order and narrative:
  1. Cubic-phase state   — non-quadratic shear, no symplectic principal
                           frame; the covariance ellipse A is necessarily
                           a coarse proxy for the geometry.
  2. Thermal state       — featureless Gaussian, large A, no structure;
                           a as an upper bound on resolution rather
                           than a marker of present structure.
  3. Asymmetric cat      — collinear unequal-weight lobes; separation and
                           fringe spacing decoupled, unlike the symmetric
                           cats of Cluster 3.
  4. Heavy-tailed state  — cusped wavefunction, large finite <p^2>;
                           strains A = pi Delta_x Delta_p as a finite,
                           sensible ellipse.

Takeaway: the portrait W-tilde is non-negative across all four, and
a marks the finest structure wherever the state has structure to
mark; where it does not (thermal), a reads as the resolution
floor. The construction holds outside the regime of the first twelve
states.

Output: tex/figures/skeptic.pdf.
"""

from __future__ import annotations

from pathlib import Path

from wigner_resolution.figures.grid import assemble_grid_5col, save_grid
from wigner_resolution.plotstyle import use_prl_style
from wigner_resolution.systems.skeptic import (
    asymmetric_cat_state,
    cubic_phase_state,
    heavy_tailed_state,
    thermal_state,
)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "skeptic.pdf"

use_prl_style(use_tex=True)

states = [
    cubic_phase_state(),
    thermal_state(),
    asymmetric_cat_state(),
    heavy_tailed_state(),
]

row_labels = [
    r"Cubic phase",
    r"Thermal, $\bar n = 3$",
    r"Asymmetric cat",
    r"Heavy-tailed",
]

for s in states:
    print(
        f"{s.name}: Δx={s.rs.Delta_x:.3f}, Δp={s.rs.Delta_p:.3f}, "
        f"⟨x⟩={s.rs.x_mean:.3f}, overlay_x={s.overlay_center_x:.3f}, "
        f"A/(h/2)={s.rs.A_over_h_half:.2f}, Wmin={s.W.min():+.4f}"
    )

fig = assemble_grid_5col(
    states,
    row_labels=row_labels,
    tilde_W_n_theta=360,
)
save_grid(fig, OUT)
