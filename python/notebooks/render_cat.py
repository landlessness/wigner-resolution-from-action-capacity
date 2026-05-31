"""Render the four-row cat-states figure: 5 columns × 4 rows.

Layout per row (same as the eigen figure):
  Col 1: W(x, p) — Wigner function, no overlays.
  Col 2: W(x, 0) — Wigner cross-section.
  Col 3: K_{π/2}(x, p) — quantum kernel, with Heisenberg cell A and
         quantum blob a_{π/2} overlays.
  Col 4: P_{π/2}(x, 0) — convolved cross-section.
  Col 5: W̃(x, p) — convolved portrait, with Heisenberg cell A and
         quorum cell ã overlays.

The 4-cat and Zurek-compass rows differ by a π/4 rotation in
phase space; cols 1 and 3 distinguish them via the cell overlay
orientations, col 5 does not because W̃ is rotation-invariant under
the family integral.

Output: tex/figures/cat.pdf.
"""

from __future__ import annotations

from pathlib import Path

from wigner_resolution.figures.grid import assemble_grid_5col, save_grid
from wigner_resolution.plotstyle import use_prl_style
from wigner_resolution.systems.cat import cat_state

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "cat.pdf"

use_prl_style(use_tex=True)

states = [
    cat_state(2),
    cat_state(3),
    cat_state(4, variant="diag"),
    cat_state(4, variant="axis"),
]

row_labels = [
    r"2-cat",
    r"3-cat",
    r"4-cat",
    r"Zurek compass",
]

for s in states:
    print(
        f"{s.name}: Δx={s.rs.Delta_x:.3f}, Δp={s.rs.Delta_p:.3f}, "
        f"⟨x⟩={s.rs.x_mean:.3f}, cell_x={s.cell_center_x:.3f}, "
        f"A/(h/2)={s.rs.A_over_h_half:.2f}"
    )

fig = assemble_grid_5col(
    states,
    row_labels=row_labels,
    tilde_W_n_theta=360,
)
save_grid(fig, OUT)
