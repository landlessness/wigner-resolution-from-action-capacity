"""Render the four-row eigenstates figure: 5 columns × 4 rows.

Layout per row (same as the cat figure):
  Col 1: W(x, p) — Wigner function, no overlays.
  Col 2: W(x, 0) — Wigner cross-section.
  Col 3: K_{π/2}(x, p) — bitangent kernel, with Heisenberg cell A and
         bitangent blob a_{π/2} overlays.
  Col 4: P_{π/2}(x, 0) — convolved cross-section.
  Col 5: W̃(x, p) — convolved portrait, with Heisenberg cell A and
         quorum cell ã overlays.

Output: tex/figures/eigen.pdf.
"""

from __future__ import annotations

from pathlib import Path

from wigner_resolution.figures.grid import assemble_grid_5col, save_grid
from wigner_resolution.plotstyle import use_prl_style
from wigner_resolution.systems.double_well import double_well_state
from wigner_resolution.systems.harmonic import harmonic_state
from wigner_resolution.systems.morse import morse_state
from wigner_resolution.systems.squeezed_vacuum import squeezed_vacuum_state

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "eigen.pdf"

use_prl_style(use_tex=True)

states = [
    squeezed_vacuum_state(r=0.5),
    harmonic_state(n=1),
    morse_state(n=8),
    double_well_state(n=5),
]

row_labels = [
    r"Squeezed vacuum, $r = 0.5$",
    r"Harmonic, $n = 1$",
    r"Morse, $n = 8$",
    r"Double-well, $n = 5$",
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
