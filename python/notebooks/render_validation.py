"""Render the Supplement numerical-validation figures and tables.

Position-based comparison of the bare Wigner function W and the non-negative
portrait W-tilde, every instrument run identically on both. The twelve states
are the same library shown across the Letter's data figures.

Outputs (into tex/figures/):
  colocation_composite.pdf  feature radius in W vs in W-tilde, all 12 states
  colocation_grid.pdf       W-tilde heat map with W's positive maxima, all 12
  shared_support.pdf        chord power of W and W-tilde, Fock n=20
  ring_colocation.pdf       radial profiles of W and W-tilde, Fock n=20

Tables A (coherence and phase), B (co-location), and C (non-negativity) print
to stdout so the values can be checked against the Supplement LaTeX.

Note on n=20: built with harmonic_state(n=20), the same 401-grid state as the
Letter's figures. Its rings undersample at that grid, which inflates its
co-location offset to about 1.2 delta. To tighten only that one row, build the
state on a finer grid before passing it in; the instruments are grid-agnostic.

Output: tex/figures/{colocation_composite,colocation_grid,shared_support,ring_colocation}.pdf.
"""

from __future__ import annotations

from pathlib import Path

from wigner_resolution.figures.grid import save_grid
from wigner_resolution.figures.validation import (
    build_composite,
    build_grid,
    build_ring_colocation,
    build_shared_support,
    print_tables,
    run_validation,
)
from wigner_resolution.plotstyle import use_prl_style
from wigner_resolution.systems.cat import cat_state
from wigner_resolution.systems.double_well import double_well_state
from wigner_resolution.systems.harmonic import harmonic_state
from wigner_resolution.systems.kerr import kerr_state
from wigner_resolution.systems.morse import morse_state
from wigner_resolution.systems.squeezed_vacuum import squeezed_vacuum_state

HERE = Path(__file__).resolve().parent
FIG = HERE.parent.parent / "tex" / "figures"

use_prl_style(use_tex=True)

states = [
    harmonic_state(n=0),
    harmonic_state(n=1),
    harmonic_state(n=2),
    harmonic_state(n=20),
    squeezed_vacuum_state(r=0.5),
    morse_state(n=8),
    double_well_state(n=5),
    kerr_state(),
    cat_state(2),
    cat_state(3),
    cat_state(4, variant="diag"),
    cat_state(4, variant="axis"),
]

labels = [
    "QHO n=0",
    "QHO n=1",
    "QHO n=2",
    "QHO n=20",
    "Squeezed",
    "Morse",
    "Double-well",
    "Kerr",
    "2-cat",
    "3-cat",
    "4-cat diag",
    "Compass",
]

for s in states:
    print(
        f"{s.name}: Δx={s.rs.Delta_x:.3f}, Δp={s.rs.Delta_p:.3f}, "
        f"⟨x⟩={s.rs.x_mean:.3f}, cell_x={s.cell_center_x:.3f}, "
        f"A/(h/2)={s.rs.A_over_h_half:.2f}"
    )

rows, scatter, portraits = run_validation(states, labels, n_theta=360)
print_tables(rows)

i20 = labels.index("QHO n=20")
save_grid(build_composite(scatter, labels), FIG / "colocation_composite.pdf")
save_grid(build_grid(states, labels, portraits), FIG / "colocation_grid.pdf")
save_grid(build_shared_support(states[i20], portraits["QHO n=20"]), FIG / "shared_support.pdf")
save_grid(build_ring_colocation(states[i20], portraits["QHO n=20"]), FIG / "ring_colocation.pdf")
