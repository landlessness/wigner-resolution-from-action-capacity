"""Render the Cluster 1 baseline figure: the quantum harmonic oscillator.

Five columns x four rows. Layout per row identical to the eigen and cat
figures:
  Col 1: W(x, p) with Heisenberg cell A and quorum cell a-tilde overlays.
  Col 2: W(x, 0) cross-section.
  Col 3: K_{pi/2}(x, p) bitangent kernel, with A and a_{pi/2} overlays.
  Col 4: P_{pi/2}(x, 0) convolved cross-section.
  Col 5: W-tilde(x, p) convolved portrait, with A and a-tilde overlays.

Narrative (the control group): start from the fundamental Gaussian blob
of the vacuum (n=0), introduce Wigner negativity and rotational symmetry
with the low Fock states (n=1, n=2), then bridge back toward classical
physics at n=50, where the rings condense onto a classical energy
trajectory. Takeaway: as A grows, a-tilde shrinks, and the rotational
structure of the QHO is carried in a-tilde.

Note on n=50: the integration grid and the theta-family convolution both
handle it (Delta_x = Delta_p = sqrt(101) ~ 10.05, A/(h/2) = 101); the
360-theta portrait takes appreciably longer than the low-n rows but is
non-negative to floating-point precision.

Output: tex/figures/harmonic.pdf.
"""

from __future__ import annotations

from pathlib import Path

from wigner_resolution.figures.grid import assemble_grid_5col, save_grid
from wigner_resolution.plotstyle import use_prl_style
from wigner_resolution.systems.harmonic import harmonic_state

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "harmonic.pdf"

use_prl_style(use_tex=True)

states = [
    harmonic_state(n=0),
    harmonic_state(n=1),
    harmonic_state(n=2),
    harmonic_state(n=20),
]

row_labels = [
    r"QHO, $n = 0$",
    r"QHO, $n = 1$",
    r"QHO, $n = 2$",
    r"QHO, $n = 20$",
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
