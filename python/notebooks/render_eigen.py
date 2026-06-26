"""Render the Cluster 2 figure: deformations and nonlinearities.

Five columns x four rows, layout identical to the other data figures.
This is the former eigen figure, recast as the "break the symmetry"
cluster. The n=1 harmonic Fock state has moved to the baseline figure
(render_harmonic.py); the fourth row is now the Kerr crescent.

Order and narrative (systematically breaking the QHO's symmetry):
  1. Squeezed ground state  — still Gaussian, rotational symmetry broken,
                              the n=0 blob stretched along one axis.
  2. Morse eigenstate       — spatial symmetry broken; an asymmetric,
                              teardrop wavepacket in an uneven potential.
  3. Double-well eigenstate — the wavepacket splits into two wells with
                              faint inter-well interference: a spatial
                              precursor to a cat.
  4. Kerr crescent          — a coherent state sheared by H = chi n^2;
                              the same Hamiltonian run to chi t = pi makes
                              a cat, so the crescent is the dynamical
                              bridge into Cluster 3.

Takeaway: the state sets the shape of A, A sets a, and the shape of
a matches the state's finest details.

The Kerr state is rotated into its covariance principal frame so the
overlay ellipses are axis-aligned (see systems/kerr.py); the theta
integral in column 5 is rotation-invariant regardless.

Output: tex/figures/eigen.pdf.
"""

from __future__ import annotations

from pathlib import Path

from wigner_resolution.figures.grid import assemble_grid_5col, save_grid
from wigner_resolution.plotstyle import use_prl_style
from wigner_resolution.systems.double_well import double_well_state
from wigner_resolution.systems.kerr import kerr_state
from wigner_resolution.systems.morse import morse_state
from wigner_resolution.systems.squeezed_vacuum import squeezed_vacuum_state

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "eigen.pdf"

use_prl_style(use_tex=True)

states = [
    squeezed_vacuum_state(r=0.5),
    morse_state(n=8),
    double_well_state(n=5),
    kerr_state(),
]

row_labels = [
    r"Squeezed vacuum, $r = 0.5$",
    r"Morse, $n = 8$",
    r"Double-well, $n = 5$",
    r"Kerr crescent",
]

for s in states:
    print(
        f"{s.name}: Δx={s.rs.Delta_x:.3f}, Δp={s.rs.Delta_p:.3f}, "
        f"⟨x⟩={s.rs.x_mean:.3f}, overlay_x={s.overlay_center_x:.3f}, "
        f"A/(h/2)={s.rs.A_over_h_half:.2f}"
    )

fig = assemble_grid_5col(
    states,
    row_labels=row_labels,
    tilde_W_n_theta=360,
)
save_grid(fig, OUT)
