# Code and figures for "Wigner Resolution from Action Capacity"

Repository for the manuscript *Wigner Resolution from Action Capacity*
(B. S. Mulloy, 2026).

The paper identifies the characteristic scales of Zurek as the axes of a
family of de Gosson quantum blobs inscribed within the state's Heisenberg
cell, and integrates Planck-scale convolutions of the Wigner function across
all angles to produce a non-negative phase-space portrait at sub-Planck
resolution.

This repository contains everything needed to reproduce the paper: the Python
code that generates the data figures, the OmniGraffle source for the
schematic figures, and the LaTeX source of the manuscript.

## Structure

- `python/` — Python project (managed with [uv](https://docs.astral.sh/uv/))
  - `src/wigner_resolution/` — reusable modules
    - `cells.py`, `kernels.py`, `wigner.py` — phase-space cells, quantum kernels, Wigner function
    - `convolve.py` — quantum blob family convolution across the quadrature angle
    - `quantum.py`, `state.py` — quantum-state representation
    - `systems/` — the eight states used in the paper (squeezed vacuum, harmonic, Morse, double-well, cat states)
    - `figures/` — panel construction, grid layout, overlay graphics
    - `plotstyle.py`, `ticks.py` — Matplotlib styling
  - `notebooks/` — one script per data figure (`render_eigen.py`, `render_cat.py`) plus the symplectic verification (`verify_blobs.py`)
- `omnigraffle/` — OmniGraffle source for the schematic figures
  - `heisenberg_cells.graffle` — schematic of the Heisenberg cell, quantum blobs, and quorum cell
  - `equations.tex` and friends — LaTeX-rendered equations exported as SVG for use inside OmniGraffle
- `tex/` — LaTeX manuscript source
  - `main.tex`, `main.bib` — manuscript and bibliography
  - `supplement.tex` — supplemental material
  - `figures/` — PDF figures included by `main.tex` (schematics from `omnigraffle/`, data figures from `python/`)

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) for environment management
- LaTeX with the `revtex4-2` class (for the manuscript)
- OmniGraffle (optional, only to edit the schematic figures)

## Reproducing the data figures

```bash
cd python
uv sync
uv run python notebooks/render_eigen.py
uv run python notebooks/render_cat.py
```

Each script writes its figure to `tex/figures/`, where `main.tex` includes it.

## Verifying the symplectic blob property

The manuscript identifies the quantum blob family `a_θ` as a family of *de Gosson
quantum blobs* — a symplectic statement: each `a_θ` has symplectic capacity
`πℏ = h/2` and is self-dual under the symplectic polar duality of de Gosson &
de Gosson, *Symmetry* **14**, 1890 (2022). In one degree of freedom the
symplectic capacity of a planar ellipse equals its Euclidean area, so the
reciprocal-axes identity `r_∥ · r_⊥ = ℏ` enforced in `cells.py` is exactly the
quantum-blob condition.

`notebooks/verify_blobs.py` checks this directly at the symplectic level —
computing the symplectic capacity and symplectic eigenvalue of `a_θ` from its
covariance via `J Σ`, rather than relying on the Euclidean proxy — across a
dense sweep in `θ ∈ [0, π)` and across the capacity range `A/(h/2) ∈ [1, 70]`.
It confirms:

1. the symplectic capacity of `a_θ` equals `h/2` for every `θ`;
2. the symplectic eigenvalue equals `ℏ/2` for every `θ` (the self-duality /
   quantum-blob condition);
3. the capacity is `θ`-independent (constant action across the family);
4. the kernel `K_θ` (`kernels.py`) and the cell `a_θ` (`cells.py`) describe the
   same ellipse;
5. the Heisenberg cell `A` circumscribes the quantum blobs `a_θ` and the quantum blobs circumscribe the quorum cell `ã`.

Run it:

```bash
cd python
uv run python notebooks/verify_blobs.py
```

The script fails fast: it prints a per-regime table and exits non-zero on any
violation, so it can be wired into CI. A passing run reports symplectic
capacity `π` (= `h/2` in the paper's `ℏ = 1` units) constant to floating-point
tolerance at every `θ` and every capacity, confirming that every member of the
family is a genuine symplectic quantum blob — not merely an area-`h/2` ellipse
coinciding with a blob at the principal angles.

## Building the manuscript

```bash
cd tex
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Citation

This release is archived on Zenodo: [10.5281/zenodo.20279094](https://doi.org/10.5281/zenodo.20279094)

Suggested citation:

> Mulloy, B. S. (2026). *Code and figures for "Wigner Resolution from Action Capacity"* (v1.2.3). Zenodo. https://doi.org/10.5281/zenodo.20279094

A BibTeX entry is available from the Zenodo deposit page, or from the
"Cite this repository" button on GitHub (populated from `CITATION.cff`).

For citing the paper itself, see the preprint (not yet peer-reviewed) on
ResearchGate: https://www.researchgate.net/publication/405021611_Wigner_Resolution_from_Action_Capacity

## License

- **Code** (`python/`): MIT License — see `LICENSE` in the repo root.
- **Manuscript and figures** (`tex/`, `omnigraffle/`): Creative Commons
  Attribution 4.0 International (CC BY 4.0) — see `tex/LICENSE` and `omnigraffle/LICENSE`.

The figures in `tex/figures/` are part of the manuscript and fall under the
manuscript license, even though the data figures are generated by the code.

## Contact

Brian S. Mulloy — bmulloy@umich.edu