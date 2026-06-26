# Code and figures for "Wigner Resolution from Action Capacity"

Repository for the manuscript *Wigner Resolution from Action Capacity*
(B. S. Mulloy, 2026).

Smoothing the Wigner function yields a non-negative phase-space portrait, and
the smoothing scale has conventionally been set from outside the state. This
paper shows that the state sets it. A state's action capacity
`A = π Δx Δp ≥ h/2` fixes a family of de Gosson quantum blobs `β_θ` that deform
at constant action `h/2`. Integrating the Planck-scale convolutions of the
Wigner function across all quadrature angles `θ` produces a portrait that is
everywhere non-negative, resolved at the symplectic polar dual of the capacity,

    a = πℏ²/(Δx Δp) ≤ h/2,   with   a · A = (h/2)².

There is no free parameter: the capacity alone fixes the resolution, which
sharpens as the capacity grows. The interference scales Zurek located are
recovered as the principal axes of the blob family.

This repository contains everything needed to reproduce the paper: the Python
code that generates the data figures, the OmniGraffle source for the
schematic figures, and the LaTeX source of the manuscript.

## Structure

- `python/` — Python project (managed with [uv](https://docs.astral.sh/uv/))
  - `src/wigner_resolution/` — reusable modules
    - `geometry.py`, `kernels.py`, `wigner.py` — phase-space geometry (capacity `A`, quantum blobs `β_θ`, resolution `a`), quantum kernels, Wigner function
    - `convolve.py` — quantum blob family convolution across the quadrature angle
    - `quantum.py`, `state.py` — quantum-state representation
    - `systems/` — the states used in the paper (squeezed vacuum, harmonic oscillator, Morse and double-well eigenstates, the Kerr crescent, and Schrödinger cat states)
    - `figures/` — panel construction, grid layout, overlay graphics
    - `plotstyle.py`, `ticks.py` — Matplotlib styling
  - `notebooks/` — one script per data figure (`render_eigen.py`, `render_cat.py`) plus the symplectic verification (`verify_blobs.py`)
- `omnigraffle/` — OmniGraffle source for the schematic figures
  - `capacity_blobs_resolution.graffle` — schematic of the capacity, quantum blobs, and resolution
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

There is one `render_*.py` script per data figure; each writes its figure to
`tex/figures/`, where `main.tex` includes it.

## Verifying the symplectic blob property

The manuscript identifies the quantum blob family `β_θ` as a family of *de Gosson
quantum blobs* — a symplectic statement: each `β_θ` has symplectic capacity
`πℏ = h/2` and is self-dual under the symplectic polar duality of de Gosson &
de Gosson, *Symmetry* **14**, 1890 (2022). In one degree of freedom the
symplectic capacity of a planar ellipse equals its Euclidean area, so the
reciprocal-axes identity `r_∥ · r_⊥ = ℏ` enforced in `geometry.py` is exactly the
quantum-blob condition.

`notebooks/verify_blobs.py` checks this directly at the symplectic level —
computing the symplectic capacity and symplectic eigenvalue of `β_θ` from its
covariance via `J Σ`, rather than relying on the Euclidean proxy — across a
dense sweep in `θ ∈ [0, π)` and across the capacity range `A/(h/2) ∈ [1, 70]`.
It confirms:

1. the symplectic capacity of `β_θ` equals `h/2` for every `θ`;
2. the symplectic eigenvalue equals `ℏ/2` for every `θ` (the self-duality /
   quantum-blob condition);
3. the capacity is `θ`-independent (constant action across the family);
4. the kernel `K_θ` (`kernels.py`) and the blob `β_θ` (`geometry.py`) describe the
   same ellipse;
5. the capacity `A` circumscribes the quantum blobs `β_θ` and the quantum blobs circumscribe the resolution `a`.

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

A companion check, `notebooks/verify_non_negative.py`, confirms the convolved
portrait `W̃` is non-negative to floating-point precision across the paper's
states, and `notebooks/verify_blobs_skeptic.py` repeats the symplectic battery
on the skeptic states.

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