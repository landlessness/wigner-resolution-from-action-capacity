# wigner_resolution

Code and manuscript source for *Wigner Resolution from the Action Capacity of the State*.

This repository contains the Python code (QuTiP) used to generate the figures,
and the LaTeX source of the manuscript.

## Structure

- `code/` — Python project (managed with [uv](https://docs.astral.sh/uv/))
  - `src/wigner_resolution/` — reusable modules (plot style, kernels, etc.)
  - `notebooks/` — Jupyter notebooks that produce each figure
- `tex/` — LaTeX manuscript source
  - `figures/` — generated PDF figures included by `main.tex`

## Reproducing the figures

```bash
cd code
uv sync
uv run jupyter lab
```

Then run the notebooks in `notebooks/`.

## Citation

A Zenodo DOI will be minted at submission.
