# wigner_resolution

Code and manuscript source for *Wigner Resolution from the Action Capacity of the State*.

This repository contains the Python code used to generate the data figures,
the OmniGraffle source for the schematic figures, and the LaTeX source of the
manuscript.

## Structure

- `python/` — Python project (managed with [uv](https://docs.astral.sh/uv/))
  - `src/wigner_resolution/` — reusable modules (cells, kernels, Wigner, panels)
  - `notebooks/` — scripts that produce each data figure
- `omnigraffle/` — OmniGraffle source files for the schematic figures
- `tex/` — LaTeX manuscript source
  - `figures/` — generated PDF figures included by `main.tex`

## Reproducing the figures

```bash
cd python
uv sync
uv run python notebooks/render_eigen.py
```

The rendered figure is written to `tex/figures/eigen.pdf`, which `main.tex` then
includes.

## Citation

A Zenodo DOI will be minted at submission.