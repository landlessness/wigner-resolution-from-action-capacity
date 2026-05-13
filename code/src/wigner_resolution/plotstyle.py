"""PRL-style matplotlib configuration.

Usage:
    from wigner_resolution.plotstyle import use_prl_style, COLUMN_WIDTH, DOUBLE_WIDTH
    use_prl_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# PRL column widths in inches
COLUMN_WIDTH = 3.375   # single column
DOUBLE_WIDTH = 7.0     # full page width
GOLDEN = (1 + 5**0.5) / 2  # for default aspect ratio


def use_prl_style(use_tex: bool = True) -> None:
    """Apply PRL-style matplotlib rcParams.

    Set use_tex=False if you don't have a LaTeX installation;
    figures will use matplotlib's mathtext instead.
    """
    mpl.rcParams.update({
        # Fonts
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # LaTeX preamble for symbols you'll likely need
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{physics}",

        # Figure
        "figure.figsize": (COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN),
        "figure.dpi": 150,           # for on-screen preview
        "savefig.dpi": 600,          # for raster fallback
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Axes
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,

        # Lines
        "lines.linewidth": 1.0,
        "lines.markersize": 3,

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.5,
    })


def save_figure(fig, name: str, figdir: str = "../../tex/figures") -> None:
    """Save a figure to the tex/figures directory as PDF."""
    import os
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, f"{name}.pdf")
    fig.savefig(path)
    print(f"Saved: {path}")
