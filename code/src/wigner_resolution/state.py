"""Per-state bundle and the shared pipeline that builds it.

A ``State`` carries the wavefunction, its RS geometry, and an evaluated
Wigner function on a uniform integration grid. Downstream consumers (the
figure code, the convolution code) read fields off this bundle.

For every state in the manuscript, the path from ψ to a populated bundle
is identical: sample ψ, compute its RS geometry, run wigner_fft on an
integration grid that pads the display window. That path lives in
``build_state``; per-system modules just deliver ψ and pick a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .quantum import RSGeometry, numerical_covariance
from .wigner import wigner_fft, wigner_norm


@dataclass(frozen=True)
class DisplayWindow:
    """Phase-space display window for a single panel row.

    ``x_lim`` and ``p_lim`` are *half-widths* relative to the window's
    center. The window covers [x_center - x_lim, x_center + x_lim] in x
    and [p_center - p_lim, p_center + p_lim] in p. For symmetric states
    the center sits at the origin and the legacy "limit = half-width"
    reading is intact; for asymmetric states (Morse) the center shifts
    to where the state actually lives.
    """

    x_lim: float
    p_lim: float
    x_center: float = 0.0
    p_center: float = 0.0
    x_ticks: tuple[float, ...] = ()
    p_ticks: tuple[float, ...] = ()

    @property
    def x_min(self) -> float:
        return self.x_center - self.x_lim

    @property
    def x_max(self) -> float:
        return self.x_center + self.x_lim

    @property
    def p_min(self) -> float:
        return self.p_center - self.p_lim

    @property
    def p_max(self) -> float:
        return self.p_center + self.p_lim


@dataclass
class State:
    """A fully-built phase-space state ready for plotting."""

    name: str
    psi: np.ndarray              # wavefunction on x_grid_psi
    x_grid_psi: np.ndarray       # the grid ψ was sampled on
    rs: RSGeometry
    hbar: float
    window: DisplayWindow

    # Visual anchor for the cell overlays. By default ``build_state`` sets
    # this to the location of max |W(x, 0)| — the deepest Wigner negativity
    # along the p=0 cross-section. This is the spot where the cell's
    # resolution argument is visually most legible: the cell sits over the
    # negative dips it would erase under convolution. Systems can override
    # by passing ``cell_center_x`` to ``build_state``.
    cell_center_x: float = 0.0

    # Populated by build_wigner:
    W: np.ndarray | None = None      # 2D Wigner on (x_int, p_int)
    x_int: np.ndarray | None = None
    p_int: np.ndarray | None = None


def build_state(
    name: str,
    psi: np.ndarray,
    x_grid_psi: np.ndarray,
    window: DisplayWindow,
    *,
    hbar: float = 1.0,
    n_x_int: int = 401,
    n_p_int: int = 401,
    x_pad_factor: float = 2.5,
    p_pad: float = 2.0,
    window_margin: float = 0.40,
    cell_center_x: float | None = None,
) -> State:
    """Sample ψ, compute its RS geometry, evaluate W on an integration grid.

    The display window is enforced to contain the full extended cell A
    (with `window_margin` fractional padding past ±Δx, ±Δp). Since the
    squeezed cell is geometrically inscribed in A, this guarantees both
    ellipses are visible in every plotted panel.

    The integration grid is chosen wider than the display window so the
    convolution doesn't suffer edge effects in the display region.

    Parameters
    ----------
    name
        Short identifier, used in logs and as a default save filename.
    psi
        Wavefunction sampled on ``x_grid_psi``. May be complex (cats).
    x_grid_psi
        Uniform grid on which ``psi`` is given.
    window
        Display window. May be enlarged to enforce extended-cell visibility.
    window_margin
        Fractional margin past ±Δx, ±Δp. 0.10 puts each axis tick of the
        extended cell roughly 10% inside the panel edge.
    x_pad_factor
        Integration grid in x runs from x_lim*(1+pad) on each side.
    p_pad
        Integration grid in p extends by ``p_pad`` beyond ±p_lim.
    """
    rs = numerical_covariance(psi, x_grid_psi, hbar=hbar)

    # Integration grid: wide enough that ψ has decayed to zero at the
    # boundary and kernel tails aren't clipped during convolution.
    # Centered on ⟨x⟩ so asymmetric states (Morse) aren't crammed against
    # one edge.
    auto_half_width = (1.0 + window_margin) * max(rs.Delta_x, rs.Delta_p)
    x_int_half = max(auto_half_width * x_pad_factor, 3.0 * rs.Delta_x)
    p_int_half = max(auto_half_width + p_pad, 3.0 * rs.Delta_p)
    x_int = np.linspace(rs.x_mean - x_int_half, rs.x_mean + x_int_half, n_x_int)
    p_int = np.linspace(-p_int_half, p_int_half, n_p_int)

    # Interpolate ψ onto the integration grid (zero outside its support).
    from scipy.interpolate import interp1d

    if np.iscomplexobj(psi):
        psi_re = interp1d(x_grid_psi, psi.real, bounds_error=False, fill_value=0.0)(x_int)
        psi_im = interp1d(x_grid_psi, psi.imag, bounds_error=False, fill_value=0.0)(x_int)
        psi_int = psi_re + 1j * psi_im
    else:
        psi_int = interp1d(x_grid_psi, psi, bounds_error=False, fill_value=0.0)(x_int)

    W = wigner_fft(psi_int, x_int, p_int, hbar=hbar)
    norm = wigner_norm(W, x_int, p_int)
    if abs(norm - 1.0) > 5e-3:
        import warnings
        warnings.warn(
            f"State {name!r}: Wigner norm = {norm:.4f} (expected 1). "
            "Consider widening the integration grid."
        )

    # Cell-overlay anchor: location of max |W(x, 0)|. The cell sits over
    # the deepest Wigner negativity, where its resolution argument is
    # visually most legible.
    ip0 = int(np.argmin(np.abs(p_int)))
    W_at_p0 = W[:, ip0]
    if cell_center_x is None:
        ix_peak = int(np.argmax(np.abs(W_at_p0)))
        cell_center_x_resolved = float(x_int[ix_peak])
    else:
        cell_center_x_resolved = cell_center_x

    # ------------------------------------------------------------------
    # DISPLAY WINDOW — three constraints:
    #
    #   1. Cell A must not be clipped: cell_center ± Δx must fit in x,
    #      and ±Δp in p, with margin.
    #
    #   2. Panel is square in data units: x_lim = p_lim. The half-width
    #      is (1+margin) · max(Δx, Δp) so both Δx and Δp can be the
    #      binding constraint without exceeding the panel.
    #
    #   3. Window centers on the state's interesting extent (midpoint of
    #      |W(x, 0)| support above a threshold). For symmetric states
    #      this is x = 0; for asymmetric ones (Morse) it lands roughly
    #      at the orbit center, so the structure fills the panel instead
    #      of crowding one edge.
    #
    # Constraints 1 and 3 can conflict: if the state midpoint is far
    # from the cell anchor, centering on the midpoint may push the cell
    # off an edge. We clamp the center so the cell always stays inside.
    # ------------------------------------------------------------------

    half_width = (1.0 + window_margin) * max(rs.Delta_x, rs.Delta_p)

    # State extent along p=0: where |W| exceeds 5% of its peak.
    abs_W = np.abs(W_at_p0)
    threshold = 0.05 * abs_W.max()
    significant = abs_W > threshold
    if significant.any():
        ix_lo = int(np.argmax(significant))
        ix_hi = int(len(significant) - 1 - np.argmax(significant[::-1]))
        state_extent_center = 0.5 * (float(x_int[ix_lo]) + float(x_int[ix_hi]))
    else:
        state_extent_center = cell_center_x_resolved

    # Resolve x window: caller can override; otherwise auto-compute.
    if window.x_lim > 0:
        x_lim_resolved = window.x_lim
        x_center_resolved = window.x_center
    else:
        x_lim_resolved = half_width
        # Try centering on state extent, then clamp so the cell stays
        # inside the window. The half-width already includes margin past
        # the cell edge (it's (1+margin) · Δmax), so the cell just needs
        # to fit inside ±half_width from the center.
        desired_center = state_extent_center
        cell_left  = cell_center_x_resolved - rs.Delta_x
        cell_right = cell_center_x_resolved + rs.Delta_x
        min_center = cell_right - half_width   # so right edge ≥ cell_right
        max_center = cell_left  + half_width   # so left edge ≤ cell_left
        x_center_resolved = max(min_center, min(desired_center, max_center))

    if window.p_lim > 0:
        p_lim_resolved = window.p_lim
        p_center_resolved = window.p_center
    else:
        p_lim_resolved = half_width
        p_center_resolved = 0.0

    # Final display window with auto-picked ticks.
    from .ticks import nice_ticks_around
    if not window.x_ticks:
        x_ticks = nice_ticks_around(x_center_resolved, x_lim_resolved, target_count=4)
    else:
        x_ticks = window.x_ticks
    if not window.p_ticks:
        p_ticks = nice_ticks_around(p_center_resolved, p_lim_resolved, target_count=4)
    else:
        p_ticks = window.p_ticks

    window = DisplayWindow(
        x_lim=x_lim_resolved,
        p_lim=p_lim_resolved,
        x_center=x_center_resolved,
        p_center=p_center_resolved,
        x_ticks=x_ticks,
        p_ticks=p_ticks,
    )

    return State(
        name=name,
        psi=psi,
        x_grid_psi=x_grid_psi,
        rs=rs,
        hbar=hbar,
        window=window,
        cell_center_x=cell_center_x_resolved,
        W=W,
        x_int=x_int,
        p_int=p_int,
    )
