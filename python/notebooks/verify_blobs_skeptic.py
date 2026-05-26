"""Verify the bitangent blobs for the Cluster 4 states (skeptic.py).

This is the state-level companion to verify_blobs.py. Where that script
checks the blob family {a_theta} over an abstract sweep of covariances
(Delta_x, Delta_p), this one builds each of the four Cluster 4 states,
reads off the covariance the figure pipeline actually produces, and runs
the identical quantum-blob checks on the family inscribed in that
covariance. The point is to confirm that the blobs used to resolve the
states that test the construction's assumptions are themselves genuine
de Gosson quantum blobs of action h/2 — not merely that some nearby
abstract covariance passes.

Because the blob family a_theta depends only on the Heisenberg-cell
semi-axes (Delta_x, Delta_p) and nothing else about the state, the
checks reduce to feeding each state's measured (Delta_x, Delta_p) into
the same check_regime routine verify_blobs.py uses. We import that
routine directly so the two scripts can never drift apart.

A note on the cubic-phase state. Its covariance is non-diagonal in the
lab frame; the figure pipeline (systems/skeptic.py, following the Kerr
pattern) reports the state in its covariance principal frame, where the
cell is axis-aligned and (Delta_x, Delta_p) are the principal semi-axes.
The blob family is constructed in that same principal frame, so the
diagonal check_regime applies exactly. The portrait W-tilde is
rotation-invariant, so this frame choice does not affect the result.

Checks per state (inherited from verify_blobs.py, all fail-fast):
  1. Symplectic capacity of a_theta equals pi*hbar for every theta.
  2. The symplectic eigenvalue of the covariance equals hbar/2.
  3. Capacity is theta-independent.
  4. Kernel K_theta and cell a_theta describe the same ellipse.
  5. a_theta is bitangent to A and to a-tilde in the literal sense: exactly
     two antipodal contact points with each, at every theta (inscribed in A,
     circumscribed about a-tilde).

Run:
    cd python
    uv run python notebooks/verify_blobs_skeptic.py
"""

from __future__ import annotations

import sys
import warnings

from wigner_resolution.systems.skeptic import (
    asymmetric_cat_state,
    cubic_phase_state,
    heavy_tailed_state,
    thermal_state,
)
# verify_blobs.py is a sibling script in this same notebooks/ directory,
# not a package module, so we import its check routine by adding the
# script's own directory to sys.path. This keeps both scripts sharing one
# definition of the blob checks rather than duplicating them.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from verify_blobs import HBAR, check_regime  # noqa: E402


# The four Cluster 4 states, in figure order. Each builder returns a
# State whose rs.Delta_x, rs.Delta_p are the principal-frame Heisenberg
# semi-axes the blob family is built from.
STATE_BUILDERS = [
    (cubic_phase_state, "cubic phase"),
    (thermal_state, "thermal, nbar=3"),
    (asymmetric_cat_state, "asymmetric cat"),
    (heavy_tailed_state, "heavy-tailed"),
]


def main() -> int:
    print("Verifying a_theta are de Gosson quantum blobs (capacity = h/2) "
          "for the Cluster 4 states.")
    print("Covariances are read from the figure pipeline; the blob family "
          "is built from each state's measured (Delta_x, Delta_p).")

    # Building the states is quiet; suppress the Wigner-norm UserWarnings
    # that the wide-grid FFT path can emit for the cusped/heavy-tailed
    # state, which are about integration-grid coverage, not the blobs.
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for builder, label in STATE_BUILDERS:
            state = builder()
            Dx = state.rs.Delta_x
            Dp = state.rs.Delta_p
            results.append(
                check_regime(Dx, Dp, f"{label}  [{state.name}]")
            )

    print("\n" + "=" * 64)
    if all(results):
        print("ALL CHECKS PASS.")
        print("Every a_theta inscribed in the four Cluster 4 states has "
              "symplectic capacity h/2 and is self-dual: a genuine de Gosson "
              "quantum blob, at constant action across theta.")
        return 0
    print("VERIFICATION FAILED. See messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
