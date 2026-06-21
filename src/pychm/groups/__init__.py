"""Group-agnostic core for pyCHM cosets: Lie-algebra primitives, the `Coset(G,H)` abstraction,
representation lifts, and Casimir branchings.  The SO(5)/SO(4) and SO(6)/SO(5) machinery in
`ccwz`/`symbolic.core`/`symbolic.so6` are instances of this; SU(4)/Sp(4) is built on it directly.
"""
from . import lie, coset, reps, branch, so6, so6_spinors

__all__ = ["lie", "coset", "reps", "branch", "so6", "so6_spinors"]
