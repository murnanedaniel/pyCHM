"""Group-agnostic core for pyCHM cosets: Lie-algebra primitives, the `Coset(G,H)` abstraction,
representation lifts, Casimir branchings, and the SO(5)/SO(4), SO(6)/SO(5) and SU(4)/Sp(4) coset
instances.  Replaces the former duplicated `pychm.ccwz` / `pychm.symbolic.*` layer.
"""
from . import lie, coset, reps, branch, tensors, decompose, derive
from . import so5, so6, spinors, so6_spinors, models

__all__ = ["lie", "coset", "reps", "branch", "tensors", "decompose", "derive",
           "so5", "so6", "spinors", "so6_spinors", "models"]
