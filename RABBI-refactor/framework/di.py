"""Dependency injection container and policy registry.
This file references classes from RABBI-Neighbor without modifying them.
Resolves imports by adding the sibling directory `RABBI-Neighbor/` to sys.path
when necessary.
"""
from typing import Dict, Optional
from pathlib import Path
import sys


def _ensure_neighbor_on_path():
    """Ensure RABBI-Neighbor directory is on sys.path for imports like `solver`.
    Works when this module is executed from repo root or any subdir.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2] if len(here.parents) >= 3 else here.parent
    # Prefer local refactor root first
    refactor_root = repo_root / "RABBI-refactor"
    if refactor_root.is_dir():
        s = str(refactor_root)
        if s not in sys.path:
            sys.path.insert(0, s)
    neighbor = repo_root / "RABBI-Neighbor"
    if neighbor.is_dir():
        str_neighbor = str(neighbor)
        if str_neighbor not in sys.path:
            sys.path.insert(0, str_neighbor)


# We import lazily inside methods to avoid hard dependency at import time


class PolicyRegistry:
    """Static registry that maps policy names to solver classes.
    Names must match existing class names in solver.py.
    """

    _registry: Dict[str, str] = {
        "RABBI": "RABBI",
        "OFFline": "OFFline",
        "NPlusOneLP": "NPlusOneLP",
        "TopKLP": "TopKLP",
    }

    @classmethod
    def get_class(cls, name: str):
        """Return the solver class object by name (from solver.py)."""
        if name not in cls._registry:
            raise KeyError(f"Unknown policy: {name}")
        # dynamic import to avoid heavy import during package load
        _ensure_neighbor_on_path()
        module = __import__("solver", fromlist=[cls._registry[name]])
        return getattr(module, cls._registry[name])


class Container:
    """Factory-style container for constructing sims and solvers.
    Keeps configuration centralized for reproducibility and caching.
    """

    def __init__(self, config_path: str, seed: Optional[int] = None, y_prefix: Optional[str] = None):
        self.config_path = config_path
        self.seed = seed
        self.y_prefix = y_prefix

    def make_sim(self):
        _ensure_neighbor_on_path()
        module = __import__("customer", fromlist=["CustomerChoiceSimulator"])
        CustomerChoiceSimulator = getattr(module, "CustomerChoiceSimulator")
        sim = CustomerChoiceSimulator(self.config_path, random_seed=self.seed)
        return sim

    def prepare_Y(self, sim, k_val: Optional[float] = None):
        import os
        if self.y_prefix is None:
            # If no prefix configured, just (re)generate Y for current sim
            sim.generate_Y_matrix()
            return
        # With prefix, try to cache by k value
        suffix = f"_k{int(k_val)}" if k_val is not None else ""
        y_file = f"{self.y_prefix}{suffix}.npy"
        if os.path.exists(y_file):
            sim.load_Y(y_file)
        else:
            sim.generate_Y_matrix()
            sim.save_Y(y_file)

    def make_solver(self, name: str, sim):
        SolverClass = PolicyRegistry.get_class(name)
        return SolverClass(sim, debug=False)
