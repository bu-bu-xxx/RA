"""Dependency injection container and policy registry for local framework modules."""
from typing import Dict, Optional


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
    def available_names(cls):
        """Return sorted list of available solver names."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_class(cls, name: str):
        """Return the solver class object by name (from solver.py)."""
        if name not in cls._registry:
            raise KeyError(f"Unknown policy: {name}")
        # dynamic import from local framework package
        from . import solver as module
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
        from .customer import CustomerChoiceSimulator
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
            # Ensure directory exists before saving
            y_dir = os.path.dirname(y_file)
            if y_dir:
                os.makedirs(y_dir, exist_ok=True)
            sim.save_Y(y_file)

    def make_solver(self, name: str, sim, debug: bool = False):
        SolverClass = PolicyRegistry.get_class(name)
        return SolverClass(sim, debug=debug)
