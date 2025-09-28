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
        "Robust": "Robust",
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

    def __init__(self, config_path: str, seed: Optional[int] = None, qy_prefix: Optional[str] = None):
        self.config_path = config_path
        self.seed = seed
        self.qy_prefix = qy_prefix

    def make_sim(self):
        from .customer import CustomerChoiceSimulator
        sim = CustomerChoiceSimulator(self.config_path, random_seed=self.seed)
        return sim

    def prepare_qy(self, sim, k_val: Optional[float] = None):
        import os
        import numpy as np

        if self.qy_prefix is None:
            sim.generate_Y_matrix()
            sim.compute_offline_Q()
            return

        suffix = f"_k{int(k_val)}" if k_val is not None else ""
        base_path = f"{self.qy_prefix}{suffix}"
        y_file = f"{base_path}_Y.npy"
        q_file = f"{base_path}_Q.npy"

        base_dir = os.path.dirname(base_path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

        if os.path.exists(y_file):
            sim.params.Y = np.load(y_file, mmap_mode='r')
        else:
            sim.generate_Y_matrix()
            np.save(y_file, sim.params.Y)
            sim.params.Y = np.load(y_file, mmap_mode='r')

        if os.path.exists(q_file):
            sim.params.Q = np.load(q_file, mmap_mode='r')
        else:
            sim.compute_offline_Q()
            np.save(q_file, sim.params.Q)
            sim.params.Q = np.load(q_file, mmap_mode='r')

    def make_solver(self, name: str, sim, debug: bool = False):
        SolverClass = PolicyRegistry.get_class(name)
        return SolverClass(sim, debug=debug)
