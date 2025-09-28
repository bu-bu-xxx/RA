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
        import time
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

        def _acquire_lock(path: str, timeout: Optional[float] = None, poll: float = 0.1) -> str:
            lock_path = f"{path}.lock"
            start = time.monotonic()
            while True:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    os.close(fd)
                    return lock_path
                except FileExistsError:
                    if timeout is not None and (time.monotonic() - start) > timeout:
                        raise TimeoutError(f"Timed out waiting for lock on {path}")
                    time.sleep(poll)

        def _load_memmap(path: str):
            return np.load(path, mmap_mode='r')

        def _ensure_array(path: str, producer):
            try:
                return _load_memmap(path)
            except (FileNotFoundError, ValueError, OSError):
                pass

            lock_path = _acquire_lock(path)
            try:
                # Double-check after acquiring lock
                try:
                    return _load_memmap(path)
                except (FileNotFoundError, ValueError, OSError):
                    pass

                array = producer()
                tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}.npy"
                np.save(tmp_path, array)
                os.replace(tmp_path, path)
                return _load_memmap(path)
            finally:
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass

        sim.params.Y = _ensure_array(y_file, lambda: sim.generate_Y_matrix())
        sim.params.Q = _ensure_array(q_file, lambda: (sim.compute_offline_Q()))

    def make_solver(self, name: str, sim, debug: bool = False):
        SolverClass = PolicyRegistry.get_class(name)
        return SolverClass(sim, debug=debug)
