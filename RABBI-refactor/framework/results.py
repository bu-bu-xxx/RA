"""Standard result objects and metrics. Non-invasive: only read params.* fields.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from pathlib import Path
import sys


@dataclass
class RunResult:
    solver_name: str
    k_val: Optional[float]
    params: Any  # the original params object from sim
    total_reward: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiKResult:
    solver_name: str
    k_values: List[float]
    results: List[RunResult]


def compute_total_reward(params) -> float:
    return float(np.sum(params.reward_history)) if getattr(params, "reward_history", None) is not None else 0.0


def compute_lp_x_benchmark(params) -> np.ndarray:
    """Wrapper around existing main.compute_lp_x_benchmark to keep single source of truth.
    If main is not importable, fallback to a lightweight reimplementation that uses
    LPBasedPolicy.solve_lp and params.* only.
    """
    try:
        main_mod = __import__("main", fromlist=["compute_lp_x_benchmark"])
        return main_mod.compute_lp_x_benchmark(params)
    except (ImportError, AttributeError):
        # Fallback: re-implement with the same semantics but without renaming
        # Ensure neighbor path is available
        here = Path(__file__).resolve()
        repo_root = here.parents[2] if len(here.parents) >= 3 else here.parent
        neighbor = repo_root / "RABBI-Neighbor"
        if neighbor.is_dir():
            str_neighbor = str(neighbor)
            if str_neighbor not in sys.path:
                sys.path.insert(0, str_neighbor)
        from solver import LPBasedPolicy
        T = len(params.alpha_history)
        result = []
        for t in range(T):
            b = np.array(params.b_history[t])
            alpha = params.alpha_history[t]
            p_t = params.Q[t, :, :]
            x_t = LPBasedPolicy.solve_lp(
                b, p_t, t, params.n, params.m, params.d, params.f, params.A, params.T
            )
            result.append(x_t[alpha])
        return np.array(result)


def compute_ratio_to_offline(target_rewards: List[float], offline_rewards: List[float]) -> List[float]:
    ratios = []
    for t, o in zip(target_rewards, offline_rewards):
        ratios.append(float(t / o) if o != 0 else 0.0)
    return ratios


def compute_regret_vs_offline(target_rewards: List[float], offline_rewards: List[float]) -> List[float]:
    regrets = []
    for t, o in zip(target_rewards, offline_rewards):
        regrets.append(float(o - t))
    return regrets
