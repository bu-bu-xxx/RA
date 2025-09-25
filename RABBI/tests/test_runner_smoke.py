"""
Smoke test: single-run sanity check.

- Ensures DI/runner wiring works end-to-end for a minimal config.
- Runs `run_single` with solver `RABBI` on `tests/params_min.yml` (seed=123).
- Validates: non-negative `total_reward` and a returned `params` object.
- Note: `OFFline` needs precomputed Q; here we use `RABBI` for a fast path.
"""

import os
import sys

# Ensure RABBI is on sys.path so 'framework' can be imported
THIS_DIR = os.path.dirname(__file__)
REFAC_ROOT = os.path.dirname(THIS_DIR)
if REFAC_ROOT not in sys.path:
    sys.path.insert(0, REFAC_ROOT)

from framework.runner import run_single


def test_run_single_smoke():
    param = os.path.join(os.path.dirname(__file__), "params_min.yml")
    for solver_name in ("RABBI", "Robust"):
        res = run_single(param, y_prefix=None, solver_name=solver_name, seed=123)
        assert res.total_reward >= 0
        assert res.params is not None
