import os
import sys

# Ensure RABBI-refactor is on sys.path so 'framework' can be imported
THIS_DIR = os.path.dirname(__file__)
REFAC_ROOT = os.path.dirname(THIS_DIR)
if REFAC_ROOT not in sys.path:
    sys.path.insert(0, REFAC_ROOT)

from framework.runner import run_single


def test_run_single_smoke():
    param = os.path.join(os.path.dirname(__file__), "params_min.yml")
    # OFFline needs Q; we can run RABBI for a smoke test
    res = run_single(param, y_prefix=None, solver_name="RABBI", seed=123)
    assert res.total_reward >= 0
    assert res.params is not None
