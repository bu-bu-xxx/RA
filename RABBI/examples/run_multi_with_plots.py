"""Example script: run multiple solvers across k and generate plots using refactor framework.
"""
import os
import sys

# Ensure refactor root on path so 'framework' imports resolve
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from framework.runner import run_multi_k
from framework.viz import Visualizer


def main():
    param_file = os.path.join(ROOT, 'params', 'params5.yml')
    y_prefix = os.path.join(ROOT, 'data', 'Y', 'Y_matrix_params5')
    os.makedirs(os.path.join(ROOT, 'data', 'Y'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'data', 'pics'), exist_ok=True)

    # Select solvers by name
    from framework import solver as solver_mod
    solver_classes = [solver_mod.OFFline, solver_mod.NPlusOneLP]

    results = run_multi_k(param_file, y_prefix, solver_classes, max_concurrency=None, seed=42)

    viz = Visualizer()
    viz.generate_plots(results,
                       plot_keys=['multi_k_results', 'multi_k_ratio', 'multi_k_regret', 'lp_x_benchmark_ratio'],
                       save_dir=os.path.join(ROOT, 'data', 'pics'))


if __name__ == '__main__':
    main()
