"""Optional CLI entry to run experiments in RABBI-refactor without touching RABBI-Neighbor.
Usage examples:
  python -m RABBI-refactor.cli single --param params5.yml --solver OFFline --y-prefix data/Y/Y_matrix_params5
  python -m RABBI-refactor.cli multi  --param params5.yml --y-prefix data/Y/Y_matrix_params5 --solvers OFFline NPlusOneLP
"""
import argparse
import os
import sys

# Ensure this package root is on sys.path so 'framework' can be imported
REFAC_ROOT = os.path.dirname(__file__)
if REFAC_ROOT not in sys.path:
    sys.path.insert(0, REFAC_ROOT)

from framework.runner import run_single, run_multi_k, run_multi_k_with_cache
from framework.viz import Visualizer


def main():
    parser = argparse.ArgumentParser(
        prog="rabbi-refactor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single")
    p_single.add_argument("--param", required=True)
    p_single.add_argument("--solver", required=True)
    p_single.add_argument("--y-prefix", required=False, default=None)
    p_single.add_argument("--seed", type=int, default=42)

    p_multi = sub.add_parser("multi")
    p_multi.add_argument("--param", required=True)
    p_multi.add_argument("--y-prefix", required=True)
    p_multi.add_argument("--solvers", nargs="+", required=True)
    p_multi.add_argument("--max-concurrency", type=int, default=None)
    p_multi.add_argument("--seed", type=int, default=42)
    p_multi.add_argument("--plots", nargs="*", default=[], help="Plot keys: multi_k_results, multi_k_ratio, multi_k_regret, lp_x_benchmark_ratio")
    p_multi.add_argument("--save-dir", default="RABBI-refactor/data/pics")

    p_cache = sub.add_parser("cache")
    p_cache.add_argument("--param", required=True)
    p_cache.add_argument("--y-prefix", required=True)
    p_cache.add_argument("--solvers", nargs="+", required=True)
    p_cache.add_argument("--max-concurrency", type=int, default=None)
    p_cache.add_argument("--seed", type=int, default=42)
    p_cache.add_argument("--shelve-dir", default="RABBI-refactor/data/shelve")
    p_cache.add_argument("--plots", nargs="*", default=[], help="Plot keys: multi_k_results, multi_k_ratio, multi_k_regret, lp_x_benchmark_ratio")
    p_cache.add_argument("--save-dir", default="RABBI-refactor/data/pics")

    p_clear = sub.add_parser("clear-cache")
    p_clear.add_argument("--solvers", nargs="*", default=[], help="Solvers to clear; default clears all known")
    p_clear.add_argument("--shelve-dir", default="RABBI-refactor/data/shelve")
    p_clear.add_argument("--preview-clear", "--dry-run", "--preview", action="store_true", dest="preview_clear",
                         help="List files that would be removed without deleting them")

    args = parser.parse_args()

    if args.cmd == "single":
        res = run_single(args.param, args.y_prefix, args.solver, seed=args.seed)
        print(f"solver={res.solver_name}, k={res.k_val}, total_reward={res.total_reward}")
    elif args.cmd == "multi":
        # Dynamically resolve solver classes by name from solver module
        solver_mod = __import__("solver")
        solver_classes = [getattr(solver_mod, name) for name in args.solvers]
        results = run_multi_k(args.param, args.y_prefix, solver_classes,
                              max_concurrency=args.max_concurrency, seed=args.seed)
        for name, plist in results.items():
            totals = [sum(p.reward_history) for p in plist]
            print(name, totals)
        if args.plots:
            viz = Visualizer()
            viz.generate_plots(results, args.plots, args.save_dir)
    elif args.cmd == "cache":
        solver_mod = __import__("solver")
        solver_classes = [getattr(solver_mod, name) for name in args.solvers]
        os.makedirs(args.shelve_dir, exist_ok=True)
        shelve_paths = {name: os.path.join(args.shelve_dir, f"params_{name.lower()}.shelve") for name in args.solvers}
        results = run_multi_k_with_cache(args.param, args.y_prefix, solver_classes,
                                         max_concurrency=args.max_concurrency, seed=args.seed,
                                         shelve_paths=shelve_paths)
        for name, plist in results.items():
            totals = [sum(p.reward_history) if p is not None else None for p in plist]
            print(name, totals)
        if args.plots:
            viz = Visualizer()
            viz.generate_plots(results, args.plots, args.save_dir)
    elif args.cmd == "clear-cache":
        os.makedirs(args.shelve_dir, exist_ok=True)
        # Accept arbitrary solver names; if none provided, use standard set
        solvers = args.solvers or ["RABBI", "OFFline", "NPlusOneLP", "TopKLP"]
        targets = []
        for name in solvers:
            base = os.path.join(args.shelve_dir, f"params_{name.lower()}.shelve")
            # Shelve may create multiple files depending on backend; remove known suffixes
            for suffix in ("", ".db", ".dat", ".dir", ".bak"):  # tolerate variants
                path = base + suffix
                if os.path.exists(path):
                    targets.append(path)
        if args.preview_clear:
            print("Would remove:")
            for p in targets:
                print(p)
        else:
            removed = []
            for p in targets:
                try:
                    os.remove(p)
                    removed.append(p)
                except OSError:
                    pass
            print("Removed:")
            for p in removed:
                print(p)


if __name__ == "__main__":
    main()
