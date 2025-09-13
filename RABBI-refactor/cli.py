"""Optional CLI entry to run experiments in RABBI-refactor without touching RABBI-Neighbor.
Usage examples:
  python -m RABBI-refactor.cli single --param params5.yml --solver OFFline --y-prefix data/Y/Y_matrix_params5
  python -m RABBI-refactor.cli multi  --param params5.yml --y-prefix data/Y/Y_matrix_params5 --solvers OFFline NPlusOneLP
"""
import argparse
from framework.runner import run_single, run_multi_k


def main():
    parser = argparse.ArgumentParser(prog="rabbi-refactor")
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

    args = parser.parse_args()

    if args.cmd == "single":
        res = run_single(args.param, args.y_prefix, args.solver, seed=args.seed,
                         with_offline_Q=(args.solver == "OFFline"))
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


if __name__ == "__main__":
    main()
