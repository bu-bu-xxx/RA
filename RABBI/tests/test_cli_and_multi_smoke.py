"""
Smoke test: programmatic multi-k and CLI flows with all solvers.

Includes three parts:
1) Programmatic multi-k: calls `framework.runner.run_multi_k` with solvers
    (RABBI, OFFline, NPlusOneLP, TopKLP) on `tests/params_min.yml` (seed=42,
    max_concurrency=2) and asserts each solver/k returns params with a non-empty
    reward_history and non-negative total reward.
2) CLI multi + plots: runs `python -m cli multi` with all
    solvers, generates plots (multi_k_results, multi_k_ratio, multi_k_regret,
    lp_x_benchmark_ratio) and asserts the expected PNG files exist under
    `RABBI/data/pics`.
3) CLI cache: runs `python -m cli cache` with all solvers and
    asserts shelve artifacts exist under `RABBI/data/shelve` (accepts
    varying dbm extensions).
"""

import os
import sys
import subprocess

# Ensure RABBI is on sys.path so 'framework' and 'solver' can be imported
THIS_DIR = os.path.dirname(__file__)
REFAC_ROOT = os.path.dirname(THIS_DIR)
REPO_ROOT = os.path.dirname(REFAC_ROOT)
if REFAC_ROOT not in sys.path:
    sys.path.insert(0, REFAC_ROOT)

from framework.runner import run_multi_k


def test_run_multi_k_all_solvers_smoke():
    # Use params_min.yml for speed and determinism
    param = os.path.join(THIS_DIR, "params_min.yml")
    qy_prefix = os.path.join(REFAC_ROOT, "data", "QY", "qy_params_min")
    os.makedirs(os.path.dirname(qy_prefix), exist_ok=True)

    # Import solver classes dynamically
    from framework import solver as solver_mod
    solver_classes = [getattr(solver_mod, name) for name in ("RABBI", "OFFline", "NPlusOneLP", "TopKLP", "Robust")]

    results = run_multi_k(param, qy_prefix, solver_classes, max_concurrency=2, seed=42)

    # Basic assertions: keys exist, each k entry has params with non-empty reward_history
    assert set(results.keys()) == {"RABBI", "OFFline", "NPlusOneLP", "TopKLP", "Robust"}
    for name, plist in results.items():
        assert len(plist) >= 1
        for params in plist:
            assert params is not None
            assert hasattr(params, "reward_history")
            assert len(params.reward_history) > 0
            assert sum(params.reward_history) >= 0


def test_cli_multi_plots_and_cache_smoke():
    # Prepare paths
    param = os.path.join(THIS_DIR, "params_min.yml")
    qy_prefix = os.path.join(REFAC_ROOT, "data", "QY", "qy_params_min")
    pics_dir = os.path.join(REFAC_ROOT, "data", "pics")
    shelve_dir = os.path.join(REFAC_ROOT, "data", "shelve")
    os.makedirs(os.path.dirname(qy_prefix), exist_ok=True)
    os.makedirs(pics_dir, exist_ok=True)
    os.makedirs(shelve_dir, exist_ok=True)

    # Run CLI multi with all solvers and plots
    cmd_multi = [
        sys.executable,
        "-m",
        "cli",
        "multi",
        "--param",
        param,
        "--qy-prefix",
        qy_prefix,
        "--solvers",
        "RABBI",
        "OFFline",
        "NPlusOneLP",
        "TopKLP",
        "Robust",
        "--plots",
        "multi_k_results",
        "multi_k_ratio",
        "multi_k_regret",
        "lp_x_benchmark_ratio",
        "--plot-dir",
        pics_dir,
    ]
    # Run from repo root so module resolution works
    proc = subprocess.run(cmd_multi, cwd=REFAC_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, f"CLI multi failed: stdout={proc.stdout}\nstderr={proc.stderr}"

    # Check expected plot files exist
    expected_plots = [
        "multi_k_results.png",
        "multi_k_ratio_results.png",
        "multi_k_regret_results.png",
        "lp_x_benchmark_ratio_vs_k.png",
    ]
    for fname in expected_plots:
        assert os.path.exists(os.path.join(pics_dir, fname)), f"Missing plot: {fname}"

    # Run CLI cache to create shelve files
    cmd_cache = [
        sys.executable,
        "-m",
        "cli",
        "cache",
        "--param",
        param,
        "--qy-prefix",
        qy_prefix,
        "--solvers",
        "RABBI",
        "OFFline",
        "NPlusOneLP",
        "TopKLP",
        "Robust",
        "--shelve-dir",
        shelve_dir,
        "--plots",
        "multi_k_results",
        "--plot-dir",
        pics_dir,
    ]
    proc2 = subprocess.run(cmd_cache, cwd=REFAC_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc2.returncode == 0, f"CLI cache failed: stdout={proc2.stdout}\nstderr={proc2.stderr}"

    # Verify cache files exist
    for name in ("RABBI", "OFFline", "NPlusOneLP", "TopKLP", "Robust"):
        path = os.path.join(shelve_dir, f"params_{name.lower()}.shelve.db")
        # Different dbm implementations vary extensions; check prefix existence
        prefix = os.path.join(shelve_dir, f"params_{name.lower()}.shelve")
        exists = any(os.path.exists(prefix + ext) for ext in ("", ".db", ".dat", ".dir", ".bak"))
        assert exists, f"Shelve file for {name} not created"
