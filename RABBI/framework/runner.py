"""Runner orchestration using DI and standardized results.
Preserves existing behavior without renaming functions.
"""
from typing import Dict, List, Optional, Sequence, Tuple
import os
import concurrent.futures
from concurrent.futures import as_completed

from .di import Container
from .results import RunResult, compute_total_reward


def _collect_extra_fields(solver_name: str, params, k_val: Optional[float]) -> List[str]:
    extras: List[str] = []
    if solver_name == "Robust":
        history = getattr(params, "A_prime_size_history", None)
        if history:
            avg_size = sum(history) / len(history)
            max_size = max(history)
            extras.append(f"avg_A_prime_size={avg_size:.2f}")
            extras.append(f"max_A_prime_size={max_size}")
            if k_val is not None:
                try:
                    k_numeric = float(k_val)
                except (TypeError, ValueError):
                    k_numeric = k_val
                extras.append(f"k={k_numeric}")
            m_val = getattr(params, "m", None)
            if m_val is not None:
                extras.append(f"m={m_val}")

    no_sell_cnt = getattr(params, "no_sell_cnt", None)
    if no_sell_cnt is not None:
        extras.append(f"no_sell_cnt={no_sell_cnt}")

    return extras


def _extract_error_detail(exc: BaseException) -> str:
    if exc is None:
        return ""
    detail = str(exc) or exc.__class__.__name__
    return detail


def _describe_task(task_args: Tuple[int, float, str, str, str, Optional[int]]) -> str:
    task_idx, k_val, _param_file, _y_prefix, solver_name, _seed = task_args
    try:
        k_numeric = float(k_val)
    except (TypeError, ValueError):
        k_numeric = k_val
    return f"task#{task_idx} solver={solver_name} k={k_numeric} (scheduled)"


def _log_task_error(context: str, task_args: Tuple[int, float, str, str, str, Optional[int]], exc: BaseException) -> None:
    description = _describe_task(task_args)
    detail = _extract_error_detail(exc)
    print(f"[{context}:error] {description} failed: {detail}", flush=True)


def _universal_worker(args: Tuple[int, float, str, str, str, Optional[int]]):
    """Worker compatible with concurrent.futures for multi-k runs.
    Args: (task_idx, k_val, param_file, y_prefix, solver_name, seed)
    Returns: (task_idx, params, solver_name)
    """
    task_idx, k_val, param_file, y_prefix, solver_name, seed = args
    container = Container(param_file, seed=seed, y_prefix=y_prefix)
    sim = container.make_sim()
    # scale by k
    sim.params.B = sim.params.B * k_val
    sim.params.T = int(sim.params.T * k_val)
    container.prepare_Y(sim, k_val=k_val)
    # compute offline Q for all solvers to enable LP benchmark and parity with original main
    sim.compute_offline_Q()
    solver = container.make_solver(solver_name, sim, debug=False)
    solver.run()
    return (task_idx, sim.params, solver_name)


def run_single(param_file: str, y_prefix: Optional[str], solver_name: str, seed: Optional[int] = None,
               k_val: Optional[float] = None, debug: bool = False) -> RunResult:
    print(f"[run_single] param={param_file}, y_prefix={y_prefix}, solver={solver_name}, seed={seed}, k={k_val}", flush=True)
    container = Container(param_file, seed=seed, y_prefix=y_prefix)
    sim = container.make_sim()
    if k_val is not None:
        sim.params.B = sim.params.B * k_val
        sim.params.T = int(sim.params.T * k_val)
    container.prepare_Y(sim, k_val=k_val)
    # For compatibility with compute_lp_x_benchmark, compute Q regardless of solver
    sim.compute_offline_Q()
    solver = container.make_solver(solver_name, sim, debug=debug)
    try:
        solver.run()
    except Exception as exc:  # pragma: no cover - pass-through for logging at caller
        effective_k = k_val if k_val is not None else 1.0
        _log_task_error("run_single", (0, effective_k, param_file, y_prefix, solver_name, seed), exc)
        raise
    total = compute_total_reward(sim.params)
    steps = len(sim.params.reward_history)
    extras = _collect_extra_fields(solver_name, sim.params, k_val)
    extras_suffix = f", {', '.join(extras)}" if extras else ""
    print(
        f"[run_single:done] solver={solver_name}, k={k_val}, total_reward={total:.4f}, steps={steps}{extras_suffix}",
        flush=True,
    )
    return RunResult(solver_name=solver_name, k_val=k_val, params=sim.params, total_reward=total)


def run_multi_k(param_file: str, y_prefix: str, solver_classes: Sequence[type], max_concurrency: Optional[int] = None,
                seed: Optional[int] = None) -> Dict[str, List[object]]:
    """Compatibility wrapper that returns dict[solver_name] -> List[params] like main.run_multi_k."""
    if max_concurrency is None:
        max_concurrency = os.cpu_count()

    # Resolve class names
    solver_names = [cls.__name__ for cls in solver_classes]

    # Prepare k values using a temporary sim
    container = Container(param_file, seed=seed, y_prefix=y_prefix)
    sim = container.make_sim()
    k_values = sim.params.k

    all_args = []
    for solver_name in solver_names:
        for i, k_val in enumerate(k_values):
            all_args.append((len(all_args), k_val, param_file, y_prefix, solver_name, seed))
    print(
        f"[run_multi_k] param={param_file}, y_prefix={y_prefix}, solvers={solver_names}, "
        f"k_values={[float(k) for k in k_values]}, max_concurrency={max_concurrency}, seed={seed}, tasks={len(all_args)}",
        flush=True,
    )
    for task_idx, k_val, _p, _y, s, _seed in all_args:
        print(f"  - task#{task_idx} solver={s} k={float(k_val)}", flush=True)

    future_map: Dict[concurrent.futures.Future, Tuple[int, float, str, str, str, Optional[int]]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrency) as executor:
        for args in all_args:
            future = executor.submit(_universal_worker, args)
            future_map[future] = args
        all_results = []
        for fut in as_completed(future_map):
            args = future_map[fut]
            try:
                task_idx, params, solver_name = fut.result()
            except Exception as exc:
                _log_task_error("run_multi_k", args, exc)
                raise
            all_results.append((task_idx, params, solver_name))
            # Derive k via position in schedule
            k_val = all_args[task_idx][1]
            total = float(sum(getattr(params, 'reward_history', []) or [0.0]))
            steps = len(getattr(params, 'reward_history', []) or [])
            extras = _collect_extra_fields(solver_name, params, k_val)
            extras_suffix = f", {', '.join(extras)}" if extras else ""
            print(
                f"[done] task#{task_idx} solver={solver_name} k={float(k_val)} total_reward={total:.4f} steps={steps}{extras_suffix}",
                flush=True,
            )

    # Group
    results_dict: Dict[str, List[object]] = {name: [None] * len(k_values) for name in solver_names}
    for task_idx, params, solver_name in all_results:
        # recover index order by task_idx by recomputing mapping
        # we simply append then reorder by k index below; simpler: map by first-come order
        # Here, we assume tasks added in strict order per solver_name and k
        pass

    # Build order map
    ptr = 0
    index_map: Dict[int, Tuple[str, int]] = {}
    for solver_name in solver_names:
        for i, _ in enumerate(k_values):
            index_map[ptr] = (solver_name, i)
            ptr += 1

    for task_idx, params, solver_name in all_results:
        solver_name_mapped, k_idx = index_map[task_idx]
        results_dict[solver_name_mapped][k_idx] = params

    return results_dict


def run_multi_k_with_cache(param_file: str, y_prefix: str, solver_classes: Sequence[type], max_concurrency: Optional[int] = None,
                            shelve_paths: Optional[Dict[str, str]] = None, seed: Optional[int] = None) -> Dict[str, List[object]]:
    """Compatibility wrapper mirroring main.run_multi_k_with_cache signature/behavior, but centralized here.
    Returns dict[solver_name] -> List[params], where each list aligns with k values.
    """
    import shelve

    container = Container(param_file, seed=seed, y_prefix=y_prefix)
    sim = container.make_sim()
    k_values = sim.params.k

    solver_names = [cls.__name__ for cls in solver_classes]
    shelve_paths = shelve_paths or {}

    pending_tasks = []
    index_map: Dict[int, Tuple[str, int]] = {}
    task_ptr = 0

    for solver_name in solver_names:
        shelve_path = shelve_paths.get(solver_name)
        if not shelve_path:
            # schedule all tasks without caching
            for k_idx, k_val in enumerate(k_values):
                index_map[task_ptr] = (solver_name, k_idx)
                pending_tasks.append((task_ptr, k_val, param_file, y_prefix, solver_name, seed))
                task_ptr += 1
            continue
        # try read cache keys
        try:
            with shelve.open(shelve_path, flag='c') as db:
                for k_idx, k_val in enumerate(k_values):
                    key = f"params_{int(k_val)}"
                    if key in db:
                        # cached: no task
                        continue
                    else:
                        index_map[task_ptr] = (solver_name, k_idx)
                        pending_tasks.append((task_ptr, k_val, param_file, y_prefix, solver_name, seed))
                        task_ptr += 1
        except (OSError, RuntimeError):
            # if shelve broken, schedule all
            for k_idx, k_val in enumerate(k_values):
                index_map[task_ptr] = (solver_name, k_idx)
                pending_tasks.append((task_ptr, k_val, param_file, y_prefix, solver_name, seed))
                task_ptr += 1

    # run pending
    results_dict: Dict[str, List[object]] = {name: [None] * len(k_values) for name in solver_names}

    if pending_tasks:
        total_tasks = len(pending_tasks)
        print(
            f"[run_multi_k_with_cache] param={param_file}, y_prefix={y_prefix}, solvers={solver_names}, "
            f"k_values={[float(k) for k in k_values]}, pending_tasks={total_tasks}, max_concurrency={max_concurrency}, seed={seed}",
            flush=True,
        )
        for task_idx, k_val, _p, _y, s, _seed in pending_tasks:
            print(f"  - task#{task_idx} solver={s} k={float(k_val)} (scheduled)", flush=True)
        max_workers = max_concurrency or os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_universal_worker, t): t for t in pending_tasks}
            for fut in as_completed(future_map):
                task_args = future_map[fut]
                try:
                    task_idx, params, solver_name = fut.result()
                except Exception as exc:
                    _log_task_error("run_multi_k_with_cache", task_args, exc)
                    raise
                solver_name_mapped, k_idx = index_map[task_idx]
                results_dict[solver_name_mapped][k_idx] = params
                # write-through cache
                shelve_path = shelve_paths.get(solver_name_mapped)
                if shelve_path:
                    try:
                        with shelve.open(shelve_path, flag='c') as db:
                            key = f"params_{int(k_values[k_idx])}"
                            db[key] = params
                    except (OSError, RuntimeError):
                        pass
                k_val = pending_tasks[[t[0] for t in pending_tasks].index(task_idx)][1]
                total = float(sum(getattr(params, 'reward_history', []) or [0.0]))
                steps = len(getattr(params, 'reward_history', []) or [])
                extras = _collect_extra_fields(solver_name, params, k_val)
                extras_suffix = f", {', '.join(extras)}" if extras else ""
                print(
                    f"[done] task#{task_idx} solver={solver_name} k={float(k_val)} total_reward={total:.4f} steps={steps}{extras_suffix}",
                    flush=True,
                )

    # fill from cache for existing ones
    for solver_name in solver_names:
        shelve_path = shelve_paths.get(solver_name)
        if not shelve_path:
            continue
        try:
            with shelve.open(shelve_path, flag='r') as db:
                for k_idx, k_val in enumerate(k_values):
                    key = f"params_{int(k_val)}"
                    if results_dict[solver_name][k_idx] is None and key in db:
                        try:
                            results_dict[solver_name][k_idx] = db[key]
                        except (OSError, RuntimeError, KeyError, ModuleNotFoundError, AttributeError):
                            # Leave as None to be recomputed in fallback below
                            pass
        except (OSError, RuntimeError, KeyError):
            pass

    # Fallback: if any entries remain None, schedule and run those tasks now
    missing_tasks: List[Tuple[int, float, str, str, str, Optional[int]]] = []
    missing_index_map: Dict[int, Tuple[str, int]] = {}
    miss_ptr = 0
    for solver_name in solver_names:
        for k_idx, k_val in enumerate(k_values):
            if results_dict[solver_name][k_idx] is None:
                missing_index_map[miss_ptr] = (solver_name, k_idx)
                missing_tasks.append((miss_ptr, float(k_val), param_file, y_prefix, solver_name, seed))
                miss_ptr += 1

    if missing_tasks:
        max_workers = max_concurrency or os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_universal_worker, t): t for t in missing_tasks}
            for fut in as_completed(future_map):
                task_args = future_map[fut]
                try:
                    task_idx, params, solver_name = fut.result()
                except Exception as exc:
                    _log_task_error("run_multi_k_with_cache", task_args, exc)
                    raise
                solver_name_mapped, k_idx = missing_index_map[task_idx]
                results_dict[solver_name_mapped][k_idx] = params
                # write-through cache
                shelve_path = shelve_paths.get(solver_name_mapped)
                if shelve_path:
                    try:
                        with shelve.open(shelve_path, flag='c') as db:
                            key = f"params_{int(k_values[k_idx])}"
                            db[key] = params
                    except (OSError, RuntimeError):
                        pass
                k_val = k_values[k_idx]
                total = float(sum(getattr(params, 'reward_history', []) or [0.0]))
                steps = len(getattr(params, 'reward_history', []) or [])
                extras = _collect_extra_fields(solver_name, params, k_val)
                extras_suffix = f", {', '.join(extras)}" if extras else ""
                print(
                    f"[done] task#{task_idx} solver={solver_name} k={float(k_val)} total_reward={total:.4f} steps={steps}{extras_suffix}",
                    flush=True,
                )

    return results_dict
