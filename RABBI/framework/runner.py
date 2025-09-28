"""Runner orchestration using DI and standardized results.
Preserves existing behavior without renaming functions.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import os
import concurrent.futures
from concurrent.futures import as_completed

import json
import yaml

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
    task_idx, k_val, _param_file, _qy_prefix, solver_name, _seed = task_args
    try:
        k_numeric = float(k_val)
    except (TypeError, ValueError):
        k_numeric = k_val
    return f"task#{task_idx} solver={solver_name} k={k_numeric} (scheduled)"


def _log_task_error(context: str, task_args: Tuple[int, float, str, str, str, Optional[int]], exc: BaseException) -> None:
    description = _describe_task(task_args)
    detail = _extract_error_detail(exc)
    print(f"[{context}:error] {description} failed: {detail}", flush=True)


def _normalize_k_value(k_val: float) -> str:
    try:
        k_float = float(k_val)
    except (TypeError, ValueError):
        return str(k_val)
    if k_float.is_integer():
        return str(int(k_float))
    normalized = f"{k_float:.10f}".rstrip("0").rstrip(".")
    return normalized


def _cache_key_for_k(k_val: float) -> str:
    return f"params_{_normalize_k_value(k_val)}"


def _strip_scaling_list_from_config(config_data):
    if not isinstance(config_data, dict):
        return config_data
    sanitized = dict(config_data)
    sanitized.pop("scaling_list", None)
    return sanitized


def _load_config_data(param_file: str):
    try:
        with open(param_file, "r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp) or {}
    except FileNotFoundError:
        return {}
    return _strip_scaling_list_from_config(raw)


def _compute_config_signature_from_data(config_data) -> str:
    return json.dumps(config_data, sort_keys=True, ensure_ascii=False)


def _entry_config_signature(entry) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    if "config" not in entry:
        return None
    return _compute_config_signature_from_data(entry["config"])


def _entry_matches_signature(entry, signature: str) -> bool:
    entry_sig = _entry_config_signature(entry)
    if entry_sig is None:
        return False
    return entry_sig == signature


def _prepare_cache_entry(params, config_data) -> Dict[str, object]:
    heavy_attrs = {}
    for attr in ("Y", "Q"):
        if hasattr(params, attr):
            heavy_attrs[attr] = getattr(params, attr)
            setattr(params, attr, None)

    try:
        stored_params = copy.deepcopy(params)
    finally:
        for attr, value in heavy_attrs.items():
            setattr(params, attr, value)

    for attr in ("Y", "Q", "x_history"):
        if hasattr(stored_params, attr):
            setattr(stored_params, attr, None)
    if hasattr(stored_params, "k"):
        stored_params.k = None
    return {"params": stored_params, "config": copy.deepcopy(config_data)}


def _log_task_schedule(context: str, param_file: str, qy_prefix: str, solver_names: Sequence[str],
                       k_values: Iterable[float], tasks: Sequence[Tuple[int, float, str, str, str, Optional[int]]],
                       max_concurrency: Optional[int], seed: Optional[int]) -> None:
    total_tasks = len(tasks)
    print(
        f"[{context}] param={param_file}, qy_prefix={qy_prefix}, solvers={list(solver_names)}, "
        f"k_values={[float(k) for k in k_values]}, pending_tasks={total_tasks}, max_concurrency={max_concurrency}, seed={seed}",
        flush=True,
    )
    for task_idx, k_val, _p, _qy, solver_name, _seed in tasks:
        print(f"  - task#{task_idx} solver={solver_name} k={float(k_val)} (scheduled)", flush=True)


def _universal_worker(args: Tuple[int, float, str, str, str, Optional[int]]):
    """Worker compatible with concurrent.futures for multi-k runs.
    Args: (task_idx, k_val, param_file, qy_prefix, solver_name, seed)
    Returns: (task_idx, params, solver_name)
    """
    task_idx, k_val, param_file, qy_prefix, solver_name, seed = args
    container = Container(param_file, seed=seed, qy_prefix=qy_prefix)
    sim = container.make_sim()
    # scale by k
    sim.params.B = sim.params.B * k_val
    sim.params.T = int(sim.params.T * k_val)
    container.prepare_qy(sim, k_val=k_val)
    # compute offline Q for all solvers to enable LP benchmark and parity with original main
    if getattr(sim.params, "Q", None) is None:
        sim.compute_offline_Q()
    solver = container.make_solver(solver_name, sim, debug=False)
    solver.run()
    return (task_idx, sim.params, solver_name)


def run_single(param_file: str, qy_prefix: Optional[str], solver_name: str, seed: Optional[int] = None,
               k_val: Optional[float] = None, debug: bool = False) -> RunResult:
    print(f"[run_single] param={param_file}, qy_prefix={qy_prefix}, solver={solver_name}, seed={seed}, k={k_val}", flush=True)
    container = Container(param_file, seed=seed, qy_prefix=qy_prefix)
    sim = container.make_sim()
    if k_val is not None:
        sim.params.B = sim.params.B * k_val
        sim.params.T = int(sim.params.T * k_val)
    container.prepare_qy(sim, k_val=k_val)
    # For compatibility with compute_lp_x_benchmark, compute Q regardless of solver
    if getattr(sim.params, "Q", None) is None:
        sim.compute_offline_Q()
    solver = container.make_solver(solver_name, sim, debug=debug)
    try:
        solver.run()
    except Exception as exc:  # pragma: no cover - pass-through for logging at caller
        effective_k = k_val if k_val is not None else 1.0
        _log_task_error("run_single", (0, effective_k, param_file, qy_prefix, solver_name, seed), exc)
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


def run_multi_k(param_file: str, qy_prefix: str, solver_classes: Sequence[type], max_concurrency: Optional[int] = None,
                seed: Optional[int] = None) -> Dict[str, List[object]]:
    """Compatibility wrapper that returns dict[solver_name] -> List[params] like main.run_multi_k."""
    if max_concurrency is None:
        max_concurrency = os.cpu_count()

    # Resolve class names
    solver_names = [cls.__name__ for cls in solver_classes]

    # Prepare k values using a temporary sim
    container = Container(param_file, seed=seed, qy_prefix=qy_prefix)
    sim = container.make_sim()
    k_values = sim.params.k

    all_args = []
    for solver_name in solver_names:
        for i, k_val in enumerate(k_values):
            all_args.append((len(all_args), k_val, param_file, qy_prefix, solver_name, seed))
    print(
    f"[run_multi_k] param={param_file}, qy_prefix={qy_prefix}, solvers={solver_names}, "
        f"k_values={[float(k) for k in k_values]}, max_concurrency={max_concurrency}, seed={seed}, tasks={len(all_args)}",
        flush=True,
    )
    for task_idx, k_val, _p, _qy, s, _seed in all_args:
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


def run_multi_k_with_cache(param_file: str, qy_prefix: str, solver_classes: Sequence[type], max_concurrency: Optional[int] = None,
                            shelve_paths: Optional[Dict[str, str]] = None, seed: Optional[int] = None) -> Dict[str, List[object]]:
    """Compatibility wrapper mirroring main.run_multi_k_with_cache signature/behavior, but centralized here.
    Returns dict[solver_name] -> List[params], where each list aligns with k values.
    """
    import shelve

    container = Container(param_file, seed=seed, qy_prefix=qy_prefix)
    sim = container.make_sim()
    k_values = sim.params.k
    config_data = _load_config_data(param_file)
    config_signature = _compute_config_signature_from_data(config_data)

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
                pending_tasks.append((task_ptr, k_val, param_file, qy_prefix, solver_name, seed))
                task_ptr += 1
            continue
        # try read cache keys
        try:
            with shelve.open(shelve_path, flag='c') as db:
                for k_idx, k_val in enumerate(k_values):
                    key = _cache_key_for_k(k_val)
                    entry = db.get(key)
                    if _entry_matches_signature(entry, config_signature):
                        continue
                    if entry is not None:
                        try:
                            del db[key]
                        except Exception:
                            pass
                        index_map[task_ptr] = (solver_name, k_idx)
                        pending_tasks.append((task_ptr, k_val, param_file, qy_prefix, solver_name, seed))
                        task_ptr += 1
        except (OSError, RuntimeError):
            # if shelve broken, schedule all
            for k_idx, k_val in enumerate(k_values):
                index_map[task_ptr] = (solver_name, k_idx)
                pending_tasks.append((task_ptr, k_val, param_file, qy_prefix, solver_name, seed))
                task_ptr += 1

    # run pending
    results_dict: Dict[str, List[object]] = {name: [None] * len(k_values) for name in solver_names}

    failed_entries: set[Tuple[str, int]] = set()

    if pending_tasks:
        _log_task_schedule(
            "run_multi_k_with_cache",
            param_file,
            qy_prefix,
            solver_names,
            k_values,
            pending_tasks,
            max_concurrency,
            seed,
        )
        max_workers = max_concurrency or os.cpu_count()
        task_info_by_idx = {task_idx: (k_val, solver_name) for task_idx, k_val, _p, _qy, solver_name, _seed in pending_tasks}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_universal_worker, t): t for t in pending_tasks}
            for fut in as_completed(future_map):
                task_args = future_map[fut]
                try:
                    task_idx, params, solver_name = fut.result()
                except Exception as exc:
                    _log_task_error("run_multi_k_with_cache", task_args, exc)
                    solver_name_mapped, k_idx = index_map[task_args[0]]
                    failed_entries.add((solver_name_mapped, k_idx))
                    continue
                solver_name_mapped, k_idx = index_map[task_idx]
                if hasattr(params, "k"):
                    params.k = k_values
                results_dict[solver_name_mapped][k_idx] = params
                # write-through cache
                shelve_path = shelve_paths.get(solver_name_mapped)
                if shelve_path:
                    try:
                        with shelve.open(shelve_path, flag='c') as db:
                            key = _cache_key_for_k(k_values[k_idx])
                            db[key] = _prepare_cache_entry(params, config_data)
                    except (OSError, RuntimeError):
                        pass
                k_val, _ = task_info_by_idx.get(task_idx, (k_values[k_idx], solver_name_mapped))
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
                    if results_dict[solver_name][k_idx] is None:
                        key = _cache_key_for_k(k_val)
                        entry = db.get(key)
                        if not _entry_matches_signature(entry, config_signature):
                            continue
                        try:
                            cached_params = entry["params"]
                            setattr(cached_params, "k", k_values)
                            results_dict[solver_name][k_idx] = cached_params
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
            if results_dict[solver_name][k_idx] is None and (solver_name, k_idx) not in failed_entries:
                missing_index_map[miss_ptr] = (solver_name, k_idx)
                missing_tasks.append((miss_ptr, float(k_val), param_file, qy_prefix, solver_name, seed))
                miss_ptr += 1

    if missing_tasks:
        _log_task_schedule(
            "run_multi_k_with_cache:fallback",
            param_file,
            qy_prefix,
            solver_names,
            k_values,
            missing_tasks,
            max_concurrency,
            seed,
        )
        max_workers = max_concurrency or os.cpu_count()
        task_info_by_idx = {task_idx: (k_val, solver_name) for task_idx, k_val, _p, _qy, solver_name, _seed in missing_tasks}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_universal_worker, t): t for t in missing_tasks}
            for fut in as_completed(future_map):
                task_args = future_map[fut]
                try:
                    task_idx, params, solver_name = fut.result()
                except Exception as exc:
                    _log_task_error("run_multi_k_with_cache", task_args, exc)
                    solver_name_mapped, k_idx = missing_index_map[task_args[0]]
                    failed_entries.add((solver_name_mapped, k_idx))
                    continue
                solver_name_mapped, k_idx = missing_index_map[task_idx]
                if hasattr(params, "k"):
                    params.k = k_values
                results_dict[solver_name_mapped][k_idx] = params
                # write-through cache
                shelve_path = shelve_paths.get(solver_name_mapped)
                if shelve_path:
                    try:
                        with shelve.open(shelve_path, flag='c') as db:
                            key = _cache_key_for_k(k_values[k_idx])
                            db[key] = _prepare_cache_entry(params, config_data)
                    except (OSError, RuntimeError):
                        pass
                k_val, _ = task_info_by_idx.get(task_idx, (k_values[k_idx], solver_name_mapped))
                total = float(sum(getattr(params, 'reward_history', []) or [0.0]))
                steps = len(getattr(params, 'reward_history', []) or [])
                extras = _collect_extra_fields(solver_name, params, k_val)
                extras_suffix = f", {', '.join(extras)}" if extras else ""
                print(
                    f"[done] task#{task_idx} solver={solver_name} k={float(k_val)} total_reward={total:.4f} steps={steps}{extras_suffix}",
                    flush=True,
                )

    if not pending_tasks and not missing_tasks:
        print(
            f"[run_multi_k_with_cache] param={param_file}, qy_prefix={qy_prefix}, solvers={solver_names}, "
            f"k_values={[float(k) for k in k_values]}, pending_tasks=0, reused_from_cache=True",
            flush=True,
        )

    return results_dict
