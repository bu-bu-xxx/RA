import os
from solver import RABBI, OFFline, NPlusOneLP
import numpy as np
from customer import CustomerChoiceSimulator
import concurrent.futures
import shelve


# 顶层worker函数，支持多进程pickle

def rabbi_worker(args):
    i, k_val, param_file, y_filename = args
    from solver import RABBI
    from customer import CustomerChoiceSimulator
    import numpy as np
    import os
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    sim.B = sim.B * k_val
    sim.T = int(sim.T * k_val)
    y_file = f"{y_filename}_k{int(k_val)}.npy"
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    sim.compute_offline_Q()
    rabbi = RABBI(sim)
    rabbi.run()
    print(f"[RABBI][k={k_val}] x_history shape:", np.array(sim.x_history).shape)
    print(f"[RABBI][k={k_val}] alpha_history:", sim.alpha_history)
    print(f"[RABBI][k={k_val}] j_history:", sim.j_history)
    print(f"[RABBI][k={k_val}] b_history:", sim.b_history)
    print(f"[RABBI][k={k_val}] reward_history:", sim.reward_history)
    print(f"[RABBI][k={k_val}] Final inventory:", sim.b)
    print(f"[RABBI][k={k_val}] total reward:", sum(sim.reward_history))
    return (i, sim)

def offline_worker(args):
    i, k_val, param_file, y_filename = args
    from solver import OFFline
    from customer import CustomerChoiceSimulator
    import numpy as np
    import os
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    sim.B = sim.B * k_val
    sim.T = int(sim.T * k_val)
    y_file = f"{y_filename}_k{int(k_val)}.npy"
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    sim.compute_offline_Q()
    offline = OFFline(sim)
    offline.run()
    print(f"[OFFline][k={k_val}] x_history shape:", np.array(sim.x_history).shape)
    print(f"[OFFline][k={k_val}] alpha_history:", sim.alpha_history)
    print(f"[OFFline][k={k_val}] j_history:", sim.j_history)
    print(f"[OFFline][k={k_val}] b_history:", sim.b_history)
    print(f"[OFFline][k={k_val}] reward_history:", sim.reward_history)
    print(f"[OFFline][k={k_val}] Final inventory:", sim.b)
    print(f"[OFFline][k={k_val}] total reward:", sum(sim.reward_history))
    return (i, sim)

def nplusonelp_worker(args):
    i, k_val, param_file, y_filename = args
    from solver import NPlusOneLP
    from customer import CustomerChoiceSimulator
    import numpy as np
    import os
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    sim.B = sim.B * k_val
    sim.T = int(sim.T * k_val)
    y_file = f"{y_filename}_k{int(k_val)}.npy"
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    sim.compute_offline_Q()
    rabbi_nplus1 = NPlusOneLP(sim, debug=False)
    rabbi_nplus1.run()
    print(f"[NPlusOneLP][k={k_val}] x_history shape:", np.array(sim.x_history).shape)
    print(f"[NPlusOneLP][k={k_val}] alpha_history:", sim.alpha_history)
    print(f"[NPlusOneLP][k={k_val}] j_history:", sim.j_history)
    print(f"[NPlusOneLP][k={k_val}] b_history:", sim.b_history)
    print(f"[NPlusOneLP][k={k_val}] reward_history:", sim.reward_history)
    print(f"[NPlusOneLP][k={k_val}] Final inventory:", sim.b)
    print(f"[NPlusOneLP][k={k_val}] total reward:", sum(sim.reward_history))
    return (i, sim)

def run_rabbi(param_file, y_file):
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    rabbi = RABBI(sim)
    rabbi.run()
    print("[RABBI] x_history shape:", np.array(sim.x_history).shape)
    print("[RABBI] alpha_history:", sim.alpha_history)
    print("[RABBI] j_history:", sim.j_history)
    print("[RABBI] b_history:", sim.b_history)
    print("[RABBI] reward_history:", sim.reward_history)
    print("[RABBI] Final inventory:", sim.b)
    print("[RABBI] total reward:", sum(sim.reward_history))

def run_offline(param_file, y_file):
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    sim.compute_offline_Q()  # 计算Q矩阵
    offline = OFFline(sim)
    offline.run()
    print("[OFFline] x_history shape:", np.array(sim.x_history).shape)
    print("[OFFline] alpha_history:", sim.alpha_history)
    print("[OFFline] j_history:", sim.j_history)
    print("[OFFline] b_history:", sim.b_history)
    print("[OFFline] reward_history:", sim.reward_history)
    print("[OFFline] Final inventory:", sim.b)
    print("[OFFline] total reward:", sum(sim.reward_history))

def run_nplusonelp(param_file, y_file):
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    rabbi_nplus1 = NPlusOneLP(sim, debug=False)
    rabbi_nplus1.run()
    print("[NPlusOneLP] x_history shape:", np.array(sim.x_history).shape)
    print("[NPlusOneLP] alpha_history:", sim.alpha_history)
    print("[NPlusOneLP] j_history:", sim.j_history)
    print("[NPlusOneLP] b_history:", sim.b_history)
    print("[NPlusOneLP] reward_history:", sim.reward_history)
    print("[NPlusOneLP] Final inventory:", sim.b)
    print("[NPlusOneLP] total reward:", sum(sim.reward_history))

def run_rabbi_multi_k(param_file, y_filename, max_concurrency=None):
    if max_concurrency is None:
        max_concurrency = os.cpu_count()
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    k = sim.k  # 获取倍率列表, 所有sim实例都使用同一倍率列表
    args_list = [(i, k_val, param_file, y_filename) for i, k_val in enumerate(k)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrency) as executor:
        results = list(executor.map(rabbi_worker, args_list))
    results.sort(key=lambda x: x[0])
    sim_list = [sim for i, sim in results]
    print(f"[RABBI] sim_list length: {len(sim_list)}")
    return sim_list

def run_offline_multi_k(param_file, y_filename, max_concurrency=None):
    if max_concurrency is None:
        max_concurrency = os.cpu_count()
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    k = sim.k
    args_list = [(i, k_val, param_file, y_filename) for i, k_val in enumerate(k)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrency) as executor:
        results = list(executor.map(offline_worker, args_list))
    results.sort(key=lambda x: x[0])
    sim_list = [sim for i, sim in results]
    print(f"[OFFline] sim_list length: {len(sim_list)}")
    return sim_list

def run_nplusonelp_multi_k(param_file, y_filename, max_concurrency=None):
    if max_concurrency is None:
        max_concurrency = os.cpu_count()
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    k = sim.k
    args_list = [(i, k_val, param_file, y_filename) for i, k_val in enumerate(k)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrency) as executor:
        results = list(executor.map(nplusonelp_worker, args_list))
    results.sort(key=lambda x: x[0])
    sim_list = [sim for i, sim in results]
    print(f"[NPlusOneLP] sim_list length: {len(sim_list)}")
    return sim_list

def compute_lp_x_benchmark(sim) -> np.ndarray:
    """
    对于给定sim，依次用历史b_history和alpha_history，计算每一步的LP解x_t，
    并输出每一步对应alpha的x_t[alpha]，返回长度为T的列表。
    """
    from solver import LPBasedPolicy
    T = len(sim.alpha_history)
    assert T == sim.T 
    result = []
    for t in range(T):
        b = np.array(sim.b_history[t])
        alpha = sim.alpha_history[t]
        p_t = sim.Q[t, :, :]
        # 下面参数依赖sim的结构
        x_t = LPBasedPolicy.solve_lp(
            b, p_t, t, sim.n, sim.m, sim.d, sim.f, sim.A, sim.T
        )
        result.append(x_t[alpha])
    result = np.array(result)
    return result

def save_sim_list_to_shelve(sim_list, shelve_path):
    """
    将sim_list中的每个sim对象的关键参数保存到shelve文件。
    每个sim以sim_{idx}为key，保存为dict，内容包括：
    b_history, reward_history, alpha_history, j_history, x_history, Q, Y, T, n, m, d, A, f, B, k, p, mnl_params, linear_params
    """
    with shelve.open(shelve_path) as db:
        for idx, sim in enumerate(sim_list):
            sim_data = {
                'b_history': getattr(sim, 'b_history', None),
                'reward_history': getattr(sim, 'reward_history', None),
                'alpha_history': getattr(sim, 'alpha_history', None),
                'j_history': getattr(sim, 'j_history', None),
                'x_history': getattr(sim, 'x_history', None),
                'Q': getattr(sim, 'Q', None),
                'Y': getattr(sim, 'Y', None),
                'T': getattr(sim, 'T', None),
                'n': getattr(sim, 'n', None),
                'm': getattr(sim, 'm', None),
                'd': getattr(sim, 'd', None),
                'A': getattr(sim, 'A', None),
                'f': getattr(sim, 'f', None),
                'B': getattr(sim, 'B', None),
                'k': getattr(sim, 'k', None),
                'p': getattr(sim, 'p', None),
                'mnl': vars(getattr(sim, 'mnl', {})) if hasattr(sim, 'mnl') else None,
                'linear': vars(getattr(sim, 'linear', {})) if hasattr(sim, 'linear') else None,
            }
            db[f'sim_{idx}'] = sim_data
    print(f"已将{len(sim_list)}个sim对象保存到shelve文件: {shelve_path}")

def load_sim_list_from_shelve(shelve_path):
    """
    从指定shelve文件读取所有sim数据，返回sim_list（每个元素为dict，结构与save_sim_list_to_shelve一致）。
    """
    sim_list = []
    with shelve.open(shelve_path) as db:
        # 按key顺序还原
        keys = sorted([k for k in db.keys() if k.startswith('sim_')], key=lambda x: int(x.split('_')[1]))
        for k in keys:
            sim_data = db[k]
            sim_list.append(sim_data)
    print(f"已从shelve文件{shelve_path}读取{len(sim_list)}个sim对象")
    return sim_list

if __name__ == "__main__":
    param_file = 'params.yml'
    # y_file = os.path.join("data", 'Y_matrix_debug.npy')
    # print("===== RABBI 示例 =====")
    # run_rabbi(param_file, y_file)
    # print("\n===== OFFline 示例 =====")
    # run_offline(param_file, y_file)
    # print("\n===== NPlusOneLP 示例 =====")
    # run_nplusonelp(param_file, y_file)


    y_filename = os.path.join("data", 'Y_matrix_debug')
    print("\n===== RABBI 多倍率示例 =====")
    sim_rabbi = run_rabbi_multi_k(param_file, y_filename)
    print("\n===== OFFline 多倍率示例 =====")
    sim_offline = run_offline_multi_k(param_file, y_filename)
    print("\n===== NPlusOneLP 多倍率示例 =====")
    sim_nplusonelp = run_nplusonelp_multi_k(param_file, y_filename)

    # 计算x_benchmark
    print("\n===== 计算LP解基准 =====\n")
    rabbi_x_benchmark = compute_lp_x_benchmark(sim_rabbi[0])  # 只取第一个sim作为基准
    offline_x_benchmark = compute_lp_x_benchmark(sim_offline[0])
    nplus1_x_benchmark = compute_lp_x_benchmark(sim_nplusonelp[0])
    print("[RABBI] x_benchmark:", rabbi_x_benchmark)
    print("[OFFline] x_benchmark:", offline_x_benchmark)
    print("[NPlusOneLP] x_benchmark:", nplus1_x_benchmark)

    # 保存sim_list到shelve文件
    shelve_path_rabbi = os.path.join("data", "sim_rabbi.shelve")
    shelve_path_offline = os.path.join("data", "sim_offline.shelve")
    shelve_path_nplusonelp = os.path.join("data", "sim_nplusonelp.shelve")
    save_sim_list_to_shelve(sim_rabbi, shelve_path_rabbi)
    save_sim_list_to_shelve(sim_offline, shelve_path_offline)
    save_sim_list_to_shelve(sim_nplusonelp, shelve_path_nplusonelp)

    # 从shelve文件加载sim_list示例
    print("\n===== 从shelve文件加载sim_list示例 =====")
    loaded_sim_rabbi = load_sim_list_from_shelve(shelve_path_rabbi)
    loaded_sim_offline = load_sim_list_from_shelve(shelve_path_offline)
    loaded_sim_nplusonelp = load_sim_list_from_shelve(shelve_path_nplusonelp)
    print(f"[Loaded RABBI] sim_list length: {len(loaded_sim_rabbi)}")
    print(f"[Loaded OFFline] sim_list length: {len(loaded_sim_offline)}")
    print(f"[Loaded NPlusOneLP] sim_list length: {len(loaded_sim_nplusonelp)}")



