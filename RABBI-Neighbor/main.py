import os
from solver import RABBI, OFFline, NPlusOneLP, TopKLP
import numpy as np
from customer import CustomerChoiceSimulator
import concurrent.futures
import shelve


# 顶层worker函数，支持多进程pickle

def universal_worker(args):
    i, k_val, param_file, y_filename, solver_class_name = args
    from customer import CustomerChoiceSimulator
    import numpy as np
    import os
    
    # 动态导入solver类
    exec(f"from solver import {solver_class_name}")
    solver_class = locals()[solver_class_name]
    
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    sim.params.B = sim.params.B * k_val
    sim.params.T = int(sim.params.T * k_val)
    y_file = f"{y_filename}_k{int(k_val)}.npy"
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    sim.compute_offline_Q()
    
    # 创建求解器实例
    if solver_class_name == 'TopKLP':
        solver = solver_class(sim, topk=3, debug=False)  ################## 使用默认topk
    else:
        solver = solver_class(sim, debug=False)
    
    solver.run()
    print(f"[{solver_class_name}][k={k_val}] x_history shape:", np.array(sim.params.x_history).shape)
    print(f"[{solver_class_name}][k={k_val}] alpha_history:", sim.params.alpha_history)
    print(f"[{solver_class_name}][k={k_val}] j_history:", sim.params.j_history)
    print(f"[{solver_class_name}][k={k_val}] b_history:", sim.params.b_history)
    print(f"[{solver_class_name}][k={k_val}] reward_history:", sim.params.reward_history)
    print(f"[{solver_class_name}][k={k_val}] Final inventory:", sim.params.b)
    print(f"[{solver_class_name}][k={k_val}] total reward:", sum(sim.params.reward_history))
    return (i, sim.params, solver_class_name)

def run_rabbi(param_file, y_file):
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    rabbi = RABBI(sim)
    rabbi.run()
    print("[RABBI] x_history shape:", np.array(sim.params.x_history).shape)
    print("[RABBI] alpha_history:", sim.params.alpha_history)
    print("[RABBI] j_history:", sim.params.j_history)
    print("[RABBI] b_history:", sim.params.b_history)
    print("[RABBI] reward_history:", sim.params.reward_history)
    print("[RABBI] Final inventory:", sim.params.b)
    print("[RABBI] total reward:", sum(sim.params.reward_history))

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
    print("[OFFline] x_history shape:", np.array(sim.params.x_history).shape)
    print("[OFFline] alpha_history:", sim.params.alpha_history)
    print("[OFFline] j_history:", sim.params.j_history)
    print("[OFFline] b_history:", sim.params.b_history)
    print("[OFFline] reward_history:", sim.params.reward_history)
    print("[OFFline] Final inventory:", sim.params.b)
    print("[OFFline] total reward:", sum(sim.params.reward_history))

def run_nplusonelp(param_file, y_file):
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    rabbi_nplus1 = NPlusOneLP(sim, debug=False)
    rabbi_nplus1.run()
    print("[NPlusOneLP] x_history shape:", np.array(sim.params.x_history).shape)
    print("[NPlusOneLP] alpha_history:", sim.params.alpha_history)
    print("[NPlusOneLP] j_history:", sim.params.j_history)
    print("[NPlusOneLP] b_history:", sim.params.b_history)
    print("[NPlusOneLP] reward_history:", sim.params.reward_history)
    print("[NPlusOneLP] Final inventory:", sim.params.b)
    print("[NPlusOneLP] total reward:", sum(sim.params.reward_history))

def run_multi_k(param_file, y_filename, solver_classes, max_concurrency=None):
    """
    通用的多倍率运行函数
    param_file: 参数文件路径
    y_filename: Y矩阵文件前缀
    solver_classes: 求解器类列表，如[RABBI, OFFline, NPlusOneLP]
    max_concurrency: 最大并发数
    返回: 字典，键为求解器名称，值为对应的params_list
    """
    if max_concurrency is None:
        max_concurrency = os.cpu_count()
    
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    k = sim.params.k  # 获取倍率列表
    
    # 合并所有solver_class的args_list
    all_args_list = []
    solver_k_mapping = {}  # 记录每个结果对应的solver和k的索引
    
    for solver_class in solver_classes:
        solver_name = solver_class.__name__
        print(f"\n===== 准备运行 {solver_name} 多倍率示例 =====")
        
        for i, k_val in enumerate(k):
            args = (len(all_args_list), k_val, param_file, y_filename, solver_name)
            all_args_list.append(args)
            solver_k_mapping[len(all_args_list) - 1] = (solver_name, i)
    
    print(f"\n===== 开始并行运行所有任务，总共 {len(all_args_list)} 个任务 =====")
    
    # 一次性运行所有任务
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrency) as executor:
        all_results = list(executor.map(universal_worker, all_args_list))
    
    # 按solver分组整理结果
    results_dict = {}
    for solver_class in solver_classes:
        solver_name = solver_class.__name__
        results_dict[solver_name] = [None] * len(k)
    
    # 将结果分配到对应的solver和k位置
    for result in all_results:
        task_idx, params, solver_name = result
        solver_name_mapped, k_idx = solver_k_mapping[task_idx]
        results_dict[solver_name_mapped][k_idx] = params
    
    # 输出结果统计
    for solver_name in results_dict:
        print(f"[{solver_name}] params_list length: {len(results_dict[solver_name])}")
    
    return results_dict

def compute_lp_x_benchmark(params) -> np.ndarray:
    """
    对于给定params，依次用历史b_history和alpha_history，计算每一步的LP解x_t，
    并输出每一步对应alpha的x_t[alpha]，返回长度为T的列表。
    """
    from solver import LPBasedPolicy
    T = len(params.alpha_history)
    assert T == params.T 
    result = []
    for t in range(T):
        b = np.array(params.b_history[t])
        alpha = params.alpha_history[t]
        p_t = params.Q[t, :, :]
        # 下面参数依赖params的结构
        x_t = LPBasedPolicy.solve_lp(
            b, p_t, t, params.n, params.m, params.d, params.f, params.A, params.T
        )
        result.append(x_t[alpha])
    result = np.array(result)
    return result

def save_params_list_to_shelve(params_list, shelve_path):
    """
    将params_list中的每个params对象直接保存到shelve文件。
    每个params以params_{idx}为key保存。
    """
    with shelve.open(shelve_path) as db:
        for idx, params in enumerate(params_list):
            db[f'params_{idx}'] = params
    print(f"已将{len(params_list)}个params对象保存到shelve文件: {shelve_path}")

def load_params_list_from_shelve(shelve_path):
    """
    从指定shelve文件读取所有params数据，返回params_list。
    """
    params_list = []
    with shelve.open(shelve_path) as db:
        # 按key顺序还原
        keys = sorted([k for k in db.keys() if k.startswith('params_')], key=lambda x: int(x.split('_')[1]))
        for k in keys:
            params_data = db[k]
            params_list.append(params_data)
    print(f"已从shelve文件{shelve_path}读取{len(params_list)}个params对象")
    return params_list

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
    
    # 使用新的统一函数运行所有求解器
    solver_classes = [RABBI, OFFline, NPlusOneLP, TopKLP]
    results = run_multi_k(param_file, y_filename, solver_classes)
    
    # 从结果字典中提取各个params_list
    params_rabbi = results['RABBI']
    params_offline = results['OFFline']
    params_nplusonelp = results['NPlusOneLP']
    params_topklp = results['TopKLP']

    # 计算x_benchmark
    print("\n===== 计算LP解基准 =====\n")
    rabbi_x_benchmark = compute_lp_x_benchmark(params_rabbi[0])  # 只取第一个params作为基准
    offline_x_benchmark = compute_lp_x_benchmark(params_offline[0])
    nplus1_x_benchmark = compute_lp_x_benchmark(params_nplusonelp[0])
    topklp_x_benchmark = compute_lp_x_benchmark(params_topklp[0])
    print("[RABBI] x_benchmark:", rabbi_x_benchmark)
    print("[OFFline] x_benchmark:", offline_x_benchmark)
    print("[NPlusOneLP] x_benchmark:", nplus1_x_benchmark)
    print("[TopKLP] x_benchmark:", topklp_x_benchmark)

    # 保存params_list到shelve文件
    shelve_path_rabbi = os.path.join("data", "shelve", "params_rabbi.shelve")
    shelve_path_offline = os.path.join("data", "shelve", "params_offline.shelve")
    shelve_path_nplusonelp = os.path.join("data", "shelve", "params_nplusonelp.shelve")
    shelve_path_topklp = os.path.join("data", "shelve", "params_topklp.shelve")
    save_params_list_to_shelve(params_rabbi, shelve_path_rabbi)
    save_params_list_to_shelve(params_offline, shelve_path_offline)
    save_params_list_to_shelve(params_nplusonelp, shelve_path_nplusonelp)
    save_params_list_to_shelve(params_topklp, shelve_path_topklp)

    # 从shelve文件加载params_list示例
    print("\n===== 从shelve文件加载params_list示例 =====")
    loaded_params_rabbi = load_params_list_from_shelve(shelve_path_rabbi)
    loaded_params_offline = load_params_list_from_shelve(shelve_path_offline)
    loaded_params_nplusonelp = load_params_list_from_shelve(shelve_path_nplusonelp)
    loaded_params_topklp = load_params_list_from_shelve(shelve_path_topklp)
    print(f"[Loaded RABBI] params_list length: {len(loaded_params_rabbi)}")
    print(f"[Loaded OFFline] params_list length: {len(loaded_params_offline)}")
    print(f"[Loaded NPlusOneLP] params_list length: {len(loaded_params_nplusonelp)}")
    print(f"[Loaded TopKLP] params_list length: {len(loaded_params_topklp)}")



