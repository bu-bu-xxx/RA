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

def run_multi_k_with_cache(param_file, y_filename, solver_classes, max_concurrency=4, 
                          shelve_path_rabbi=None, shelve_path_offline=None, 
                          shelve_path_nplusonelp=None, shelve_path_topklp=None):
    """
    智能运行多倍率任务，带缓存功能
    - 检测shelve文件中已存在的params，跳过已完成的任务
    - 只运行缺失的任务并实时保存到对应的shelve文件
    - 任务失败时记录日志但不抛出异常
    
    参数:
        param_file: 参数文件路径
        y_filename: Y矩阵文件前缀
        solver_classes: 求解器类列表
        max_concurrency: 最大并发数
        shelve_path_*: 各求解器对应的shelve文件路径
    
    返回: 字典，键为求解器名称，值为对应的params_list
    """
    import shelve
    from pathlib import Path
    
    # 获取k值列表
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    k = sim.params.k
    
    # 建立求解器名称到shelve路径的映射
    shelve_paths = {
        'RABBI': shelve_path_rabbi,
        'OFFline': shelve_path_offline, 
        'NPlusOneLP': shelve_path_nplusonelp,
        'TopKLP': shelve_path_topklp
    }
    
    # 检查已存在的params并构建待运行任务列表
    pending_tasks = []
    existing_tasks_count = 0  # 统计已存在的任务数
    
    for solver_class in solver_classes:
        solver_name = solver_class.__name__
        shelve_path = shelve_paths.get(solver_name)
        
        if shelve_path is None:
            print(f"警告: {solver_name} 没有提供shelve路径，跳过该求解器")
            continue
        
        # 检查shelve文件中已存在的params，但不加载到内存
        if os.path.exists(shelve_path + '.dat') or os.path.exists(shelve_path + '.db'):
            try:
                with shelve.open(shelve_path, 'r') as db:
                    for k_idx in range(len(k)):
                        key = f'params_{k_idx}'
                        if key in db:
                            existing_tasks_count += 1
                            print(f"[{solver_name}][k={k[k_idx]}] 从缓存跳过")
                        else:
                            # 添加到待运行任务
                            task_args = (len(pending_tasks), k[k_idx], param_file, y_filename, solver_name, k_idx, shelve_path)
                            pending_tasks.append(task_args)
                            print(f"[{solver_name}][k={k[k_idx]}] 添加到待运行队列")
            except Exception as e:
                print(f"读取{shelve_path}时出错: {e}，将重新运行所有任务")
                for k_idx in range(len(k)):
                    task_args = (len(pending_tasks), k[k_idx], param_file, y_filename, solver_name, k_idx, shelve_path)
                    pending_tasks.append(task_args)
        else:
            # shelve文件不存在，添加所有k值任务
            print(f"[{solver_name}] shelve文件不存在，添加所有任务到队列")
            for k_idx in range(len(k)):
                task_args = (len(pending_tasks), k[k_idx], param_file, y_filename, solver_name, k_idx, shelve_path)
                pending_tasks.append(task_args)
    
    print(f"\n===== 总计 {len(pending_tasks)} 个待运行任务 =====")
    
    # 运行待完成的任务
    success_count = 0  # 成功任务计数
    failed_count = 0   # 失败任务计数
    
    if pending_tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrency) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(cached_worker, task): task for task in pending_tasks}
            
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                task_idx, k_val, param_file, y_filename, solver_name, k_idx, shelve_path = task
                
                try:
                    result = future.result()
                    if result is not None:
                        task_idx, params, solver_class_name = result
                        total_reward = sum(params.reward_history)
                        success_count += 1
                        print(f"[{solver_name}][k={k_val}] 任务完成，total reward: {total_reward}")
                    else:
                        failed_count += 1
                        print(f"[{solver_name}][k={k_val}] 任务返回None")
                except Exception as e:
                    failed_count += 1
                    print(f"[{solver_name}][k={k_val}] 任务执行失败: {e}")
    else:
        print("所有任务都已从缓存加载，无需运行新任务")
    
    # 打印任务执行统计
    total_tasks = len(solver_classes) * len(k)  # 总任务数
    print(f"\n===== 任务执行统计 =====")
    print(f"总任务数: {total_tasks}")
    print(f"已存在任务数: {existing_tasks_count}")
    print(f"待运行任务数: {len(pending_tasks)}")
    print(f"运行成功任务数: {success_count}")
    print(f"运行失败任务数: {failed_count}")
    if len(pending_tasks) > 0:
        success_rate = (success_count / len(pending_tasks)) * 100
        print(f"运行成功率: {success_rate:.1f}%")
    
    # 任务完成后，统一从shelve文件读取所有结果
    print(f"\n===== 从shelve文件统一加载所有结果 =====")
    results_dict = {}
    for solver_class in solver_classes:
        solver_name = solver_class.__name__
        shelve_path = shelve_paths.get(solver_name)
        
        if shelve_path is None:
            continue
            
        # 从shelve文件加载该求解器的所有params
        params_list = [None] * len(k)
        completed_count = 0
        
        if os.path.exists(shelve_path + '.dat') or os.path.exists(shelve_path + '.db'):
            try:
                with shelve.open(shelve_path, 'r') as db:
                    for k_idx in range(len(k)):
                        key = f'params_{k_idx}'
                        if key in db:
                            params_list[k_idx] = db[key]
                            completed_count += 1
                print(f"[{solver_name}] 从shelve文件加载 {completed_count}/{len(k)} 个结果")
            except Exception as e:
                print(f"读取{shelve_path}时出错: {e}")
        
        results_dict[solver_name] = params_list
        print(f"[{solver_name}] 完成 {completed_count}/{len(k)} 个任务")
    
    return results_dict


def cached_worker(args):
    """
    带缓存保存功能的worker函数
    """
    task_idx, k_val, param_file, y_filename, solver_class_name, k_idx, shelve_path = args
    
    try:
        # 执行任务（复用universal_worker的逻辑）
        from customer import CustomerChoiceSimulator
        import numpy as np
        import os
        import shelve
        
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
        
        # 创建求解器实例并运行
        solver = solver_class(sim, debug=False)
        solver.run()
        
        # 立即保存到shelve文件
        with shelve.open(shelve_path) as db:
            db[f'params_{k_idx}'] = sim.params
        
        # 不在这里打印，让主进程打印
        return (task_idx, sim.params, solver_class_name)
        
    except Exception as e:
        print(f"[{solver_class_name}][k={k_val}] 任务失败: {e}")
        return None

if __name__ == "__main__":
    param_file = 'params.yml'
    # y_file = os.path.join("data", "Y", 'Y_matrix_debug.npy')
    # print("===== RABBI 示例 =====")
    # run_rabbi(param_file, y_file)
    # print("\n===== OFFline 示例 =====")
    # run_offline(param_file, y_file)
    # print("\n===== NPlusOneLP 示例 =====")
    # run_nplusonelp(param_file, y_file)
    y_filename = os.path.join("data", "Y", 'Y_matrix_debug')
    
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

    # 使用带缓存的智能运行示例
    print("\n===== 带缓存的智能运行示例 =====")
    # 注释掉以避免重复运行，如需测试请取消注释
    # cache_results = run_multi_k_with_cache(
    #     param_file, y_filename, solver_classes, max_concurrency=4,
    #     shelve_path_rabbi=shelve_path_rabbi,
    #     shelve_path_offline=shelve_path_offline,
    #     shelve_path_nplusonelp=shelve_path_nplusonelp,
    #     shelve_path_topklp=shelve_path_topklp
    # )



