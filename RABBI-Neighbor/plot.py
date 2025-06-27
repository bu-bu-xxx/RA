import os
import matplotlib.pyplot as plt
from main import run_multi_k
from solver import RABBI, OFFline, NPlusOneLP, TopKLP
import numpy as np

def plot_multi_k_results(rabbi_params, offline_params, nplus1_params):
    # 获取k_list
    k_list = None
    for params in [rabbi_params, offline_params, nplus1_params]:
        if params is not None and len(params) > 0:
            k_list = params[0].k if hasattr(params[0], 'k') else list(range(len(params)))
            break
    
    if k_list is None:
        print("Warning: No valid params provided for plotting")
        return

    plt.figure(figsize=(8,6))
    
    # 只绘制非None的params
    if rabbi_params is not None:
        rabbi_rewards = [sum(params.reward_history) for params in rabbi_params]
        plt.plot(k_list, rabbi_rewards, marker='o', label='RABBI')
    
    if offline_params is not None:
        offline_rewards = [sum(params.reward_history) for params in offline_params]
        plt.plot(k_list, offline_rewards, marker='s', label='OFFline')
    
    if nplus1_params is not None:
        nplus1_rewards = [sum(params.reward_history) for params in nplus1_params]
        plt.plot(k_list, nplus1_rewards, marker='^', label='NPlusOneLP')
    
    plt.xlabel('k (scaling factor)')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs k for Different Policies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multi_k_ratio_results(rabbi_params, offline_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
    # 获取k_list
    k_list = None
    for params in [rabbi_params, offline_params, nplus1_params, topklp_params]:
        if params is not None and len(params) > 0:
            k_list = params[0].k if hasattr(params[0], 'k') else list(range(len(params)))
            break
    
    if k_list is None:
        raise ValueError("No valid params provided for plotting")
        return
    
    # 获取offline_rewards作为基准，如果offline_params为None则无法计算比例
    if offline_params is None:
        raise ValueError("offline_params is None, cannot compute ratios")
        return
    
    offline_rewards = [sum(params.reward_history) for params in offline_params]

    plt.figure(figsize=(8,6))
    
    # 只绘制非None的params
    if rabbi_params is not None:
        rabbi_rewards = [sum(params.reward_history) for params in rabbi_params]
        rabbi_ratio = [r/o if o != 0 else 0 for r, o in zip(rabbi_rewards, offline_rewards)]
        plt.plot(k_list, rabbi_ratio, marker='o', label='RABBI / OFFline')
    
    if nplus1_params is not None:
        nplus1_rewards = [sum(params.reward_history) for params in nplus1_params]
        nplus1_ratio = [n/o if o != 0 else 0 for n, o in zip(nplus1_rewards, offline_rewards)]
        plt.plot(k_list, nplus1_ratio, marker='^', label='NPlusOneLP / OFFline')
    
    if topklp_params is not None:
        topklp_rewards = [sum(params.reward_history) for params in topklp_params]
        topklp_ratio = [t/o if o != 0 else 0 for t, o in zip(topklp_rewards, offline_rewards)]
        plt.plot(k_list, topklp_ratio, marker='s', label='TopKLP / OFFline')
    
    plt.xlabel('k (scaling factor)')
    plt.ylabel('Reward Ratio to OFFline')
    plt.title('Reward Ratio vs k')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_lp_x_benchmark_ratio_vs_k(rabbi_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
    """
    输入rabbi_params、nplus1_params和topklp_params，分别计算每个params的compute_lp_x_benchmark(params)，
    统计每个输出结果中>=1的数所占比例，
    并画plot，x轴为params.k，y轴为比例，三组曲线同图展示。
    """
    from main import compute_lp_x_benchmark
    
    # 获取k_list
    k_list = None
    for params in [rabbi_params, nplus1_params, topklp_params]:
        if params is not None and len(params) > 0:
            k_list = params[0].k if hasattr(params[0], 'k') else list(range(len(params)))
            break
    if k_list is None:
        print("Warning: No valid params provided for plotting")
        return

    plt.figure(figsize=(8,6))
    
    # 只计算和绘制非None的params
    if rabbi_params is not None:
        rabbi_ratios = []
        for params in rabbi_params:
            x_bench = compute_lp_x_benchmark(params)
            ratio = np.mean(np.array(x_bench) >= 1)
            rabbi_ratios.append(ratio)
        plt.plot(k_list, rabbi_ratios, marker='o', label='RABBI LP x_benchmark >= 1(satisfy) ratio')

    if nplus1_params is not None:
        nplus1_ratios = []
        for params in nplus1_params:
            x_bench = compute_lp_x_benchmark(params)
            ratio = np.mean(np.array(x_bench) >= 1)
            nplus1_ratios.append(ratio)
        plt.plot(k_list, nplus1_ratios, marker='^', label='NPlusOneLP LP x_benchmark >= 1(satisfy) ratio')
    
    if topklp_params is not None:
        topklp_ratios = []
        for params in topklp_params:
            x_bench = compute_lp_x_benchmark(params)
            ratio = np.mean(np.array(x_bench) >= 1)
            topklp_ratios.append(ratio)
        plt.plot(k_list, topklp_ratios, marker='s', label='TopKLP LP x_benchmark >= 1(satisfy) ratio')
    
    plt.xlabel('k (scaling factor)')
    plt.ylabel('Proportion of x_benchmark >= 1')
    plt.title('Proportion of LP x_benchmark >= 1 vs k')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # 文件路径定义
    param_file = 'params2.yml'
    y_filename = os.path.join("data", 'Y_matrix_params2')
    shelve_path_rabbi = os.path.join("data", "shelve", "params_rabbi_params2.shelve")
    shelve_path_offline = os.path.join("data", "shelve", "params_offline_params2.shelve")
    shelve_path_nplusonelp = os.path.join("data", "shelve", "params_nplusonelp_params2.shelve")
    shelve_path_topklp = os.path.join("data", "shelve", "params_topklp_params2.shelve")
    save_path_ratio_results = os.path.join("data", "pics", "multi_k_ratio_results2.png")
    save_path_lp_benchmark = os.path.join("data", "pics", "lp_x_benchmark_ratio_vs_k2.png")

    print("\n===== 运行多倍率示例 =====")
    # 使用新的统一函数运行所有求解器
    solver_classes = [RABBI, OFFline, NPlusOneLP, TopKLP]
    results = run_multi_k(param_file, y_filename, solver_classes, max_concurrency=4)
    
    # 从结果字典中提取各个params_list
    rabbi_params = results['RABBI']
    offline_params = results['OFFline']
    nplus1_params = results['NPlusOneLP']
    topklp_params = results['TopKLP']

    # 保存params_list到shelve文件
    from main import save_params_list_to_shelve
    save_params_list_to_shelve(rabbi_params, shelve_path_rabbi)
    save_params_list_to_shelve(offline_params, shelve_path_offline)
    save_params_list_to_shelve(nplus1_params, shelve_path_nplusonelp)
    save_params_list_to_shelve(topklp_params, shelve_path_topklp)

    print("\n===== 正在绘制ratio result结果 =====")
    plot_multi_k_ratio_results(rabbi_params, offline_params, nplus1_params, topklp_params, save_path_ratio_results, show_plot=False)
    print("\n===== 正在绘制LP解基准比例 =====")
    plot_lp_x_benchmark_ratio_vs_k(rabbi_params, nplus1_params, topklp_params, save_path_lp_benchmark, show_plot=False)
