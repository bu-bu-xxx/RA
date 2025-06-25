import os
import matplotlib.pyplot as plt
from main import run_multi_k
from solver import RABBI, OFFline, NPlusOneLP, TopKLP
import numpy as np

def plot_multi_k_results(rabbi_params, offline_params, nplus1_params):
    # 提取k和reward
    rabbi_rewards = [sum(params.reward_history) for params in rabbi_params]
    offline_rewards = [sum(params.reward_history) for params in offline_params]
    nplus1_rewards = [sum(params.reward_history) for params in nplus1_params]
    # 直接用params.k
    k_list = rabbi_params[0].k if hasattr(rabbi_params[0], 'k') else list(range(len(rabbi_params)))

    plt.figure(figsize=(8,6))
    plt.plot(k_list, rabbi_rewards, marker='o', label='RABBI')
    plt.plot(k_list, offline_rewards, marker='s', label='OFFline')
    plt.plot(k_list, nplus1_rewards, marker='^', label='NPlusOneLP')
    plt.xlabel('k (scaling factor)')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs k for Different Policies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multi_k_ratio_results(rabbi_params, offline_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
    rabbi_rewards = [sum(params.reward_history) for params in rabbi_params]
    offline_rewards = [sum(params.reward_history) for params in offline_params]
    nplus1_rewards = [sum(params.reward_history) for params in nplus1_params]
    topklp_rewards = [sum(params.reward_history) for params in topklp_params]
    k_list = rabbi_params[0].k if hasattr(rabbi_params[0], 'k') else list(range(len(rabbi_params)))
    rabbi_ratio = [r/o if o != 0 else 0 for r, o in zip(rabbi_rewards, offline_rewards)]
    nplus1_ratio = [n/o if o != 0 else 0 for n, o in zip(nplus1_rewards, offline_rewards)]
    topklp_ratio = [t/o if o != 0 else 0 for t, o in zip(topklp_rewards, offline_rewards)]

    plt.figure(figsize=(8,6))
    plt.plot(k_list, rabbi_ratio, marker='o', label='RABBI / OFFline')
    plt.plot(k_list, nplus1_ratio, marker='^', label='NPlusOneLP / OFFline')
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
    k_list = rabbi_params[0].k
    rabbi_ratios = []
    nplus1_ratios = []
    topklp_ratios = []
    for params in rabbi_params:
        x_bench = compute_lp_x_benchmark(params)
        ratio = np.mean(np.array(x_bench) >= 1)
        rabbi_ratios.append(ratio)
    for params in nplus1_params:
        x_bench = compute_lp_x_benchmark(params)
        ratio = np.mean(np.array(x_bench) >= 1)
        nplus1_ratios.append(ratio)
    for params in topklp_params:
        x_bench = compute_lp_x_benchmark(params)
        ratio = np.mean(np.array(x_bench) >= 1)
        topklp_ratios.append(ratio)
    plt.figure(figsize=(8,6))
    plt.plot(k_list, rabbi_ratios, marker='o', label='RABBI LP x_benchmark >= 1(satisfy) ratio')
    plt.plot(k_list, nplus1_ratios, marker='^', label='NPlusOneLP LP x_benchmark >= 1(satisfy) ratio')
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
    param_file = 'params.yml'
    y_filename = os.path.join("data", 'Y_matrix_params')
    shelve_path_rabbi = os.path.join("data", "shelve", "params_rabbi_params.shelve")
    shelve_path_offline = os.path.join("data", "shelve", "params_offline_params.shelve")
    shelve_path_nplusonelp = os.path.join("data", "shelve", "params_nplusonelp_params.shelve")
    shelve_path_topklp = os.path.join("data", "shelve", "params_topklp_params.shelve")
    save_path_results = os.path.join("data", "pics", "multi_k_results.png")
    save_path_lp_benchmark = os.path.join("data", "pics", "lp_x_benchmark_ratio_vs_k.png")

    print("\n===== 运行多倍率示例 =====")
    # 使用新的统一函数运行所有求解器
    solver_classes = [RABBI, OFFline, NPlusOneLP, TopKLP]
    results = run_multi_k(param_file, y_filename, solver_classes)
    
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

    print("\n===== 绘制结果 =====")
    plot_multi_k_ratio_results(rabbi_params, offline_params, nplus1_params, topklp_params, save_path_results, show_plot=False)
    print("\n===== 绘制LP解基准比例 =====")
    plot_lp_x_benchmark_ratio_vs_k(rabbi_params, nplus1_params, topklp_params, save_path_lp_benchmark, show_plot=False)
