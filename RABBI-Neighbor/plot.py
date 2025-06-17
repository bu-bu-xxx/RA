import os
import matplotlib.pyplot as plt
from main import run_rabbi_multi_k, run_offline_multi_k, run_nplusonelp_multi_k
import numpy as np

def plot_multi_k_results(rabbi_sims, offline_sims, nplus1_sims):
    # 提取k和reward
    rabbi_rewards = [sum(sim.reward_history) for sim in rabbi_sims]
    offline_rewards = [sum(sim.reward_history) for sim in offline_sims]
    nplus1_rewards = [sum(sim.reward_history) for sim in nplus1_sims]
    # 直接用sim.k
    k_list = rabbi_sims[0].k if hasattr(rabbi_sims[0], 'k') else list(range(len(rabbi_sims)))

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

def plot_multi_k_ratio_results(rabbi_sims, offline_sims, nplus1_sims, save_path=None, show_plot=False):
    rabbi_rewards = [sum(sim.reward_history) for sim in rabbi_sims]
    offline_rewards = [sum(sim.reward_history) for sim in offline_sims]
    nplus1_rewards = [sum(sim.reward_history) for sim in nplus1_sims]
    k_list = rabbi_sims[0].k if hasattr(rabbi_sims[0], 'k') else list(range(len(rabbi_sims)))
    rabbi_ratio = [r/o if o != 0 else 0 for r, o in zip(rabbi_rewards, offline_rewards)]
    nplus1_ratio = [n/o if o != 0 else 0 for n, o in zip(nplus1_rewards, offline_rewards)]

    plt.figure(figsize=(8,6))
    plt.plot(k_list, rabbi_ratio, marker='o', label='RABBI / OFFline')
    plt.plot(k_list, nplus1_ratio, marker='^', label='NPlusOneLP / OFFline')
    plt.xlabel('k (scaling factor)')
    plt.ylabel('Reward Ratio to OFFline')
    plt.title('Reward Ratio vs k')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_lp_x_benchmark_ratio_vs_k(rabbi_sims, nplus1_sims, save_path=None, show_plot=False):
    """
    输入rabbi_sims和nplus1_sims，分别计算每个sim的compute_lp_x_benchmark(sim)，
    统计每个输出结果中>=1的数所占比例，
    并画plot，x轴为sim.k，y轴为比例，两组曲线同图展示。
    """
    from main import compute_lp_x_benchmark
    k_list = rabbi_sims[0].k
    rabbi_ratios = []
    nplus1_ratios = []
    for sim in rabbi_sims:
        x_bench = compute_lp_x_benchmark(sim)
        ratio = np.mean(np.array(x_bench) >= 1)
        rabbi_ratios.append(ratio)
    for sim in nplus1_sims:
        x_bench = compute_lp_x_benchmark(sim)
        ratio = np.mean(np.array(x_bench) >= 1)
        nplus1_ratios.append(ratio)
    plt.figure(figsize=(8,6))
    plt.plot(k_list, rabbi_ratios, marker='o', label='RABBI LP x_benchmark >= 1(satisfy) ratio')
    plt.plot(k_list, nplus1_ratios, marker='^', label='NPlusOneLP LP x_benchmark >= 1(satisfy) ratio')
    plt.xlabel('k (scaling factor)')
    plt.ylabel('Proportion of x_benchmark >= 1')
    plt.title('Proportion of LP x_benchmark >= 1 vs k')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    param_file = 'params1.yml'
    y_filename = os.path.join("data", 'Y_matrix_params1')

    print("\n===== RABBI 多倍率示例 =====")
    rabbi_sims = run_rabbi_multi_k(param_file, y_filename)
    print("\n===== OFFline 多倍率示例 =====")
    offline_sims = run_offline_multi_k(param_file, y_filename)
    print("\n===== NPlusOneLP 多倍率示例 =====")
    nplus1_sims = run_nplusonelp_multi_k(param_file, y_filename)

    print("\n===== 绘制结果 =====")
    save_path = os.path.join("data", "multi_k_results.png")
    plot_multi_k_ratio_results(rabbi_sims, offline_sims, nplus1_sims, save_path, show_plot=True)
    print("\n===== 绘制LP解基准比例 =====")
    save_path = os.path.join("data", "lp_x_benchmark_ratio_vs_k.png")
    plot_lp_x_benchmark_ratio_vs_k(rabbi_sims, nplus1_sims, save_path, show_plot=True)
