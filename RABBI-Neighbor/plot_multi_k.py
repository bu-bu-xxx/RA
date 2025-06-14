import os
import matplotlib.pyplot as plt
from main import run_rabbi_multi_k, run_offline_multi_k, run_nplusonelp_multi_k

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

def plot_multi_k_ratio_results(rabbi_sims, offline_sims, nplus1_sims):
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
    plt.show()

if __name__ == "__main__":
    param_file = 'params.yml'
    y_filename = os.path.join("data", 'Y_matrix_debug')

    print("\n===== RABBI 多倍率示例 =====")
    rabbi_sims = run_rabbi_multi_k(param_file, y_filename)
    print("\n===== OFFline 多倍率示例 =====")
    offline_sims = run_offline_multi_k(param_file, y_filename)
    print("\n===== NPlusOneLP 多倍率示例 =====")
    nplus1_sims = run_nplusonelp_multi_k(param_file, y_filename)

    # plot_multi_k_results(rabbi_sims, offline_sims, nplus1_sims)
    plot_multi_k_ratio_results(rabbi_sims, offline_sims, nplus1_sims)
