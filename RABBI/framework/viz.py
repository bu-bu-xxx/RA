"""Visualization adapter.
Accepts standardized results or raw params_list and renders plots directly.
"""
from typing import Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt
from .results import compute_lp_x_benchmark


class Visualizer:
    def __init__(self):
        pass

    def plot_multi_k_results(self, rabbi_params, offline_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
        k_list = None
        for params in [rabbi_params, offline_params, nplus1_params, topklp_params]:
            if params is not None and len(params) > 0:
                k_list = params[0].k if hasattr(params[0], 'k') else list(range(len(params)))
                break
        if k_list is None:
            print("Warning: No valid params provided for plotting")
            return

        plt.figure(figsize=(8, 6))
        if rabbi_params is not None:
            rabbi_rewards = [sum(p.reward_history) for p in rabbi_params]
            plt.plot(k_list, rabbi_rewards, marker='o', label='RABBI')
        if offline_params is not None:
            offline_rewards = [sum(p.reward_history) for p in offline_params]
            plt.plot(k_list, offline_rewards, marker='s', label='OFFline')
        if nplus1_params is not None:
            nplus1_rewards = [sum(p.reward_history) for p in nplus1_params]
            plt.plot(k_list, nplus1_rewards, marker='^', label='NPlusOneLP')
        if topklp_params is not None:
            topklp_rewards = [sum(p.reward_history) for p in topklp_params]
            plt.plot(k_list, topklp_rewards, marker='d', label='TopKLP')

        plt.xlabel('k (scaling factor)')
        plt.ylabel('Total Reward')
        plt.title('Total Reward vs k for Different Policies')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_multi_k_ratio_results(self, rabbi_params, offline_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
        k_list = None
        for params in [rabbi_params, offline_params, nplus1_params, topklp_params]:
            if params is not None and len(params) > 0:
                k_list = params[0].k if hasattr(params[0], 'k') else list(range(len(params)))
                break
        if k_list is None:
            raise ValueError("No valid params provided for plotting")
        if offline_params is None:
            raise ValueError("offline_params is None, cannot compute ratios")

        offline_rewards = [sum(p.reward_history) for p in offline_params]
        plt.figure(figsize=(8, 6))
        if rabbi_params is not None:
            rabbi_rewards = [sum(p.reward_history) for p in rabbi_params]
            rabbi_ratio = [r / o if o != 0 else 0 for r, o in zip(rabbi_rewards, offline_rewards)]
            plt.plot(k_list, rabbi_ratio, marker='o', label='RABBI / OFFline')
        if nplus1_params is not None:
            nplus1_rewards = [sum(p.reward_history) for p in nplus1_params]
            nplus1_ratio = [n / o if o != 0 else 0 for n, o in zip(nplus1_rewards, offline_rewards)]
            plt.plot(k_list, nplus1_ratio, marker='^', label='NPlusOneLP / OFFline')
        if topklp_params is not None:
            topklp_rewards = [sum(p.reward_history) for p in topklp_params]
            topklp_ratio = [t / o if o != 0 else 0 for t, o in zip(topklp_rewards, offline_rewards)]
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

    def plot_multi_k_regret(self, rabbi_params, offline_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
        k_list = None
        for params in [rabbi_params, offline_params, nplus1_params, topklp_params]:
            if params is not None and len(params) > 0:
                k_list = params[0].k if hasattr(params[0], 'k') else list(range(len(params)))
                break
        if k_list is None:
            raise ValueError("No valid params provided for plotting")
        if offline_params is None:
            raise ValueError("offline_params is None, cannot compute regret")

        offline_rewards = [sum(p.reward_history) for p in offline_params]
        plt.figure(figsize=(8, 6))
        if rabbi_params is not None:
            rabbi_rewards = [sum(p.reward_history) for p in rabbi_params]
            rabbi_regret = [o - r for r, o in zip(rabbi_rewards, offline_rewards)]
            plt.plot(k_list, rabbi_regret, marker='o', label='OFFline - RABBI')
        if nplus1_params is not None:
            nplus1_rewards = [sum(p.reward_history) for p in nplus1_params]
            nplus1_regret = [o - n for n, o in zip(nplus1_rewards, offline_rewards)]
            plt.plot(k_list, nplus1_regret, marker='^', label='OFFline - NPlusOneLP')
        if topklp_params is not None:
            topklp_rewards = [sum(p.reward_history) for p in topklp_params]
            topklp_regret = [o - t for t, o in zip(topklp_rewards, offline_rewards)]
            plt.plot(k_list, topklp_regret, marker='s', label='OFFline - TopKLP')

        plt.xlabel('k (scaling factor)')
        plt.ylabel('Regret (OFFline - Other)')
        plt.title('Regret vs k')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def plot_lp_x_benchmark_ratio_vs_k(self, rabbi_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
        k_list = None
        for params in [rabbi_params, nplus1_params, topklp_params]:
            if params is not None and len(params) > 0:
                k_list = params[0].k if hasattr(params[0], 'k') else list(range(len(params)))
                break
        if k_list is None:
            print("Warning: No valid params provided for plotting")
            return

        plt.figure(figsize=(8, 6))
        if rabbi_params is not None:
            rabbi_ratios = []
            for p in rabbi_params:
                x_bench = compute_lp_x_benchmark(p)
                ratio = np.mean(np.array(x_bench) >= 1)
                rabbi_ratios.append(ratio)
            plt.plot(k_list, rabbi_ratios, marker='o', label='RABBI LP x_benchmark >= 1(satisfy) ratio')
        if nplus1_params is not None:
            nplus1_ratios = []
            for p in nplus1_params:
                x_bench = compute_lp_x_benchmark(p)
                ratio = np.mean(np.array(x_bench) >= 1)
                nplus1_ratios.append(ratio)
            plt.plot(k_list, nplus1_ratios, marker='^', label='NPlusOneLP LP x_benchmark >= 1(satisfy) ratio')
        if topklp_params is not None:
            topklp_ratios = []
            for p in topklp_params:
                x_bench = compute_lp_x_benchmark(p)
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

    def generate_plots(self, results: Dict[str, List[object]], plot_keys: List[str], save_dir: str, file_prefix: str = ""):
        os.makedirs(save_dir, exist_ok=True)
        rabbi_params = results.get('RABBI')
        offline_params = results.get('OFFline')
        nplus1_params = results.get('NPlusOneLP')
        topklp_params = results.get('TopKLP')

        def out(name: str) -> str:
            return os.path.join(save_dir, f"{file_prefix}{name}") if file_prefix else os.path.join(save_dir, name)

        if 'multi_k_results' in plot_keys:
            self.plot_multi_k_results(rabbi_params, offline_params, nplus1_params, topklp_params,
                                      save_path=out('multi_k_results.png'), show_plot=False)
        if 'multi_k_ratio' in plot_keys and offline_params is not None:
            self.plot_multi_k_ratio_results(rabbi_params, offline_params, nplus1_params, topklp_params,
                                            save_path=out('multi_k_ratio_results.png'), show_plot=False)
        if 'multi_k_regret' in plot_keys and offline_params is not None:
            self.plot_multi_k_regret(rabbi_params, offline_params, nplus1_params, topklp_params,
                                     save_path=out('multi_k_regret_results.png'), show_plot=False)
        if 'lp_x_benchmark_ratio' in plot_keys:
            self.plot_lp_x_benchmark_ratio_vs_k(rabbi_params, nplus1_params, topklp_params,
                                                save_path=out('lp_x_benchmark_ratio_vs_k.png'), show_plot=False)
