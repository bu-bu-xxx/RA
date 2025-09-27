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

    @staticmethod
    def _extract_reward_entries(params_list: List[object]) -> List[tuple]:
        entries: List[tuple] = []
        if not params_list:
            return entries
        k_values = None
        for p in params_list:
            if p is not None and hasattr(p, "k"):
                k_values = list(p.k)
                break
        if k_values is None:
            k_values = list(range(len(params_list)))
        for idx, (k_val, params) in enumerate(zip(k_values, params_list)):
            if params is None:
                continue
            rewards = getattr(params, "reward_history", None)
            if rewards is None:
                continue
            entries.append((idx, float(k_val), float(sum(rewards))))
        return entries

    @staticmethod
    def _prepare_series(entries: List[tuple]) -> tuple:
        if not entries:
            return [], []
        _, ks, rewards = zip(*entries)
        return list(ks), list(rewards)

    def plot_multi_k_results(self, rabbi_params, offline_params, nplus1_params, topklp_params, robust_params=None, save_path=None, show_plot=False):
        series = [
            ("RABBI", rabbi_params, "o"),
            ("OFFline", offline_params, "s"),
            ("NPlusOneLP", nplus1_params, "^"),
            ("TopKLP", topklp_params, "d"),
            ("Robust", robust_params, "x"),
        ]

        plt.figure(figsize=(8, 6))
        for label, params, marker in series:
            if params is None:
                continue
            entries = self._extract_reward_entries(params)
            ks, rewards = self._prepare_series(entries)
            if not ks:
                continue
            plt.plot(ks, rewards, marker=marker, label=label)

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

    def plot_multi_k_ratio_results(self, rabbi_params, offline_params, nplus1_params, topklp_params, robust_params=None, save_path=None, show_plot=False):
        series = [
            ("RABBI", rabbi_params, "o"),
            ("NPlusOneLP", nplus1_params, "^"),
            ("TopKLP", topklp_params, "s"),
            ("Robust", robust_params, "x"),
        ]

        if offline_params is None:
            raise ValueError("offline_params is None, cannot compute ratios")

        offline_entries = self._extract_reward_entries(offline_params)
        if not offline_entries:
            raise ValueError("offline_params does not contain valid reward history")
        offline_map = {idx: (k, reward) for idx, k, reward in offline_entries}

        plt.figure(figsize=(8, 6))
        for label, params, marker in series:
            if params is None:
                continue
            entries = self._extract_reward_entries(params)
            ratios = []
            ks = []
            for idx, _, reward in entries:
                if idx not in offline_map:
                    continue
                k_val, offline_reward = offline_map[idx]
                if offline_reward == 0:
                    continue
                ks.append(k_val)
                ratios.append(reward / offline_reward)
            if not ks:
                continue
            plt.plot(ks, ratios, marker=marker, label=f"{label} / OFFline")

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

    def plot_multi_k_regret(self, rabbi_params, offline_params, nplus1_params, topklp_params, robust_params=None, save_path=None, show_plot=False):
        series = [
            ("RABBI", rabbi_params, "o"),
            ("NPlusOneLP", nplus1_params, "^"),
            ("TopKLP", topklp_params, "s"),
            ("Robust", robust_params, "x"),
        ]

        if offline_params is None:
            raise ValueError("offline_params is None, cannot compute regret")

        offline_entries = self._extract_reward_entries(offline_params)
        if not offline_entries:
            raise ValueError("offline_params does not contain valid reward history")
        offline_map = {idx: (k, reward) for idx, k, reward in offline_entries}

        plt.figure(figsize=(8, 6))
        for label, params, marker in series:
            if params is None:
                continue
            entries = self._extract_reward_entries(params)
            regrets = []
            ks = []
            for idx, _, reward in entries:
                if idx not in offline_map:
                    continue
                k_val, offline_reward = offline_map[idx]
                ks.append(k_val)
                regrets.append(offline_reward - reward)
            if not ks:
                continue
            plt.plot(ks, regrets, marker=marker, label=f"OFFline - {label}")

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

    def plot_lp_x_benchmark_ratio_vs_k(self, rabbi_params, nplus1_params, topklp_params, robust_params=None, save_path=None, show_plot=False):
        k_list = None
        for params in [rabbi_params, nplus1_params, topklp_params, robust_params]:
            if params is not None and len(params) > 0:
                entry = next((p for p in params if p is not None), None)
                if entry is not None and hasattr(entry, 'k'):
                    k_list = entry.k
                else:
                    k_list = list(range(len(params)))
                break
        if k_list is None:
            print("Warning: No valid params provided for plotting")
            return

        plt.figure(figsize=(8, 6))
        if rabbi_params is not None:
            rabbi_ratios = []
            for p in rabbi_params:
                if p is None:
                    continue
                x_bench = compute_lp_x_benchmark(p)
                ratio = np.mean(np.array(x_bench) >= 1)
                rabbi_ratios.append(ratio)
            plt.plot(k_list, rabbi_ratios, marker='o', label='RABBI LP x_benchmark >= 1(satisfy) ratio')
        if nplus1_params is not None:
            nplus1_ratios = []
            for p in nplus1_params:
                if p is None:
                    continue
                x_bench = compute_lp_x_benchmark(p)
                ratio = np.mean(np.array(x_bench) >= 1)
                nplus1_ratios.append(ratio)
            plt.plot(k_list, nplus1_ratios, marker='^', label='NPlusOneLP LP x_benchmark >= 1(satisfy) ratio')
        if topklp_params is not None:
            topklp_ratios = []
            for p in topklp_params:
                if p is None:
                    continue
                x_bench = compute_lp_x_benchmark(p)
                ratio = np.mean(np.array(x_bench) >= 1)
                topklp_ratios.append(ratio)
            plt.plot(k_list, topklp_ratios, marker='s', label='TopKLP LP x_benchmark >= 1(satisfy) ratio')
        if robust_params is not None:
            robust_ratios = []
            for p in robust_params:
                if p is None:
                    continue
                x_bench = compute_lp_x_benchmark(p)
                ratio = np.mean(np.array(x_bench) >= 1)
                robust_ratios.append(ratio)
            plt.plot(k_list, robust_ratios, marker='x', label='Robust LP x_benchmark >= 1(satisfy) ratio')

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
        robust_params = results.get('Robust')

        def out(name: str) -> str:
            return os.path.join(save_dir, f"{file_prefix}{name}") if file_prefix else os.path.join(save_dir, name)

        if 'multi_k_results' in plot_keys:
            self.plot_multi_k_results(rabbi_params, offline_params, nplus1_params, topklp_params, robust_params,
                                      save_path=out('multi_k_results.png'), show_plot=False)
        if 'multi_k_ratio' in plot_keys and offline_params is not None:
            self.plot_multi_k_ratio_results(rabbi_params, offline_params, nplus1_params, topklp_params, robust_params,
                                            save_path=out('multi_k_ratio_results.png'), show_plot=False)
        if 'multi_k_regret' in plot_keys and offline_params is not None:
            self.plot_multi_k_regret(rabbi_params, offline_params, nplus1_params, topklp_params, robust_params,
                                     save_path=out('multi_k_regret_results.png'), show_plot=False)
        if 'lp_x_benchmark_ratio' in plot_keys:
            self.plot_lp_x_benchmark_ratio_vs_k(rabbi_params, nplus1_params, topklp_params, robust_params,
                                                save_path=out('lp_x_benchmark_ratio_vs_k.png'), show_plot=False)
