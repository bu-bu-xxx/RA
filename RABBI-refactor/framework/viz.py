"""Visualization adapter.
Accepts standardized results or raw params_list and delegates to existing plot functions.
"""
from typing import Dict, List, Optional
import os


class Visualizer:
    def __init__(self):
        pass

    def plot_multi_k_results(self, rabbi_params, offline_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
        # Delegate to existing plot.py function to keep behavior identical
        mod = __import__("plot", fromlist=["plot_multi_k_results"])
        fn = getattr(mod, "plot_multi_k_results")
        return fn(rabbi_params, offline_params, nplus1_params, topklp_params, save_path=save_path, show_plot=show_plot)

    def plot_multi_k_ratio_results(self, rabbi_params, offline_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
        mod = __import__("plot", fromlist=["plot_multi_k_ratio_results"])
        fn = getattr(mod, "plot_multi_k_ratio_results")
        return fn(rabbi_params, offline_params, nplus1_params, topklp_params, save_path=save_path, show_plot=show_plot)

    def plot_multi_k_regret(self, rabbi_params, offline_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
        # Some repositories have this in plot.py as imported by plot_offline_nplus1.py
        mod = __import__("plot", fromlist=["plot_multi_k_regret"])
        fn = getattr(mod, "plot_multi_k_regret")
        return fn(rabbi_params, offline_params, nplus1_params, topklp_params, save_path=save_path, show_plot=show_plot)

    def plot_lp_x_benchmark_ratio_vs_k(self, rabbi_params, nplus1_params, topklp_params, save_path=None, show_plot=False):
        mod = __import__("plot", fromlist=["plot_lp_x_benchmark_ratio_vs_k"])
        fn = getattr(mod, "plot_lp_x_benchmark_ratio_vs_k")
        return fn(rabbi_params, nplus1_params, topklp_params, save_path=save_path, show_plot=show_plot)

    def generate_plots(self, results: Dict[str, List[object]], plot_keys: List[str], save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        rabbi_params = results.get('RABBI')
        offline_params = results.get('OFFline')
        nplus1_params = results.get('NPlusOneLP')
        topklp_params = results.get('TopKLP')

        if 'multi_k_results' in plot_keys:
            self.plot_multi_k_results(rabbi_params, offline_params, nplus1_params, topklp_params,
                                      save_path=os.path.join(save_dir, 'multi_k_results.png'), show_plot=False)
        if 'multi_k_ratio' in plot_keys and offline_params is not None:
            self.plot_multi_k_ratio_results(rabbi_params, offline_params, nplus1_params, topklp_params,
                                            save_path=os.path.join(save_dir, 'multi_k_ratio_results.png'), show_plot=False)
        if 'multi_k_regret' in plot_keys and offline_params is not None:
            self.plot_multi_k_regret(rabbi_params, offline_params, nplus1_params, topklp_params,
                                     save_path=os.path.join(save_dir, 'multi_k_regret_results.png'), show_plot=False)
        if 'lp_x_benchmark_ratio' in plot_keys:
            self.plot_lp_x_benchmark_ratio_vs_k(rabbi_params, nplus1_params, topklp_params,
                                                save_path=os.path.join(save_dir, 'lp_x_benchmark_ratio_vs_k.png'), show_plot=False)
