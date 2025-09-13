"""Visualization adapter.
Accepts standardized results or raw params_list and delegates to existing plot functions.
"""
from typing import Dict, List, Optional


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
