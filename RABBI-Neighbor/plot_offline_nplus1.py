import os
import matplotlib.pyplot as plt
from main import run_multi_k
from solver import RABBI, OFFline, NPlusOneLP, TopKLP
import numpy as np
from plot import (
    plot_multi_k_ratio_results,
    plot_multi_k_regret,
    plot_multi_k_results,
    plot_lp_x_benchmark_ratio_vs_k
)

if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(os.path.join("data", "Y"), exist_ok=True)
    os.makedirs(os.path.join("data", "shelve"), exist_ok=True)
    os.makedirs(os.path.join("data", "pics"), exist_ok=True)

    # 文件路径定义
    param_file = 'params5.yml'
    y_filename = os.path.join("data", "Y", 'Y_matrix_params5')
    shelve_path_rabbi = os.path.join("data", "shelve", "params_rabbi_params5.shelve")
    shelve_path_offline = os.path.join("data", "shelve", "params_offline_params5.shelve")
    shelve_path_nplusonelp = os.path.join("data", "shelve", "params_nplusonelp_params5.shelve")
    shelve_path_topklp = os.path.join("data", "shelve", "params_topklp_params5.shelve")
    save_path_ratio_results = os.path.join("data", "pics", "multi_k_ratio_results5.png")
    save_path_lp_benchmark = os.path.join("data", "pics", "lp_x_benchmark_ratio_vs_k5.png")
    save_path_regret_results = os.path.join("data", "pics", "multi_k_regret_results5.png")
    save_path_multi_k_results = os.path.join("data", "pics", "multi_k_results5.png")

    print("\n===== 运行多倍率示例 (带缓存) =====")
    # 使用带缓存的智能函数运行所有求解器
    from main import run_multi_k_with_cache
    solver_classes = [OFFline, NPlusOneLP]
    results = run_multi_k_with_cache(
        param_file, y_filename, solver_classes, max_concurrency=None,
        shelve_path_rabbi=shelve_path_rabbi,
        shelve_path_offline=shelve_path_offline,
        shelve_path_nplusonelp=shelve_path_nplusonelp,
        shelve_path_topklp=shelve_path_topklp
    )

    # 从结果字典中提取各个params_list
    rabbi_params = None
    offline_params = results['OFFline']
    nplus1_params = results['NPlusOneLP']
    topklp_params = None

    # 检查结果并过滤掉None值
    if offline_params is not None:
        offline_params = [p for p in offline_params if p is not None]
        if len(offline_params) == 0:
            offline_params = None
    
    if nplus1_params is not None:
        nplus1_params = [p for p in nplus1_params if p is not None]
        if len(nplus1_params) == 0:
            nplus1_params = None

    # 不需要再单独保存，因为已经在运行过程中实时保存了
    print("数据已在运行过程中自动保存到shelve文件")

    # 检查是否有有效数据进行绘图
    if offline_params is None:
        print("警告: offline_params 为空，无法绘制需要baseline的图表")
    elif nplus1_params is None:
        print("警告: nplus1_params 为空，无法绘制比较图表")
    else:
        print("\n===== 正在绘制ratio result结果 =====")
        plot_multi_k_ratio_results(rabbi_params, offline_params, nplus1_params, topklp_params, save_path_ratio_results, show_plot=False)
        print("\n===== 正在绘制regret结果 =====")
        plot_multi_k_regret(rabbi_params, offline_params, nplus1_params, topklp_params, save_path_regret_results, show_plot=False)
        print("\n===== 正在绘制多算法总奖励结果 =====")
        plot_multi_k_results(rabbi_params, offline_params, nplus1_params, topklp_params, save_path_multi_k_results, show_plot=False)
        print("\n===== 正在绘制LP解基准比例 =====")
        plot_lp_x_benchmark_ratio_vs_k(rabbi_params, nplus1_params, topklp_params, save_path_lp_benchmark, show_plot=False)
