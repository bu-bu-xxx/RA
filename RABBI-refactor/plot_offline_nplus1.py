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

    print("\n===== 运行多倍率示例 =====")
    # 使用 run_multi_k 函数运行求解器
    solver_classes = [OFFline, NPlusOneLP]
    
    print("正在运行求解器...")
    results = run_multi_k(param_file, y_filename, solver_classes, max_concurrency=None)
    
    # 从结果字典中提取各个params_list
    rabbi_params = None
    offline_params = results.get('OFFline', None)
    nplus1_params = results.get('NPlusOneLP', None)
    topklp_params = None

    print(f"OFFline结果: {len(offline_params) if offline_params else 0} 个参数组")
    print(f"NPlusOneLP结果: {len(nplus1_params) if nplus1_params else 0} 个参数组")

    # 保存params_list到shelve文件
    from main import save_params_list_to_shelve
    if offline_params:
        save_params_list_to_shelve(offline_params, shelve_path_offline)
        print(f"OFFline结果已保存到: {shelve_path_offline}")
    
    if nplus1_params:
        save_params_list_to_shelve(nplus1_params, shelve_path_nplusonelp)
        print(f"NPlusOneLP结果已保存到: {shelve_path_nplusonelp}")

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
