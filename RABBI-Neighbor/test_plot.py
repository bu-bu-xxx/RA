import os
import matplotlib.pyplot as plt
from main import run_multi_k
from solver import RABBI, OFFline, NPlusOneLP, TopKLP
import numpy as np
from plot import plot_multi_k_ratio_results, plot_lp_x_benchmark_ratio_vs_k

def test_plot_functions():
    """
    测试绘图功能，使用 params.yml 配置文件
    """
    print("===== 开始测试绘图功能 =====")
    
    # 文件路径定义 - 使用 params.yml 进行测试
    param_file = 'params.yml'
    y_filename = os.path.join("data", 'Y_matrix_debug')
    shelve_path_rabbi = os.path.join("data", "shelve", "debug_params_rabbi.shelve")
    shelve_path_offline = os.path.join("data", "shelve", "debug_params_offline.shelve")
    shelve_path_nplusonelp = os.path.join("data", "shelve", "debug_params_nplusonelp.shelve")
    shelve_path_topklp = os.path.join("data", "shelve", "debug_params_topklp.shelve")
    save_path_ratio_results = os.path.join("data", "pics", "debug_multi_k_ratio_results.png")
    save_path_lp_benchmark = os.path.join("data", "pics", "debug_lp_x_benchmark_ratio_vs_k.png")

    # 确保输出目录存在
    os.makedirs(os.path.join("data", "shelve"), exist_ok=True)
    os.makedirs(os.path.join("data", "pics"), exist_ok=True)

    print(f"使用配置文件: {param_file}")
    print(f"Y矩阵文件前缀: {y_filename}")
    
    try:
        print("\n===== 运行多倍率测试 =====")
        # 使用新的统一函数运行所有求解器
        solver_classes = [RABBI, OFFline, NPlusOneLP, TopKLP]  # 包含所有求解器
        
        print("正在运行求解器...")
        results = run_multi_k(param_file, y_filename, solver_classes, max_concurrency=2)
        
        # 从结果字典中提取各个params_list
        rabbi_params = results.get('RABBI', None)
        offline_params = results.get('OFFline', None)
        nplus1_params = results.get('NPlusOneLP', None)
        topklp_params = results.get('TopKLP', None)
        
        print(f"RABBI结果: {len(rabbi_params) if rabbi_params else 0} 个参数组")
        print(f"OFFline结果: {len(offline_params) if offline_params else 0} 个参数组")
        print(f"NPlusOneLP结果: {len(nplus1_params) if nplus1_params else 0} 个参数组")
        print(f"TopKLP结果: {len(topklp_params) if topklp_params else 0} 个参数组")

        # 保存params_list到shelve文件
        from main import save_params_list_to_shelve
        if rabbi_params:
            save_params_list_to_shelve(rabbi_params, shelve_path_rabbi)
            print(f"RABBI结果已保存到: {shelve_path_rabbi}")
        
        if offline_params:
            save_params_list_to_shelve(offline_params, shelve_path_offline)
            print(f"OFFline结果已保存到: {shelve_path_offline}")
        
        if nplus1_params:
            save_params_list_to_shelve(nplus1_params, shelve_path_nplusonelp)
            print(f"NPlusOneLP结果已保存到: {shelve_path_nplusonelp}")
        
        if topklp_params:
            save_params_list_to_shelve(topklp_params, shelve_path_topklp)
            print(f"TopKLP结果已保存到: {shelve_path_topklp}")

        print("\n===== 测试绘制ratio result结果 =====")
        try:
            plot_multi_k_ratio_results(
                rabbi_params, offline_params, nplus1_params, topklp_params, 
                save_path_ratio_results, show_plot=False
            )
            print(f"Ratio结果图已保存到: {save_path_ratio_results}")
        except Exception as e:
            print(f"绘制ratio结果时出错: {e}")

        print("\n===== 测试绘制LP解基准比例 =====")
        try:
            plot_lp_x_benchmark_ratio_vs_k(
                rabbi_params, nplus1_params, topklp_params, 
                save_path_lp_benchmark, show_plot=False
            )
            print(f"LP基准比例图已保存到: {save_path_lp_benchmark}")
        except Exception as e:
            print(f"绘制LP基准比例时出错: {e}")

        print("\n===== 显示部分结果统计 =====")
        if rabbi_params and len(rabbi_params) > 0:
            print(f"RABBI - k值范围: {[p.k for p in rabbi_params]}")
            print(f"RABBI - 总奖励范围: {[sum(p.reward_history) for p in rabbi_params]}")
        
        if offline_params and len(offline_params) > 0:
            print(f"OFFline - k值范围: {[p.k for p in offline_params]}")
            print(f"OFFline - 总奖励范围: {[sum(p.reward_history) for p in offline_params]}")
        
        if nplus1_params and len(nplus1_params) > 0:
            print(f"NPlusOneLP - k值范围: {[p.k for p in nplus1_params]}")
            print(f"NPlusOneLP - 总奖励范围: {[sum(p.reward_history) for p in nplus1_params]}")
        
        if topklp_params and len(topklp_params) > 0:
            print(f"TopKLP - k值范围: {[p.k for p in topklp_params]}")
            print(f"TopKLP - 总奖励范围: {[sum(p.reward_history) for p in topklp_params]}")

        print("\n===== 测试完成 =====")
        print("所有绘图功能测试完成!")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 主测试函数
    test_plot_functions()
    
    print("\n===== 所有测试完成 =====")
