import os
import matplotlib.pyplot as plt
from main import run_multi_k, run_multi_k_with_cache
from solver import RABBI, OFFline, NPlusOneLP, TopKLP
import numpy as np
from plot import plot_multi_k_ratio_results, plot_lp_x_benchmark_ratio_vs_k, plot_multi_k_regret, plot_multi_k_results

def test_plot_functions(param_file='params.yml', max_concurrency=2, show_plots=False):
    """
    测试绘图功能，使用指定的配置文件
    
    参数:
        param_file: 参数配置文件路径
        max_concurrency: 最大并发数
        show_plots: 是否显示图形界面
    """
    print("===== 开始测试绘图功能 =====")
    
    # 文件路径定义
    y_filename = os.path.join("data", "Y", 'Y_matrix_debug')
    shelve_path_rabbi = os.path.join("data", "shelve", "debug_params_rabbi.shelve")
    shelve_path_offline = os.path.join("data", "shelve", "debug_params_offline.shelve")
    shelve_path_nplusonelp = os.path.join("data", "shelve", "debug_params_nplusonelp.shelve")
    shelve_path_topklp = os.path.join("data", "shelve", "debug_params_topklp.shelve")
    save_path_ratio_results = os.path.join("data", "pics", "debug_multi_k_ratio_results.png")
    save_path_lp_benchmark = os.path.join("data", "pics", "debug_lp_x_benchmark_ratio_vs_k.png")
    save_path_regret_results = os.path.join("data", "pics", "debug_multi_k_regret_results.png")
    save_path_multi_k_results = os.path.join("data", "pics", "debug_multi_k_results.png")

    # 确保输出目录存在
    os.makedirs(os.path.join("data", "Y"), exist_ok=True)
    os.makedirs(os.path.join("data", "shelve"), exist_ok=True)
    os.makedirs(os.path.join("data", "pics"), exist_ok=True)
    
    print(f"使用配置文件: {param_file}")
    print(f"Y矩阵文件前缀: {y_filename}")
    print(f"最大并发数: {max_concurrency}")
    
    try:
        print("\n===== 运行多倍率测试 =====")
        # 使用新的统一函数运行所有求解器
        solver_classes = [RABBI, OFFline, NPlusOneLP, TopKLP]  # 包含所有求解器
        
        print("正在运行求解器...")
        results = run_multi_k(param_file, y_filename, solver_classes, max_concurrency=max_concurrency)
        
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
                save_path_ratio_results, show_plot=show_plots
            )
            print(f"Ratio结果图已保存到: {save_path_ratio_results}")
        except Exception as e:
            print(f"绘制ratio结果时出错: {e}")

        print("\n===== 测试绘制regret结果 =====")
        try:
            plot_multi_k_regret(
                rabbi_params, offline_params, nplus1_params, topklp_params, 
                save_path_regret_results, show_plot=show_plots
            )
            print(f"Regret结果图已保存到: {save_path_regret_results}")
        except Exception as e:
            print(f"绘制regret结果时出错: {e}")

        print("\n===== 测试绘制多算法总奖励结果 =====")
        try:
            plot_multi_k_results(
                rabbi_params, offline_params, nplus1_params, topklp_params,
                save_path=save_path_multi_k_results, show_plot=show_plots
            )
            print(f"多算法总奖励图已保存到: {save_path_multi_k_results}")
        except Exception as e:
            print(f"绘制多算法总奖励结果时出错: {e}")

        print("\n===== 测试绘制LP解基准比例 =====")
        try:
            plot_lp_x_benchmark_ratio_vs_k(
                rabbi_params, nplus1_params, topklp_params, 
                save_path_lp_benchmark, show_plot=show_plots
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

def test_plot_functions_with_cache(param_file='params.yml', max_concurrency=2, show_plots=False):
    """
    测试绘图功能，使用 params.yml 配置文件和 run_multi_k_with_cache
    """
    print("===== 开始测试绘图功能 (带缓存) =====")
    
    # 文件路径定义 - 使用 params.yml 进行测试，和test_plot_functions相同的路径
    param_file = 'params.yml'
    y_filename = os.path.join("data", "Y", 'Y_matrix_debug')
    shelve_path_rabbi = os.path.join("data", "shelve", "debug_params_rabbi.shelve")
    shelve_path_offline = os.path.join("data", "shelve", "debug_params_offline.shelve")
    shelve_path_nplusonelp = os.path.join("data", "shelve", "debug_params_nplusonelp.shelve")
    shelve_path_topklp = os.path.join("data", "shelve", "debug_params_topklp.shelve")
    save_path_ratio_results = os.path.join("data", "pics", "debug_multi_k_ratio_results.png")
    save_path_lp_benchmark = os.path.join("data", "pics", "debug_lp_x_benchmark_ratio_vs_k.png")
    save_path_regret_results = os.path.join("data", "pics", "debug_multi_k_regret_results.png")
    save_path_multi_k_results = os.path.join("data", "pics", "debug_multi_k_results.png")

    # 确保输出目录存在
    os.makedirs(os.path.join("data", "Y"), exist_ok=True)
    os.makedirs(os.path.join("data", "shelve"), exist_ok=True)
    os.makedirs(os.path.join("data", "pics"), exist_ok=True)

    print(f"使用配置文件: {param_file}")
    print(f"Y矩阵文件前缀: {y_filename}")
    
    try:
        print("\n===== 运行多倍率测试 (带缓存) =====")
        # 使用带缓存的智能函数运行所有求解器
        from main import run_multi_k_with_cache
        solver_classes = [RABBI, OFFline, NPlusOneLP, TopKLP]  # 包含所有求解器
        
        print("正在运行求解器 (带缓存功能)...")
        results = run_multi_k_with_cache(
            param_file, y_filename, solver_classes, max_concurrency=max_concurrency,
            shelve_path_rabbi=shelve_path_rabbi,
            shelve_path_offline=shelve_path_offline,
            shelve_path_nplusonelp=shelve_path_nplusonelp,
            shelve_path_topklp=shelve_path_topklp
        )
        
        # 从结果字典中提取各个params_list
        rabbi_params = results.get('RABBI', None)
        offline_params = results.get('OFFline', None)
        nplus1_params = results.get('NPlusOneLP', None)
        topklp_params = results.get('TopKLP', None)
        
        print(f"RABBI结果: {len(rabbi_params) if rabbi_params else 0} 个参数组")
        print(f"OFFline结果: {len(offline_params) if offline_params else 0} 个参数组")
        print(f"NPlusOneLP结果: {len(nplus1_params) if nplus1_params else 0} 个参数组")
        print(f"TopKLP结果: {len(topklp_params) if topklp_params else 0} 个参数组")

        # 不需要额外保存，因为run_multi_k_with_cache已经自动保存到shelve文件
        print("数据已在运行过程中自动保存到shelve文件")

        print("\n===== 测试绘制ratio result结果 =====")
        try:
            plot_multi_k_ratio_results(
                rabbi_params, offline_params, nplus1_params, topklp_params, 
                save_path_ratio_results, show_plot=show_plots
            )
            print(f"Ratio结果图已保存到: {save_path_ratio_results}")
        except Exception as e:
            print(f"绘制ratio结果时出错: {e}")

        print("\n===== 测试绘制regret结果 =====")
        try:
            plot_multi_k_regret(
                rabbi_params, offline_params, nplus1_params, topklp_params, 
                save_path_regret_results, show_plot=show_plots
            )
            print(f"Regret结果图已保存到: {save_path_regret_results}")
        except Exception as e:
            print(f"绘制regret结果时出错: {e}")

        print("\n===== 测试绘制多算法总奖励结果 =====")
        try:
            plot_multi_k_results(
                rabbi_params, offline_params, nplus1_params, topklp_params,
                save_path=save_path_multi_k_results, show_plot=show_plots
            )
            print(f"多算法总奖励图已保存到: {save_path_multi_k_results}")
        except Exception as e:
            print(f"绘制多算法总奖励结果时出错: {e}")

        print("\n===== 测试绘制LP解基准比例 =====")
        try:
            plot_lp_x_benchmark_ratio_vs_k(
                rabbi_params, nplus1_params, topklp_params, 
                save_path_lp_benchmark, show_plot=show_plots
            )
            print(f"LP基准比例图已保存到: {save_path_lp_benchmark}")
        except Exception as e:
            print(f"绘制LP基准比例时出错: {e}")

        print("\n===== 显示部分结果统计 =====")
        if rabbi_params and len(rabbi_params) > 0:
            # 过滤掉None值
            valid_rabbi_params = [p for p in rabbi_params if p is not None]
            if valid_rabbi_params:
                print(f"RABBI - k值范围: {[p.k for p in valid_rabbi_params]}")
                print(f"RABBI - 总奖励范围: {[sum(p.reward_history) for p in valid_rabbi_params]}")
        
        if offline_params and len(offline_params) > 0:
            valid_offline_params = [p for p in offline_params if p is not None]
            if valid_offline_params:
                print(f"OFFline - k值范围: {[p.k for p in valid_offline_params]}")
                print(f"OFFline - 总奖励范围: {[sum(p.reward_history) for p in valid_offline_params]}")
        
        if nplus1_params and len(nplus1_params) > 0:
            valid_nplus1_params = [p for p in nplus1_params if p is not None]
            if valid_nplus1_params:
                print(f"NPlusOneLP - k值范围: {[p.k for p in valid_nplus1_params]}")
                print(f"NPlusOneLP - 总奖励范围: {[sum(p.reward_history) for p in valid_nplus1_params]}")
        
        if topklp_params and len(topklp_params) > 0:
            valid_topklp_params = [p for p in topklp_params if p is not None]
            if valid_topklp_params:
                print(f"TopKLP - k值范围: {[p.k for p in valid_topklp_params]}")
                print(f"TopKLP - 总奖励范围: {[sum(p.reward_history) for p in valid_topklp_params]}")

        print("\n===== 缓存测试完成 =====")
        print("所有绘图功能测试完成 (带缓存)!")
        
    except Exception as e:
        print(f"缓存测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    import sys
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='测试绘图功能，支持原版和缓存版本的多倍率算法运行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python test_plot.py                    # 运行原版测试函数
  python test_plot.py --cached           # 运行带缓存的测试函数
  python test_plot.py --both             # 运行两个版本进行对比
  python test_plot.py --cached --max-concurrency 4  # 指定并发数
        '''
    )
    
    # 添加命令行参数
    parser.add_argument(
        '--cached', 
        action='store_true',
        help='使用带缓存的测试函数 (run_multi_k_with_cache)'
    )
    
    parser.add_argument(
        '--both', 
        action='store_true',
        help='运行原版和缓存版本进行对比'
    )
    
    parser.add_argument(
        '--max-concurrency', 
        type=int, 
        default=2,
        help='最大并发数 (默认: 2)'
    )
    
    parser.add_argument(
        '--param-file',
        type=str,
        default='params.yml',
        help='参数配置文件路径 (默认: params.yml)'
    )
    
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='显示图形界面 (默认只保存图片)'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 验证参数冲突
    if args.cached and args.both:
        print("错误: --cached 和 --both 参数不能同时使用")
        sys.exit(1)
    
    # 打印运行配置
    print("=" * 60)
    print("测试绘图功能配置")
    print("=" * 60)
    print(f"参数文件: {args.param_file}")
    print(f"最大并发数: {args.max_concurrency}")
    print(f"显示图片: {'是' if args.show_plots else '否'}")
    
    # 根据参数执行相应的测试函数
    if args.both:
        print("模式: 对比运行两个版本")
        print("\n" + "="*50)
        print("第一阶段: 运行原版测试函数")
        print("="*50)
        test_plot_functions(args.param_file, args.max_concurrency, args.show_plots)
        
        print("\n" + "="*50)
        print("第二阶段: 运行带缓存的测试函数")
        print("="*50)
        test_plot_functions_with_cache(args.param_file, args.max_concurrency, args.show_plots)
        
    elif args.cached:
        print("模式: 运行带缓存的测试函数")
        print("="*60)
        test_plot_functions_with_cache(args.param_file, args.max_concurrency, args.show_plots)
        
    else:
        print("模式: 运行原版测试函数")
        print("="*60)
        test_plot_functions(args.param_file, args.max_concurrency, args.show_plots)
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
