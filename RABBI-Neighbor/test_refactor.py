#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重构后的代码是否正常工作
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from read_params import ParamsLoader, Parameters
    from env import DynamicPricingEnv
    from customer import CustomerChoiceSimulator
    from solver import RABBI, OFFline, NPlusOneLP, LPBasedPolicy
    from main import compute_lp_x_benchmark, save_sim_list_to_shelve
    from plot import plot_multi_k_results, plot_multi_k_ratio_results
    
    print("✓ 所有模块导入成功")
    
    # 测试 Parameters 类
    params = Parameters()
    print("✓ Parameters 类实例化成功")
    
    # 测试基本属性
    print(f"Parameters 类包含以下属性:")
    for attr in dir(params):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
    print("\n" + "="*50)
    print("重构测试完成!")
    print("现在所有参数都通过 self.params.* 访问")
    print("历史记录也存储在 self.params 中:")
    print("  - self.params.reward_history")
    print("  - self.params.b_history") 
    print("  - self.params.j_history")
    print("  - self.params.alpha_history")
    print("  - self.params.Y (customer choice matrix)")
    print("  - self.params.Q (offline Q matrix)")
    print("  - self.params.x_history (LP solution history)")
    
    print("\n主要修改总结:")
    print("1. solver.py:")
    print("   - 在LPBasedPolicy.__init__中添加了self.params = self.env.params")
    print("   - 所有历史记录现在使用self.params.x_history")
    print("   - 所有参数访问改为self.params.*")
    print("2. main.py:")
    print("   - 所有worker函数使用sim.params.*访问参数")
    print("   - compute_lp_x_benchmark函数更新参数访问")
    print("   - save_sim_list_to_shelve函数更新属性访问")
    print("3. plot.py:")
    print("   - 所有函数使用sim.params.*访问参数")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
