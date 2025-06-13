import os
from solver import RABBI, OFFline
import numpy as np

def run_rabbi(param_file, y_file):
    from customer import CustomerChoiceSimulator
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    rabbi = RABBI(sim)
    rabbi.run()
    print("[RABBI] x_history shape:", np.array(sim.x_history).shape)
    print("[RABBI] alpha_history:", sim.alpha_history)
    print("[RABBI] j_history:", sim.j_history)
    print("[RABBI] b_history:", sim.b_history)
    print("[RABBI] reward_history:", sim.reward_history)
    print("[RABBI] Final inventory:", sim.b)
    print("[RABBI] total reward:", sum(sim.reward_history))

def run_offline(param_file, y_file):
    from customer import CustomerChoiceSimulator
    sim_off = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim_off.load_Y(y_file)
    else:
        sim_off.generate_Y_matrix()
        sim_off.save_Y(y_file)
    sim_off.compute_offline_Q()  # 计算Q矩阵
    offline = OFFline(sim_off)
    offline.run()
    print("[OFFline] x_history shape:", np.array(sim_off.x_history).shape)
    print("[OFFline] alpha_history:", sim_off.alpha_history)
    print("[OFFline] j_history:", sim_off.j_history)
    print("[OFFline] b_history:", sim_off.b_history)
    print("[OFFline] reward_history:", sim_off.reward_history)
    print("[OFFline] Final inventory:", sim_off.b)
    print("[OFFline] total reward:", sum(sim_off.reward_history))

if __name__ == "__main__":
    param_file = 'params.yml'
    y_file = os.path.join("data", 'Y_matrix_debug.npy')
    print("===== RABBI 示例 =====")
    run_rabbi(param_file, y_file)
    print("\n===== OFFline 示例 =====")
    run_offline(param_file, y_file)
