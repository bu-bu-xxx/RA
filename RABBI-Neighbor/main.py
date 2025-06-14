import os
from solver import RABBI, OFFline, NPlusOneLP
import numpy as np
from customer import CustomerChoiceSimulator


def run_rabbi(param_file, y_file):
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
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    sim.compute_offline_Q()  # 计算Q矩阵
    offline = OFFline(sim)
    offline.run()
    print("[OFFline] x_history shape:", np.array(sim.x_history).shape)
    print("[OFFline] alpha_history:", sim.alpha_history)
    print("[OFFline] j_history:", sim.j_history)
    print("[OFFline] b_history:", sim.b_history)
    print("[OFFline] reward_history:", sim.reward_history)
    print("[OFFline] Final inventory:", sim.b)
    print("[OFFline] total reward:", sum(sim.reward_history))

def run_nplusonelp(param_file, y_file):
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    if os.path.exists(y_file):
        sim.load_Y(y_file)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(y_file)
    rabbi_nplus1 = NPlusOneLP(sim, debug=False)
    rabbi_nplus1.run()
    print("[NPlusOneLP] x_history shape:", np.array(sim.x_history).shape)
    print("[NPlusOneLP] alpha_history:", sim.alpha_history)
    print("[NPlusOneLP] j_history:", sim.j_history)
    print("[NPlusOneLP] b_history:", sim.b_history)
    print("[NPlusOneLP] reward_history:", sim.reward_history)
    print("[NPlusOneLP] Final inventory:", sim.b)
    print("[NPlusOneLP] total reward:", sum(sim.reward_history))

if __name__ == "__main__":
    param_file = 'params.yml'
    y_file = os.path.join("data", 'Y_matrix_debug.npy')
    print("===== RABBI 示例 =====")
    run_rabbi(param_file, y_file)
    print("\n===== OFFline 示例 =====")
    run_offline(param_file, y_file)
    print("\n===== NPlusOneLP 示例 =====")
    run_nplusonelp(param_file, y_file)
