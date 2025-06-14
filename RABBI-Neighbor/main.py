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

def run_rabbi_multi_k(param_file, y_filename):
    sim_list = []
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    for i, k_val in enumerate(sim.k):
        sim.B = sim.B * k_val
        sim.T = int(sim.T * k_val)
        y_file = f"{y_filename}_k{int(k_val)}.npy"
        if os.path.exists(y_file):
            sim.load_Y(y_file)
        else:
            sim.generate_Y_matrix()
            sim.save_Y(y_file)
        rabbi = RABBI(sim)
        rabbi.run()
        print(f"[RABBI][k={k_val}] x_history shape:", np.array(sim.x_history).shape)
        print(f"[RABBI][k={k_val}] alpha_history:", sim.alpha_history)
        print(f"[RABBI][k={k_val}] j_history:", sim.j_history)
        print(f"[RABBI][k={k_val}] b_history:", sim.b_history)
        print(f"[RABBI][k={k_val}] reward_history:", sim.reward_history)
        print(f"[RABBI][k={k_val}] Final inventory:", sim.b)
        print(f"[RABBI][k={k_val}] total reward:", sum(sim.reward_history))
        sim_list.append(sim)
        sim = CustomerChoiceSimulator(param_file, random_seed=42)
    print(f"[RABBI] sim_list length: {len(sim_list)}")
    return sim_list

def run_offline_multi_k(param_file, y_filename):
    sim_list = []
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    for i, k_val in enumerate(sim.k):
        sim.B = sim.B * k_val
        sim.T = int(sim.T * k_val)
        y_file = f"{y_filename}_k{int(k_val)}.npy"
        if os.path.exists(y_file):
            sim.load_Y(y_file)
        else:
            sim.generate_Y_matrix()
            sim.save_Y(y_file)
        sim.compute_offline_Q()
        offline = OFFline(sim)
        offline.run()
        print(f"[OFFline][k={k_val}] x_history shape:", np.array(sim.x_history).shape)
        print(f"[OFFline][k={k_val}] alpha_history:", sim.alpha_history)
        print(f"[OFFline][k={k_val}] j_history:", sim.j_history)
        print(f"[OFFline][k={k_val}] b_history:", sim.b_history)
        print(f"[OFFline][k={k_val}] reward_history:", sim.reward_history)
        print(f"[OFFline][k={k_val}] Final inventory:", sim.b)
        print(f"[OFFline][k={k_val}] total reward:", sum(sim.reward_history))
        sim_list.append(sim)
        sim = CustomerChoiceSimulator(param_file, random_seed=42)
    print(f"[OFFline] sim_list length: {len(sim_list)}")
    return sim_list

def run_nplusonelp_multi_k(param_file, y_filename):
    sim_list = []
    sim = CustomerChoiceSimulator(param_file, random_seed=42)
    for i, k_val in enumerate(sim.k):
        sim.B = sim.B * k_val
        sim.T = int(sim.T * k_val)
        y_file = f"{y_filename}_k{int(k_val)}.npy"
        if os.path.exists(y_file):
            sim.load_Y(y_file)
        else:
            sim.generate_Y_matrix()
            sim.save_Y(y_file)
        rabbi_nplus1 = NPlusOneLP(sim, debug=False)
        rabbi_nplus1.run()
        print(f"[NPlusOneLP][k={k_val}] x_history shape:", np.array(sim.x_history).shape)
        print(f"[NPlusOneLP][k={k_val}] alpha_history:", sim.alpha_history)
        print(f"[NPlusOneLP][k={k_val}] j_history:", sim.j_history)
        print(f"[NPlusOneLP][k={k_val}] b_history:", sim.b_history)
        print(f"[NPlusOneLP][k={k_val}] reward_history:", sim.reward_history)
        print(f"[NPlusOneLP][k={k_val}] Final inventory:", sim.b)
        print(f"[NPlusOneLP][k={k_val}] total reward:", sum(sim.reward_history))
        sim_list.append(sim)
        sim = CustomerChoiceSimulator(param_file, random_seed=42)
    print(f"[NPlusOneLP] sim_list length: {len(sim_list)}")
    return sim_list

if __name__ == "__main__":
    param_file = 'params.yml'
    # y_file = os.path.join("data", 'Y_matrix_debug.npy')
    # print("===== RABBI 示例 =====")
    # run_rabbi(param_file, y_file)
    # print("\n===== OFFline 示例 =====")
    # run_offline(param_file, y_file)
    # print("\n===== NPlusOneLP 示例 =====")
    # run_nplusonelp(param_file, y_file)


    y_filename = os.path.join("data", 'Y_matrix_debug')
    print("\n===== RABBI 多倍率示例 =====")
    run_rabbi_multi_k(param_file, y_filename)
    print("\n===== OFFline 多倍率示例 =====")
    run_offline_multi_k(param_file, y_filename)
    print("\n===== NPlusOneLP 多倍率示例 =====")
    run_nplusonelp_multi_k(param_file, y_filename)
