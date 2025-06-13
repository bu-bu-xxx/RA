import numpy as np
from scipy.optimize import linprog
import traceback

class LPBasedPolicy:
    def __init__(self, env):
        """
        env: CustomerChoiceSimulator 实例
        history:
        - x_history: 存储每一步的x向量 (T, m)
        """
        self.env = env
        self.reset()

    def reset(self):
        """
        重置环境和历史记录
        """
        self.env.reset()
        self.env.x_history = [] # (T, m)

    @staticmethod
    def solve_lp(b, p, t, n, m, d, f, A_mat, T):
        """
        求解 P[b, p, t] 线性规划
        b: 当前库存 (d,)
        p: 当前购买概率矩阵 (n, m)
        t: 当前时间步 (int)
        n, m, d: 维度参数
        f: 价格矩阵 (n, m)
        A_mat: 资源消耗矩阵 (n, d)
        T: 总时间步
        返回: x (m,)
        """
        # 目标函数: maximize sum_{alpha} x_alpha * sum_j f_{j,alpha} p_{j,alpha}
        c = -np.sum(f * p, axis=0)  # (m,) 负号因为linprog默认minimize
        # 约束1: sum_{alpha} sum_{j} a_{ij} p_{j,alpha} x_alpha <= b_i, ∀i∈[d]
        A_ub = np.zeros((d, m))
        for i in range(d):
            for alpha in range(m):
                A_ub[i, alpha] = np.sum(A_mat[:, i] * p[:, alpha])
        b_ub = b
        # 约束2: sum_{alpha} x_alpha = T-t
        A_eq = np.ones((1, m))
        b_eq = [T - t]  # T-t, 即剩余时间步数
        # 约束3: x >= 0
        bounds = [(0, None) for _ in range(m)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            raise RuntimeError("LP solver failed: " + res.message)

class RABBI(LPBasedPolicy):
    def __init__(self, env):
        super().__init__(env)

    def run(self):
        """
        运行RABBI算法，自动推进T步
        """
        env = self.env
        env.reset()
        Y = env.Y if hasattr(env, 'Y') and env.Y is not None else env.generate_Y_matrix() # (T, m)
        b = env.B.copy() # (d,)
        for t in range(env.T):
            p_t = env.p  # 每一步都用env.p (n, m)
            try:
                x_t = self.solve_lp(b, p_t, t, env.n, env.m, env.d, env.f, env.A, env.T)  # (m,)
            except Exception as e:
                print(f"Error solving LP at time {t}: {e}")
                import traceback
                traceback.print_exc()
                break
            env.x_history.append(x_t)
            alpha = int(np.argmax(x_t)) 
            j = Y[t, alpha]
            _, _, done, _ = env.step(j, alpha)
            b = env.b.copy()
            if done:
                break

class OFFline(LPBasedPolicy):
    def __init__(self, env):
        super().__init__(env)

    def run(self):
        """
        运行RABBI算法，自动推进T步
        """
        env = self.env
        env.reset()
        Y = env.Y if hasattr(env, 'Y') and env.Y is not None else env.generate_Y_matrix() # (T, m)
        b = env.B.copy() # (d,)
        for t in range(env.T): 
            p_t = env.Q[t, :, :]  # 使用Q[t]矩阵 (n, m)
            try:
                x_t = self.solve_lp(b, p_t, t, env.n, env.m, env.d, env.f, env.A, env.T)  # (m,)
            except Exception as e:
                print(f"Error solving LP at time {t}: {e}")
                import traceback
                traceback.print_exc()
                break
            env.x_history.append(x_t)
            alpha = int(np.argmax(x_t)) 
            j = Y[t, alpha]
            _, _, done, _ = env.step(j, alpha)
            b = env.b.copy()
            if done:
                break

# 示例用法
if __name__ == "__main__":
    import os
    from customer import CustomerChoiceSimulator
    # 直接用CustomerChoiceSimulator读取params.yml
    sim = CustomerChoiceSimulator('params.yml', random_seed=42)
    # 生成Y矩阵
    # sim.generate_Y_matrix()  
    # 加载已保存的Y矩阵
    Y_path = os.path.join("data", 'Y_matrix_debug.npy')
    if os.path.exists(Y_path):
        sim.load_Y(Y_path)
    else:
        sim.generate_Y_matrix()
        sim.save_Y(Y_path)
    rabbi = RABBI(sim)
    rabbi.run()
    print("[RABBI] x_history shape:", np.array(sim.x_history).shape)
    print("[RABBI] alpha_history:", sim.alpha_history)
    print("[RABBI] j_history:", sim.j_history)
    print("[RABBI] b_history:", sim.b_history)
    print("[RABBI] reward_history:", sim.reward_history)
    print("[RABBI] Final inventory:", sim.b)
    print("[RABBI] total reward:", sum(sim.reward_history))

    # OFFline策略示例
    sim_off = CustomerChoiceSimulator('params.yml', random_seed=42)
    if os.path.exists(Y_path):
        sim_off.load_Y(Y_path)
    else:
        sim_off.generate_Y_matrix()
        sim_off.save_Y(Y_path)
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


