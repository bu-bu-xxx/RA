import numpy as np
from scipy.optimize import linprog

class LPBasedPolicy:
    def __init__(self, env):
        """
        env: CustomerChoiceSimulator 实例
        history:
        - x_history: 存储每一步的x向量 (T, m)
        """
        self.env = env
        self.env.x_history = []  # 存储每一步的x向量 (T, m)
        self.env.reset()

class RABBI(LPBasedPolicy):
    def __init__(self, env):
        super().__init__(env)

    def solve_lp(self, b, p, t):
        """
        求解 P[b, p, t] 线性规划
        b: 当前库存 (d,)
        p: 当前购买概率矩阵 (n, m)
        t: 当前时间步 (int)
        返回: x (m,)
        """
        n, m, d = self.env.n, self.env.m, self.env.d
        f = self.env.f  # (n, m)
        A_mat = self.env.A  # (n, d)
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
        b_eq = [self.env.T - t]  # T-t, 即剩余时间步数

        # 约束3: x >= 0
        bounds = [(0, None) for _ in range(m)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            raise RuntimeError("LP solver failed: " + res.message)

    def run(self):
        """
        运行RABBI算法，自动推进T步
        """
        env = self.env
        env.reset()
        Y = env.Y if hasattr(env, 'Y') and env.Y is not None else env.generate_choice_matrix()
        b = env.B.copy()
        for t in range(env.T):
            p_t = env.p  # 每一步都用env.p
            x_t = self.solve_lp(b, p_t, t)  
            env.x_history.append(x_t)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            _, _, done, _ = env.step(j, alpha)
            b = env.b.copy()
            if done:
                break

# 示例用法
if __name__ == "__main__":
    from read_params import read_params
    from customer import CustomerChoiceSimulator
    n, d, m, A, f, p, T, B, k = read_params('params.yml')
    sim = CustomerChoiceSimulator(n, d, m, A, f, p, T, B, k, random_seed=42)
    sim.generate_choice_matrix()
    rabbi = RABBI(sim)
    rabbi.run()
    print("x_history shape:", np.array(sim.x_history).shape)
    print("alpha_history:", sim.alpha_history)
    print("j_history:", sim.j_history)