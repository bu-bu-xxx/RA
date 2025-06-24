import numpy as np
from scipy.optimize import linprog, minimize
import traceback

class LPBasedPolicy:
    def __init__(self, env):
        """
        env: CustomerChoiceSimulator 实例
        history:
        - x_history: 存储每一步的x向量 (T, m)
        """
        self.env = env
        self.params = self.env.params  # 添加params引用
        self.reset()

    def reset(self):
        """
        重置环境和历史记录
        """
        self.env.reset()
        self.params.x_history = [] # (T, m)

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
        # 约束3: 0 <= x <= T-t
        bounds = [(0, T - t) for _ in range(m)]

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
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix() # (T, m)
        b = env.params.B.copy() # (d,)
        for t in range(env.params.T):
            p_t = env.params.p  # 每一步都用env.params.p (n, m)
            try:
                x_t = self.solve_lp(b, p_t, t, env.params.n, env.params.m, env.params.d, env.params.f, env.params.A, env.params.T)  # (m,)
            except Exception as e:
                print(f"Error solving LP at time {t}: {e}")
                import traceback
                traceback.print_exc()
                break
            self.params.x_history.append(x_t)
            alpha = int(np.argmax(x_t)) 
            j = Y[t, alpha]
            _, _, done, _ = env.step(j, alpha)
            b = env.params.b.copy()
            # if done:
            #     break

class OFFline(LPBasedPolicy):
    def __init__(self, env):
        super().__init__(env)

    def run(self):
        """
        运行RABBI算法，自动推进T步
        """
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix() # (T, m)
        b = env.params.B.copy() # (d,)
        for t in range(env.params.T): 
            p_t = env.params.Q[t, :, :]  # 使用Q[t]矩阵 (n, m)
            try:
                x_t = self.solve_lp(b, p_t, t, env.params.n, env.params.m, env.params.d, env.params.f, env.params.A, env.params.T)  # (m,)
            except Exception as e:
                print(f"Error solving LP at time {t}: {e}")
                raise e
            self.params.x_history.append(x_t)
            alpha = int(np.argmax(x_t)) 
            j = Y[t, alpha]
            _, _, done, _ = env.step(j, alpha)
            b = env.params.b.copy()
            # if done:
            #     break

class NPlusOneLP(LPBasedPolicy):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.n = env.params.n  # 产品数量
        self.d = env.params.d  # 资源种类数量
        self.b = env.params.b  # 当前库存 (d维向量)
        self.price_grid = env.params.f_split  # 离散价格集合 (列表的列表)
        self.d_attract = env.params.mnl.d  # 产品吸引力参数
        self.mu = env.params.mnl.mu  # MNL尺度参数
        self.u0 = env.params.mnl.u0  # 不购买选项的效用
        self.gamma = env.params.mnl.gamma  # 顾客到达率
        self.debug = debug
        self.A = env.params.A  # 资源消耗矩阵 (n, d)
        self.T = env.params.T  # 总时间步数
        self.env = env  # 环境实例
    
    @staticmethod
    def mnl_demand(prices, d, mu, u0=0, gamma=1.0):
        """
        计算MNL需求
        :param prices: 价格向量 [p1, p2, ..., pN]
        :param d: 产品吸引力向量 [d1, d2, ..., dN]
        :param mu: 理性参数 (μ > 0)
        :param u0: 不购买的效用
        :param gamma: 概率缩放系数
        :return: 需求向量 [λ1, λ2, ..., λN]
        """
        exponents = np.exp((np.array(d) - np.array(prices)) / mu)
        denominator = np.sum(exponents) + np.exp(u0 / mu)
        return gamma * exponents / denominator
    
    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def solve_continuous_relaxation(self):
        """
        步骤1: 求解连续松弛问题 (P_C)
        :return: 最优连续价格向量 p_star
        """
        # 定义目标函数 (最大化收益 = 最小化负收益)
        def objective(p):
            demand = self.mnl_demand(p, self.d_attract, self.mu, self.u0, self.gamma)
            self._debug_print("[DEBUG][objective] p:", p, "shape:", np.shape(p))
            self._debug_print("[DEBUG][objective] demand:", demand, "shape:", np.shape(demand))
            return -np.dot(p, demand)  # 负号用于最小化
        
        # 定义约束条件: A' * λ(p) <= b / (T-t)
        def constraint(p):
            demand = self.mnl_demand(p, self.d_attract, self.mu, self.u0, self.gamma)
            self._debug_print("[DEBUG][constraint] p:", p, "shape:", np.shape(p))
            self._debug_print("[DEBUG][constraint] demand:", demand, "shape:", np.shape(demand))
            self._debug_print("[DEBUG][constraint] self.params.b:", self.params.b, "shape:", np.shape(self.params.b))
            self._debug_print("[DEBUG][constraint] self.A:", self.A, "shape:", np.shape(self.A))
            self._debug_print("[DEBUG][constraint] self.A.T @ demand:", self.A.T @ demand, "shape:", np.shape(self.A.T @ demand))
            return self.params.b/(self.T-self.params.t) - self.A.T @ demand  # A' * λ(p) <= b
        
        # 价格上下界
        bounds = [(min(prices), max(prices)) for prices in self.price_grid]
        
        # 初始点 (取每个产品价格范围的中点)
        x0 = [np.mean(prices) for prices in self.price_grid]
        
        # 约束定义
        cons = [{'type': 'ineq', 'fun': constraint}]
        
        self._debug_print("[DEBUG][solve_continuous_relaxation] bounds:", bounds)
        self._debug_print("[DEBUG][solve_continuous_relaxation] x0:", x0)
        # 求解连续松弛问题
        res = minimize(objective, x0, bounds=bounds, constraints=cons)
        
        self._debug_print("[DEBUG][solve_continuous_relaxation] res.x:", res.x, "shape:", np.shape(res.x))
        self._debug_print("[DEBUG][solve_continuous_relaxation] res.success:", res.success, "message:", res.message)
        if not res.success:
            raise ValueError(f"连续松弛问题求解失败: {res.message}")
            
        return res.x
    
    def find_neighbors(self, p_star):
        """
        步骤2: 找到每个产品的最优连续价格对应的离散价格邻居
        :param p_star: 最优连续价格向量
        :return: 邻居价格向量列表 (N+1个)
        """
        neighbors = []
        # 找到每个产品的上下界离散价格
        lower_bounds = []
        upper_bounds = []
        
        for i in range(self.n):
            prices = self.price_grid[i]
            # 找到下界 (小于等于p_star[i]的最大值)
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            # 找到上界 (大于等于p_star[i]的最小值)
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        # 构造邻居向量 (N+1个)
        # 第一个: 所有产品都使用下界
        neighbors.append(lower_bounds)
        # 后N个: 前i个位置用上界，其余用下界
        for i in range(1, self.n+1):
            neighbor = [upper_bounds[j] if j < i else lower_bounds[j] for j in range(self.n)]
            neighbors.append(neighbor)
        
        self._debug_print("[DEBUG][find_neighbors] p_star:", p_star, "shape:", np.shape(p_star))
        self._debug_print("[DEBUG][find_neighbors] lower_bounds:", lower_bounds, "shape:", np.shape(lower_bounds))
        self._debug_print("[DEBUG][find_neighbors] upper_bounds:", upper_bounds, "shape:", np.shape(upper_bounds))
        self._debug_print("[DEBUG][find_neighbors] neighbors:", neighbors, "shape:", np.shape(neighbors))
        # print("[DEBUG][find_neighbors] neighbors:", neighbors)
        return neighbors
    
    def solve_n_plus_one_lp(self, neighbors):
        """
        步骤3: 求解(N+1) LP问题
        :param neighbors: 邻居价格向量列表 (N+1, n)
        :return: 最优时间分配比例 zeta_star (N+1,)
        """
        num_neighbors = len(neighbors)
        
        # 计算每个邻居的需求向量和收益
        demands = []
        revenues = []
        
        for neighbor in neighbors:
            demand = self.mnl_demand(neighbor, self.d_attract, self.mu, self.u0, self.gamma)
            demands.append(demand)
            revenues.append(np.dot(neighbor, demand))
        
        # 构建线性规划问题
        # 目标函数: 最大化收益
        c = -np.array(revenues)  # 负号用于最小化
        
        # 约束: A' * (∑ζ_i * λ_i) <= b
        # 计算每个资源约束的系数
        A_ub = []
        for k in range(self.d):
            row = []
            for i in range(num_neighbors):
                # 计算第i个邻居对资源k的消耗
                # self.A shape (n, d), demands[i] shape (n,)
                # 对于资源k, 取A的第k列，对应所有产品的消耗
                resource_consumption = np.dot(self.A[:, k], demands[i])
                row.append(resource_consumption)
            A_ub.append(row)
          # 添加时间总和约束: ∑ζ_i = T-t
        A_eq = [np.ones(num_neighbors)]
        b_eq = [self.T- self.params.t] 
        
        # 变量边界: 0 <= ζ_i <= T-t
        bounds = [(0, self.T- self.params.t)] * num_neighbors
        
        self._debug_print("[DEBUG][solve_n_plus_one_lp] neighbors:", neighbors, "shape:", np.shape(neighbors))
        self._debug_print("[DEBUG][solve_n_plus_one_lp] demands:", demands, "shape:", np.shape(demands))
        self._debug_print("[DEBUG][solve_n_plus_one_lp] revenues:", revenues, "shape:", np.shape(revenues))
        self._debug_print("[DEBUG][solve_n_plus_one_lp] c:", c, "shape:", np.shape(c))
        self._debug_print("[DEBUG][solve_n_plus_one_lp] A_ub:", A_ub, "shape:", np.shape(A_ub))
        self._debug_print("[DEBUG][solve_n_plus_one_lp] b_ub:", self.b, "shape:", np.shape(self.b))
        self._debug_print("[DEBUG][solve_n_plus_one_lp] A_eq:", A_eq, "b_eq:", b_eq)        # 求解线性规划
        res = linprog(c, A_ub=A_ub, b_ub=self.params.b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        self._debug_print("[DEBUG][solve_n_plus_one_lp] res.x:", res.x if res.success else None, "shape:", np.shape(res.x) if res.success else None)
        self._debug_print("[DEBUG][solve_n_plus_one_lp] res.success:", res.success, "message:", res.message)
        if not res.success:
            print("[DEBUG][solve_n_plus_one_lp] neighbors:", neighbors)
            print("[DEBUG][solve_n_plus_one_lp] demands:", demands)
            print("[DEBUG][solve_n_plus_one_lp] revenues:", revenues)
            print("[DEBUG][solve_n_plus_one_lp] c:", c)
            print("[DEBUG][solve_n_plus_one_lp] A_ub:", A_ub)
            print("[DEBUG][solve_n_plus_one_lp] b_ub:", self.params.b)
            print("[DEBUG][solve_n_plus_one_lp] A_eq:", A_eq, "b_eq:", b_eq)
            print("[DEBUG][solve_n_plus_one_lp] bounds:", bounds)
            raise ValueError(f"(N+1) LP求解失败: {res.message}")
            
        return res.x

    def get_pricing_policy(self):
        """
        获取定价策略
        :return: (邻居价格向量, 最优时间分配比例)
        """
        # 步骤1: 求解连续松弛问题
        p_star = self.solve_continuous_relaxation()
        self._debug_print("[DEBUG][get_pricing_policy] p_star:", p_star, "shape:", np.shape(p_star))
        
        # 步骤2: 找到邻居价格向量
        neighbors = self.find_neighbors(p_star)
        self._debug_print("[DEBUG][get_pricing_policy] neighbors:", neighbors, "shape:", np.shape(neighbors))
        
        # 步骤3: 求解(N+1) LP
        zeta_star = self.solve_n_plus_one_lp(neighbors)
        self._debug_print("[DEBUG][get_pricing_policy] zeta_star:", zeta_star, "shape:", np.shape(zeta_star))
        self._debug_print("[DEBUG][get_pricing_policy] p_star:", p_star, "shape:", np.shape(p_star))
        self._debug_print("[DEBUG][get_pricing_policy] neighbors:", neighbors, "shape:", np.shape(neighbors))
        self._debug_print("[DEBUG][get_pricing_policy] zeta_star:", zeta_star, "shape:", np.shape(zeta_star))
        return neighbors, zeta_star

    def map_zeta_to_xt(self, neighbors, zeta_star, env_f):
        """
        将zeta_star（neighbors的解）映射为x_t（env.f的解）。
        :param neighbors: 邻居价格向量列表 (N+1, n)
        :param zeta_star: 每个邻居的时间分配比例 (N+1,)
        :param env_f: 环境中的所有价格列 (n, m)
        :return: x_t (m,)
        """
        m = env_f.shape[1]
        x_t = np.zeros(m)
        for neighbor_idx, neighbor in enumerate(neighbors):
            for idx in range(m):
                if np.allclose(env_f[:, idx], neighbor):
                    x_t[idx] = zeta_star[neighbor_idx]
                    break  # 一个neighbor只对应一个env.f中的列
        return x_t

    def run(self):
        """
        运行NPlusOneLP算法，自动推进T步，每步根据get_pricing_policy选择最优邻居价格
        """
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()  # (T, m)
        b = env.params.B.copy()  # (d,)
        for t in range(env.params.T):
            try:
                neighbors, zeta_star = self.get_pricing_policy()
            except Exception as e:
                print(f"Error in get_pricing_policy at time {t}: {e}")
                raise e
            # 新增：将zeta_star映射为x_t
            x_t = self.map_zeta_to_xt(neighbors, zeta_star, env.params.f)
            # 选择x_t最大的index为alpha
            alpha = int(np.argmax(x_t))
            price_vec = env.params.f[:, alpha]
            j = Y[t, alpha]
            _, _, done, _ = env.step(j, alpha)
            self.params.x_history.append(x_t)
            b = env.params.b.copy()
            # if done:
            #     break


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
    print("[RABBI] x_history shape:", np.array(sim.params.x_history).shape)
    print("[RABBI] x_history:", sim.params.x_history)
    print("[RABBI] alpha_history:", sim.params.alpha_history)
    print("[RABBI] j_history:", sim.params.j_history)
    print("[RABBI] b_history:", sim.params.b_history)
    print("[RABBI] reward_history:", sim.params.reward_history)
    print("[RABBI] Final inventory:", sim.params.b)
    print("[RABBI] total reward:", sum(sim.params.reward_history))

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
    print("[OFFline] x_history shape:", np.array(sim_off.params.x_history).shape)
    print("[OFFline] x_history:", sim_off.params.x_history)
    print("[OFFline] alpha_history:", sim_off.params.alpha_history)
    print("[OFFline] j_history:", sim_off.params.j_history)
    print("[OFFline] b_history:", sim_off.params.b_history)
    print("[OFFline] reward_history:", sim_off.params.reward_history)
    print("[OFFline] Final inventory:", sim_off.params.b)
    print("[OFFline] total reward:", sum(sim_off.params.reward_history))

    # NPlusOneLP策略示例
    sim_nplus1 = CustomerChoiceSimulator('params.yml', random_seed=42)
    if os.path.exists(Y_path):
        sim_nplus1.load_Y(Y_path)
    else:
        sim_nplus1.generate_Y_matrix()
        sim_nplus1.save_Y(Y_path)
    rabbi_nplus1 = NPlusOneLP(sim_nplus1, debug=False)
    rabbi_nplus1.run()
    print("[NPlusOneLP] x_history shape:", np.array(sim_nplus1.params.x_history).shape)
    print("[NPlusOneLP] x_history:", sim_nplus1.params.x_history)
    print("[NPlusOneLP] alpha_history:", sim_nplus1.params.alpha_history)
    print("[NPlusOneLP] j_history:", sim_nplus1.params.j_history)
    print("[NPlusOneLP] b_history:", sim_nplus1.params.b_history)
    print("[NPlusOneLP] reward_history:", sim_nplus1.params.reward_history)
    print("[NPlusOneLP] Final inventory:", sim_nplus1.params.b)
    print("[NPlusOneLP] total reward:", sum(sim_nplus1.params.reward_history))





