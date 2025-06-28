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
    def __init__(self, env, debug=False):
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
    def __init__(self, env, debug=False):
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

    @staticmethod
    def _debug_print(debug_tag, debug, *args, **kwargs):
        if debug:
            print(f"[{debug_tag}]", *args, **kwargs)

    @staticmethod
    def solve_continuous_relaxation(price_grid, d_attract, mu, u0, gamma, A, b, T, t, debug=False, debug_tag="solve_continuous_relaxation"):
        """
        步骤1: 求解连续松弛问题 (P_C)
        :param price_grid: 离散价格集合 (列表的列表)
        :param d_attract: 产品吸引力参数
        :param mu: MNL尺度参数
        :param u0: 不购买选项的效用
        :param gamma: 顾客到达率
        :param A: 资源消耗矩阵 (n, d)
        :param b: 当前库存 (d,)
        :param T: 总时间步数
        :param t: 当前时间步
        :param debug: 是否打印调试信息
        :param debug_tag: 调试标签
        :return: 最优连续价格向量 p_star
        """
        # 定义目标函数 (最大化收益 = 最小化负收益)
        # 目标函数: maximize sum_{j} p_j * λ(p_j)
        def objective(p):
            demand = NPlusOneLP.mnl_demand(p, d_attract, mu, u0, gamma)
            NPlusOneLP._debug_print(f"{debug_tag}_objective", debug, "p:", p, "shape:", np.shape(p))
            NPlusOneLP._debug_print(f"{debug_tag}_objective", debug, "demand:", demand, "shape:", np.shape(demand))
            return -np.dot(p, demand)  # 负号用于最小化
        
        # 定义约束条件: A' * λ(p) <= b / (T-t)
        def constraint(p):
            demand = NPlusOneLP.mnl_demand(p, d_attract, mu, u0, gamma)
            NPlusOneLP._debug_print(f"{debug_tag}_constraint", debug, "p:", p, "shape:", np.shape(p))
            NPlusOneLP._debug_print(f"{debug_tag}_constraint", debug, "demand:", demand, "shape:", np.shape(demand))
            NPlusOneLP._debug_print(f"{debug_tag}_constraint", debug, "b:", b, "shape:", np.shape(b))
            NPlusOneLP._debug_print(f"{debug_tag}_constraint", debug, "A:", A, "shape:", np.shape(A))
            NPlusOneLP._debug_print(f"{debug_tag}_constraint", debug, "A.T @ demand:", A.T @ demand, "shape:", np.shape(A.T @ demand))
            return b/(T-t) - A.T @ demand  # A' * λ(p) <= b
        
        # 价格上下界
        bounds = [(min(prices), max(prices)) for prices in price_grid]
        
        # 初始点 (取每个产品价格范围的中点)
        x0 = [np.mean(prices[:-1]) for prices in price_grid]
        
        # 约束定义
        cons = [{'type': 'ineq', 'fun': constraint}]
        
        NPlusOneLP._debug_print(debug_tag, debug, "bounds:", bounds)
        NPlusOneLP._debug_print(debug_tag, debug, "x0:", x0)
        # 求解连续松弛问题
        res = minimize(objective, x0, bounds=bounds, constraints=cons)
        
        NPlusOneLP._debug_print(debug_tag, debug, "res.x:", res.x, "shape:", np.shape(res.x))
        NPlusOneLP._debug_print(debug_tag, debug, "res.success:", res.success, "message:", res.message)
        if not res.success:
            raise ValueError(f"连续松弛问题求解失败: {res.message}")
            
        return res.x

    @staticmethod
    def get_relative_position(p_star, price_grid, n):
        """
        计算每个产品价格的相对位置
        :param p_star: 最优连续价格向量
        :param price_grid: 离散价格集合 (列表的列表)
        :param n: 产品数量
        :return: 相对位置向量
        """
        relative_positions = []
        
        for i in range(n):
            prices = price_grid[i]
            # 找到下界和上界
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            
            # 计算相对位置：(p_i - lower) / (upper - lower)
            if upper == lower:
                relative_pos = 0.0  # 如果上下界相等，相对位置为0
            else:
                relative_pos = (p_star[i] - lower) / (upper - lower)
            
            relative_positions.append(relative_pos)
        
        return relative_positions

    @staticmethod
    def find_neighbors(p_star, price_grid, n, debug=False, debug_tag="find_neighbors"):
        """
        步骤2: 找到每个产品的最优连续价格对应的离散价格邻居
        使用相对位置排序的方法生成邻居
        :param p_star: 最优连续价格向量
        :param price_grid: 离散价格集合 (列表的列表)
        :param n: 产品数量
        :param debug: 是否打印调试信息
        :param debug_tag: 调试标签
        :return: 邻居价格向量列表 (N+1个)
        """
        # 找到每个产品的上下界离散价格
        lower_bounds = []
        upper_bounds = []
        
        for i in range(n):
            prices = price_grid[i]
            # 找到下界 (小于等于p_star[i]的最大值)
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            # 找到上界 (大于等于p_star[i]的最小值)
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        # 计算每个产品的相对位置
        relative_positions = NPlusOneLP.get_relative_position(p_star, price_grid, n)
        
        # 根据相对位置进行排序，获得排序后的索引
        # 相对位置越大的产品，在排序中越靠前
        sorted_indices = sorted(range(n), key=lambda i: relative_positions[i], reverse=True)
        
        NPlusOneLP._debug_print(debug_tag, debug, "relative_positions:", relative_positions)
        NPlusOneLP._debug_print(debug_tag, debug, "sorted_indices:", sorted_indices)
        
        neighbors = []
        
        # 构造邻居向量 (N+1个)
        # 第一个: 所有产品都使用下界
        base_neighbor = lower_bounds.copy()
        neighbors.append(base_neighbor)
        
        # 后N个: 按照排序后的顺序，前i个位置用上界，其余用下界
        for i in range(1, n+1):
            neighbor = lower_bounds.copy()  # 从下界开始
            # 按排序后的顺序，前i个位置用上界
            for j in range(i):
                idx = sorted_indices[j]  # 获取原始索引
                neighbor[idx] = upper_bounds[idx]  # 在原始位置设置上界
            neighbors.append(neighbor)
        
        # 去除重复的邻居 (当relative_pos=0时，upper_bound == lower_bound会产生重复)
        unique_neighbors = []
        seen = set()
        for neighbor in neighbors:
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple not in seen:
                seen.add(neighbor_tuple)
                unique_neighbors.append(neighbor)
        
        NPlusOneLP._debug_print(debug_tag, debug, "p_star:", p_star, "shape:", np.shape(p_star))
        NPlusOneLP._debug_print(debug_tag, debug, "lower_bounds:", lower_bounds, "shape:", np.shape(lower_bounds))
        NPlusOneLP._debug_print(debug_tag, debug, "upper_bounds:", upper_bounds, "shape:", np.shape(upper_bounds))
        NPlusOneLP._debug_print(debug_tag, debug, f"Generated {len(neighbors)} neighbors before deduplication")
        NPlusOneLP._debug_print(debug_tag, debug, f"Generated {len(unique_neighbors)} unique neighbors after deduplication")
        NPlusOneLP._debug_print(debug_tag, debug, "unique_neighbors:", unique_neighbors, "shape:", np.shape(unique_neighbors))
        return unique_neighbors

    @staticmethod
    def solve_n_plus_one_lp(neighbors, d_attract, mu, u0, gamma, A, d, b, T, t, debug=False, debug_tag="solve_n_plus_one_lp"):
        """
        步骤3: 求解(N+1) LP问题
        :param neighbors: 邻居价格向量列表 (N+1, n)
        :param d_attract: 产品吸引力参数
        :param mu: MNL尺度参数
        :param u0: 不购买选项的效用
        :param gamma: 顾客到达率
        :param A: 资源消耗矩阵 (n, d)
        :param d: 资源种类数量
        :param b: 当前库存 (d,)
        :param T: 总时间步数
        :param t: 当前时间步
        :param debug: 是否打印调试信息
        :param debug_tag: 调试标签
        :return: 最优时间分配比例 zeta_star (N+1,)
        """
        num_neighbors = len(neighbors)
        
        # 计算每个邻居的需求向量和收益
        demands = []
        revenues = []
        
        for neighbor in neighbors:
            demand = NPlusOneLP.mnl_demand(neighbor, d_attract, mu, u0, gamma)
            demands.append(demand)
            revenues.append(np.dot(neighbor, demand))
        
        # 构建线性规划问题
        # 目标函数: 最大化收益
        c = -np.array(revenues)  # 负号用于最小化
        
        # 约束: A' * (∑ζ_i * λ_i) <= b
        # 计算每个资源约束的系数
        A_ub = []
        for k in range(d):
            row = []
            for i in range(num_neighbors):
                # 计算第i个邻居对资源k的消耗
                # A shape (n, d), demands[i] shape (n,)
                # 对于资源k, 取A的第k列，对应所有产品的消耗
                resource_consumption = np.dot(A[:, k], demands[i])
                row.append(resource_consumption)
            A_ub.append(row)
        # 添加时间总和约束: ∑ζ_i = T-t
        A_eq = [np.ones(num_neighbors)]
        b_eq = [T - t] 
        
        # 变量边界: 0 <= ζ_i <= T-t
        bounds = [(0, T - t)] * num_neighbors
        
        NPlusOneLP._debug_print(debug_tag, debug, "neighbors:", neighbors, "shape:", np.shape(neighbors))
        NPlusOneLP._debug_print(debug_tag, debug, "demands:", demands, "shape:", np.shape(demands))
        NPlusOneLP._debug_print(debug_tag, debug, "revenues:", revenues, "shape:", np.shape(revenues))
        NPlusOneLP._debug_print(debug_tag, debug, "c:", c, "shape:", np.shape(c))
        NPlusOneLP._debug_print(debug_tag, debug, "A_ub:", A_ub, "shape:", np.shape(A_ub))
        NPlusOneLP._debug_print(debug_tag, debug, "b_ub:", b, "shape:", np.shape(b))
        NPlusOneLP._debug_print(debug_tag, debug, "A_eq:", A_eq, "b_eq:", b_eq)
        # 求解线性规划
        res = linprog(c, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        NPlusOneLP._debug_print(debug_tag, debug, "res.x:", res.x if res.success else None, "shape:", np.shape(res.x) if res.success else None)
        NPlusOneLP._debug_print(debug_tag, debug, "res.success:", res.success, "message:", res.message)
        if not res.success:
            print(f"[{debug_tag}] neighbors:", neighbors)
            print(f"[{debug_tag}] demands:", demands)
            print(f"[{debug_tag}] revenues:", revenues)
            print(f"[{debug_tag}] c:", c)
            print(f"[{debug_tag}] A_ub:", A_ub)
            print(f"[{debug_tag}] b_ub:", b)
            print(f"[{debug_tag}] A_eq:", A_eq, "b_eq:", b_eq)
            print(f"[{debug_tag}] bounds:", bounds)
            raise ValueError(f"(N+1) LP求解失败: {res.message}")
            
        return res.x

    def get_pricing_policy(self):
        """
        获取定价策略
        :return: (邻居价格向量, 最优时间分配比例)
        """
        # 步骤1: 求解连续松弛问题
        p_star = self.solve_continuous_relaxation(
            self.price_grid, self.d_attract, self.mu, self.u0, self.gamma,
            self.A, self.params.b, self.T, self.params.t, self.debug, "get_pricing_policy_step1"
        )
        self._debug_print("get_pricing_policy", self.debug, "p_star:", p_star, "shape:", np.shape(p_star))
        
        # 步骤2: 找到邻居价格向量
        neighbors = self.find_neighbors(p_star, self.price_grid, self.n, self.debug, "get_pricing_policy_step2")
        self._debug_print("get_pricing_policy", self.debug, "neighbors:", neighbors, "shape:", np.shape(neighbors))
        
        # 步骤3: 求解(N+1) LP
        zeta_star = self.solve_n_plus_one_lp(
            neighbors, self.d_attract, self.mu, self.u0, self.gamma,
            self.A, self.d, self.params.b, self.T, self.params.t, self.debug, "get_pricing_policy_step3"
        )
        self._debug_print("get_pricing_policy", self.debug, "zeta_star:", zeta_star, "shape:", np.shape(zeta_star))
        self._debug_print("get_pricing_policy", self.debug, "p_star:", p_star, "shape:", np.shape(p_star))
        self._debug_print("get_pricing_policy", self.debug, "neighbors:", neighbors, "shape:", np.shape(neighbors))
        self._debug_print("get_pricing_policy", self.debug, "zeta_star:", zeta_star, "shape:", np.shape(zeta_star))
        return neighbors, zeta_star

    @staticmethod
    def map_zeta_to_xt(neighbors, zeta_star, env_f):
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


class TopKLP(LPBasedPolicy):
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
        self.topk = env.params.topk  # 从params中读取Top-K参数
        
        # 验证topk参数
        if self.topk > self.n:
            self.topk = self.n  # 如果topk大于产品数量，限制为n
            print(f"Top-K parameter {self.topk} exceeds product count {self.n}, resetting to {self.n}")
        # 存储上一轮的结果
        self.p_star_prev = None
        self.zeta_star_prev = None
        self.iteration = 0

    def find_neighbors_topk(self, p_star, debug=False, debug_tag="find_neighbors_topk"):
        """
        TopK版本的find_neighbors方法
        第一轮：返回所有2^n个价格组合
        后续轮次：基于需求变化选择top-k个产品，返回2^k个局部组合和2^n个完整组合
        """
        if self.iteration == 0:
            # 第一轮：返回所有可能的组合
            neighbors = self._get_all_neighbors(p_star, debug, debug_tag)
            full_neighbors = neighbors
            self._debug_print(debug_tag, debug, f"First iteration: generated {len(neighbors)} neighbors")
        else:            # 后续轮次：基于需求变化选择top-k
            topk_indices = self._select_topk_products(p_star, debug, debug_tag)
            neighbors = self._get_topk_neighbors(p_star, topk_indices, debug, debug_tag)
            full_neighbors = self._get_all_neighbors(p_star, debug, debug_tag)
            self._debug_print(debug_tag, debug, f"Iteration {self.iteration}: selected top-{self.topk} products {topk_indices}")
            self._debug_print(debug_tag, debug, f"Generated {len(neighbors)} topk neighbors and {len(full_neighbors)} full neighbors")
        
        return neighbors, full_neighbors

    def _get_all_neighbors(self, p_star, debug=False, debug_tag="get_all_neighbors"):
        """
        获取所有2^n个价格组合
        """
        neighbors = []
        # 找到每个产品的上下界
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
        
        # 生成所有2^n个组合
        for i in range(2**self.n):
            neighbor = []
            for j in range(self.n):
                if (i >> j) & 1:
                    neighbor.append(upper_bounds[j])
                else:
                    neighbor.append(lower_bounds[j])
            neighbors.append(neighbor)
        
        self._debug_print(debug_tag, debug, "p_star:", p_star, "shape:", np.shape(p_star))
        self._debug_print(debug_tag, debug, "lower_bounds:", lower_bounds)
        self._debug_print(debug_tag, debug, "upper_bounds:", upper_bounds)
        self._debug_print(debug_tag, debug, f"Generated {len(neighbors)} neighbors")
        self._debug_print(debug_tag, debug, "check if len(neighbors) == 2**self.n:", len(neighbors) == 2**self.n)
        return neighbors

    def _select_topk_products(self, p_star, debug=False, debug_tag="select_topk_products"):
        """
        基于需求变化选择top-k个产品
        """
        # if self.p_star_prev is None:
        #     # 如果没有上一轮结果，随机选择k个
        #     topk_indices = list(range(min(self.topk, self.n)))
        #     self._debug_print(debug_tag, debug, f"No previous p_star, selecting first {len(topk_indices)} products")
        #     return topk_indices
        
        # 计算当前和上一轮的需求
        demand_prev = NPlusOneLP.mnl_demand(self.p_star_prev, self.d_attract, self.mu, self.u0, self.gamma)
        demand_curr = NPlusOneLP.mnl_demand(p_star, self.d_attract, self.mu, self.u0, self.gamma)
        
        # 计算需求变化
        demand_changes = np.abs(demand_curr - demand_prev)
          # 选择变化最大的top-k个产品
        topk_indices = np.argsort(demand_changes)[-self.topk:].tolist()
        
        self._debug_print(debug_tag, debug, "demand_prev:", demand_prev)
        self._debug_print(debug_tag, debug, "demand_curr:", demand_curr)
        self._debug_print(debug_tag, debug, "demand_changes:", demand_changes)
        self._debug_print(debug_tag, debug, f"Selected top-{self.topk} indices:", topk_indices)
        
        return topk_indices

    def _get_topk_neighbors(self, p_star, topk_indices, debug=False, debug_tag="get_topk_neighbors"):
        """
        为top-k个产品生成所有2^k个价格组合
        """
        neighbors = []
        
        # 为所有产品找到上下界
        bounds = []
        for i in range(self.n):
            prices = self.price_grid[i]
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            bounds.append((lower, upper))
          # 生成2^k个组合，只变化top-k个产品的价格
        for i in range(2**self.topk):
            neighbor = []
            for j in range(self.n):
                if j in topk_indices:
                    # 找到在topk_indices中的位置
                    k_idx = topk_indices.index(j)
                    if (i >> k_idx) & 1:
                        neighbor.append(bounds[j][1])  # 上界
                    else:
                        neighbor.append(bounds[j][0])  # 下界
                else:
                    # 不在top-k中的产品使用下界
                    neighbor.append(bounds[j][0])
            neighbors.append(neighbor)
        
        self._debug_print(debug_tag, debug, f"Generated {len(neighbors)} topk neighbors for indices {topk_indices}")
        self._debug_print(debug_tag, debug, "check if len(neighbors) == 2**self.topk:", len(neighbors) == 2**self.topk)
        return neighbors

    def solve_n_plus_one_lp_topk(self, neighbors, full_neighbors, debug=False, debug_tag="solve_n_plus_one_lp_topk"):
        """
        TopK版本的solve_n_plus_one_lp方法
        第一轮：对所有neighbors进行求解
        后续轮次：对neighbors进行求解，其他使用上一轮的结果
        """
        if self.iteration == 0:
            # 第一轮：对所有neighbors进行求解
            zeta_star = NPlusOneLP.solve_n_plus_one_lp(
                neighbors, self.d_attract, self.mu, self.u0, self.gamma,
                self.A, self.d, self.params.b, self.T, self.params.t, debug, debug_tag
            )
            full_zeta_star = zeta_star
            self._debug_print(debug_tag, debug, f"First iteration: solved for {len(neighbors)} neighbors")
        else:
            # 后续轮次：只对neighbors进行求解，其他使用上一轮结果
            # 首先计算已经使用上一轮结果的邻居所消耗的资源和时间
              # 构建neighbors_dict: key为tuple(neighbor), value为neighbors的idx
            neighbors_dict = {tuple(neighbor): idx for idx, neighbor in enumerate(neighbors)}
            used_resources = np.zeros(self.d)  # 已消耗的资源
            used_time = 0  # 已分配的时间
            
            for i, full_neighbor in enumerate(full_neighbors):
                full_neighbor_tuple = tuple(full_neighbor)
                if full_neighbor_tuple not in neighbors_dict:
                    # 这个邻居使用上一轮的结果
                    if self.zeta_star_prev is not None and i < len(self.zeta_star_prev):
                        zeta_prev = self.zeta_star_prev[i]
                        # 计算这个邻居的需求
                        demand_prev = NPlusOneLP.mnl_demand(full_neighbor, self.d_attract, self.mu, self.u0, self.gamma)
                        # 累计资源消耗: A' * (ζ_i * λ_i)
                        used_resources += self.A.T @ (zeta_prev * demand_prev)
                        # 累计时间消耗
                        used_time += zeta_prev
            
            # 调整约束条件
            adjusted_b = self.params.b - used_resources  # 剩余资源
            adjusted_remaining_time = (self.T - self.params.t) - used_time  # 剩余时间
            
            self._debug_print(debug_tag, debug, f"Used resources: {used_resources}")
            self._debug_print(debug_tag, debug, f"Used time: {used_time}")
            self._debug_print(debug_tag, debug, f"Adjusted b: {adjusted_b}")
            self._debug_print(debug_tag, debug, f"Adjusted remaining time: {adjusted_remaining_time}")
            
            # 确保剩余时间非负
            if adjusted_remaining_time <= 0:
                self._debug_print(debug_tag, debug, "No remaining time for optimization")
                zeta_star = np.zeros(len(neighbors))
            else:
                # 用调整后的约束条件求解neighbors的LP问题
                # 计算调整后的t值，使得T - t_adjusted = adjusted_remaining_time
                t_adjusted = self.T - adjusted_remaining_time
                zeta_star = NPlusOneLP.solve_n_plus_one_lp(
                    neighbors, self.d_attract, self.mu, self.u0, self.gamma,
                    self.A, self.d, adjusted_b, self.T, t_adjusted, debug, debug_tag + "_adjusted"
                )
              # 构建完整的zeta_star
            full_zeta_star = np.zeros(len(full_neighbors))
            
            for i, full_neighbor in enumerate(full_neighbors):
                full_neighbor_tuple = tuple(full_neighbor)
                if full_neighbor_tuple in neighbors_dict:
                    # 使用neighbors_dict中的索引来获取对应的zeta_star值
                    neighbor_idx = neighbors_dict[full_neighbor_tuple]
                    full_zeta_star[i] = zeta_star[neighbor_idx]
                else:
                    # 使用上一轮的结果
                    if self.zeta_star_prev is not None and i < len(self.zeta_star_prev):
                        full_zeta_star[i] = self.zeta_star_prev[i]
                    else:
                        full_zeta_star[i] = 0
            
            self._debug_print(debug_tag, debug, f"Iteration {self.iteration}: solved for {len(neighbors)} neighbors, filled {len(full_neighbors)} full neighbors")
        
        return full_neighbors, full_zeta_star

    def get_pricing_policy(self):
        """
        获取定价策略 (TopK version)
        """
        # 步骤1: 求解连续松弛问题
        p_star = NPlusOneLP.solve_continuous_relaxation(
            self.price_grid, self.d_attract, self.mu, self.u0, self.gamma,
            self.A, self.params.b, self.T, self.params.t, self.debug, "get_pricing_policy_step1"
        )
        self._debug_print("get_pricing_policy", self.debug, "p_star:", p_star, "shape:", np.shape(p_star))
        
        # 步骤2: 找到邻居价格向量 (TopK version)
        neighbors, full_neighbors = self.find_neighbors_topk(p_star, self.debug, "get_pricing_policy_step2")
        self._debug_print("get_pricing_policy", self.debug, "neighbors count:", len(neighbors))
        self._debug_print("get_pricing_policy", self.debug, "full_neighbors count:", len(full_neighbors))
        
        # 步骤3: 求解LP (TopK version)
        final_neighbors, zeta_star = self.solve_n_plus_one_lp_topk(
            neighbors, full_neighbors, self.debug, "get_pricing_policy_step3"
        )
        self._debug_print("get_pricing_policy", self.debug, "zeta_star:", zeta_star, "shape:", np.shape(zeta_star))
        
        # 更新历史记录
        self.p_star_prev = p_star.copy()
        self.zeta_star_prev = zeta_star.copy()
        
        return final_neighbors, zeta_star

    @staticmethod
    def _debug_print(debug_tag, debug, *args, **kwargs):
        if debug:
            print(f"[{debug_tag}]", *args, **kwargs)

    def run(self):
        """
        运行TopKLP算法，自动推进T步，每步根据get_pricing_policy选择最优邻居价格
        """
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()  # (T, m)
        b = env.params.B.copy()  # (d,)
        
        for t in range(env.params.T):
            self.iteration = t
            try:
                neighbors, zeta_star = self.get_pricing_policy()
            except Exception as e:
                print(f"Error in get_pricing_policy at time {t}: {e}")
                raise e
              # 将zeta_star映射为x_t
            x_t = NPlusOneLP.map_zeta_to_xt(neighbors, zeta_star, env.params.f)
            
            # 选择x_t最大的index为alpha
            alpha = int(np.argmax(x_t))
            price_vec = env.params.f[:, alpha]
            j = Y[t, alpha]
            _, _, done, _ = env.step(j, alpha)
            env.params.x_history.append(x_t)
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
    print("[OFFline] total reward:", sum(sim_off.params.reward_history))    # NPlusOneLP策略示例
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
    print("[NPlusOneLP] total reward:", sum(sim_nplus1.params.reward_history))    # TopKLP策略示例
    sim_topk = CustomerChoiceSimulator('params.yml', random_seed=42)
    if os.path.exists(Y_path):
        sim_topk.load_Y(Y_path)
    else:
        sim_topk.generate_Y_matrix()
        sim_topk.save_Y(Y_path)
    rabbi_topk = TopKLP(sim_topk, debug=False)
    rabbi_topk.run()
    print("[TopKLP] x_history shape:", np.array(sim_topk.params.x_history).shape)
    print("[TopKLP] x_history:", sim_topk.params.x_history)
    print("[TopKLP] alpha_history:", sim_topk.params.alpha_history)
    print("[TopKLP] j_history:", sim_topk.params.j_history)
    print("[TopKLP] b_history:", sim_topk.params.b_history)
    print("[TopKLP] reward_history:", sim_topk.params.reward_history)
    print("[TopKLP] Final inventory:", sim_topk.params.b)
    print("[TopKLP] total reward:", sum(sim_topk.params.reward_history))