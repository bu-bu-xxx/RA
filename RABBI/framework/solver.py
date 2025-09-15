import numpy as np
from scipy.optimize import linprog, minimize


class LPBasedPolicy:
    def __init__(self, env):
        self.env = env
        self.params = self.env.params
        self.reset()

    def reset(self):
        self.env.reset()
        self.params.x_history = []

    @staticmethod
    def solve_lp(b, p, t, n, m, d, f, A_mat, T):
        c = -np.sum(f * p, axis=0)
        A_ub = np.zeros((d, m))
        for i in range(d):
            for alpha in range(m):
                A_ub[i, alpha] = np.sum(A_mat[:, i] * p[:, alpha])
        b_ub = b
        A_eq = np.ones((1, m))
        b_eq = [T - t]
        bounds = [(0, T - t) for _ in range(m)]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'maxiter': 10000})
        if res.success:
            return res.x
        else:
            raise RuntimeError("LP solver failed: " + res.message)


class RABBI(LPBasedPolicy):
    def __init__(self, env, debug=False):
        super().__init__(env)

    def run(self):
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()
        b = env.params.B.copy()
        for t in range(env.params.T):
            p_t = env.params.p
            x_t = self.solve_lp(b, p_t, t, env.params.n, env.params.m, env.params.d, env.params.f, env.params.A, env.params.T)
            self.params.x_history.append(x_t)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            env.step(j, alpha)
            b = env.params.b.copy()


class OFFline(LPBasedPolicy):
    def __init__(self, env, debug=False):
        super().__init__(env)

    def run(self):
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()
        b = env.params.B.copy()
        for t in range(env.params.T):
            p_t = env.params.Q[t, :, :]
            x_t = self.solve_lp(b, p_t, t, env.params.n, env.params.m, env.params.d, env.params.f, env.params.A, env.params.T)
            self.params.x_history.append(x_t)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            env.step(j, alpha)
            b = env.params.b.copy()


class NPlusOneLP(LPBasedPolicy):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.n = env.params.n
        self.d = env.params.d
        self.b = env.params.b
        self.price_grid = env.params.f_split
        self.d_attract = env.params.mnl.d
        self.mu = env.params.mnl.mu
        self.u0 = env.params.mnl.u0
        self.gamma = env.params.mnl.gamma
        self.debug = debug
        self.A = env.params.A
        self.T = env.params.T
        self.env = env

    @staticmethod
    def mnl_demand(prices, d, mu, u0=0, gamma=1.0):
        exponents = np.exp((np.array(d) - np.array(prices)) / mu)
        denominator = np.sum(exponents) + np.exp(u0 / mu)
        return gamma * exponents / denominator

    @staticmethod
    def _debug_print(debug_tag, debug, *args, **kwargs):
        if debug:
            print(f"[{debug_tag}]", *args, **kwargs)

    @staticmethod
    def solve_continuous_relaxation(price_grid, d_attract, mu, u0, gamma, A, b, T, t, debug=False, debug_tag="solve_continuous_relaxation"):
        def objective(p):
            demand = NPlusOneLP.mnl_demand(p, d_attract, mu, u0, gamma)
            return -np.dot(p, demand)
        def constraint(p):
            demand = NPlusOneLP.mnl_demand(p, d_attract, mu, u0, gamma)
            return b/(T-t) - A.T @ demand
        bounds = [(min(prices), max(prices)) for prices in price_grid]
        x0 = [np.mean(prices[:-1]) for prices in price_grid]
        cons = [{'type': 'ineq', 'fun': constraint}]
        res = minimize(objective, x0, bounds=bounds, constraints=cons, options={'maxiter': 10000})
        if not res.success:
            raise ValueError(f"连续松弛问题求解失败: {res.message}")
        return res.x

    @staticmethod
    def get_relative_position(p_star, price_grid, n):
        relative_positions = []
        for i in range(n):
            prices = price_grid[i]
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            if upper == lower:
                relative_pos = 0.0
            else:
                relative_pos = (p_star[i] - lower) / (upper - lower)
            relative_positions.append(relative_pos)
        return relative_positions

    @staticmethod
    def find_neighbors(p_star, price_grid, n, debug=False, debug_tag="find_neighbors"):
        lower_bounds = []
        upper_bounds = []
        for i in range(n):
            prices = price_grid[i]
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        relative_positions = NPlusOneLP.get_relative_position(p_star, price_grid, n)
        sorted_indices = sorted(range(n), key=lambda i: relative_positions[i], reverse=True)
        neighbors = []
        base_neighbor = lower_bounds.copy()
        neighbors.append(base_neighbor)
        for i in range(1, n+1):
            neighbor = lower_bounds.copy()
            for j in range(i):
                idx = sorted_indices[j]
                neighbor[idx] = upper_bounds[idx]
            neighbors.append(neighbor)
        unique_neighbors = []
        seen = set()
        for neighbor in neighbors:
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple not in seen:
                seen.add(neighbor_tuple)
                unique_neighbors.append(neighbor)
        return unique_neighbors

    @staticmethod
    def solve_n_plus_one_lp(neighbors, d_attract, mu, u0, gamma, A, d, b, T, t, debug=False, debug_tag="solve_n_plus_one_lp"):
        num_neighbors = len(neighbors)
        demands = []
        revenues = []
        for neighbor in neighbors:
            demand = NPlusOneLP.mnl_demand(neighbor, d_attract, mu, u0, gamma)
            demands.append(demand)
            revenues.append(np.dot(neighbor, demand))
        c = -np.array(revenues)
        A_ub = []
        for k in range(d):
            row = []
            for i in range(num_neighbors):
                resource_consumption = np.dot(A[:, k], demands[i])
                row.append(resource_consumption)
            A_ub.append(row)
        A_eq = [np.ones(num_neighbors)]
        b_eq = [T - t]
        bounds = [(0, T - t)] * num_neighbors
        res = linprog(c, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'maxiter': 10000})
        if not res.success:
            raise ValueError(f"(N+1) LP求解失败: {res.message}")
        return res.x

    def get_pricing_policy(self):
        p_star = self.solve_continuous_relaxation(
            self.price_grid, self.d_attract, self.mu, self.u0, self.gamma,
            self.A, self.params.b, self.T, self.params.t, self.debug, "get_pricing_policy_step1"
        )
        neighbors = self.find_neighbors(p_star, self.price_grid, self.n, self.debug, "get_pricing_policy_step2")
        zeta_star = self.solve_n_plus_one_lp(
            neighbors, self.d_attract, self.mu, self.u0, self.gamma,
            self.A, self.d, self.params.b, self.T, self.params.t, self.debug, "get_pricing_policy_step3"
        )
        return neighbors, zeta_star

    @staticmethod
    def map_zeta_to_xt(neighbors, zeta_star, env_f):
        m = env_f.shape[1]
        x_t = np.zeros(m)
        for neighbor_idx, neighbor in enumerate(neighbors):
            for idx in range(m):
                if np.allclose(env_f[:, idx], neighbor):
                    x_t[idx] = zeta_star[neighbor_idx]
                    break
        return x_t

    def run(self):
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()
        b = env.params.B.copy()
        for t in range(env.params.T):
            neighbors, zeta_star = self.get_pricing_policy()
            x_t = self.map_zeta_to_xt(neighbors, zeta_star, env.params.f)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            env.step(j, alpha)
            self.params.x_history.append(x_t)
            b = env.params.b.copy()


class TopKLP(LPBasedPolicy):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.n = env.params.n
        self.d = env.params.d
        self.b = env.params.b
        self.price_grid = env.params.f_split
        self.d_attract = env.params.mnl.d
        self.mu = env.params.mnl.mu
        self.u0 = env.params.mnl.u0
        self.gamma = env.params.mnl.gamma
        self.debug = debug
        self.A = env.params.A
        self.T = env.params.T
        self.env = env
        self.topk = env.params.topk
        if self.topk > self.n:
            self.topk = self.n
        self.p_star_prev = None
        self.zeta_star_prev = None
        self.iteration = 0

    def find_neighbors_topk(self, p_star, debug=False, debug_tag="find_neighbors_topk"):
        if self.iteration == 0:
            neighbors = self._get_all_neighbors(p_star, debug, debug_tag)
            full_neighbors = neighbors
        else:
            topk_indices = self._select_topk_products(p_star, debug, debug_tag)
            neighbors = self._get_topk_neighbors(p_star, topk_indices, debug, debug_tag)
            full_neighbors = self._get_all_neighbors(p_star, debug, debug_tag)
        return neighbors, full_neighbors

    def _get_all_neighbors(self, p_star, debug=False, debug_tag="get_all_neighbors"):
        neighbors = []
        lower_bounds = []
        upper_bounds = []
        for i in range(self.n):
            prices = self.price_grid[i]
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        for i in range(2**self.n):
            neighbor = []
            for j in range(self.n):
                if (i >> j) & 1:
                    neighbor.append(upper_bounds[j])
                else:
                    neighbor.append(lower_bounds[j])
            neighbors.append(neighbor)
        return neighbors

    def _select_topk_products(self, p_star, debug=False, debug_tag="select_topk_products"):
        demand_prev = NPlusOneLP.mnl_demand(self.p_star_prev, self.d_attract, self.mu, self.u0, self.gamma)
        demand_curr = NPlusOneLP.mnl_demand(p_star, self.d_attract, self.mu, self.u0, self.gamma)
        demand_changes = np.abs(demand_curr - demand_prev)
        topk_indices = np.argsort(demand_changes)[-self.topk:].tolist()
        return topk_indices

    def _get_topk_neighbors(self, p_star, topk_indices, debug=False, debug_tag="get_topk_neighbors"):
        neighbors = []
        bounds = []
        for i in range(self.n):
            prices = self.price_grid[i]
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            bounds.append((lower, upper))
        for i in range(2**self.topk):
            neighbor = []
            for j in range(self.n):
                if j in topk_indices:
                    k_idx = topk_indices.index(j)
                    if (i >> k_idx) & 1:
                        neighbor.append(bounds[j][1])
                    else:
                        neighbor.append(bounds[j][0])
                else:
                    neighbor.append(bounds[j][0])
            neighbors.append(neighbor)
        return neighbors

    def solve_n_plus_one_lp_topk(self, neighbors, full_neighbors, debug=False, debug_tag="solve_n_plus_one_lp_topk"):
        if self.iteration == 0:
            zeta_star = NPlusOneLP.solve_n_plus_one_lp(
                neighbors, self.d_attract, self.mu, self.u0, self.gamma,
                self.A, self.d, self.params.b, self.T, self.params.t, debug, debug_tag
            )
            full_zeta_star = zeta_star
        else:
            neighbors_dict = {tuple(neighbor): idx for idx, neighbor in enumerate(neighbors)}
            used_resources = np.zeros(self.d)
            used_time = 0
            for i, full_neighbor in enumerate(full_neighbors):
                full_neighbor_tuple = tuple(full_neighbor)
                if full_neighbor_tuple not in neighbors_dict:
                    if self.zeta_star_prev is not None and i < len(self.zeta_star_prev):
                        zeta_prev = self.zeta_star_prev[i]
                        demand_prev = NPlusOneLP.mnl_demand(full_neighbor, self.d_attract, self.mu, self.u0, self.gamma)
                        used_resources += self.A.T @ (zeta_prev * demand_prev)
                        used_time += zeta_prev
            adjusted_b = self.params.b - used_resources
            adjusted_remaining_time = (self.T - self.params.t) - used_time
            if adjusted_remaining_time <= 0:
                zeta_star = np.zeros(len(neighbors))
            else:
                t_adjusted = self.T - adjusted_remaining_time
                zeta_star = NPlusOneLP.solve_n_plus_one_lp(
                    neighbors, self.d_attract, self.mu, self.u0, self.gamma,
                    self.A, self.d, adjusted_b, self.T, t_adjusted, debug, debug_tag + "_adjusted"
                )
            full_zeta_star = np.zeros(len(full_neighbors))
            for i, full_neighbor in enumerate(full_neighbors):
                full_neighbor_tuple = tuple(full_neighbor)
                if full_neighbor_tuple in neighbors_dict:
                    neighbor_idx = neighbors_dict[full_neighbor_tuple]
                    full_zeta_star[i] = zeta_star[neighbor_idx]
                else:
                    if self.zeta_star_prev is not None and i < len(self.zeta_star_prev):
                        full_zeta_star[i] = self.zeta_star_prev[i]
                    else:
                        full_zeta_star[i] = 0
        return full_neighbors, full_zeta_star

    def get_pricing_policy(self):
        p_star = NPlusOneLP.solve_continuous_relaxation(
            self.price_grid, self.d_attract, self.mu, self.u0, self.gamma,
            self.A, self.params.b, self.T, self.params.t, self.debug, "get_pricing_policy_step1"
        )
        neighbors, full_neighbors = self.find_neighbors_topk(p_star, self.debug, "get_pricing_policy_step2")
        final_neighbors, zeta_star = self.solve_n_plus_one_lp_topk(
            neighbors, full_neighbors, self.debug, "get_pricing_policy_step3"
        )
        self.p_star_prev = p_star.copy()
        self.zeta_star_prev = zeta_star.copy()
        return final_neighbors, zeta_star

    @staticmethod
    def _debug_print(debug_tag, debug, *args, **kwargs):
        if debug:
            print(f"[{debug_tag}]", *args, **kwargs)

    def run(self):
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()
        b = env.params.B.copy()
        for t in range(env.params.T):
            self.iteration = t
            neighbors, zeta_star = self.get_pricing_policy()
            x_t = NPlusOneLP.map_zeta_to_xt(neighbors, zeta_star, env.params.f)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            env.step(j, alpha)
            env.params.x_history.append(x_t)
            b = env.params.b.copy()
