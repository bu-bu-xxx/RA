import numpy as np
from scipy.optimize import linprog, minimize


def _format_prices_vec(vec):
    def _fmt(x):
        xv = float(x)
        xr = round(xv)
        if abs(xv - xr) < 1e-9:
            return str(int(xr))
        s = ("{:f}".format(xv)).rstrip('0').rstrip('.')
        return s or "0"
    arr = np.asarray(vec).tolist()
    return "[" + " ".join(_fmt(v) for v in arr) + "]"

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
        self.debug = debug

    def run(self):
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()
        b = env.params.B.copy()
        if getattr(self, 'debug', False):
            print(f"[RABBI] start T={env.params.T} n={env.params.n} m={env.params.m} d={env.params.d}")
        for t in range(env.params.T):
            p_t = env.params.p
            x_t = self.solve_lp(b, p_t, t, env.params.n, env.params.m, env.params.d, env.params.f, env.params.A, env.params.T)
            self.params.x_history.append(x_t)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                print(f"[RABBI] t={t} b={b} alpha={alpha} j={j} max_x={x_t[alpha]:.4f} selected_prices={_format_prices_vec(sel_prices)}")
            env.step(j, alpha)
            b = env.params.b.copy()


class OFFline(LPBasedPolicy):
    def __init__(self, env, debug=False):
        super().__init__(env)
        self.debug = debug

    def run(self):
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()
        b = env.params.B.copy()
        if getattr(self, 'debug', False):
            print(f"[OFFline] start T={env.params.T} n={env.params.n} m={env.params.m} d={env.params.d}")
        for t in range(env.params.T):
            p_t = env.params.Q[t, :, :]
            x_t = self.solve_lp(b, p_t, t, env.params.n, env.params.m, env.params.d, env.params.f, env.params.A, env.params.T)
            self.params.x_history.append(x_t)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                print(f"[OFFline] t={t} b={b} alpha={alpha} j={j} max_x={x_t[alpha]:.4f} selected_prices={_format_prices_vec(sel_prices)}")
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
        if getattr(self, 'debug', False):
            print(f"[NPlusOneLP] start T={env.params.T} n={env.params.n} m={env.params.m} d={env.params.d} topk=None")
        for t in range(env.params.T):
            neighbors, zeta_star = self.get_pricing_policy()
            x_t = self.map_zeta_to_xt(neighbors, zeta_star, env.params.f)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                print(f"[NPlusOneLP] t={t} b={b} alpha={alpha} j={j} zeta_sum={zeta_star.sum():.4f} selected_prices={_format_prices_vec(sel_prices)}")
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
        if getattr(self, 'debug', False):
            print(f"[TopKLP] start T={env.params.T} n={env.params.n} m={env.params.m} d={env.params.d} topk={self.topk}")
        for t in range(env.params.T):
            self.iteration = t
            neighbors, zeta_star = self.get_pricing_policy()
            x_t = NPlusOneLP.map_zeta_to_xt(neighbors, zeta_star, env.params.f)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                print(f"[TopKLP] t={t} b={b} alpha={alpha} j={j} zeta_sum={zeta_star.sum():.4f} selected_prices={_format_prices_vec(sel_prices)}")
            env.step(j, alpha)
            env.params.x_history.append(x_t)
            b = env.params.b.copy()


class Robust(LPBasedPolicy):
    """Robust-RABBI solver (abbr. robust).
    New-added class implementing the Robust-RABBI algorithm described in Robust-RABBI.md.
    It performs per-period column generation using a dual (resource/time) pricing LP and a
    separation step over the menu lattice to build a restricted primal LP.
    """

    def __init__(self, env, debug: bool = False):
        # new-add: initialize base LP policy holder
        super().__init__(env)
        self.debug = debug  # new-add: debug flag for verbose logging
        # new-add: tolerance for separation stopping ε, η from params if present or defaults
        self.sep_eps = float(getattr(env.params, 'sep_eps', 1e-6))  # math: ε_sep > 0
        self.eta = float(getattr(env.params, 'eta', 1e-8))  # math: η > 0
        # new-add: how many violators to add each iteration (top-k)
        self.topk = min(int(getattr(env.params, 'topk', 10000000)), env.params.m)

    # --------- helpers implementing mathematics in Robust-RABBI ---------
    def _feasible_products_mask(self, b: np.ndarray) -> np.ndarray:
        """New-added function: determine which products i are instantly fulfillable under remaining budget b.
        Returns a boolean mask of shape (n,) where mask[i] is True if b_k >= A[i,k] for all k.
        Implements S4 feasibility filter (see doc).
        """
        A = self.env.params.A
        # math: CanFulfill(b, A_i) = ∧_k [ b_k ≥ A_{i,k} ]
        return np.all(A <= b[None, :], axis=1)

    def _raise_lp_failure(self, step: str, detail: str, t: int | None = None):
        """New-added function: raise RuntimeError with explicit step id when LP solving fails."""
        t_suffix = f" at t={t}" if t is not None else ""
        raise RuntimeError(f"[Robust Step {step}] LP solver{t_suffix} failed: {detail}")

    def _filtered_coefficients(self, b: np.ndarray) -> tuple:
        """New-added function: compute r_α and c_{k,α} after feasibility-aware filtering.
        Sets p_{i,α} = 0 if product i is infeasible under current b (Step 1 + S4), then computes:
          r_α = sum_i f_{i,α} p_{i,α}
          c_{k,α} = sum_i A_{i,k} p_{i,α}
        Returns (r: (m,), C: (d,m)).
        """
        params = self.env.params
        n, m, d = params.n, params.m, params.d
        p_full = params.p  # shape (n, m)
        f = params.f       # shape (n, m)
        A = params.A       # shape (n, d)
        mask = self._feasible_products_mask(b)  # math: feasibility per product i
        # math: p_{i,α} := 0 if not CanFulfill(b, A_i)
        p_eff = p_full.copy()
        p_eff[~mask, :] = 0.0
        # math: r_α = Σ_i f_{i,α} p_{i,α}
        r = np.sum(f * p_eff, axis=0)
        # math: c_{k,α} = Σ_i A_{i,k} p_{i,α}; compute for all k as C ∈ R^{d×m}
        C = np.zeros((d, m))
        for k in range(d):
            C[k, :] = np.sum(A[:, k][:, None] * p_eff, axis=0)
        return r, C

    def _solve_restricted_primal(self, columns: list, b: np.ndarray, r: np.ndarray, C: np.ndarray, t: int) -> np.ndarray:
        """New-added function: solve restricted primal LP over given columns A'_t.
        Primal:
          max Σ_{α∈A'} x_α r_α
          s.t. Σ_{α∈A'} x_α C_{k,α} ≤ b_k, ∀k
               Σ_{α∈A'} x_α = T - t
               x_α ≥ 0
        Returns x over the subset (len(columns),).
        """
        params = self.env.params
        m = len(columns)
        if m == 0:
            return np.zeros(0)
        # math: objective c = -r for maximization via linprog
        c = -r[columns]
        d = params.d
        # math: A_ub rows for resources, entries C_{k,α}
        A_ub = C[:, columns]
        b_ub = b  # math: b_k
        # math: equality Σ x_α = T - t
        A_eq = np.ones((1, m))
        b_eq = [params.T - t]
        bounds = [(0, params.T - t) for _ in range(m)]  # x_α ≥ 0
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs', options={'maxiter': 10000})
        if not res.success:
            self._raise_lp_failure("4", res.message, t)
        return res.x

    def _solve_restricted_dual(self, columns: list, r: np.ndarray, C: np.ndarray, b: np.ndarray, t: int) -> tuple:
        """New-added function: solve restricted dual (resource prices λ ≥ 0 and time price μ ≥ 0).
        Dual (specialized as in doc):
          min b^T λ + (T - t) μ
          s.t. Σ_k C_{k,α} λ_k + μ ≥ r_α, ∀ α ∈ A'
               λ ≥ 0, μ ≥ 0
        Returns (lambda: (d,), mu: float).
        """
        params = self.env.params
        d = params.d
        if len(columns) == 0:
            return np.zeros(d), 0.0
        # math: variables z = [λ_1..λ_d, μ]
        # math: objective minimize [b, (T - t)] • z
        c_obj = np.concatenate([b, [params.T - t]])
        # math: constraints - (C^T λ + μ) ≤ -r for α ∈ A'_t
        A_ub = []
        b_ub = []
        for alpha in columns:
            row = -np.concatenate([C[:, alpha], [1.0]])  # math: -[C_{·,α}; 1]
            A_ub.append(row)
            b_ub.append(-r[alpha])  # math: ≤ -r_α
        A_ub = np.asarray(A_ub)
        b_ub = np.asarray(b_ub)
        bounds = [(0, None)] * (d + 1)  # math: λ_k ≥ 0, μ ≥ 0
        # math: linprog solves the dual LP:
        #   minimize    b^T λ + (T - t) μ
        #   subject to  Σ_k C_{k,α} λ_k + μ ≥ r_α   for every α in the current column set A'_t
        #               λ ≥ 0,  μ ≥ 0
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'maxiter': 10000})
        if not res.success:
            self._raise_lp_failure("3", res.message, t)
        z = res.x
        lam = z[:d]
        mu = float(z[d])
        return lam, mu

    def _max_reduced_costs(self, r: np.ndarray, C: np.ndarray, lam: np.ndarray, mu: float) -> tuple:
        """New-added function: compute reduced costs and top violating columns.
        Reduced cost per α: \bar{c}_α(λ, μ) = r_α - Σ_k C_{k,α} λ_k - μ.
        Returns (violations: np.ndarray, order: np.ndarray sorted indices desc).
        """
        # math: compute v = r - (C^T λ + μ)
        v = r - (C.T @ lam + mu)
        order = np.argsort(-v)
        return v, order

    def run(self):
        """New-added method: main loop of Robust-RABBI, per Robust-RABBI.md.
        Steps per t: Feasibility filter → compute coefficients → Column generation
        (dual separation + restricted primal) → choose menu by max score → simulate and update.
        Maintains x_history with full dimension m (fill 0 for non-selected columns).
        """
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()
        b = env.params.B.copy()
        if getattr(self, 'debug', False):
            print(f"[Robust] start T={env.params.T} n={env.params.n} m={env.params.m} d={env.params.d} topk={self.topk}")
        self.params.A_prime_size_history = []  # new-add: record |A'_t| for each time t

        for t in range(env.params.T):
            # Step 1 & 2: feasibility-aware coefficients
            r_all, C_all = self._filtered_coefficients(b)  # math: r_α, C_{k,α}

            # Step 3: dual-guided column generation
            A_prime: list[int] = []  # new-add: discovered columns set 𝒜'_t
            # initialize with best revenue column to seed
            seed_alpha = int(np.argmax(r_all))
            A_prime.append(seed_alpha)

            iter_cnt = 0
            while True:
                iter_cnt += 1
                # solve restricted dual to get (λ, μ)
                lam, mu = self._solve_restricted_dual(A_prime, r_all, C_all, b, t)  # math: dual prices
                # separation: compute violations over all α
                viol, order = self._max_reduced_costs(r_all, C_all, lam, mu)  # math: \bar{c}_α(λ, μ)
                max_v = float(viol[order[0]]) if order.size else -np.inf
                if getattr(self, 'debug', False):
                    print(f"[Robust] t={t} iter={iter_cnt} max_violation={max_v:.6g} |A'|={len(A_prime)}")
                if max_v <= self.eta:  # math: stop if \tilde{V}(λ, μ) ≤ η
                    break
                # add top-k violating columns this iteration
                added = 0
                for idx in order:
                    if viol[idx] <= self.eta:
                        break
                    if idx not in A_prime:
                        A_prime.append(int(idx))
                        added += 1
                        if added >= self.topk:
                            break
                # guard
                if added == 0:
                    break
                # optional max iterations to avoid pathological loops
                if iter_cnt > 5000:
                    break

            self.params.A_prime_size_history.append(len(A_prime))  # new-add: store discovered column count |A'_t|
            # Step 4: solve restricted primal to get scores x on A'
            x_sub = self._solve_restricted_primal(A_prime, b, r_all, C_all, t)  # math: x^{(t)}
            # expand to full m with zeros as required
            x_full = np.zeros(self.env.params.m)
            for i, alpha in enumerate(A_prime):
                x_full[alpha] = x_sub[i]
            env.params.x_history.append(x_full)

            # Step 5: choose menu with largest score among feasible ones
            alpha = int(np.argmax(x_full))  # math: α_t ∈ argmax score(α)
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                print(f"[Robust] t={t} b={b} alpha={alpha} j={j} max_x={x_full[alpha]:.4f} selected_prices={_format_prices_vec(sel_prices)}")

            # Step 6: simulate and update
            env.step(j, alpha)
            b = env.params.b.copy()
