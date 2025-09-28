import numpy as np
from scipy.optimize import linprog, minimize


def _format_prices_vec(vec, mask=None):
    def _fmt(x):
        xv = float(x)
        xr = round(xv)
        if abs(xv - xr) < 1e-9:
            return str(int(xr))
        s = ("{:f}".format(xv)).rstrip('0').rstrip('.')
        return s or "0"
    arr = np.asarray(vec).tolist()
    if mask is None:
        return "[" + " ".join(_fmt(v) for v in arr) + "]"
    entries = []
    for offered, value in zip(mask, arr):
        if not offered:
            entries.append("--")
        else:
            entries.append(_fmt(value))
    return "[" + " ".join(entries) + "]"

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
            if getattr(self, 'debug', False) and isinstance(self.params.x_history, list):
                self.params.x_history.append(x_t)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                sel_mask = getattr(env.params, 'offer_mask', None)
                col_mask = sel_mask[:, alpha] if sel_mask is not None else None
                print(f"[RABBI] t={t} b={b} alpha={alpha} j={j} max_x={x_t[alpha]:.4f} selected_prices={_format_prices_vec(sel_prices, col_mask)}")
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
            if getattr(self, 'debug', False) and isinstance(self.params.x_history, list):
                self.params.x_history.append(x_t)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                sel_mask = getattr(env.params, 'offer_mask', None)
                col_mask = sel_mask[:, alpha] if sel_mask is not None else None
                print(f"[OFFline] t={t} b={b} alpha={alpha} j={j} max_x={x_t[alpha]:.4f} selected_prices={_format_prices_vec(sel_prices, col_mask)}")
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
        allow_skip = getattr(env.params, 'allow_skip', None)
        if allow_skip is None:
            allow_skip = np.zeros(self.n, dtype=bool)
        self.allow_skip = np.array(allow_skip, dtype=bool)

    @staticmethod
    def mnl_demand(prices, d, mu, u0=0, gamma=1.0, offer_mask=None):
        prices = np.array(prices, dtype=float)
        if offer_mask is None:
            offer_mask = np.ones_like(prices, dtype=bool)
        else:
            offer_mask = np.array(offer_mask, dtype=bool)
        exponents = np.zeros_like(prices, dtype=float)
        active = offer_mask
        if np.any(active):
            exponents[active] = np.exp((np.array(d)[active] - prices[active]) / mu)
        denominator = np.sum(exponents[active]) + np.exp(u0 / mu)
        if denominator == 0:
            return np.zeros_like(prices, dtype=float)
        probabilities = np.zeros_like(prices, dtype=float)
        probabilities[active] = gamma * exponents[active] / denominator
        return probabilities

    @staticmethod
    def _debug_print(debug_tag, debug, *args, **kwargs):
        if debug:
            print(f"[{debug_tag}]", *args, **kwargs)

    @staticmethod
    def _neighbor_key(prices, mask):
        prices = np.asarray(prices, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        rounded = tuple(np.round(prices.tolist(), 9))
        mask_tuple = tuple(bool(x) for x in mask.tolist())
        return rounded, mask_tuple

    def _expand_with_skips(self, base_prices):
        neighbors = []
        seen = set()

        base_prices = np.asarray(base_prices, dtype=float)

        def recurse(idx, price_acc, mask_acc):
            if idx == self.n:
                prices_arr = np.array(price_acc, dtype=float)
                mask_arr = np.array(mask_acc, dtype=bool)
                key = self._neighbor_key(prices_arr, mask_arr)
                if key not in seen:
                    seen.add(key)
                    neighbors.append((prices_arr, mask_arr))
                return

            # Offered option
            price_acc.append(base_prices[idx])
            mask_acc.append(True)
            recurse(idx + 1, price_acc, mask_acc)
            price_acc.pop()
            mask_acc.pop()

            # Skip option if allowed
            if self.allow_skip[idx]:
                price_acc.append(0.0)
                mask_acc.append(False)
                recurse(idx + 1, price_acc, mask_acc)
                price_acc.pop()
                mask_acc.pop()

        recurse(0, [], [])
        return neighbors

    @staticmethod
    def _dedupe_neighbors(neighbors):
        deduped = []
        seen = set()
        for prices, mask in neighbors:
            key = NPlusOneLP._neighbor_key(prices, mask)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((np.array(prices, dtype=float), np.array(mask, dtype=bool)))
        return deduped

    @staticmethod
    def solve_continuous_relaxation(price_grid, d_attract, mu, u0, gamma, A, b, T, t, allow_skip=None, debug=False, debug_tag="solve_continuous_relaxation"):
        if allow_skip is None:
            allow_skip = np.zeros(len(price_grid), dtype=bool)
        allow_skip = np.array(allow_skip, dtype=bool)

        bounds = []
        x0 = []
        for idx, prices in enumerate(price_grid):
            prices = np.asarray(prices, dtype=float)
            lower = float(np.min(prices))
            upper = float(np.max(prices))
            span = max(upper - lower, 1.0)
            if allow_skip[idx]:
                upper += max(10.0, span * 5.0)
            bounds.append((lower, upper))
            x0.append(float(np.clip(np.mean(prices), lower, upper)))

        def objective(p):
            demand = NPlusOneLP.mnl_demand(p, d_attract, mu, u0, gamma)
            return -np.dot(p, demand)

        def constraint(p):
            demand = NPlusOneLP.mnl_demand(p, d_attract, mu, u0, gamma)
            return b/(T-t) - A.T @ demand
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

    def find_neighbors(self, p_star, price_grid, n, debug=False, debug_tag="find_neighbors"):
        lower_bounds = []
        upper_bounds = []
        for i in range(n):
            prices = price_grid[i]
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        relative_positions = self.get_relative_position(p_star, price_grid, n)
        sorted_indices = sorted(range(n), key=lambda i: relative_positions[i], reverse=True)

        base_vectors = []
        base_vectors.append(lower_bounds.copy())
        for i in range(1, n + 1):
            neighbor = lower_bounds.copy()
            for j in range(i):
                idx = sorted_indices[j]
                neighbor[idx] = upper_bounds[idx]
            base_vectors.append(neighbor)

        candidates = []
        for base in base_vectors:
            candidates.extend(self._expand_with_skips(base))

        neighbors = self._dedupe_neighbors(candidates)
        self._debug_print(debug_tag, debug, f"generated {len(neighbors)} neighbors")
        return neighbors

    @staticmethod
    def solve_n_plus_one_lp(neighbors, d_attract, mu, u0, gamma, A, d, b, T, t, debug=False, debug_tag="solve_n_plus_one_lp"):
        num_neighbors = len(neighbors)
        if num_neighbors == 0:
            return np.zeros(0)

        demands = []
        revenues = []
        for prices, mask in neighbors:
            demand = NPlusOneLP.mnl_demand(prices, d_attract, mu, u0, gamma, offer_mask=mask)
            demands.append(demand)
            revenues.append(float(np.dot(prices, demand)))

        c = -np.array(revenues)
        A_ub = np.zeros((d, num_neighbors))
        for k in range(d):
            for idx, demand in enumerate(demands):
                A_ub[k, idx] = np.dot(A[:, k], demand)

        A_eq = np.ones((1, num_neighbors))
        b_eq = [T - t]
        bounds = [(0, T - t)] * num_neighbors

        res = linprog(c, A_ub=A_ub, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method='highs', options={'maxiter': 10000})
        if not res.success:
            raise ValueError(f"(N+1) LP求解失败: {res.message}")
        return res.x

    def get_pricing_policy(self):
        p_star = self.solve_continuous_relaxation(
            self.price_grid, self.d_attract, self.mu, self.u0, self.gamma,
            self.A, self.params.b, self.T, self.params.t, self.allow_skip,
            self.debug, "get_pricing_policy_step1"
        )
        neighbors = self.find_neighbors(p_star, self.price_grid, self.n, self.debug, "get_pricing_policy_step2")
        zeta_star = self.solve_n_plus_one_lp(
            neighbors, self.d_attract, self.mu, self.u0, self.gamma,
            self.A, self.d, self.params.b, self.T, self.params.t, self.debug, "get_pricing_policy_step3"
        )
        return neighbors, zeta_star

    @staticmethod
    def map_zeta_to_xt(neighbors, zeta_star, env):
        f_matrix = env.params.f
        offer_mask_matrix = getattr(env.params, 'offer_mask', None)
        if offer_mask_matrix is None:
            offer_mask_matrix = np.ones_like(f_matrix, dtype=bool)

        m = f_matrix.shape[1]
        x_t = np.zeros(m)

        for neighbor_idx, (prices, mask) in enumerate(neighbors):
            for col in range(m):
                col_mask = offer_mask_matrix[:, col]
                if not np.array_equal(col_mask, mask):
                    continue
                if np.allclose(f_matrix[mask, col], prices[mask]):
                    x_t[col] = zeta_star[neighbor_idx]
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
            x_t = self.map_zeta_to_xt(neighbors, zeta_star, env)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                sel_mask_matrix = getattr(env.params, 'offer_mask', None)
                col_mask = sel_mask_matrix[:, alpha] if sel_mask_matrix is not None else None
                print(f"[NPlusOneLP] t={t} b={b} alpha={alpha} j={j} zeta_sum={zeta_star.sum():.4f} selected_prices={_format_prices_vec(sel_prices, col_mask)}")
            env.step(j, alpha)
            if getattr(self, 'debug', False) and isinstance(self.params.x_history, list):
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
        allow_skip = getattr(env.params, 'allow_skip', None)
        if allow_skip is None:
            allow_skip = np.zeros(self.n, dtype=bool)
        self.allow_skip = np.array(allow_skip, dtype=bool)

    def find_neighbors_topk(self, p_star, debug=False, debug_tag="find_neighbors_topk"):
        if self.iteration == 0:
            neighbors = self._get_all_neighbors(p_star, debug, debug_tag)
            full_neighbors = neighbors
        else:
            topk_indices = self._select_topk_products(p_star, debug, debug_tag)
            neighbors = self._get_topk_neighbors(p_star, topk_indices, debug, debug_tag)
            full_neighbors = self._get_all_neighbors(p_star, debug, debug_tag)
        return neighbors, full_neighbors

    def _expand_with_skips(self, base_prices):
        neighbors = []
        seen = set()
        base_prices = np.asarray(base_prices, dtype=float)

        def recurse(idx, price_acc, mask_acc):
            if idx == self.n:
                prices_arr = np.array(price_acc, dtype=float)
                mask_arr = np.array(mask_acc, dtype=bool)
                key = NPlusOneLP._neighbor_key(prices_arr, mask_arr)
                if key not in seen:
                    seen.add(key)
                    neighbors.append((prices_arr, mask_arr))
                return

            price_acc.append(base_prices[idx])
            mask_acc.append(True)
            recurse(idx + 1, price_acc, mask_acc)
            price_acc.pop()
            mask_acc.pop()

            if self.allow_skip[idx]:
                price_acc.append(0.0)
                mask_acc.append(False)
                recurse(idx + 1, price_acc, mask_acc)
                price_acc.pop()
                mask_acc.pop()

        recurse(0, [], [])
        return neighbors

    @staticmethod
    def _dedupe_neighbors(neighbors):
        deduped = []
        seen = set()
        for prices, mask in neighbors:
            key = NPlusOneLP._neighbor_key(prices, mask)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((np.array(prices, dtype=float), np.array(mask, dtype=bool)))
        return deduped

    def _get_all_neighbors(self, p_star, debug=False, debug_tag="get_all_neighbors"):
        lower_bounds = []
        upper_bounds = []
        for i in range(self.n):
            prices = self.price_grid[i]
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        candidates = []
        for mask_bits in range(2 ** self.n):
            base = []
            for j in range(self.n):
                if (mask_bits >> j) & 1:
                    base.append(upper_bounds[j])
                else:
                    base.append(lower_bounds[j])
            candidates.extend(self._expand_with_skips(base))

        neighbors = self._dedupe_neighbors(candidates)
        self._debug_print(debug_tag, debug, f"all neighbors={len(neighbors)}")
        return neighbors

    def _select_topk_products(self, p_star, debug=False, debug_tag="select_topk_products"):
        mask_all = np.ones(self.n, dtype=bool)
        demand_prev = NPlusOneLP.mnl_demand(self.p_star_prev, self.d_attract, self.mu, self.u0, self.gamma, mask_all)
        demand_curr = NPlusOneLP.mnl_demand(p_star, self.d_attract, self.mu, self.u0, self.gamma, mask_all)
        demand_changes = np.abs(demand_curr - demand_prev)
        topk_indices = np.argsort(demand_changes)[-self.topk:].tolist()
        return topk_indices

    def _get_topk_neighbors(self, p_star, topk_indices, debug=False, debug_tag="get_topk_neighbors"):
        bounds = []
        for i in range(self.n):
            prices = self.price_grid[i]
            lower = max([p for p in prices if p <= p_star[i]], default=min(prices))
            upper = min([p for p in prices if p >= p_star[i]], default=max(prices))
            bounds.append((lower, upper))
        candidates = []
        topk_len = len(topk_indices)
        index_pos = {idx: pos for pos, idx in enumerate(topk_indices)}
        for mask_bits in range(2 ** topk_len):
            base = []
            for j in range(self.n):
                if j in index_pos:
                    pos = index_pos[j]
                    if (mask_bits >> pos) & 1:
                        base.append(bounds[j][1])
                    else:
                        base.append(bounds[j][0])
                else:
                    base.append(bounds[j][0])
            candidates.extend(self._expand_with_skips(base))

        neighbors = self._dedupe_neighbors(candidates)
        self._debug_print(debug_tag, debug, f"topk neighbors={len(neighbors)}")
        return neighbors

    def solve_n_plus_one_lp_topk(self, neighbors, full_neighbors, debug=False, debug_tag="solve_n_plus_one_lp_topk"):
        if self.iteration == 0:
            zeta_star = NPlusOneLP.solve_n_plus_one_lp(
                neighbors, self.d_attract, self.mu, self.u0, self.gamma,
                self.A, self.d, self.params.b, self.T, self.params.t, debug, debug_tag
            )
            full_zeta_star = zeta_star
        else:
            neighbors_dict = {NPlusOneLP._neighbor_key(prices, mask): idx for idx, (prices, mask) in enumerate(neighbors)}
            used_resources = np.zeros(self.d)
            used_time = 0
            for i, (prices_full, mask_full) in enumerate(full_neighbors):
                key = NPlusOneLP._neighbor_key(prices_full, mask_full)
                if key not in neighbors_dict:
                    if self.zeta_star_prev is not None and i < len(self.zeta_star_prev):
                        zeta_prev = self.zeta_star_prev[i]
                        demand_prev = NPlusOneLP.mnl_demand(prices_full, self.d_attract, self.mu, self.u0, self.gamma, mask_full)
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
            for i, (prices_full, mask_full) in enumerate(full_neighbors):
                key = NPlusOneLP._neighbor_key(prices_full, mask_full)
                if key in neighbors_dict:
                    neighbor_idx = neighbors_dict[key]
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
            self.A, self.params.b, self.T, self.params.t, self.allow_skip,
            self.debug, "get_pricing_policy_step1"
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
            x_t = NPlusOneLP.map_zeta_to_xt(neighbors, zeta_star, env)
            alpha = int(np.argmax(x_t))
            j = Y[t, alpha]
            if getattr(self, 'debug', False):
                sel_prices = env.params.f[:, alpha]
                sel_mask_matrix = getattr(env.params, 'offer_mask', None)
                col_mask = sel_mask_matrix[:, alpha] if sel_mask_matrix is not None else None
                print(f"[TopKLP] t={t} b={b} alpha={alpha} j={j} zeta_sum={zeta_star.sum():.4f} selected_prices={_format_prices_vec(sel_prices, col_mask)}")
            env.step(j, alpha)
            if getattr(self, 'debug', False) and isinstance(env.params.x_history, list):
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
        # locate existing all-null assortment column if available (all products skipped)
        self.null_alpha_idx = self._locate_null_assortment()

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
                      bounds=bounds, method='highs', options={'maxiter': 30000})
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

    def _locate_null_assortment(self) -> int | None:
        """Locate an existing column where no products are offered (all skipped)."""
        offer_mask = getattr(self.env.params, 'offer_mask', None)
        if offer_mask is None:
            return None
        null_cols = np.where(~offer_mask.any(axis=0))[0]
        if null_cols.size > 0:
            return int(null_cols[0])
        return None

    def run(self):
        """New-added method: main loop of Robust-RABBI, per Robust-RABBI.md.
        Steps per t: Feasibility filter → compute coefficients → Column generation
        (dual separation + restricted primal) → choose menu by max score → simulate and update.
    When debug is enabled, maintains x_history with full dimension m (fill 0 for non-selected columns).
        """
        env = self.env
        env.reset()
        Y = env.params.Y if env.params.Y is not None else env.generate_Y_matrix()
        b = env.params.B.copy()
        if getattr(self, 'debug', False):
            print(f"[Robust] start T={env.params.T} n={env.params.n} m={env.params.m} d={env.params.d} topk={self.topk}")
        self.params.A_prime_size_history = []  # new-add: record |A'_t| for each time t

        current_t: int | None = None
        current_b_snapshot: np.ndarray | None = None
        current_A_prime_size: int | None = None
        try:
            original_m = self.env.params.m
            for t in range(env.params.T):
                current_t = t
                current_b_snapshot = b.copy()
                current_A_prime_size = 0
                # Step 1 & 2: feasibility-aware coefficients
                r_all, C_all = self._filtered_coefficients(b)  # math: r_α, C_{k,α}

                null_alpha = self.null_alpha_idx
                synthetic_null = False
                if null_alpha is None:
                    # fallback: ensure feasibility by adding synthetic null column
                    null_alpha = r_all.shape[0]
                    r_all = np.append(r_all, 0.0)
                    C_all = np.concatenate([C_all, np.zeros((C_all.shape[0], 1))], axis=1)
                    synthetic_null = True

                # Step 3: dual-guided column generation
                A_prime: list[int] = []  # new-add: discovered columns set 𝒜'_t
                # initialize with best revenue column to seed
                if original_m > 0:
                    seed_alpha = int(np.argmax(r_all[:original_m]))
                    A_prime.append(seed_alpha)
                if null_alpha is not None and null_alpha not in A_prime:
                    A_prime.append(null_alpha)
                current_A_prime_size = len(A_prime)

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
                        if idx == null_alpha and not synthetic_null:
                            # keep existing null assortment but avoid re-adding
                            continue
                        if idx == null_alpha and synthetic_null and idx in A_prime:
                            continue
                        if viol[idx] <= self.eta:
                            break
                        if idx not in A_prime:
                            A_prime.append(int(idx))
                            added += 1
                            current_A_prime_size = len(A_prime)
                            if added >= self.topk:
                                break
                    # guard
                    if added == 0:
                        break
                    # optional max iterations to avoid pathological loops
                    if iter_cnt > 5000:
                        break

                current_A_prime_size = len(A_prime)
                self.params.A_prime_size_history.append(len(A_prime))  # new-add: store discovered column count |A'_t|
                # Step 4: solve restricted primal to get scores x on A'
                x_sub = self._solve_restricted_primal(A_prime, b, r_all, C_all, t)  # math: x^{(t)}
                # expand to full m with zeros as required
                x_full = np.zeros(self.env.params.m)
                for i, alpha in enumerate(A_prime):
                    if alpha < original_m:
                        x_full[alpha] = x_sub[i]
                    elif synthetic_null and alpha == null_alpha:
                        # synthetic null column is outside original m; skip assignment
                        continue
                if getattr(self, 'debug', False) and isinstance(env.params.x_history, list):
                    env.params.x_history.append(x_full)

                # Step 5: choose menu with largest score among feasible ones
                alpha = int(np.argmax(x_full))  # math: α_t ∈ argmax score(α)
                j = Y[t, alpha]
                if getattr(self, 'debug', False):
                    sel_prices = env.params.f[:, alpha]
                    sel_mask_matrix = getattr(env.params, 'offer_mask', None)
                    col_mask = sel_mask_matrix[:, alpha] if sel_mask_matrix is not None else None
                    print(f"[Robust] t={t} b={b} alpha={alpha} j={j} max_x={x_full[alpha]:.4f} selected_prices={_format_prices_vec(sel_prices, col_mask)}")

                # Step 6: simulate and update
                env.step(j, alpha)
                b = env.params.b.copy()
        except Exception:
            t_repr = current_t if current_t is not None else "unknown"
            if current_b_snapshot is None:
                b_repr = "unknown"
            else:
                b_repr = np.array2string(current_b_snapshot, precision=6, separator=', ')
            a_repr = current_A_prime_size if current_A_prime_size is not None else "unknown"
            print(f"[Robust:error_state] t={t_repr} b={b_repr} A_prime_size={a_repr}", flush=True)
            raise
