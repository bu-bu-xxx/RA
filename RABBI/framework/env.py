import numpy as np
import yaml
import itertools


class MNLParams:
    def __init__(self):
        self.d = None
        self.mu = None
        self.u0 = 0
        self.gamma = 1.0


class LinearParams:
    def __init__(self):
        self.psi = None
        self.theta = None


class Parameters:
    def __init__(self):
        self.n = None
        self.d = None
        self.A = None
        self.f_split = None
        self.T = None
        self.B = None
        self.k = None
        self.f = None
        self.m = None
        self.demand_model = None
        self.tolerance = None
        self.mnl = None
        self.linear = None
        self.p = None
        self.topk = None
        self.b = None
        self.t = None
        self.reward_history = []
        self.b_history = []
        self.j_history = []
        self.alpha_history = []
        self.Y = None
        self.Q = None


class ParamsLoader:
    def __init__(self, yaml_path):
        self.params = Parameters()

        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_params = yaml.safe_load(f)

        self.params.n = int(yaml_params['product_number'])
        self.params.d = int(yaml_params['resource_number'])
        self.params.A = np.array(yaml_params['resource_matrix'], dtype=float)
        self.params.f_split = yaml_params['price_set_matrix']
        self.params.T = int(yaml_params['horizon'])
        self.params.B = np.array(yaml_params['budget'], dtype=float)
        self.params.k = np.array(yaml_params['scaling_list'], dtype=float)
        self.params.topk = int(yaml_params.get('topk', 9999))
        self.params.f = self.generate_price_combinations(self.params.f_split)
        self.params.m = self.params.f.shape[1]

        self.check_A_matrix()
        if self.params.f_split.__len__() != self.params.n:
            raise ValueError(f"价格集矩阵f_split的行数({self.params.f_split.shape[0]})应与产品数量n({self.params.n})一致")
        if self.params.B.shape[0] != self.params.d:
            raise ValueError(f"预算B的长度({self.params.B.shape[0]})应与资源数量d({self.params.d})一致")

        self.params.demand_model = yaml_params.get('demand_model', 'MNL')
        self.params.tolerance = float(yaml_params.get('tolerance', 1e-4))
        self.params.mnl = MNLParams()
        self.params.linear = LinearParams()

        if 'MNL' in yaml_params:
            self.params.mnl.d = np.array(yaml_params['MNL'].get('d'), dtype=float)
            self.params.mnl.mu = float(yaml_params['MNL'].get('mu'))
            self.params.mnl.u0 = float(yaml_params['MNL'].get('u0', 0))
            self.params.mnl.gamma = float(yaml_params['MNL'].get('gamma', 1.0))
            if self.params.mnl.d.shape[0] != self.params.n:
                raise ValueError(f"MNL.d的长度({self.params.mnl.d.shape[0]})应与产品数量n({self.params.n})一致")
        if 'Linear' in yaml_params:
            self.params.linear.psi = np.array(yaml_params['Linear'].get('psi'), dtype=float)
            self.params.linear.theta = np.array(yaml_params['Linear'].get('theta'), dtype=float)

        if self.params.demand_model.upper() == 'MNL':
            if self.params.mnl.d is None or self.params.mnl.mu is None:
                raise ValueError('MNL模型需要在yaml中提供MNL.d和MNL.mu')
            self.params.p = self.compute_demand_matrix(d=self.params.mnl.d, mu=self.params.mnl.mu, u0=self.params.mnl.u0, tolerance=self.params.tolerance)
        elif self.params.demand_model.upper() == 'LINEAR':
            if self.params.linear.psi is None or self.params.linear.theta is None:
                raise ValueError('Linear模型需要在yaml中提供Linear.psi和Linear.theta')
            self.params.p = self.compute_demand_matrix(psi=self.params.linear.psi, theta=self.params.linear.theta, tolerance=self.params.tolerance)
        else:
            raise NotImplementedError(f'暂不支持的需求模型: {self.params.demand_model}')

    def generate_price_combinations(self, f_split):
        combos = list(itertools.product(*f_split))
        f_matrix = np.array(combos, dtype=float)
        f_matrix = f_matrix.transpose()
        return f_matrix

    def check_A_matrix(self):
        if self.params.A.shape != (self.params.n, self.params.d):
            raise ValueError(f"A矩阵维度应为({self.params.n}, {self.params.d})，实际为{self.params.A.shape}")
        if not np.issubdtype(self.params.A.dtype, np.number):
            raise ValueError("A矩阵元素必须为数值类型")
        if np.any(self.params.A < 0):
            raise ValueError("A矩阵所有元素必须为非负数")

    @staticmethod
    def mnl_demand(prices, d, mu, u0=0, gamma=1.0):
        exponents = np.exp((np.array(d) - np.array(prices)) / mu)
        denominator = np.sum(exponents) + np.exp(u0 / mu)
        return gamma * exponents / denominator

    @staticmethod
    def linear_demand(prices, psi, theta):
        return np.array(psi) + np.dot(theta, np.array(prices))

    def compute_mnl_demand_matrix(self, d, mu, u0=0, tolerance=1e-4, gamma=1.0):
        n, m = self.params.n, self.params.m
        p_matrix = np.zeros((n, m))
        for j in range(m):
            prices = self.params.f[:, j]
            p_matrix[:, j] = self.mnl_demand(prices, d, mu, u0, gamma)
        p_matrix[np.abs(p_matrix) < tolerance] = 0
        self.params.p = p_matrix
        return p_matrix

    def compute_linear_demand_matrix(self, psi, theta, tolerance=1e-4):
        n, m = self.params.n, self.params.m
        p_matrix = np.zeros((n, m))
        for j in range(m):
            prices = self.params.f[:, j]
            p_matrix[:, j] = self.linear_demand(prices, psi, theta)
        p_matrix[np.abs(p_matrix) < tolerance] = 0
        self.params.p = p_matrix
        return p_matrix

    def compute_demand_matrix(self, **kwargs):
        tolerance = kwargs.get('tolerance', 1e-4)
        if self.params.demand_model.upper() == 'MNL':
            d = kwargs.get('d')
            mu = kwargs.get('mu')
            u0 = kwargs.get('u0', 0)
            gamma = kwargs.get('gamma', 1.0)
            if d is None or mu is None:
                raise ValueError('MNL模型需要参数d(吸引力向量)和mu(理性参数)')
            return self.compute_mnl_demand_matrix(d, mu, u0, tolerance, gamma)
        elif self.params.demand_model.upper() == 'LINEAR':
            psi = kwargs.get('psi')
            theta = kwargs.get('theta')
            if psi is None or theta is None:
                raise ValueError('Linear模型需要参数psi(截距)和theta(敏感度矩阵)')
            return self.compute_linear_demand_matrix(psi, theta, tolerance)
        else:
            raise NotImplementedError(f'暂不支持的需求模型: {self.params.demand_model}')


class DynamicPricingEnv(ParamsLoader):
    def __init__(self, yaml_path, random_seed=None):
        super().__init__(yaml_path)
        self.random_seed = random_seed
        self.reset()
        if random_seed is not None:
            np.random.seed(random_seed)

    def reset(self):
        self.params.b = self.params.B.copy()
        self.params.t = 0
        self.params.reward_history = []
        self.params.b_history = [self.params.b.copy()]
        self.params.j_history = []
        self.params.alpha_history = []
        return self._get_obs()

    def _get_obs(self):
        return {
            'b': self.params.b.copy(),
            't': self.params.t
        }

    def step(self, j: int, alpha: int):
        reward = 0
        self.params.t += 1
        done = (self.params.t >= self.params.T)
        info = {}

        if j == -1:
            info['sold'] = False
            self.params.reward_history.append(reward)
            self.params.b_history.append(self.params.b.copy())
            self.params.j_history.append(j)
            self.params.alpha_history.append(alpha)
            done = done or np.any(self.params.b < 0)
            return self._get_obs(), reward, done, info

        if j is None:
            raise ValueError("j must be an integer representing the product index or -1.")
        if j < 0 or j >= self.params.n:
            raise ValueError(f"j must be in range [0, {self.params.n - 1}] or -1 (not buy), but got {j}.")
        if np.all(self.params.b - self.params.A[j] >= 0):
            self.params.b -= self.params.A[j]
            reward = self.params.f[j, alpha]
            info['sold'] = True
        else:
            info['sold'] = False

        self.params.reward_history.append(reward)
        self.params.b_history.append(self.params.b.copy())
        self.params.j_history.append(j)
        self.params.alpha_history.append(alpha)
        done = done or np.any(self.params.b < 0)
        return self._get_obs(), reward, done, info
