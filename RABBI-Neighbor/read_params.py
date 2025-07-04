import yaml
import numpy as np
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
        self.n = None  # number of products
        self.d = None  # number of resources
        self.A = None  # resource matrix (n, d)
        self.f_split = None  # price set
        self.T = None  # horizon T
        self.B = None  # budget (d,)
        self.k = None  # scaling list
        self.f = None  # price matrix (n, m)
        self.m = None  # number of price sets
        self.demand_model = None  # demand model type
        self.tolerance = None  # tolerance for demand calculation
        self.mnl = None  # MNL parameters
        self.linear = None  # Linear parameters
        self.p = None  # demand matrix (n, m)
        
        # Environment state and history
        self.b = None  # current budget/inventory
        self.t = None  # current time step
        self.reward_history = []  # reward history
        self.b_history = []  # budget/inventory history
        self.j_history = []  # product choice history
        self.alpha_history = []  # price set choice history
        
        # Customer simulation matrices
        self.Y = None  # (T, m) customer choice matrix
        self.Q = None  # (T, n, m) offline Q matrix

class ParamsLoader:
    def __init__(self, yaml_path):
        self.params = Parameters()
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_params = yaml.safe_load(f)
        
        # Initialize basic parameters
        self.params.n = int(yaml_params['product_number']) # number of products
        self.params.d = int(yaml_params['resource_number']) # number of resources
        self.params.A = np.array(yaml_params['resource_matrix'], dtype=float) # resource (n, d)
        self.params.f_split = yaml_params['price_set_matrix'] # price set
        self.params.T = int(yaml_params['horizon']) # horizon T
        self.params.B = np.array(yaml_params['budget'], dtype=float) # budget (d,)
        self.params.k = np.array(yaml_params['scaling_list'], dtype=float) # scaling list 
        self.params.topk = int(yaml_params.get('topk', 9999))  # TopKLP的top-k参数
        self.params.f = self.generate_price_combinations(self.params.f_split) # (n, m) price matrix
        self.params.m = self.params.f.shape[1]  # 价格集数量

        # check dimensions
        self.check_A_matrix()
        if self.params.f_split.__len__() != self.params.n:
            raise ValueError(f"价格集矩阵f_split的行数({self.params.f_split.shape[0]})应与产品数量n({self.params.n})一致")
        if self.params.B.shape[0] != self.params.d:
            raise ValueError(f"预算B的长度({self.params.B.shape[0]})应与资源数量d({self.params.d})一致")
        
        self.params.demand_model = yaml_params.get('demand_model', 'MNL')  # 默认使用MNL模型
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
        
        # 自动读取需求参数并计算self.params.p (dim=n, m)
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
        f_matrix = f_matrix.transpose()  # 转置为 (n, m) 形状
        return f_matrix

    def check_A_matrix(self):
        """
        检查A矩阵是否合规：
        - 维度为(n, d)
        - 所有元素为非负数
        - 所有元素为数值类型
        """
        if self.params.A.shape != (self.params.n, self.params.d):
            raise ValueError(f"A矩阵维度应为({self.params.n}, {self.params.d})，实际为{self.params.A.shape}")
        if not np.issubdtype(self.params.A.dtype, np.number):
            raise ValueError("A矩阵元素必须为数值类型")
        if np.any(self.params.A < 0):
            raise ValueError("A矩阵所有元素必须为非负数")
    
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
    def linear_demand(prices, psi, theta):
        """
        计算线性需求
        :param prices: 价格向量 [p1, p2, ..., pN]
        :param psi: 截距向量 [ψ1, ψ2, ..., ψN]
        :param theta: 敏感度矩阵 (N x N, 对角为θ_nn)
        :return: 需求向量 [λ1, λ2, ..., λN]
        """
        return np.array(psi) + np.dot(theta, np.array(prices))

    def compute_mnl_demand_matrix(self, d, mu, u0=0, tolerance=1e-4, gamma=1.0):
        """
        计算MNL模型下的需求概率矩阵 self.params.p (n, m)
        :param d: 产品吸引力向量
        :param mu: 理性参数
        :param u0: 不购买的效用
        :param tolerance: 小于该值的概率置为0
        :param gamma: 概率缩放系数
        """
        n, m = self.params.n, self.params.m
        p_matrix = np.zeros((n, m))
        for j in range(m):
            prices = self.params.f[:, j]
            p_matrix[:, j] = self.mnl_demand(prices, d, mu, u0, gamma)
        p_matrix[np.abs(p_matrix) < tolerance] = 0
        self.params.p = p_matrix
        return p_matrix

    def compute_linear_demand_matrix(self, psi, theta, tolerance=1e-4):
        """
        计算线性模型下的需求概率矩阵 self.params.p (n, m)
        :param psi: 截距向量
        :param theta: 敏感度矩阵
        :param tolerance: 小于该值的概率置为0
        """
        n, m = self.params.n, self.params.m
        p_matrix = np.zeros((n, m))
        for j in range(m):
            prices = self.params.f[:, j]
            p_matrix[:, j] = self.linear_demand(prices, psi, theta)
        p_matrix[np.abs(p_matrix) < tolerance] = 0
        self.params.p = p_matrix
        return p_matrix

    def compute_demand_matrix(self, **kwargs):
        """
        根据self.params.demand_model选择的模型，读取对应参数，对self.params.f每一列计算需求，输出self.params.p (n, m)
        kwargs: 传递给需求模型的参数，如mu, d, psi, theta, u0, tolerance, gamma等
        """
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

# 示例用法
if __name__ == "__main__":
    yaml_path = 'params.yml'  # 替换为实际的 YAML 文件路径
    loader = ParamsLoader(yaml_path)
    print("n:", loader.params.n)
    print("d:", loader.params.d)
    print("A:", loader.params.A)
    print("f shape:", loader.params.f.shape)
    print("T:", loader.params.T)
    print("B:", loader.params.B)
    print("k:", loader.params.k)
    print("m:", loader.params.m)
    print("p (demand matrix) shape:", loader.params.p.shape)
    print("p (demand matrix):", loader.params.p)
    print("f (price matrix):", loader.params.f)
    print("Demand model:", loader.params.demand_model)

