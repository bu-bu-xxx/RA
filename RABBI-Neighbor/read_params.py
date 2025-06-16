import yaml
import numpy as np
import itertools

class ParamsLoader:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        self.n = int(params['product_number']) # number of products
        self.d = int(params['resource_number']) # number of resources
        self.A = np.array(params['resource_matrix'], dtype=float) # resource (n, d)
        self.f_split = params['price_set_matrix'] # price set
        self.T = int(params['horizon']) # horizon T
        self.B = np.array(params['budget'], dtype=float) # budget (d,)
        self.k = np.array(params['scaling_list'], dtype=float) # scaling list 
        self.f = self.generate_price_combinations(self.f_split) # (n, m) price matrix
        self.m = self.f.shape[1]  # 价格集数量

        # check dimensions
        self.check_A_matrix()
        if self.f_split.__len__() != self.n:
            raise ValueError(f"价格集矩阵f_split的行数({self.f_split.shape[0]})应与产品数量n({self.n})一致")
        if self.B.shape[0] != self.d:
            raise ValueError(f"预算B的长度({self.B.shape[0]})应与资源数量d({self.d})一致")
        
        self.demand_model = params.get('demand_model', 'MNL')  # 默认使用MNL模型
        self.tolerance = float(params.get('tolerance', 1e-4))
        self.mnl = type('MNLParams', (), {})()
        self.linear = type('LinearParams', (), {})()
        if 'MNL' in params:
            self.mnl.d = np.array(params['MNL'].get('d'), dtype=float)
            self.mnl.mu = float(params['MNL'].get('mu'))
            self.mnl.u0 = float(params['MNL'].get('u0', 0))
            self.mnl.gamma = float(params['MNL'].get('gamma', 1.0))  
            if self.mnl.d.shape[0] != self.n:
                raise ValueError(f"MNL.d的长度({self.mnl.d.shape[0]})应与产品数量n({self.n})一致")
        if 'Linear' in params:
            self.linear.psi = np.array(params['Linear'].get('psi'), dtype=float)
            self.linear.theta = np.array(params['Linear'].get('theta'), dtype=float)
        # 自动读取需求参数并计算self.p (dim=n, m)
        if self.demand_model.upper() == 'MNL':
            if self.mnl.d is None or self.mnl.mu is None:
                raise ValueError('MNL模型需要在yaml中提供MNL.d和MNL.mu')
            self.p = self.compute_demand_matrix(d=self.mnl.d, mu=self.mnl.mu, u0=self.mnl.u0, tolerance=self.tolerance)
        elif self.demand_model.upper() == 'LINEAR':
            if self.linear.psi is None or self.linear.theta is None:
                raise ValueError('Linear模型需要在yaml中提供Linear.psi和Linear.theta')
            self.p = self.compute_demand_matrix(psi=self.linear.psi, theta=self.linear.theta, tolerance=self.tolerance)
        else:
            raise NotImplementedError(f'暂不支持的需求模型: {self.demand_model}')

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
        if self.A.shape != (self.n, self.d):
            raise ValueError(f"A矩阵维度应为({self.n}, {self.d})，实际为{self.A.shape}")
        if not np.issubdtype(self.A.dtype, np.number):
            raise ValueError("A矩阵元素必须为数值类型")
        if np.any(self.A < 0):
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
        计算MNL模型下的需求概率矩阵 self.p (n, m)
        :param d: 产品吸引力向量
        :param mu: 理性参数
        :param u0: 不购买的效用
        :param tolerance: 小于该值的概率置为0
        :param gamma: 概率缩放系数
        """
        n, m = self.n, self.m
        p_matrix = np.zeros((n, m))
        for j in range(m):
            prices = self.f[:, j]
            p_matrix[:, j] = self.mnl_demand(prices, d, mu, u0, gamma)
        p_matrix[np.abs(p_matrix) < tolerance] = 0
        self.p = p_matrix
        return p_matrix

    def compute_linear_demand_matrix(self, psi, theta, tolerance=1e-4):
        """
        计算线性模型下的需求概率矩阵 self.p (n, m)
        :param psi: 截距向量
        :param theta: 敏感度矩阵
        :param tolerance: 小于该值的概率置为0
        """
        n, m = self.n, self.m
        p_matrix = np.zeros((n, m))
        for j in range(m):
            prices = self.f[:, j]
            p_matrix[:, j] = self.linear_demand(prices, psi, theta)
        p_matrix[np.abs(p_matrix) < tolerance] = 0
        self.p = p_matrix
        return p_matrix

    def compute_demand_matrix(self, **kwargs):
        """
        根据self.demand_model选择的模型，读取对应参数，对self.f每一列计算需求，输出self.p (n, m)
        kwargs: 传递给需求模型的参数，如mu, d, psi, theta, u0, tolerance, gamma等
        """
        tolerance = kwargs.get('tolerance', 1e-4)
        if self.demand_model.upper() == 'MNL':
            d = kwargs.get('d')
            mu = kwargs.get('mu')
            u0 = kwargs.get('u0', 0)
            gamma = kwargs.get('gamma', 1.0)
            if d is None or mu is None:
                raise ValueError('MNL模型需要参数d(吸引力向量)和mu(理性参数)')
            return self.compute_mnl_demand_matrix(d, mu, u0, tolerance, gamma)
        elif self.demand_model.upper() == 'LINEAR':
            psi = kwargs.get('psi')
            theta = kwargs.get('theta')
            if psi is None or theta is None:
                raise ValueError('Linear模型需要参数psi(截距)和theta(敏感度矩阵)')
            return self.compute_linear_demand_matrix(psi, theta, tolerance)
        else:
            raise NotImplementedError(f'暂不支持的需求模型: {self.demand_model}')

# 示例用法
if __name__ == "__main__":
    yaml_path = 'params.yml'  # 替换为实际的 YAML 文件路径
    params = ParamsLoader(yaml_path)
    print("n:", params.n)
    print("d:", params.d)
    print("A:", params.A)
    print("f shape:", params.f.shape)
    print("T:", params.T)
    print("B:", params.B)
    print("k:", params.k)
    print("m:", params.m)
    print("p (demand matrix) shape:", params.p.shape)
    print("p (demand matrix):", params.p)
    print("f (price matrix):", params.f)
    print("Demand model:", params.demand_model)

