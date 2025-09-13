import numpy as np
import bisect
import os
from env import DynamicPricingEnv

class CustomerChoiceSimulator(DynamicPricingEnv):
    def __init__(self, yaml_path, random_seed=None):
        super().__init__(yaml_path, random_seed=random_seed)
        # Y和Q矩阵现在存储在params中
        # self.params.Y = None  # (T, m)  - 已在Parameters类中定义
        # self.params.Q = None  # (T, n, m)  - 已在Parameters类中定义

    def generate_Y_matrix(self):
        """
        返回：Y，shape=(T, m)，
        Y[t, alpha]=j表示t时刻顾客在价格集alpha下选择了产品j（0~n-1），-1表示未购买
        每个时间步和价格集，顾客可能选择一个商品或不购买
        """
        Y = np.full((self.params.T, self.params.m), -1, dtype=int)  # 默认-1表示未购买
        for t in range(self.params.T):
            for alpha in range(self.params.m):
                p_alpha = self.params.p[:, alpha]  # (n,)
                p_cumsum = np.cumsum(p_alpha)
                xi = np.random.uniform(0, 1)
                idx = bisect.bisect_left(p_cumsum, xi)
                if idx == self.params.n:  # 如果选择了最后一个位置，表示未购买
                    Y[t, alpha] = -1  # 选择不购买
                else:
                    Y[t, alpha] = idx
        self.params.Y = Y
        return Y

    def compute_offline_Q(self):
        """
        根据self.params.Y (T, m) 生成 Q 矩阵 (T, n, m)，
        Q[t, idx, alpha] = 在 t~T 时间段内，价格集 alpha 下选择产品 idx 的频率
        """
        if self.params.Y is None:
            raise ValueError("self.params.Y is None，请先生成或加载Y矩阵！")
        Q = np.zeros((self.params.T, self.params.n, self.params.m), dtype=float)
        for t in range(self.params.T):
            # 统计 t~T 的选择
            for alpha in range(self.params.m):
                for idx in range(self.params.n):
                    count = np.sum(self.params.Y[t:self.params.T, alpha] == idx)
                    Q[t, idx, alpha] = count / (max(1, self.params.T - t))  # 避免除以0
        self.params.Q = Q
        return Q

    def save_Y(self, filename):
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.save(filename, self.params.Y)

    def load_Y(self, filename):
        self.params.Y = np.load(filename)
        return self.params.Y

# No top-level script code; this module is library-only.
