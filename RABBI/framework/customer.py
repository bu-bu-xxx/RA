import numpy as np
import bisect
import os
from .env import DynamicPricingEnv


class CustomerChoiceSimulator(DynamicPricingEnv):
    def __init__(self, yaml_path, random_seed=None):
        super().__init__(yaml_path, random_seed=random_seed)

    def generate_Y_matrix(self):
        Y = np.full((self.params.T, self.params.m), -1, dtype=int)
        for t in range(self.params.T):
            for alpha in range(self.params.m):
                p_alpha = self.params.p[:, alpha]
                p_cumsum = np.cumsum(p_alpha)
                xi = np.random.uniform(0, 1)
                idx = bisect.bisect_left(p_cumsum, xi)
                if idx == self.params.n:
                    Y[t, alpha] = -1
                else:
                    Y[t, alpha] = idx
        self.params.Y = Y
        return Y

    def compute_offline_Q(self):
        if self.params.Y is None:
            raise ValueError("self.params.Y is None，请先生成或加载Y矩阵！")
        Q = np.zeros((self.params.T, self.params.n, self.params.m), dtype=float)
        for t in range(self.params.T):
            for alpha in range(self.params.m):
                for idx in range(self.params.n):
                    count = np.sum(self.params.Y[t:self.params.T, alpha] == idx)
                    Q[t, idx, alpha] = count / (max(1, self.params.T - t))
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
