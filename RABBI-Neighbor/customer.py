import numpy as np
import bisect
import os
from env import DynamicPricingEnv

class CustomerChoiceSimulator(DynamicPricingEnv):
    def __init__(self, yaml_path, random_seed=None):
        super().__init__(yaml_path, random_seed=random_seed)
        self.Y = None  # (T, m)
        self.Q = None        # (T, n, m)

    def generate_choice_matrix(self):
        """
        返回：Y，shape=(T, m)，
        Y[t, alpha]=j表示t时刻顾客在价格集alpha下选择了产品j（0~n-1）
        每个时间步和价格集，顾客一定会选择一个商品（不考虑未购买情况）
        """
        Y = np.zeros((self.T, self.m), dtype=int)
        for t in range(self.T):
            for alpha in range(self.m):
                p_alpha = self.p[:, alpha]  # (n,)
                p_cumsum = np.cumsum(p_alpha) / np.sum(p_alpha)  # 归一化累积概率
                xi = np.random.uniform(0, 1)
                idx = bisect.bisect_left(p_cumsum, xi)
                Y[t, alpha] = idx
        self.Y = Y
        return Y

    def compute_offline_Q(self, Y):
        """
        根据Y (T, m) 生成 Q 矩阵 (T, n, m)，
        Q[t, idx, alpha] = 在 t~T 时间段内，价格集 alpha 下选择产品 idx 的频率
        """
        Q = np.zeros((self.T, self.n, self.m), dtype=float)
        for t in range(self.T):
            # 统计 t~T 的选择
            for alpha in range(self.m):
                for idx in range(self.n):
                    count = np.sum(Y[t:self.T, alpha] == idx)
                    Q[t, idx, alpha] = count / (max(1, self.T - t))  # 避免除以0
        self.Q = Q
        return Q

    def save_Q(self, Q, filename):
        np.save(filename, Q)

    def load_Q(self, filename):
        self.Q = np.load(filename)
        return self.Q

# 示例用法
if __name__ == "__main__":
    sim = CustomerChoiceSimulator('params.yml', random_seed=42)
    Y = sim.generate_choice_matrix()
    print("Y矩阵 shape:", Y.shape)
    print("示例：第0个时间步的选择\n", Y[0])
    print("示例：第1个时间步的选择\n", Y[1])
    print("示例：第19个时间步的选择\n", Y[19])
    Q = sim.compute_offline_Q(Y)
    print("示例：Q矩阵 shape:", Q.shape)
    print("示例：第0个时间步的Q矩阵\n", Q[0])
    print("示例：第1个时间步的Q矩阵\n", Q[1])
    print("示例：第19个时间步的Q矩阵\n", Q[19])
    # 保存和读取Q示例
    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)
    Q_path = os.path.join(data_dir, "Q_matrix.npy")
    sim.save_Q(Q, Q_path)
    Q_loaded = sim.load_Q(Q_path)
    print("读取后的Q矩阵 shape:", Q_loaded.shape)
