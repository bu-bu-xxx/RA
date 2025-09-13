# block3_env_class.py
import numpy as np
from read_params import ParamsLoader

class DynamicPricingEnv(ParamsLoader):
    def __init__(self, yaml_path, random_seed=None):
        super().__init__(yaml_path)
        self.random_seed = random_seed
        self.reset()
        if random_seed is not None:
            np.random.seed(random_seed)
        # 适配MNL和Linear参数 - 现在通过params访问

    def reset(self):
        """重置环境到初始状态"""
        self.params.b = self.params.B.copy()
        self.params.t = 0 # 当前time step: start from 0
        self.params.reward_history = []
        self.params.b_history = [self.params.b.copy()]
        self.params.j_history = []
        self.params.alpha_history = []
        return self._get_obs()

    def _get_obs(self):
        """返回当前观测（可自定义）"""
        return {
            'b': self.params.b.copy(),
            't': self.params.t
        }

    def step(self, j: int, alpha: int):
        """
        执行一步动作（j），更新环境状态。
        j: int, 0~(n-1)，表示顾客t时间购买的产品编号，-1表示未购买
        alpha: int, 0~(m-1)，表示顾客t时间选择的价格集编号
        history:
        - self.params.reward_history: 奖励历史
        - self.params.b_history: 库存历史
        - self.params.j_history: 产品选择历史
        - self.params.alpha_history: 价格集选择历史
        返回: obs, reward, done, info
        """
        reward = 0
        self.params.t += 1
        done = (self.params.t >= self.params.T)
        info = {}

        # 支持-1表示未购买
        if j == -1:
            info['sold'] = False
            self.params.reward_history.append(reward)
            self.params.b_history.append(self.params.b.copy())
            self.params.j_history.append(j)
            self.params.alpha_history.append(alpha)
            done = done or np.any(self.params.b < 0)
            return self._get_obs(), reward, done, info

        # 检查动作是否有效
        if j is None:
            print("j: ", j, "type: ", type(j))
            raise ValueError("j must be an integer representing the product index or -1.")
        # 检查动作范围
        if j < 0 or j >= self.params.n:
            raise ValueError(f"j must be in range [0, {self.params.n - 1}] or -1 (not buy), but got {j}.")
        
        # 检查库存是否足够
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

    # 可扩展更多方法，如 render, sample_customer, 等

# 示例用法
if __name__ == "__main__":
    env = DynamicPricingEnv('params.yml', random_seed=42)
    obs = env.reset()
    print("初始观测：", obs)
    # 示例执行一步，假设顾客买了产品0
    obs, reward, done, info = env.step(j=0, alpha=0)
    print("下一步观测：", obs)
    print("reward:", reward, "done:", done, "info:", info)
