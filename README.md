
### CustomerChoiceSimulator 及其父类变量说明表

| 变量名         | 维度/类型         | 简单解释                                      |
|----------------|-------------------|-----------------------------------------------|
| n              | int               | 产品数量（product_number）                    |
| d              | int               | 资源种类数量（resource_number）               |
| A              | (n, d) ndarray    | 资源消耗矩阵，每个产品消耗每种资源的数量      |
| f_split        | list              | 价格集拆分原始列表                            |
| T              | int               | 时间步长/销售周期（horizon）                  |
| B              | (d,) ndarray      | 资源总预算                                    |
| k              | (n,) ndarray      | scaling_list，缩放系数                        |
| f              | (n, m) ndarray    | 价格矩阵，所有产品所有价格组合                |
| m              | int               | 价格集数量                                    |
| demand_model   | str               | 需求模型类型（MNL/LINEAR）                    |
| p              | (n, m) ndarray    | 需求概率矩阵                                  |
| Y              | (T, m) ndarray    | 顾客选择矩阵，值为0~n-1或-1（不购买）         |
| Q              | (T, n, m) ndarray | 离线统计的选择频率矩阵                        |
| random_seed    | int/None          | 随机种子                                      |
| b              | (d,) ndarray      | 当前剩余资源                                  |
| t              | int               | 当前时间步                                    |
| reward_history | list              | 每步奖励历史                                  |
| b_history      | list[(d,)]        | 每步资源剩余历史                              |
| j_history      | list[int]         | 每步顾客选择的产品编号（或-1）                |
| alpha_history  | list[int]         | 每步顾客选择的价格集编号                      |
| x_history      | list[(m,)]        | 每步LP解出的x向量（RABBI/Offline策略用）      |

> 注：所有变量均为CustomerChoiceSimulator及其父类（DynamicPricingEnv, ParamsLoader）中定义。