"""
面板数据模型示例
================

演示面板数据方法：
- 固定效应 (Within) 估计
- 随机效应 (GLS) 估计
- 豪斯曼检验
"""

import numpy as np
from pyconometrics import PanelFE, PanelRE, HausmanTest

np.random.seed(123)

# ---- 生成面板数据 ----
N = 100   # 个体数
T = 5     # 时期数
n = N * T

# 个体固定效应
alpha = np.repeat(np.random.randn(N) * 0.8, T)

# 自变量
x1 = np.random.randn(n)
x2 = np.random.randn(n)

# 被解释变量: y = 2 + 1.5*x1 - 0.8*x2 + α_i + ε
y = 2.0 + 1.5 * x1 - 0.8 * x2 + alpha + np.random.randn(n) * 0.4

X = np.column_stack([x1, x2])
ids = np.repeat(np.arange(1, N + 1), T)

print("=" * 70)
print("面板固定效应 (Within) 估计")
print("=" * 70)
fe = PanelFE(y, X, ids, var_names=["x1", "x2"])
fe_result = fe.fit()
print(fe_result.summary())

print("\n" + "=" * 70)
print("面板随机效应 (GLS) 估计")
print("=" * 70)
re = PanelRE(y, X, ids, var_names=["x1", "x2"])
re_result = re.fit()
print(re_result.summary())

print("\n" + "=" * 70)
print("豪斯曼检验 (FE vs RE)")
print("=" * 70)
ht = HausmanTest(fe_result, re_result)
ht_result = ht.run()
for k, v in ht_result.items():
    print(f"  {k}: {v}")
