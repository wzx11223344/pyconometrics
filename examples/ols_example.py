"""
OLS 回归示例
============

演示 PyConometrics 中 OLS 模型的各种功能：
- 基本回归
- 同方差 vs 异方差稳健标准误
- 聚类标准误
- 多重共线性诊断 (VIF)
- Durbin-Watson 自相关检验
"""

import numpy as np
from pyconometrics import OLS

# ---- 生成模拟数据 ----
np.random.seed(42)
n = 500

# 自变量
edu = np.random.normal(12, 3, n)      # 教育年限
exp = np.random.normal(8, 5, n)       # 工作经验
exp2 = exp ** 2                        # 经验平方项

# 被解释变量: 工资对数
y = 5.0 + 0.08 * edu + 0.04 * exp - 0.001 * exp2 + np.random.normal(0, 0.3, n)

X = np.column_stack([edu, exp, exp2])
var_names = ["education", "experience", "experience²"]

print("=" * 70)
print("示例 1: 同方差 OLS")
print("=" * 70)
ols1 = OLS(y, X, var_names=var_names)
r1 = ols1.fit(se_type="homosk")
print(r1.summary())

print("\n" + "=" * 70)
print("示例 2: 异方差稳健标准误 (HC1)")
print("=" * 70)
ols2 = OLS(y, X, var_names=var_names)
r2 = ols2.fit(se_type="hc1")
print(r2.summary())

print("\n" + "=" * 70)
print("示例 3: VIF 多重共线性诊断")
print("=" * 70)
print(f"{'Variable':>16s}  {'VIF':>10s}")
print("-" * 30)
for i, name in enumerate(r1.var_names):
    print(f"{name:>16s}  {r1.vif_values[i]:>10.4f}")
if np.any(r1.vif_values > 10):
    print("\n⚠  存在 VIF > 10 的变量，可能存在多重共线性问题")
else:
    print("\n✓ 所有 VIF 值 < 10，无严重多重共线性")

print(f"\nDurbin-Watson 统计量: {r1.dw:.4f}")
if r1.dw < 1.5:
    print("⚠  D-W < 1.5，可能存在正自相关")
elif r1.dw > 2.5:
    print("⚠  D-W > 2.5，可能存在负自相关")
else:
    print("✓  D-W 在 [1.5, 2.5]，无明显自相关")

# ---- 聚类标准误示例 ----
print("\n" + "=" * 70)
print("示例 4: 聚类稳健标准误")
print("=" * 70)
# 模拟 50 个群组
cluster = np.repeat(np.arange(1, 51), 10)
ols3 = OLS(y, X, var_names=var_names)
r3 = ols3.fit(se_type="clustered", cluster=cluster)
print(r3.summary())
