"""
IV/2SLS 回归示例
================

演示工具变量方法：
- 第一阶段诊断 (F 统计量、R²)
- 第二阶段 2SLS 估计
- 弱工具变量检验
- Sargan 过度识别检验
"""

import numpy as np
from pyconometrics import IV

np.random.seed(123)
n = 500

# ---- 生成内生变量的数据 ----
# 工具变量 Z
z1 = np.random.randn(n)
z2 = np.random.randn(n)

# 外生变量
x_exog = np.random.randn(n)

# 内生变量: X_endog = 1 + 0.5*z1 + 0.3*z2 + 0.4*x_exog + error
error = np.random.randn(n) * 0.5
X_endog = 1.0 + 0.5 * z1 + 0.3 * z2 + 0.4 * x_exog + error

# 结构方程: y = 2 + 1.5*X_endog + 0.6*x_exog + noise
# X_endog 与 noise 相关 (内生性)
noise = np.random.randn(n) * 0.8 + 0.5 * error  # 通过 error 引入内生性
y = 2.0 + 1.5 * X_endog + 0.6 * x_exog + noise

# ---- IV 估计 ----
Z = np.column_stack([z1, z2])
iv = IV(y, X_endog, x_exog, Z, var_names=["X_endog", "X_exog", "const"])
result = iv.fit()

print(result.summary())

print("\n诊断说明:")
print(f"  第一阶段 F 统计量: {result.first_stage_f:.2f}")
if result.first_stage_f > 10:
    print("  ✓ F > 10: 工具变量足够强")
elif result.first_stage_f > 5:
    print("  ⚠ 5 < F ≤ 10: 工具变量可能偏弱")
else:
    print("  ✗ F ≤ 5: 弱工具变量问题严重")

if result.sargan_stat is not None:
    print(f"  Sargan 检验 p 值: {result.sargan_pvalue:.4f}")
    if result.sargan_pvalue > 0.05:
        print("  ✓ 不能拒绝工具变量外生性的原假设")
    else:
        print("  ✗ 工具变量可能不满足外生性")
