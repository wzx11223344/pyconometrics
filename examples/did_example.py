"""
DID 与 Event Study 示例
========================

演示双重差分和事件研究方法：
- 经典 2×2 DID
- 带协变量的 DID
- 事件研究 (Event Study) 动态 DID
"""

import numpy as np
from pyconometrics import DID, EventStudy

np.random.seed(42)

# ============ 经典 2×2 DID ============
print("=" * 60)
print("示例 1: 经典 2×2 DID")
print("=" * 60)

# 生成面板数据: 200 个个体 × 2 期
n_units = 200
n_periods = 2

# 处理组和对照组的潜在结果
unit_effect = np.random.randn(n_units) * 0.5
time_effect = 0.3
treat_effect = 0.8  # 真实 ATT
noise = np.random.randn(n_units * n_periods) * 0.2

treated = np.repeat(np.random.binomial(1, 0.5, n_units), n_periods)
post = np.tile([0, 1], n_units)

y0 = 2.0 + np.repeat(unit_effect, n_periods) + post * time_effect + noise
y1 = y0 + treated * post * treat_effect
y = y0 * (1 - treated * post) + y1 * (treated * post)

did = DID(y, treated, post)
result = did.fit()
print(result.summary())

# ============ 带协变量的 DID ============
print("\n" + "=" * 60)
print("示例 2: 带协变量的 DID")
print("=" * 60)
covariate = np.random.randn(n_units * n_periods)

did_cov = DID(y, treated, post, covariates=covariate)
result_cov = did_cov.fit()
print(result_cov.summary())

# ============ 事件研究 ============
print("\n" + "=" * 60)
print("示例 3: 事件研究 (Event Study)")
print("=" * 60)

n_units_es = 50
n_times_es = 10
treat_time = 5  # 处理发生在 t=5

y_es = []
treated_es = []
time_to_treat_es = []
unit_id_es = []
time_id_es = []

for i in range(n_units_es):
    is_treated = i < n_units_es // 2
    unit_fe = np.random.randn() * 0.3
    for t in range(n_times_es):
        time_fe = 0.1 * t + np.random.randn() * 0.1
        tau = 0 if not is_treated or t < treat_time else 0.5 * (1 + 0.1 * (t - treat_time))
        y_val = 5 + unit_fe + time_fe + tau + np.random.randn() * 0.15
        y_es.append(y_val)
        treated_es.append(1 if is_treated else 0)
        time_to_treat_es.append(t - treat_time)
        unit_id_es.append(i)
        time_id_es.append(t)

y_es = np.array(y_es)
treated_es = np.array(treated_es)
time_to_treat_es = np.array(time_to_treat_es)
unit_id_es = np.array(unit_id_es)
time_id_es = np.array(time_id_es)

es = EventStudy(y_es, treated_es, time_to_treat_es, unit_id_es, time_id_es,
                pre_periods=4, post_periods=4)
es_result = es.fit(base_period=-1)
print(es_result.summary())
