# PyConometrics - 从零实现的计量经济学 Python 库

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/dependency-NumPy-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/dependency-SciPy-8CAAE6.svg)](https://scipy.org/)

**PyConometrics** 是一个纯 Python + NumPy 实现的计量经济学工具库。所有模型均从底层矩阵运算构建，**不依赖 statsmodels 等高级统计库**，适合学习计量经济学原理、教学演示和轻量级实证研究。

## ✨ 特性

- 🧮 **从零实现**：所有估计量均由 NumPy 矩阵运算构建，代码即教材
- 📊 **丰富模型**：涵盖 OLS、IV/2SLS、DID、Event Study、RDD、Panel FE/RE、Logit/Probit
- 🛡️ **稳健推断**：支持异方差稳健标准误 (HC1)、聚类标准误
- 📝 **完整诊断**：t 检验、F 检验、R²、AIC/BIC、VIF、Durbin-Watson
- 🎯 **简洁 API**：与 statsmodels 类似的 `.fit().summary()` 风格
- 📦 **轻量依赖**：仅需 NumPy + SciPy

## 📦 安装

```bash
pip install pyconometrics
```

或从源码安装：

```bash
git clone https://github.com/wzx11223344/pyconometrics.git
cd pyconometrics
pip install -e .
```

## 🚀 快速开始

### OLS 回归

```python
import numpy as np
from pyconometrics import OLS

# 生成模拟数据
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
y = 3 + 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(n) * 0.5

# 拟合模型
model = OLS(y, X, var_names=["edu", "exp"])
result = model.fit(se_type="hc1")  # 异方差稳健标准误
print(result.summary())
```

### 工具变量 / 2SLS

```python
from pyconometrics import IV

# 内生变量 X_endog, 工具变量 Z
result = IV(y, X_endog, X_exog, Z_instruments).fit()
print(f"第一阶段 F: {result.first_stage_f:.2f}")
print(result.summary())
```

### 双重差分 (DID)

```python
from pyconometrics import DID

did = DID(y, treated, post)
result = did.fit()
print(f"ATT = {result.att:.4f}, p = {result.att_pvalue:.4f}")
```

### 事件研究 (Event Study)

```python
from pyconometrics import EventStudy

es = EventStudy(y, treated, time_to_treat, unit_id, time_id)
result = es.fit(base_period=-1)
print(result.summary())
```

### 断点回归 (RDD)

```python
from pyconometrics import SharpRDD

rdd = SharpRDD(y, running_var, cutoff=0)
result = rdd.fit()
print(f"处理效应 = {result.tau:.4f}")
```

### 面板固定效应

```python
from pyconometrics import PanelFE

fe = PanelFE(y, X, unit_ids)
result = fe.fit()
print(result.summary())
```

### 面板随机效应 + 豪斯曼检验

```python
from pyconometrics import PanelRE, HausmanTest

re = PanelRE(y, X, unit_ids)
re_result = re.fit()

# 豪斯曼检验
ht = HausmanTest(fe_result, re_result)
print(ht.run())
```

### Logit / Probit 二元选择模型

```python
from pyconometrics import Logit, Probit

logit = Logit(y_binary, X)
result = logit.fit()
print(result.summary())
```

## 📚 模块参考

| 模块 | 功能 | 核心方法 |
|------|------|---------|
| `ols` | 普通最小二乘 | 矩阵求逆、HC1 稳健标准误、聚类标准误 |
| `iv` | 工具变量 / 2SLS | 两阶段估计、弱工具变量检验、Sargan 过度识别 |
| `did` | 双重差分 / 事件研究 | TWFE、动态 DID、平行趋势 |
| `rdd` | 断点回归 | 局部线性回归、IK 带宽、三角核/Epanechnikov 核 |
| `panel` | 面板数据 | Within 估计、GLS 随机效应、豪斯曼检验 |
| `logit` | 二元选择 | Newton-Raphson MLE、边际效应、McFadden R² |

## 🧪 运行示例

```bash
cd examples
python ols_example.py
python iv_example.py
python did_example.py
python panel_example.py
```

## 📖 理论背景

本库的实现基于以下经典计量经济学教材：

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*. MIT Press.
- Greene, W. H. (2018). *Econometric Analysis*. Pearson.
- Angrist, J. D. & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Imbens, G. W. & Lemieux, T. (2008). Regression discontinuity designs. *Journal of Econometrics*.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data*. Springer.

## 📄 许可证

MIT License © 2024 wzx11223344
