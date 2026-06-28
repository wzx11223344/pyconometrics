# 贡献指南

感谢你考虑为 PyConometrics 做出贡献！本文档将指导你如何参与项目建设。

## 行为准则

本项目遵守 [Contributor Covenant](CODE_OF_CONDUCT.md) 行为准则。参与即视为同意遵守。

## 我可以贡献什么？

- **Bug 报告**：发现任何计量方法实现错误，请提交 Issue
- **新模型提案**：新的估计量（如 GMM、Quantile Regression、Heckman Correction）
- **文档改进**：修正拼写错误、补充理论推导、增加中文/英文注释
- **测试用例**：为现有模块增加单元测试覆盖
- **性能优化**：改进矩阵运算效率、减少内存占用

## 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/wzx11223344/pyconometrics.git
cd pyconometrics

# 创建虚拟环境
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate          # Windows

# 安装开发依赖
pip install -e ".[dev]"
```

## 开发流程

1. **Fork** 本仓库
2. 创建功能分支：`git checkout -b feat/new-estimator`
3. 编写代码和测试
4. 运行测试套件：`pytest tests/ -v`
5. 确保代码风格：代码应符合 [PEP 8](https://pep8.org/) 规范
6. 提交 PR 到 `main` 分支，附上清晰的变更说明

## 代码规范

### 模块结构

每个新估计量应遵循以下模式：

```python
# pyconometrics/new_estimator.py
import numpy as np
from scipy import stats

class NewEstimator:
    """
    Brief description of the estimator.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable, shape (n,)
    X : np.ndarray
        Covariates, shape (n, k)
    **kwargs : dict
        Additional optional parameters.
    """
    def __init__(self, y, X, **kwargs):
        ...

    def fit(self, **options):
        """Fit the model and return a result object."""
        ...


class NewEstimatorResult:
    def __init__(self, beta, se, ...):
        ...
    def summary(self):
        """Print formatted result table."""
        ...
```

### 计量方法论要求

- 所有核心计算必须使用 NumPy/SciPy 函数，不得调用 statsmodels 或 sklearn 的内部实现
- 提供引用来源：在类 docstring 中注明原始论文
- 返回标准误差必须经过理论验证

### 测试要求

- 每个新模块至少提供 2 个测试用例
- 使用模拟数据 (`np.random.seed(42)`) 确保可复现
- 对已知结果进行数值验证 (如与 R 的 `lm()`、`ivreg()` 对比)

## 提交信息规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
feat(ols): add weighted least squares support
fix(iv): correct Sargan test degrees of freedom
docs(readme): update API reference table
test(did): add parallel trends test case
```

## 问题反馈

- **安全漏洞**：请勿公开提交，直接联系作者邮箱 `3521257027@QQ.com`
- **一般问题**：在 [Issues](https://github.com/wzx11223344/pyconometrics/issues) 中提交，附上最小可复现示例

---

再次感谢你的贡献！
