"""
OLS —— 普通最小二乘回归
=======================

从矩阵代数底层实现的完整 OLS 回归模型，包含：
- 参数估计 (XTX)⁻¹XTy
- 同方差 / 异方差稳健(HC1) / 聚类稳健标准误
- t 检验、F 检验、置信区间
- R²、调整 R²、AIC、BIC
- VIF 多重共线性诊断
- Durbin-Watson 自相关检验
- 残差诊断图数据

理论参考：Wooldridge (2010), Greene (2018)
"""

import numpy as np
from numpy.linalg import inv, pinv
from scipy import stats as scipy_stats
from typing import Optional, Dict, Union
from .utils import (
    add_constant, se_homosk, se_robust_hc1, se_clustered,
    t_test, f_test, confidence_interval, r_squared,
    durbin_watson, vif,
)


class OLSResult:
    """OLS 回归结果容器。"""

    def __init__(self):
        self.beta: np.ndarray = None
        self.se: np.ndarray = None
        self.t_stats: np.ndarray = None
        self.p_values: np.ndarray = None
        self.ci: np.ndarray = None
        self.r2: float = None
        self.r2_adj: float = None
        self.f_stat: float = None
        self.f_pvalue: float = None
        self.aic: float = None
        self.bic: float = None
        self.dw: float = None
        self.vif_values: np.ndarray = None
        self.residuals: np.ndarray = None
        self.y_hat: np.ndarray = None
        self.n: int = 0
        self.k: int = 0
        self.se_type: str = "homosk"
        self.var_names: list = None

    def summary(self) -> str:
        """生成回归结果汇总表。"""
        lines = []
        lines.append("=" * 72)
        lines.append("OLS 回归结果")
        lines.append("=" * 72)
        lines.append(f"观测数: {self.n:>6d}          自由度: {self.n - self.k:>6d}")
        lines.append(f"变量数: {self.k:>6d}          标准误类型: {self.se_type}")
        lines.append(f"R²    : {self.r2:>10.4f}      调整 R²: {self.r2_adj:>10.4f}")
        lines.append(f"F-stat: {self.f_stat:>10.4f}      F-pvalue: {self.f_pvalue:>10.4f}")
        lines.append(f"AIC   : {self.aic:>10.4f}      BIC   : {self.bic:>10.4f}")
        if self.dw is not None:
            lines.append(f"D-W   : {self.dw:>10.4f}")
        lines.append("-" * 72)
        header = f"{'Variable':>16s}  {'Coef.':>10s}  {'Std.Err.':>10s}  {'t':>8s}  {'P>|t|':>8s}  {'[95% CI]':>20s}"
        lines.append(header)
        lines.append("-" * 72)
        for i in range(self.k):
            name = self.var_names[i] if self.var_names else f"x{i}"
            line = (f"{name:>16s}  {self.beta[i]:>10.4f}  "
                    f"{self.se[i]:>10.4f}  {self.t_stats[i]:>8.3f}  "
                    f"{self.p_values[i]:>8.4f}  "
                    f"[{self.ci[i, 0]:>8.4f}, {self.ci[i, 1]:>8.4f}]")
            lines.append(line)
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """导出为字典。"""
        d = {
            "beta": self.beta.tolist(),
            "se": self.se.tolist(),
            "t_stats": self.t_stats.tolist(),
            "p_values": self.p_values.tolist(),
            "r2": self.r2,
            "r2_adj": self.r2_adj,
            "n": self.n,
            "k": self.k,
            "se_type": self.se_type,
        }
        if self.var_names:
            d["var_names"] = self.var_names
        return d


class OLS:
    """普通最小二乘回归模型。"""

    def __init__(self, y: np.ndarray, X: np.ndarray,
                 add_intercept: bool = True,
                 var_names: Optional[list] = None):
        """
        参数
        ----
        y : np.ndarray, shape (n,)
            被解释变量。
        X : np.ndarray, shape (n, k)
            自变量矩阵（不含截距项）。
        add_intercept : bool
            是否自动添加截距项，默认 True。
        var_names : list of str, optional
            变量名称列表。
        """
        self.y = np.asarray(y, dtype=float).flatten()
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if add_intercept:
            self.X = add_constant(self.X)
        self.n, self.k = self.X.shape
        self.var_names = var_names
        if self.var_names is None:
            self.var_names = [f"x{i}" for i in range(self.k)]
            if add_intercept:
                self.var_names[0] = "const"
        elif add_intercept:
            self.var_names = ["const"] + self.var_names

    def fit(self, se_type: str = "homosk",
            cluster: Optional[np.ndarray] = None) -> OLSResult:
        """
        拟合OLS回归。

        参数
        ----
        se_type : str
            标准误类型: "homosk" (同方差), "hc1" (异方差稳健), "clustered" (聚类稳健)
        cluster : np.ndarray, optional
            聚类标识向量，仅 se_type="clustered" 时使用。

        返回
        ----
        OLSResult
        """
        X, y, n, k = self.X, self.y, self.n, self.k

        # 1. 参数估计: β = (X'X)⁻¹X'y
        try:
            beta = inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            beta = pinv(X.T @ X) @ X.T @ y  # 奇异矩阵回退

        result = OLSResult()
        result.beta = beta
        result.y_hat = X @ beta
        result.residuals = y - result.y_hat
        result.n, result.k = n, k
        result.se_type = se_type
        result.var_names = self.var_names

        # 2. 标准误
        if se_type == "homosk":
            result.se = se_homosk(beta, X, result.residuals)
        elif se_type == "hc1":
            result.se = se_robust_hc1(beta, X, result.residuals)
        elif se_type == "clustered":
            if cluster is None:
                raise ValueError("se_type='clustered' 需要提供 cluster 参数")
            result.se = se_clustered(beta, X, result.residuals, np.asarray(cluster))
        else:
            raise ValueError(f"未知的 se_type: {se_type}")

        # 3. t 检验
        result.t_stats, result.p_values = t_test(beta, result.se, n - k)

        # 4. 置信区间
        result.ci = confidence_interval(beta, result.se, n - k)

        # 5. R²
        result.r2, result.r2_adj = r_squared(y, result.y_hat, k)

        # 6. 整体 F 检验 (H0: 除截距外所有系数为0)
        if k > 1:
            # 排除截距的约束矩阵
            R_f = np.zeros((k - 1, k))
            for i in range(1, k):
                R_f[i - 1, i] = 1
            q_f = np.zeros(k - 1)
            sigma2 = np.sum(result.residuals ** 2) / (n - k)
            vcov = sigma2 * inv(X.T @ X)
            result.f_stat, result.f_pvalue = f_test(R_f, q_f, beta, vcov, n - k)
        else:
            result.f_stat, result.f_pvalue = float("nan"), float("nan")

        # 7. 信息准则
        sigma2 = np.sum(result.residuals ** 2) / n
        result.aic = n * np.log(sigma2) + 2 * k
        result.bic = n * np.log(sigma2) + k * np.log(n)

        # 8. Durbin-Watson
        result.dw = durbin_watson(result.residuals)

        # 9. VIF
        result.vif_values = vif(self.X)

        return result
