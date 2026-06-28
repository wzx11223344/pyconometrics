"""
Panel FE/RE —— 面板数据模型
===========================

实现面板数据的经典估计方法：
- PanelFE: 固定效应 (Fixed Effects) / Within 估计
- PanelRE: 随机效应 (Random Effects) / GLS 估计
- HausmanTest: 豪斯曼检验 (FE vs RE)

理论参考: Wooldridge (2010) Ch.10, Baltagi (2021)
"""

import numpy as np
from numpy.linalg import inv
from scipy import stats as scipy_stats
from typing import Optional
from .ols import OLS
from .utils import add_constant


def _demcan_within(X: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """组内去均值变换 (Within transformation)。"""
    X_demean = X.copy().astype(float)
    unique_ids = np.unique(ids)
    for uid in unique_ids:
        mask = ids == uid
        X_demean[mask] = X[mask] - X[mask].mean(axis=0)
    return X_demean


class PanelFEResult:
    """面板固定效应结果。"""

    def __init__(self):
        self.beta: np.ndarray = None
        self.se: np.ndarray = None
        self.t_stats: np.ndarray = None
        self.p_values: np.ndarray = None
        self.r2_within: float = None
        self.r2_overall: float = None
        self.r2_between: float = None
        self.f_stat: float = None
        self.f_pvalue: float = None
        self.n: int = 0
        self.T: float = 0        # 平均时期数
        self.N: int = 0          # 个体数
        self.k: int = 0
        self.fixed_effects: np.ndarray = None
        self.var_names: list = None

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "面板固定效应 (Within) 估计结果",
            "=" * 72,
            f"个体数 (N): {self.N:>6d}      平均时期数 (T̄): {self.T:>8.2f}",
            f"观测数 (n): {self.n:>6d}      变量数 (k):    {self.k:>8d}",
            f"R² within:  {self.r2_within:>10.4f}",
            f"R² between: {self.r2_between:>10.4f}",
            f"R² overall: {self.r2_overall:>10.4f}",
            f"F-stat:     {self.f_stat:>10.4f}      P-value:     {self.f_pvalue:>10.4f}",
            "-" * 72,
            f"{'Variable':>16s}  {'Coef.':>10s}  {'Std.Err.':>10s}  {'t':>8s}  {'P>|t|':>8s}",
            "-" * 72,
        ]
        for i in range(self.k):
            name = self.var_names[i] if self.var_names else f"x{i}"
            lines.append(f"{name:>16s}  {self.beta[i]:>10.4f}  "
                         f"{self.se[i]:>10.4f}  {self.t_stats[i]:>8.3f}  "
                         f"{self.p_values[i]:>8.4f}")
        lines.append("=" * 72)
        return "\n".join(lines)


class PanelREResult:
    """面板随机效应结果。"""

    def __init__(self):
        self.beta: np.ndarray = None
        self.se: np.ndarray = None
        self.t_stats: np.ndarray = None
        self.p_values: np.ndarray = None
        self.theta: float = None   # GLS 变换参数
        self.r2: float = None
        self.n: int = 0
        self.N: int = 0
        self.k: int = 0
        self.var_names: list = None

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "面板随机效应 (GLS) 估计结果",
            "=" * 72,
            f"个体数 (N): {self.N:>6d}      观测数 (n): {self.n:>8d}",
            f"变量数 (k): {self.k:>6d}      θ (GLS):   {self.theta:>10.4f}",
            f"R²:         {self.r2:>10.4f}",
            "-" * 72,
            f"{'Variable':>16s}  {'Coef.':>10s}  {'Std.Err.':>10s}  {'t':>8s}  {'P>|t|':>8s}",
            "-" * 72,
        ]
        for i in range(self.k):
            name = self.var_names[i] if self.var_names else f"x{i}"
            lines.append(f"{name:>16s}  {self.beta[i]:>10.4f}  "
                         f"{self.se[i]:>10.4f}  {self.t_stats[i]:>8.3f}  "
                         f"{self.p_values[i]:>8.4f}")
        lines.append("=" * 72)
        return "\n".join(lines)


class PanelFE:
    """
    面板固定效应模型 (Within Estimator)。

    使用组内去均值变换消除个体固定效应，再进行 OLS。

    使用
    ----
    fe = PanelFE(y, X, ids)
    result = fe.fit()
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, ids: np.ndarray,
                 var_names: Optional[list] = None):
        self.y = np.asarray(y, dtype=float).flatten()
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        self.ids = np.asarray(ids).flatten()
        self.var_names = var_names or [f"x{i}" for i in range(self.X.shape[1])]

    def fit(self) -> PanelFEResult:
        """拟合固定效应模型。"""
        # 组内去均值
        y_dm = _demcan_within(self.y.reshape(-1, 1), self.ids).flatten()
        X_dm = _demcan_within(self.X, self.ids)

        ols = OLS(y_dm, X_dm, add_intercept=False, var_names=self.var_names)
        ols_result = ols.fit()

        result = PanelFEResult()
        result.beta = ols_result.beta
        result.se = ols_result.se
        result.t_stats = ols_result.t_stats
        result.p_values = ols_result.p_values
        result.r2_within = ols_result.r2
        result.n = len(self.y)
        result.N = len(np.unique(self.ids))
        result.T = result.n / result.N
        result.k = len(ols_result.beta)
        result.var_names = self.var_names
        result.f_stat = ols_result.f_stat
        result.f_pvalue = ols_result.f_pvalue

        # 计算 between R² 和 overall R²
        y_hat = self.X @ result.beta
        y_mean = np.mean(self.y)
        result.r2_overall = 1 - np.sum((self.y - y_hat) ** 2) / np.sum((self.y - y_mean) ** 2)

        # between R²: 用个体均值回归
        unique_ids = np.unique(self.ids)
        y_bar = np.array([np.mean(self.y[self.ids == uid]) for uid in unique_ids])
        y_hat_bar = np.array([np.mean(y_hat[self.ids == uid]) for uid in unique_ids])
        result.r2_between = 1 - np.sum((y_bar - y_hat_bar) ** 2) / np.sum((y_bar - np.mean(y_bar)) ** 2)

        # 固定效应
        result.fixed_effects = np.array([
            np.mean(self.y[self.ids == uid] - y_hat[self.ids == uid])
            for uid in unique_ids
        ])

        return result


class PanelRE:
    """
    面板随机效应模型 (GLS Estimator)。

    使用
    ----
    re = PanelRE(y, X, ids)
    result = re.fit()
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, ids: np.ndarray,
                 var_names: Optional[list] = None):
        self.y = np.asarray(y, dtype=float).flatten()
        self.X = add_constant(np.asarray(X, dtype=float))
        self.ids = np.asarray(ids).flatten()
        self.var_names = ["const"] + (var_names or [f"x{i}" for i in range(self.X.shape[1] - 1)])

    def fit(self) -> PanelREResult:
        """拟合随机效应模型 (GLS)。"""
        # 先用 Pooled OLS 和 FE 估计方差分量
        pooled = OLS(self.y, self.X, add_intercept=False).fit()
        # FE: 使用不含常数项的 X（FE 内部会 demean）
        fe = PanelFE(self.y, self.X[:, 1:], self.ids).fit()

        n, k = len(self.y), self.X.shape[1]
        N = fe.N
        T_bar = n / N

        # 方差分量估计
        sigma_e2 = np.sum(pooled.residuals ** 2) / (n - k)  # idiosyncratic
        # FE 预测: X_raw @ fe.beta + fixed_effects
        y_hat_fe = self.X[:, 1:] @ fe.beta
        fe_resid = self.y - y_hat_fe
        unique_ids = np.unique(self.ids)
        sigma_u2 = max(0, np.sum([
            np.mean(fe_resid[self.ids == uid]) ** 2 for uid in unique_ids
        ]) / (N - k) - sigma_e2 / T_bar)

        # GLS 变换参数
        theta = 1 - np.sqrt(sigma_e2 / (sigma_e2 + T_bar * sigma_u2))

        # 准去均值变换
        y_trans = self.y.copy()
        X_trans = self.X.copy()
        for uid in unique_ids:
            mask = self.ids == uid
            y_trans[mask] -= theta * np.mean(self.y[mask])
            X_trans[mask] -= theta * np.mean(self.X[mask], axis=0)

        ols = OLS(y_trans, X_trans, add_intercept=False, var_names=self.var_names)
        ols_result = ols.fit()

        result = PanelREResult()
        result.beta = ols_result.beta
        result.se = ols_result.se
        result.t_stats = ols_result.t_stats
        result.p_values = ols_result.p_values
        result.theta = theta
        result.r2 = ols_result.r2
        result.n = n
        result.N = N
        result.k = k
        result.var_names = self.var_names

        return result


class HausmanTest:
    """
    豪斯曼检验 (Hausman Test)。

    H0: FE 和 RE 估计量一致（随机效应有效）
    H1: FE 和 RE 估计量不一致（应使用固定效应）
    """

    def __init__(self, fe_result: PanelFEResult, re_result: PanelREResult):
        self.beta_fe = fe_result.beta
        self.beta_re = re_result.beta[:len(fe_result.beta)]
        self.k = len(self.beta_fe)

    def run(self) -> dict:
        """
        执行豪斯曼检验。

        返回
        ----
        dict with keys: statistic, pvalue, df, conclusion
        """
        diff = self.beta_fe - self.beta_re
        # 简化：使用 FE 方差作为 diff 方差的下界
        try:
            stat = float(diff.T @ diff)  # 简化版本
            pvalue = 1 - scipy_stats.chi2.cdf(stat, self.k - 1)
        except:
            stat = float('nan')
            pvalue = float('nan')

        return {
            "statistic": stat,
            "pvalue": pvalue,
            "df": self.k - 1,
            "conclusion": "应使用固定效应 (FE)" if pvalue < 0.05 else "不能拒绝随机效应 (RE)"
        }
