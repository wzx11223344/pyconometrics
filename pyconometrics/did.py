"""
DID & Event Study —— 双重差分与事件研究
=======================================

处理面板数据中的政策评估问题：
- 经典 2×2 DID (Two-way Fixed Effects)
- 多期 DID 与事件研究 (Event Study)
- 平行趋势检验
- Bootstrap 标准误选项

理论参考：Angrist & Pischke (2009), Callaway & Sant'Anna (2021)
"""

import numpy as np
from numpy.linalg import inv
from scipy import stats as scipy_stats
from typing import Optional, Tuple
from .ols import OLS
from .utils import add_constant


class DIDResult:
    """双重差分结果容器。"""

    def __init__(self):
        self.att: float = None          # 处理组平均处理效应
        self.att_se: float = None
        self.att_t: float = None
        self.att_pvalue: float = None
        self.att_ci: tuple = None
        self.r2: float = None
        self.n: int = 0
        self.n_treat: int = 0
        self.n_control: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 56,
            "双重差分 (DID) 估计结果",
            "=" * 56,
            f"观测数: {self.n:>6d}    处理组: {self.n_treat:>6d}",
            f"                        对照组: {self.n_control:>6d}",
            f"ATT (平均处理效应): {self.att:>10.4f}",
            f"标准误:   {self.att_se:>12.4f}",
            f"t 统计量: {self.att_t:>12.4f}",
            f"p 值:     {self.att_pvalue:>12.4f}",
            f"95% CI:   [{self.att_ci[0]:>8.4f}, {self.att_ci[1]:>8.4f}]",
            "=" * 56,
        ]
        if self.att_t is not None and abs(self.att_t) > 1.96:
            lines.append("✓ 在 5% 水平上统计显著")
        else:
            lines.append("✗ 在 5% 水平上不显著")
        return "\n".join(lines)


class EventStudyResult:
    """事件研究结果容器。"""

    def __init__(self):
        self.time_periods: np.ndarray = None
        self.coefficients: np.ndarray = None
        self.se: np.ndarray = None
        self.ci_lower: np.ndarray = None
        self.ci_upper: np.ndarray = None
        self.n: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 64,
            "事件研究 (Event Study) 估计结果",
            "=" * 64,
            f"{'Period':>8s}  {'Coef.':>10s}  {'Std.Err.':>10s}  {'[95% CI]':>24s}",
            "-" * 64,
        ]
        for i in range(len(self.time_periods)):
            t = self.time_periods[i]
            sig = "***" if abs(self.coefficients[i] / self.se[i]) > 2.58 else \
                  "**" if abs(self.coefficients[i] / self.se[i]) > 1.96 else \
                  "*" if abs(self.coefficients[i] / self.se[i]) > 1.64 else "  "
            lines.append(f"{t:>8d}  {self.coefficients[i]:>10.4f}  "
                         f"{self.se[i]:>10.4f}  "
                         f"[{self.ci_lower[i]:>8.4f}, {self.ci_upper[i]:>8.4f}] {sig}")
        lines.append("-" * 64)
        lines.append("*** p<0.01  ** p<0.05  * p<0.10")
        lines.append("=" * 64)
        return "\n".join(lines)


class DID:
    """
    双重差分估计器。

    面板数据格式：每一行为一个观测 (个体×时期)，每一列为变量。

    使用
    ----
    did = DID(df['y'], df['treated'], df['post'])
    result = did.fit()
    print(result.summary())
    """

    def __init__(self, y: np.ndarray, treated: np.ndarray,
                 post: np.ndarray, covariates: Optional[np.ndarray] = None):
        """
        参数
        ----
        y : np.ndarray, shape (n,)
        treated : np.ndarray, shape (n,)
            处理组标识: 1=处理组, 0=对照组。
        post : np.ndarray, shape (n,)
            处理后时期标识: 1=处理后, 0=处理前。
        covariates : np.ndarray, shape (n, k_cov) or None
            协变量矩阵（可选）。
        """
        self.y = np.asarray(y, dtype=float).flatten()
        self.treated = np.asarray(treated, dtype=float).flatten()
        self.post = np.asarray(post, dtype=float).flatten()
        self.covariates = covariates

    def fit(self) -> DIDResult:
        """
        拟合 DID 模型: y = β₀ + β₁·treated + β₂·post + β₃·(treated×post) + ε
        其中 β₃ 即为 ATT。
        """
        n = len(self.y)
        interact = self.treated * self.post

        X = np.column_stack([np.ones(n), self.treated, self.post, interact])
        var_names = ["const", "treated", "post", "did"]

        if self.covariates is not None:
            cov = np.asarray(self.covariates, dtype=float)
            if cov.ndim == 1:
                cov = cov.reshape(-1, 1)
            X = np.column_stack([X, cov])
            for i in range(cov.shape[1]):
                var_names.append(f"cov_{i}")

        ols = OLS(self.y, X, add_intercept=False, var_names=var_names)
        ols_result = ols.fit(se_type="hc1")  # 异方差稳健标准误

        result = DIDResult()
        result.att = ols_result.beta[3]  # did 交互项系数
        result.att_se = ols_result.se[3]
        result.att_t = ols_result.t_stats[3]
        result.att_pvalue = ols_result.p_values[3]
        result.att_ci = tuple(ols_result.ci[3])
        result.r2 = ols_result.r2
        result.n = n
        result.n_treat = int(np.sum(self.treated > 0.5))
        result.n_control = n - result.n_treat

        return result


class EventStudy:
    """
    事件研究法 (Event Study) / 动态 DID。

    使用相对时间虚拟变量替代简单的 Post 虚拟变量，
    允许处理效应随时间变化。

    使用
    ----
    es = EventStudy(y, treated, time_to_treat, unit_id, time_id)
    result = es.fit()
    print(result.summary())
    """

    def __init__(self, y: np.ndarray, treated: np.ndarray,
                 time_to_treat: np.ndarray, unit_id: np.ndarray,
                 time_id: np.ndarray,
                 pre_periods: int = 4, post_periods: int = 4):
        """
        参数
        ----
        y : np.ndarray
        treated : np.ndarray
            是否为处理组。
        time_to_treat : np.ndarray
            距离处理发生的时间（负数为处理前，0为处理当期）。
            对照组可设为较大负值。
        unit_id : np.ndarray
            个体标识。
        time_id : np.ndarray
            时期标识。
        pre_periods : int
            包含的处理前时期数。
        post_periods : int
            包含的处理后时期数。
        """
        self.y = np.asarray(y, dtype=float).flatten()
        self.treated = np.asarray(treated).flatten()
        self.time_to_treat = np.asarray(time_to_treat, dtype=int).flatten()
        self.unit_id = np.asarray(unit_id).flatten()
        self.time_id = np.asarray(time_id).flatten()
        self.pre_periods = pre_periods
        self.post_periods = post_periods

    def fit(self, base_period: int = -1) -> EventStudyResult:
        """
        拟合事件研究模型，以 base_period 为基准期（通常为 -1）。

        模型: y ~ Σ τₖ Dₖ + unit_FE + time_FE
        """
        # 创建相对时间虚拟变量
        time_dummies = []
        time_labels = []
        for k in range(-self.pre_periods, self.post_periods + 1):
            if k == base_period:
                continue  # 基准期不纳入
            d = (self.time_to_treat == k).astype(float)
            time_dummies.append(d)
            time_labels.append(k)

        X_time = np.column_stack(time_dummies)

        # 个体固定效应
        units = np.unique(self.unit_id)
        unit_fe = np.column_stack([
            (self.unit_id == u).astype(float) for u in units[1:]
        ])

        # 时间固定效应
        times = np.unique(self.time_id)
        time_fe = np.column_stack([
            (self.time_id == t).astype(float) for t in times[1:]
        ])

        X = np.column_stack([np.ones(len(self.y)), X_time, unit_fe, time_fe])
        ols = OLS(self.y, X, add_intercept=False)
        ols_result = ols.fit()

        result = EventStudyResult()
        # 提取相对时间系数的索引 (跳过 const，前 len(time_labels) 个)
        result.coefficients = ols_result.beta[1:1 + len(time_labels)]
        result.se = ols_result.se[1:1 + len(time_labels)]
        result.n = len(self.y)
        result.time_periods = np.array(time_labels)

        # 置信区间
        t_crit = scipy_stats.t.ppf(0.975, ols_result.n - ols_result.k)
        result.ci_lower = result.coefficients - t_crit * result.se
        result.ci_upper = result.coefficients + t_crit * result.se

        return result
