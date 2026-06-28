"""
IV / 2SLS —— 工具变量与两阶段最小二乘
====================================

处理内生性问题的方法：
- 第一阶段：X_endog ~ Z_instruments + X_exog
- 第二阶段：y ~ X̂_endog + X_exog
- 提供第一阶段 F 统计量 (弱工具变量检验)、Sargan 过度识别检验

理论参考：Angrist & Pischke (2009), Wooldridge (2010)
"""

import numpy as np
from numpy.linalg import inv
from scipy import stats as scipy_stats
from typing import Optional
from .ols import OLS
from .utils import add_constant


class IVResult:
    """IV/2SLS 回归结果容器。"""

    def __init__(self):
        self.beta: np.ndarray = None
        self.se: np.ndarray = None
        self.t_stats: np.ndarray = None
        self.p_values: np.ndarray = None
        self.first_stage_f: float = None
        self.first_stage_f_pvalue: float = None
        self.first_stage_r2: float = None
        self.sargan_stat: Optional[float] = None
        self.sargan_pvalue: Optional[float] = None
        self.n: int = 0
        self.k: int = 0
        self.var_names: list = None

    def summary(self) -> str:
        """生成结果汇总表。"""
        lines = []
        lines.append("=" * 72)
        lines.append("IV / 2SLS 回归结果")
        lines.append("=" * 72)
        lines.append(f"观测数: {self.n:>6d}")
        lines.append(f"变量数: {self.k:>6d}")
        lines.append("-" * 72)
        lines.append("第一阶段诊断:")
        lines.append(f"  第一阶段 F 统计量: {self.first_stage_f:>10.4f}")
        lines.append(f"  第一阶段 F p值  : {self.first_stage_f_pvalue:>10.4f}")
        lines.append(f"  第一阶段 R²      : {self.first_stage_r2:>10.4f}")
        if self.first_stage_f < 10:
            lines.append("  ⚠ 警告: 第一阶段 F < 10，可能存在弱工具变量问题")
        else:
            lines.append("  ✓ 第一阶段 F ≥ 10，工具变量有效")
        if self.sargan_stat is not None:
            lines.append(f"  Sargan 统计量    : {self.sargan_stat:>10.4f}")
            lines.append(f"  Sargan p值       : {self.sargan_pvalue:>10.4f}")
        lines.append("-" * 72)
        header = f"{'Variable':>16s}  {'Coef.':>10s}  {'Std.Err.':>10s}  {'t':>8s}  {'P>|t|':>8s}"
        lines.append(header)
        lines.append("-" * 72)
        for i in range(self.k):
            name = self.var_names[i] if self.var_names else f"x{i}"
            lines.append(f"{name:>16s}  {self.beta[i]:>10.4f}  "
                         f"{self.se[i]:>10.4f}  {self.t_stats[i]:>8.3f}  "
                         f"{self.p_values[i]:>8.4f}")
        lines.append("=" * 72)
        return "\n".join(lines)


class IV:
    """
    工具变量 / 两阶段最小二乘回归。

    使用
    ----
    iv = IV(y, X_endog, X_exog, Z_instruments)
    result = iv.fit()
    print(result.summary())
    """

    def __init__(self, y: np.ndarray, X_endog: np.ndarray,
                 X_exog: Optional[np.ndarray] = None,
                 Z_instruments: Optional[np.ndarray] = None,
                 var_names: Optional[list] = None):
        """
        参数
        ----
        y : np.ndarray, shape (n,)
        X_endog : np.ndarray, shape (n, k_endog)
            内生自变量矩阵。
        X_exog : np.ndarray, shape (n, k_exog) or None
            外生自变量矩阵（不含截距项）。
        Z_instruments : np.ndarray, shape (n, k_iv) or None
            工具变量矩阵（不含截距项）。如果 X_exog 中存在外生变量，
            工具变量应只包含排除的工具变量 (excluded instruments)。
        var_names : list, optional
            变量名称列表。
        """
        self.y = np.asarray(y, dtype=float).flatten()
        self.X_endog = np.asarray(X_endog, dtype=float)
        if self.X_endog.ndim == 1:
            self.X_endog = self.X_endog.reshape(-1, 1)
        self.k_endog = self.X_endog.shape[1]

        if X_exog is None:
            self.X_exog = np.ones((len(y), 1))
            self.k_exog = 0
        else:
            self.X_exog = np.asarray(X_exog, dtype=float)
            if self.X_exog.ndim == 1:
                self.X_exog = self.X_exog.reshape(-1, 1)
            self.X_exog = add_constant(self.X_exog)
            self.k_exog = self.X_exog.shape[1] - 1  # 不含截距

        self.Z = np.asarray(Z_instruments, dtype=float)
        if self.Z.ndim == 1:
            self.Z = self.Z.reshape(-1, 1)

        self.n = len(y)
        self.var_names = var_names

    def fit(self) -> IVResult:
        """
        拟合 2SLS 回归。

        返回
        ----
        IVResult
        """
        y, X_en, X_ex, Z = self.y, self.X_endog, self.X_exog, self.Z
        n, k_endog = self.n, self.k_endog
        k_exog = self.k_exog

        # ---- 第一阶段: X_endog ~ [X_exog, Z] ----
        Z_all = np.hstack([X_ex, Z])  # 包含所有外生变量 + 排除的工具变量
        first_ols = OLS(X_en.flatten() if k_endog == 1 else X_en.mean(axis=1), Z_all)
        first_result = first_ols.fit()

        # 第一阶段 F 检验 (H0: 排除的工具变量系数全为0)
        # 简化：用整体 F 近似
        result = IVResult()

        # 逐个内生变量做第一阶段回归
        first_fs = []
        first_r2s = []
        for j in range(k_endog):
            fols = OLS(X_en[:, j], Z_all)
            fr = fols.fit()
            first_fs.append(fr.f_stat)
            first_r2s.append(fr.r2)

        result.first_stage_f = min(first_fs)
        result.first_stage_f_pvalue = min(1 - scipy_stats.f.cdf(f, Z_all.shape[1] - 1, n - Z_all.shape[1]) for f in first_fs)
        result.first_stage_r2 = np.mean(first_r2s)

        # ---- 第二阶段: y ~ X̂_endog + X_exog ----
        X_en_hat = np.zeros_like(X_en)
        for j in range(k_endog):
            fols = OLS(X_en[:, j], Z_all)
            fr = fols.fit()
            X_en_hat[:, j] = fr.y_hat

        X_stage2 = np.hstack([X_en_hat, X_ex])  # 截距已在 X_ex 中
        stage2_ols = OLS(y, X_stage2, add_intercept=False)
        stage2_result = stage2_ols.fit()

        result.beta = stage2_result.beta
        # 2SLS 标准误需要校正：使用原始内生变量计算残差
        X_full = np.hstack([X_en, X_ex])
        residuals_2sls = y - X_full @ stage2_result.beta
        sigma2 = np.sum(residuals_2sls ** 2) / (n - len(stage2_result.beta))
        X2 = np.hstack([X_en_hat, X_ex])
        vcov_2sls = sigma2 * inv(X2.T @ X2)
        result.se = np.sqrt(np.diag(vcov_2sls))
        result.t_stats = result.beta / result.se
        result.p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(result.t_stats), n - len(result.beta)))
        result.n = n
        result.k = len(result.beta)

        # 变量名称
        if self.var_names:
            result.var_names = self.var_names
        else:
            result.var_names = [f"endog_{i}" for i in range(k_endog)] + \
                               [f"exog_{i}" for i in range(k_exog)] + ["const"]

        # ---- 过度识别检验 (Sargan test) ----
        k_z = Z.shape[1]  # 排除的工具变量数
        if k_z > k_endog:
            # 用所有工具变量做完整回归，检验残差与工具变量的正交性
            s_ols = OLS(residuals_2sls, np.hstack([X_ex, Z]))
            s_res = s_ols.fit()
            nR2 = n * s_res.r2
            result.sargan_stat = nR2
            df_sargan = k_z - k_endog
            result.sargan_pvalue = 1 - scipy_stats.chi2.cdf(nR2, df_sargan)

        return result


# 别名
TSLS = IV
