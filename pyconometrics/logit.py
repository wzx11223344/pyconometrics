"""
Logit / Probit —— 二元选择模型
==============================

使用 Newton-Raphson 算法 (Fisher Scoring) 从零实现：
- Logit 模型 (Logistic 回归)
- Probit 模型

包含：
- 系数估计、标准误、z 检验
- 边际效应 (Average Marginal Effects)
- 准 R² (McFadden)
- 预测概率、混淆矩阵

理论参考: Wooldridge (2010) Ch.15, Greene (2018) Ch.17
"""

import numpy as np
from numpy.linalg import inv
from scipy import stats as scipy_stats
from scipy.special import expit as logistic
from scipy.stats import norm
from typing import Optional
from .utils import add_constant


def _probit_link(x: np.ndarray) -> np.ndarray:
    """Probit 链接函数 (标准正态 CDF)。"""
    return norm.cdf(x)


def _probit_deriv(x: np.ndarray) -> np.ndarray:
    """Probit 一阶导数 (标准正态 PDF)。"""
    return norm.pdf(x)


class BinaryChoiceResult:
    """二元选择模型结果容器。"""

    def __init__(self):
        self.beta: np.ndarray = None
        self.se: np.ndarray = None
        self.z_stats: np.ndarray = None
        self.p_values: np.ndarray = None
        self.mfx: np.ndarray = None      # 边际效应
        self.mfx_se: np.ndarray = None
        self.pseudo_r2: float = None     # McFadden R²
        self.log_likelihood: float = None
        self.aic: float = None
        self.bic: float = None
        self.n: int = 0
        self.k: int = 0
        self.model_type: str = "logit"
        self.var_names: list = None

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"{'Logit' if self.model_type == 'logit' else 'Probit'} 回归结果",
            "=" * 72,
            f"观测数: {self.n:>6d}          变量数: {self.k:>6d}",
            f"Log-Likelihood: {self.log_likelihood:>12.4f}",
            f"Pseudo R²:      {self.pseudo_r2:>12.4f}",
            f"AIC:            {self.aic:>12.4f}   BIC: {self.bic:>12.4f}",
            "-" * 72,
            f"{'Variable':>16s}  {'Coef.':>10s}  {'Std.Err.':>10s}  {'z':>8s}  {'P>|z|':>8s}",
            "-" * 72,
        ]
        for i in range(self.k):
            name = self.var_names[i] if self.var_names else f"x{i}"
            lines.append(f"{name:>16s}  {self.beta[i]:>10.4f}  "
                         f"{self.se[i]:>10.4f}  {self.z_stats[i]:>8.3f}  "
                         f"{self.p_values[i]:>8.4f}")
        lines.append("-" * 72)
        lines.append("边际效应 (AME):")
        for i in range(self.k):
            name = self.var_names[i] if self.var_names else f"x{i}"
            lines.append(f"  {name:>14s}  {self.mfx[i]:>10.4f}  "
                         f"({self.mfx_se[i]:>8.4f})")
        lines.append("=" * 72)
        return "\n".join(lines)


class Logit:
    """
    Logit 模型 (Logistic 回归)。

    使用 Newton-Raphson / Fisher Scoring 进行极大似然估计。

    使用
    ----
    logit = Logit(y, X)
    result = logit.fit()
    """

    def __init__(self, y: np.ndarray, X: np.ndarray,
                 var_names: Optional[list] = None):
        self.y = np.asarray(y, dtype=float).flatten()
        self.X = add_constant(np.asarray(X, dtype=float))
        self.n, self.k = self.X.shape
        self.var_names = var_names or ["const"] + [f"x{i}" for i in range(self.X.shape[1] - 1)]

    def fit(self, max_iter: int = 100, tol: float = 1e-8) -> BinaryChoiceResult:
        """Newton-Raphson 估计。"""
        X, y = self.X, self.y

        # 初始化 (用 OLS 系数作为初值)
        from .ols import OLS
        beta = OLS(y, X, add_intercept=False).fit().beta

        for iteration in range(max_iter):
            eta = X @ beta
            mu = logistic(eta)    # p(x)
            # 得分函数 (Score)
            score = X.T @ (y - mu)
            # Hessian (Fisher Information 的负值)
            W = np.diag(mu * (1 - mu))
            hessian = -X.T @ W @ X
            # Newton step
            delta = inv(-hessian) @ score
            beta_new = beta + delta

            if np.max(np.abs(delta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        # 最终统计量
        eta = X @ beta
        mu = logistic(eta)
        W = np.diag(mu * (1 - mu))
        vcov = inv(X.T @ W @ X)
        se = np.sqrt(np.diag(vcov))

        # 对数似然
        logL = np.sum(y * np.log(np.maximum(mu, 1e-15)) +
                      (1 - y) * np.log(np.maximum(1 - mu, 1e-15)))
        # 仅截距模型的对数似然
        ybar = np.mean(y)
        logL0 = np.sum(y * np.log(ybar) + (1 - y) * np.log(1 - ybar))

        result = BinaryChoiceResult()
        result.beta = beta
        result.se = se
        result.z_stats = beta / se
        result.p_values = 2 * (1 - norm.cdf(np.abs(beta / se)))
        result.log_likelihood = logL
        result.pseudo_r2 = 1 - logL / logL0
        result.aic = -2 * logL + 2 * self.k
        result.bic = -2 * logL + self.k * np.log(self.n)
        result.n = self.n
        result.k = self.k
        result.model_type = "logit"
        result.var_names = self.var_names

        # 平均边际效应 (AME)
        mu_bar = np.mean(mu)
        mfx_base = mu_bar * (1 - mu_bar)
        result.mfx = beta * mfx_base
        result.mfx_se = se * mfx_base

        self._fitted_beta = beta  # 缓存拟合结果供 predict 使用

        return result

    def predict(self, X_new: Optional[np.ndarray] = None,
                threshold: float = 0.5) -> np.ndarray:
        """预测分类结果。需要先调用 fit()。"""
        if X_new is None:
            X_new = self.X
        else:
            X_new = add_constant(np.asarray(X_new, dtype=float))
        probs = logistic(X_new @ self._fitted_beta)
        return (probs >= threshold).astype(int)

    def predict_proba(self, X_new: Optional[np.ndarray] = None) -> np.ndarray:
        """预测概率。需要先调用 fit()。"""
        if X_new is None:
            X_new = self.X
        else:
            X_new = add_constant(np.asarray(X_new, dtype=float))
        return logistic(X_new @ self._fitted_beta)


class Probit:
    """
    Probit 模型。

    使用 Newton-Raphson 进行极大似然估计。

    使用
    ----
    probit = Probit(y, X)
    result = probit.fit()
    """

    def __init__(self, y: np.ndarray, X: np.ndarray,
                 var_names: Optional[list] = None):
        self.y = np.asarray(y, dtype=float).flatten()
        self.X = add_constant(np.asarray(X, dtype=float))
        self.n, self.k = self.X.shape
        self.var_names = var_names or ["const"] + [f"x{i}" for i in range(self.X.shape[1] - 1)]

    def fit(self, max_iter: int = 100, tol: float = 1e-8) -> BinaryChoiceResult:
        """Newton-Raphson 估计。"""
        X, y = self.X, self.y

        # 初始化
        from .ols import OLS
        beta = OLS(y, X, add_intercept=False).fit().beta

        for iteration in range(max_iter):
            eta = X @ beta
            mu = norm.cdf(eta)
            phi = norm.pdf(eta)

            # 得分函数
            score = X.T @ ((y - mu) * phi / (mu * (1 - mu) + 1e-15))
            # 期望 Hessian (Fisher)
            q = phi ** 2 / (mu * (1 - mu) + 1e-15)
            hessian = -X.T @ np.diag(q) @ X

            delta = inv(X.T @ np.diag(q) @ X) @ X.T @ ((y - mu) * phi / (mu * (1 - mu) + 1e-15))
            beta_new = beta + delta

            if np.max(np.abs(delta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        # 最终统计量
        eta = X @ beta
        mu = norm.cdf(eta)
        phi = norm.pdf(eta)
        q = phi ** 2 / (mu * (1 - mu) + 1e-15)
        vcov = inv(X.T @ np.diag(q) @ X)
        se = np.sqrt(np.diag(vcov))

        logL = np.sum(y * np.log(np.maximum(mu, 1e-15)) +
                      (1 - y) * np.log(np.maximum(1 - mu, 1e-15)))
        ybar = np.mean(y)
        logL0 = np.sum(y * np.log(ybar) + (1 - y) * np.log(1 - ybar))

        result = BinaryChoiceResult()
        result.beta = beta
        result.se = se
        result.z_stats = beta / se
        result.p_values = 2 * (1 - norm.cdf(np.abs(beta / se)))
        result.log_likelihood = logL
        result.pseudo_r2 = 1 - logL / logL0
        result.aic = -2 * logL + 2 * self.k
        result.bic = -2 * logL + self.k * np.log(self.n)
        result.n = self.n
        result.k = self.k
        result.model_type = "probit"
        result.var_names = self.var_names

        # 平均边际效应
        phi_bar = np.mean(norm.pdf(X @ beta))
        result.mfx = beta * phi_bar
        result.mfx_se = se * phi_bar

        return result
