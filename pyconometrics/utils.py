"""
PyConometrics 工具函数模块

提供计量分析中常用的统计函数：系数标准误计算、t检验、F检验、
置信区间、残差诊断等。
"""

import numpy as np
from numpy.linalg import inv, pinv
from scipy import stats as scipy_stats


def add_constant(X: np.ndarray) -> np.ndarray:
    """
    为自变量矩阵添加截距项（常数列）。

    参数
    ----
    X : np.ndarray, shape (n, k)
        自变量矩阵。

    返回
    ----
    np.ndarray, shape (n, k+1)
        添加上常数列后的矩阵。
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1))
    # 检查是否已经包含常数项（全1列）
    for j in range(X.shape[1]):
        if np.allclose(X[:, j], 1.0):
            return X
    return np.hstack([ones, X])


def se_homosk(beta: np.ndarray, X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """
    同方差假设下的系数标准误（经典OLS标准误）。

    参数
    ----
    beta : np.ndarray, shape (k,)
    X : np.ndarray, shape (n, k)
    residuals : np.ndarray, shape (n,)

    返回
    ----
    np.ndarray, shape (k,)
    """
    n, k = X.shape
    sigma2 = np.sum(residuals ** 2) / (n - k)
    cov = sigma2 * inv(X.T @ X)
    return np.sqrt(np.diag(cov))


def se_robust_hc1(beta: np.ndarray, X: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """
    异方差稳健标准误（HC1，Stata 默认）。

    参数
    ----
    beta : np.ndarray, shape (k,)
    X : np.ndarray, shape (n, k)
    residuals : np.ndarray, shape (n,)

    返回
    ----
    np.ndarray, shape (k,)
    """
    n, k = X.shape
    XtX_inv = inv(X.T @ X)
    # 残差对角矩阵 (n×n) 的二次型
    S = (X * residuals.reshape(-1, 1) ** 2).T @ X  # 等价于 X' diag(e²) X
    cov = XtX_inv @ S @ XtX_inv
    # HC1 修正因子
    cov *= n / (n - k)
    return np.sqrt(np.diag(cov))


def se_clustered(beta: np.ndarray, X: np.ndarray, residuals: np.ndarray,
                 cluster: np.ndarray) -> np.ndarray:
    """
    聚类稳健标准误（Clustered Standard Errors）。

    参数
    ----
    beta : np.ndarray, shape (k,)
    X : np.ndarray, shape (n, k)
    residuals : np.ndarray, shape (n,)
    cluster : np.ndarray, shape (n,)
        聚类标识向量。

    返回
    ----
    np.ndarray, shape (k,)
    """
    n, k = X.shape
    cluster_ids = np.unique(cluster)
    g = len(cluster_ids)
    XtX_inv = inv(X.T @ X)
    
    # 计算聚类层面的得分贡献
    score_sum = np.zeros((k, k))
    for cid in cluster_ids:
        mask = cluster == cid
        X_c = X[mask]
        e_c = residuals[mask]
        s_c = X_c.T @ e_c
        score_sum += np.outer(s_c, s_c)
    
    cov = XtX_inv @ score_sum @ XtX_inv
    # 小样本修正
    cov *= (g / (g - 1)) * ((n - 1) / (n - k))
    return np.sqrt(np.diag(cov))


def t_test(beta: np.ndarray, se: np.ndarray, df: int) -> tuple:
    """
    对每个系数进行 t 检验。

    参数
    ----
    beta : np.ndarray
    se : np.ndarray
    df : int

    返回
    ----
    t_stats, p_values : tuple of np.ndarray
    """
    t_stats = beta / se
    p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stats), df))
    return t_stats, p_values


def f_test(r_matrix: np.ndarray, q_vector: np.ndarray,
           beta: np.ndarray, vcov: np.ndarray, df_resid: int) -> tuple:
    """
    线性假设的 F 检验：H0: Rβ = q

    参数
    ----
    r_matrix : np.ndarray, shape (m, k)
    q_vector : np.ndarray, shape (m,)
    beta : np.ndarray, shape (k,)
    vcov : np.ndarray, shape (k, k)
    df_resid : int

    返回
    ----
    f_stat, p_value : tuple of float
    """
    m = r_matrix.shape[0]
    diff = r_matrix @ beta - q_vector
    f_stat = (diff.T @ inv(r_matrix @ vcov @ r_matrix.T) @ diff) / m
    p_value = 1 - scipy_stats.f.cdf(f_stat, m, df_resid)
    return float(f_stat), float(p_value)


def confidence_interval(beta: np.ndarray, se: np.ndarray,
                        df: int, alpha: float = 0.05) -> np.ndarray:
    """
    计算系数的置信区间。

    返回
    ----
    np.ndarray, shape (k, 2)
    """
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
    lower = beta - t_crit * se
    upper = beta + t_crit * se
    return np.column_stack([lower, upper])


def r_squared(y: np.ndarray, y_hat: np.ndarray, k: int) -> tuple:
    """
    计算 R² 和调整后 R²。

    返回
    ----
    r2, r2_adj : tuple of float
    """
    n = len(y)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return r2, r2_adj


def durbin_watson(residuals: np.ndarray) -> float:
    """计算 Durbin-Watson 统计量（检验一阶自相关）。"""
    diff = np.diff(residuals)
    return np.sum(diff ** 2) / np.sum(residuals ** 2)


def vif(X: np.ndarray) -> np.ndarray:
    """计算方差膨胀因子（VIF），用于诊断多重共线性。"""
    k = X.shape[1]
    vif_values = np.zeros(k)
    for j in range(k):
        y_j = X[:, j]
        X_others = np.delete(X, j, axis=1)
        beta_j = pinv(X_others.T @ X_others) @ X_others.T @ y_j
        y_hat_j = X_others @ beta_j
        ss_res = np.sum((y_j - y_hat_j) ** 2)
        ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
        r2_j = 1 - ss_res / ss_tot
        vif_values[j] = 1 / (1 - r2_j) if r2_j < 1 else np.inf
    return vif_values
