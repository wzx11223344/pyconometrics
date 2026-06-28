"""
PyConometrics 单元测试
=====================
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyconometrics import OLS, IV, DID, PanelFE, PanelRE, Logit, Probit


def test_ols_basic():
    """测试 OLS 基本功能。"""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    y = 3 + 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(n) * 0.5
    
    ols = OLS(y, X)
    result = ols.fit()
    
    # 检查基本输出
    assert result.beta is not None
    assert result.se is not None
    assert result.r2 is not None
    assert result.r2_adj is not None
    assert len(result.beta) == 3  # const + 2 vars
    
    # 系数应接近真实值
    assert abs(result.beta[0] - 3) < 1.0
    assert abs(result.beta[1] - 2) < 0.5
    assert abs(result.beta[2] + 1.5) < 0.5


def test_ols_hc1():
    """测试异方差稳健标准误。"""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 1)
    # 异方差: 误差标准差随 X 增大而增大
    y = 1 + 0.5 * X[:, 0] + np.random.randn(n) * (1 + 0.3 * np.abs(X[:, 0]))
    
    ols = OLS(y, X)
    r_homo = ols.fit(se_type="homosk")
    r_hc1 = ols.fit(se_type="hc1")
    
    assert len(r_homo.se) == 2
    assert len(r_hc1.se) == 2


def test_iv():
    """测试 IV/2SLS。"""
    np.random.seed(123)
    n = 500
    
    z = np.random.randn(n)
    error = np.random.randn(n) * 0.3
    X_endog = 1 + 0.8 * z + error
    noise = np.random.randn(n) * 0.5 + 0.4 * error  # 内生性
    y = 2 + 1.5 * X_endog + noise
    
    iv = IV(y, X_endog, Z_instruments=z.reshape(-1, 1))
    result = iv.fit()
    
    assert hasattr(result, 'first_stage_f')
    assert result.first_stage_f > 5  # 强工具变量
    assert abs(result.beta[0] - 1.5) < 0.5  # 应接近真实值


def test_did():
    """测试 DID。"""
    np.random.seed(42)
    n_units, n_periods = 200, 2
    treated = np.repeat(np.random.binomial(1, 0.5, n_units), n_periods)
    post = np.tile([0, 1], n_units)
    
    unit_fe = np.repeat(np.random.randn(n_units) * 0.3, n_periods)
    y = 5 + unit_fe + 0.5 * post + 1.2 * treated * post + np.random.randn(n_units * n_periods) * 0.2
    
    did = DID(y, treated, post)
    result = did.fit()
    
    assert abs(result.att - 1.2) < 0.3
    assert result.att_pvalue < 0.01  # 应显著


def test_panel_fe():
    """测试面板固定效应。"""
    np.random.seed(42)
    N, T = 50, 5
    n = N * T
    
    alpha = np.repeat(np.random.randn(N) * 0.5, T)
    x = np.random.randn(n)
    y = 2 + 1.5 * x + alpha + np.random.randn(n) * 0.3
    ids = np.repeat(range(N), T)
    
    fe = PanelFE(y, x, ids)
    result = fe.fit()
    
    assert abs(result.beta[0] - 1.5) < 0.15
    assert result.p_values[0] < 0.01


def test_panel_re():
    """测试面板随机效应。"""
    np.random.seed(42)
    N, T = 50, 5
    n = N * T
    
    x = np.random.randn(n)
    y = 2 + 1.5 * x + np.random.randn(n) * 0.4
    ids = np.repeat(range(N), T)
    
    re = PanelRE(y, x, ids)
    result = re.fit()
    
    assert abs(result.beta[1] - 1.5) < 0.3  # beta[0] is const


def test_logit():
    """测试 Logit 模型。"""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 2)
    
    # 生成二元因变量
    eta = 0.5 + 1.5 * X[:, 0] - 1.0 * X[:, 1]
    prob = 1 / (1 + np.exp(-eta))
    y = (np.random.rand(n) < prob).astype(float)
    
    logit = Logit(y, X)
    result = logit.fit()
    
    assert result.pseudo_r2 is not None
    assert result.log_likelihood < 0
    assert len(result.mfx) == 3  # beta + 2 vars


def test_probit():
    """测试 Probit 模型。"""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 1)
    
    eta = 0.5 + 2.0 * X[:, 0]
    prob = 1 / (1 + np.exp(-eta))
    y = (np.random.rand(n) < prob).astype(float)
    
    probit = Probit(y, X)
    result = probit.fit()
    
    assert result.pseudo_r2 is not None
    assert len(result.beta) == 2


if __name__ == "__main__":
    tests = [
        ("OLS 基本功能", test_ols_basic),
        ("OLS HC1 标准误", test_ols_hc1),
        ("IV/2SLS", test_iv),
        ("DID", test_did),
        ("面板固定效应", test_panel_fe),
        ("面板随机效应", test_panel_re),
        ("Logit 模型", test_logit),
        ("Probit 模型", test_probit),
    ]
    
    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    print(f"\n{passed}/{len(tests)} 测试通过")
