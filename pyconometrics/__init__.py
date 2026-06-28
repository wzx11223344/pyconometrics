"""
PyConometrics — 从零实现的计量经济学 Python 库
=============================================

一个纯 Python + NumPy 实现的计量经济学工具库，包含常用计量模型的完整实现。
所有模型均从底层矩阵运算构建，不依赖 statsmodels 等高级统计库，
适合学习计量经济学原理、教学演示和轻量级实证研究。

Models included:
    - OLS (普通最小二乘回归)
    - IV/2SLS (工具变量 / 两阶段最小二乘)
    - DID (双重差分)
    - RDD (断点回归设计)
    - Panel FE/RE (面板数据固定/随机效应)
    - Logit/Probit (二元选择模型)
"""

from .ols import OLS
from .iv import IV, TSLS
from .did import DID, EventStudy
from .rdd import SharpRDD, FuzzyRDD
from .panel import PanelFE, PanelRE, HausmanTest
from .logit import Logit, Probit

__version__ = "1.0.0"
__author__ = "wzx11223344"
__all__ = [
    "OLS",
    "IV", "TSLS",
    "DID", "EventStudy",
    "SharpRDD", "FuzzyRDD",
    "PanelFE", "PanelRE", "HausmanTest",
    "Logit", "Probit",
]
