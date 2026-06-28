# Changelog

All notable changes to PyConometrics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2024-06-15

### Added

- **OLS**: Ordinary Least Squares with HC1 robust standard errors, clustered SE, VIF, Durbin-Watson, AIC/BIC
- **IV / TSLS**: Two-Stage Least Squares with first-stage F-statistic and Sargan over-identification test
- **DID**: Classic Difference-in-Differences (2x2) estimator with cluster-robust inference
- **Event Study**: Dynamic TWFE event study with customizable base period
- **Sharp RDD**: Sharp Regression Discontinuity Design with local linear regression, triangular / Epanechnikov / uniform kernels, IK optimal bandwidth selection
- **Fuzzy RDD**: Fuzzy RDD with two-stage local linear estimation
- **Panel FE**: Panel Fixed Effects (Within transformation / LSDV)
- **Panel RE**: Panel Random Effects (GLS estimation with Swamy-Arora variance components)
- **Hausman Test**: FE vs RE specification test
- **Logit / Probit**: Binary choice models via Newton-Raphson MLE with marginal effects, McFadden pseudo-R^2
- **Utility functions**: `add_constant()`, `cluster_se()`, `f_stat()`
- Example scripts for all model families
- GitHub Actions CI pipeline (`test.yml`)
- Full documentation in README.md

### Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0

---

## [Unreleased]

### Planned

- GMM (Generalized Method of Moments) with optimal weighting matrix
- Quantile Regression
- Heckman Two-Step Selection Correction
- Synthetic Control Method
- Bartlett kernel for Newey-West HAC standard errors
- Bootstrap confidence intervals as alternative to asymptotic SE
