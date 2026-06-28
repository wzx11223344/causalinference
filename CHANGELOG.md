# Changelog

All notable changes to CausalInference will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2024-06-15

### Added

- **Double/Debiased ML (DML)**: Full Chernozhukov et al. (2018) implementation including:
  - Neyman orthogonal score for ATE and ATTE
  - K-fold cross-fitting to remove Donsker conditions
  - Bootstrap confidence intervals
  - Support for both continuous (Partialling-out) and binary treatments
- **Causal Forest**: Athey et al. (2019) Generalized Random Forest implementation including:
  - Honest splitting (half-sampling for splitting vs estimation)
  - Double-robust treatment effect estimation in each leaf
  - Conditional Average Treatment Effects (CATE)
  - Variable importance for heterogeneity drivers
- **Meta-Learners (Kunzel et al. 2019)**:
  - `SLearner`: Single model with treatment as feature
  - `TLearner`: Separate outcome models by treatment status
  - `XLearner`: Cross-estimation of CATE with propensity weighting
- **Inference toolkit**:
  - `bootstrap_ci()`: Bootstrap percentile confidence intervals
  - `doubly_robust_score()`: DR score computation
  - `cross_fitting()`: K-fold sample splitting utility
  - `permutation_test()`: Fisher exact randomization test
  - `summary_table()`: Multi-method comparison table
- Example scripts and demo
- Full README with theory documentation

### Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

---

## [Unreleased]

### Planned

- Orthogonal Random Forest (Oprescu et al. 2019)
- Causal BART (Hill 2011)
- Interactive Regression Model (IRM) for DML
- Riesz Representer approach for continuous treatments (Chernozhukov et al. 2022)
- Conformal inference for CATE intervals
- Uniform confidence bands for CATE functions
- Support for panel/panel-data treatment effect estimation
