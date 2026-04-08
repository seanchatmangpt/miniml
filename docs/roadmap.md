# miniml Development Roadmap

## Overview

miniml is a **browser-native AutoML library** that combines comprehensive algorithm coverage with genetic algorithm feature selection and PSO hyperparameter optimization — all in ~145KB gzipped.

**Current Status:** ✅ **Production Ready** — 70+ algorithms, 15 families, 62 modules, 110 benchmarks

---

## Completed Features ✅

### Core ML Algorithms (70+ across 15 families)

| Family | Algorithms | Status |
|--------|------------|--------|
| **Classification** | KNN, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Naive Bayes, Logistic Regression, Perceptron, Linear SVM | ✅ Complete |
| **Regression** | Linear, Ridge, Lasso, Polynomial, Exponential, Logarithmic, Power, Elastic Net, SVR, Quantile | ✅ Complete |
| **Clustering** | K-Means, K-Means++, DBSCAN, Hierarchical | ✅ Complete |
| **Preprocessing** | Standard Scaler, MinMax Scaler, Robust Scaler, Normalizer, Label Encoder, One-Hot Encoder, Ordinal Encoder, Imputer | ✅ Complete |
| **Dimensionality Reduction** | PCA | ✅ Complete |
| **Time Series** | SMA, EMA, WMA, Exponential Smoothing, Trend Forecast, Peak/Trough Detection, Momentum, Rate of Change | ✅ Complete |
| **Metrics** | Confusion Matrix, Classification Report, Silhouette Score, ROC AUC, Log Loss, Accuracy, R², RMSE, MAE | ✅ Complete |
| **Ensemble Methods** | Random Forest, Gradient Boosting, AdaBoost | ✅ Complete |
| **Optimization Suite** | Genetic Algorithms, PSO, Simulated Annealing, Multi-Armed Bandit | ✅ Complete |
| **Advanced Analytics** | Drift Detection (Jaccard, Statistical, Page-Hinkley), Anomaly Detection (Isolation Forest, Statistical), Prediction Intervals | ✅ Complete |
| **Probabilistic Methods** | Monte Carlo (Integration, Multidim, Bootstrap, Pi Estimation), Markov Chains (Steady State, N-Step, Simulation), HMM (Forward, Viterbi, Backward, Baum-Welch), MCMC (Metropolis-Hastings) | ✅ Complete |
| **Statistical Distributions** | Normal, Binomial, Poisson, Exponential, Chi-Squared, Student's t, F Distribution | ✅ Complete |
| **Statistical Inference** | t-Tests (One-Sample, Two-Sample, Paired, Welch's), Mann-Whitney U, Wilcoxon Signed-Rank, Chi-Square, ANOVA, Descriptive Statistics | ✅ Complete |
| **Kernel Methods** | RBF, Polynomial, Sigmoid | ✅ Complete |
| **Bayesian Methods** | Bayesian Estimation (MCMC), Bayesian Linear Regression | ✅ Complete |
| **Gaussian Processes** | GP Fit, GP Predict | ✅ Complete |
| **Survival Analysis** | Kaplan-Meier, Cox Proportional Hazards | ✅ Complete |
| **Association Rules** | Apriori | ✅ Complete |
| **Recommendation Systems** | Matrix Factorization, User-User Collaborative | ✅ Complete |
| **Graph Algorithms** | PageRank, Shortest Path, Community Detection | ✅ Complete |
| **Extended Regression** | Elastic Net, SVR, Quantile Regression | ✅ Complete |

### AutoML Suite

| Feature | Status | Notes |
|---------|--------|-------|
| Genetic Algorithm Feature Selection | ✅ Complete | 50-100 generations, 5-fold CV |
| PSO Hyperparameter Optimization | ✅ Complete | Swarm-based optimization |
| Algorithm Selection | ✅ Complete | Tests multiple algorithms, selects best |
| Progress Monitoring | ✅ Complete | Real-time callbacks |
| Result Interpretation | ✅ Complete | Human-readable rationales |

### Performance Features

| Feature | Status | Notes |
|---------|--------|-------|
| SIMD Acceleration | ✅ Complete | WASM v128 intrinsics, 4-100x speedup |
| Zero-Allocation Hot Paths | ✅ Complete | Pre-allocated buffers |
| Algorithmic Optimizations | ✅ Complete | Partial sort, priority queues |
| Multi-Worker Parallelism | ✅ Complete | Parallel cross-validation |

### Benchmark Coverage

| Metric | Value |
|--------|-------|
| Total Benchmarks | 110 |
| Categories | 26 |
| Fast (<1ms) | 56 (51%) |
| Moderate (1-100ms) | 44 (40%) |
| Slow (>100ms) | 10 (9%) |

---

## Future Enhancements 🚧

### High Priority

| Feature | Impact | Complexity | Timeline |
|---------|--------|------------|----------|
| **Model Persistence** | High | Low | 2-3 weeks |
| **Explainability (SHAP/LIME)** | High | Medium | 3-4 weeks |
| **Neural Network Primitives** | High | Medium | 4-6 weeks |
| **Transfer Learning (ONNX)** | High | Medium | 3-4 weeks |

### Medium Priority

| Feature | Impact | Complexity | Timeline |
|---------|--------|------------|----------|
| **DataFrame Operations** | Medium | Low | 2-3 weeks |
| **Data Augmentation (SMOTE)** | Medium | Low | 1-2 weeks |
| **Ensemble Stacking** | Medium | Medium | 2-3 weeks |
| **Advanced CV (Stratified, Time Series)** | Medium | Low | 1-2 weeks |

### Low Priority

| Feature | Impact | Complexity | Timeline |
|---------|--------|------------|----------|
| **Multi-Objective Optimization** | Low | High | 4-6 weeks |
| **Causal Inference Extensions** | Low | High | 4-6 weeks |
| **Model Compression** | Low | Medium | 3-4 weeks |

---

## Completed Milestones

### ✅ Phase 1: Core ML Foundation (Complete)
- [x] Classification algorithms (9)
- [x] Regression algorithms (7)
- [x] Clustering algorithms (4)
- [x] Preprocessing (8)
- [x] Dimensionality reduction (1)
- [x] Metrics (9)
- [x] Time series (10)

### ✅ Phase 2: Advanced Algorithms (Complete)
- [x] Ensemble methods (3)
- [x] Optimization suite (4)
- [x] Advanced analytics (3)
- [x] Neural network primitives (2)

### ✅ Phase 3: AutoML (Complete)
- [x] Genetic algorithm feature selection
- [x] PSO hyperparameter optimization
- [x] Algorithm selection
- [x] Progress monitoring
- [x] Result interpretation

### ✅ Phase 4: Probabilistic Methods (Complete)
- [x] Monte Carlo methods (4)
- [x] Markov chains (4)
- [x] Hidden Markov models (4)
- [x] MCMC (1)

### ✅ Phase 5: Statistical Foundations (Complete)
- [x] Statistical distributions (7)
- [x] Hypothesis testing (7)

### ✅ Phase 6: Advanced ML (Complete)
- [x] Kernel methods (3)
- [x] Bayesian methods (2)
- [x] Gaussian processes (2)
- [x] Extended regression (3)

### ✅ Phase 7: Specialized Domains (Complete)
- [x] Survival analysis (2)
- [x] Association rules (1)
- [x] Recommendation systems (2)
- [x] Graph algorithms (3)

---

## Version History

| Version | Date | Changes |
|--------|------|---------|
| **1.0** | 2026-04-08 | 70+ algorithms, 15 families, 110 benchmarks |
| **0.9** | 2026-03-XX | Added probabilistic methods and statistical inference |
| **0.8** | 2026-03-XX | Initial AutoML suite |
| **0.5** | 2026-01-XX | Core ML algorithms |

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to miniml.

---

## License

BSL 1.1 © 2026 Sean Chatman

See [LICENSE](../LICENSE) for details.
