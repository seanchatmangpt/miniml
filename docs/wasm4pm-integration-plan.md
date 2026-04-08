# wasm4pm → micro-ml Integration Plan

## Overview

Analysis of `~/chatmangpt/wasm4pm` reveals several statistical/ML algorithms that would enhance **micro-ml**. Below is a categorized plan for integration.

---

## 1. Survival Analysis (NEW CATEGORY)

### Source: `prediction_remaining_time.rs`

| Algorithm | Description | Priority |
|-----------|-------------|----------|
| **Weibull Distribution** | Survival analysis with shape/scale parameters | High |
| **Hazard Rate Estimation** | Instantaneous failure rate from Weibull model | High |
| **Gamma Function** | Lanczos approximation for Γ(x) | Medium |
| **Survival Probability** | P(T > t) from cumulative hazard | High |

**Use cases:** Reliability engineering, churn prediction, time-to-event modeling

**Implementation complexity:** Medium (requires gamma function, statistical fitting)

---

## 2. Feature Engineering (NEW CATEGORY)

### Source: `feature_extraction.rs`, `prediction_features.rs`

| Algorithm | Description | Priority |
|-----------|-------------|----------|
| **Prefix Features** | Extract features from sequence prefixes | Medium |
| **Rework Score** | Count activity repetitions | Low |
| **Activity Counts** | Frequency encoding of categorical features | Low |
| **Trace Statistics** | Length, elapsed time, unique activities | Low |
| **Inter-Event Time** | Average time between events | Low |

**Use cases:** Preprocessing for sequence prediction, feature extraction from logs

**Implementation complexity:** Low-Medium (data transformation, no training)

---

## 3. Feature Importance Enhancement

### Source: `feature_importance.rs`

**Note:** micro-ml already has `permutation_importance.rs`

| Enhancement | Description | Priority |
|------------|-------------|----------|
| **Global Permutation Importance** | Aggregate importance across dataset | Medium |
| **Normalized Importance** | Sum-to-1 normalization (already exists) | Low |
| **Confidence-Weighted Importance** | Weight by sample count/variance | Low |

**Use cases:** Model interpretation, feature selection

---

## 4. Ensemble Methods Enhancement

### Source: `ensemble.rs`

| Enhancement | Description | Priority |
|------------|-------------|----------|
| **Ensemble Discovery** | Run multiple algorithms, rank by quality | Medium |
| **Consensus Scoring** | Agreement score between models | Low |
| **Quality-Weighted Prediction** | Weight predictions by model quality | Medium |
| **Pruned Ensemble** | Remove low-quality models/edges | Low |

**Use cases:** Model comparison, robustness through diversity

---

## 5. Statistical Distributions (NEW CATEGORY)

### Source: `prediction_remaining_time.rs` (Weibull)

| Distribution | Description | Priority |
|-------------|-------------|----------|
| **Weibull** | Shape/scale parameters, survival function | High |
| **Exponential** | Special case of Weibull (k=1) | Medium |
| **Log-Normal** | Log-normal survival (future addition) | Low |
| **Gamma** | Gamma distribution (future addition) | Low |

**Use cases:** Reliability, survival analysis, churn modeling

---

## 6. Sequence Prediction (NEW CATEGORY)

### Source: `prediction_next_activity.rs`

| Algorithm | Description | Priority |
|-----------|-------------|----------|
| **NGram Predictor** | Markov chain of order n | Medium |
| **Sequence Prefix Features** | Encode sequences for ML | Medium |
| **Next-Activity Probability** | Top-k predictions | Low |

**Use cases:** Process mining, sequence prediction, next-item recommendation

---

## 7. Drift Detection (NEW CATEGORY)

### Source: `prediction_drift.rs`

| Algorithm | Description | Priority |
|-----------|-------------|----------|
| **EWMA Drift** | Exponentially weighted moving average drift | Medium |
| **Jaccard Window Drift** | Sliding window similarity | Medium |
| **Concept Drift Detection** | Detect process changes over time | High |

**Use cases:** Model monitoring, data pipeline validation, adaptive systems

---

## 8. Anomaly Detection Enhancements

### Source: `anomaly.rs`

| Enhancement | Description | Priority |
|------------|-------------|----------|
| **Statistical Anomaly** | Z-score, IQR-based detection | Medium |
| **Sequence Anomaly** | Deviation from expected patterns | High |
| **Boundary Coverage** | Out-of-bounds detection | Medium |

---

## 9. Validation Metrics (ENHANCEMENT)

### Source: `validation.rs`, `data_quality.rs`

| Metric | Description | Priority |
|--------|-------------|----------|
| **Statistical Tests** | Chi-square, Kolmogorov-Smirnov | Medium |
| **Quality Scores** | Completeness, consistency metrics | Low |
| **Coverage Metrics** | Trace/attribute coverage | Low |

---

## Priority Implementation Order

### Phase 1 (High Value, Low-Medium Complexity)
1. **Weibull Distribution + Survival Analysis** - New category for micro-ml
2. **Drift Detection** - EWMA, Jaccard window
3. **NGram Sequence Prediction** - Markov chains for sequences
4. **Global Feature Importance** - Enhancement to existing module

### Phase 2 (Medium Value, Medium Complexity)
5. **Feature Engineering Module** - Prefix features, rework score
6. **Ensemble Discovery** - Quality-based model ranking
7. **Statistical Distributions** - Expand beyond Weibull

### Phase 3 (Nice to Have)
8. **Quality-Weighted Predictions** - Ensemble enhancement
9. **Statistical Tests** - Chi-square, KS test
10. **Boundary Coverage** - Anomaly detection enhancement

---

## Estimated Implementation Effort

| Phase | Algorithms | Estimated Lines | Complexity |
|-------|------------|-----------------|------------|
| Phase 1 | 4 | ~800 LOC | Medium |
| Phase 2 | 3 | ~600 LOC | Medium |
| Phase 3 | 3 | ~400 LOC | Low-Medium |

**Total:** ~10 new modules, ~1800 LOC

---

## Files to Reference from wasm4pm

1. `wasm4pm/src/prediction_remaining_time.rs` - Weibull, gamma, survival
2. `wasm4pm/src/prediction_drift.rs` - Drift detection
3. `wasm4pm/src/prediction_next_activity.rs` - NGram predictor
4. `wasm4pm/src/feature_extraction.rs` - Feature engineering
5. `wasm4pm/src/ensemble.rs` - Ensemble discovery
6. `wasm4pm/src/feature_importance.rs` - Global importance
7. `wasm4pm/src/validation.rs` - Statistical tests
8. `wasm4pm/src/anomaly.rs` - Statistical anomaly detection

---

## Integration Notes

1. **WASM Constraints:** All implementations must use zero external deps
2. **Flat Arrays:** Follow micro-ml convention for matrix representation
3. **Error Handling:** Use `Result<T, MlError>` pattern
4. **Testing:** Each module needs `#[cfg(test)]` tests
5. **Documentation:** Add to `docs/api/reference.md` and `docs/overview/algorithm-guide.md`

---

## Next Steps

1. Confirm priority with user
2. Start with Phase 1 (Weibull, Drift, NGram)
3. Create feature branch: `feat/wasm4pm-integration`
4. Implement modules following micro-ml patterns
5. Update documentation and tests
