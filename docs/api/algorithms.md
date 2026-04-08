# Algorithm Details

Per-algorithm deep-dive with mathematical formulas, complexity, and implementation notes.

---

## Linear Regression

**Formula:** `y = slope * x + intercept`

`slope = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²`
`intercept = ȳ - slope * x̄`

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |
| WASM impact | ~0.5KB |

---

## Ridge Regression

**Formula:** Minimize `Σ(yi - ŷi)² + α * Σ(wj²)`

Closed-form solution: `w = (XᵀX + αI)⁻¹Xᵀy`

| Metric | Value |
|--------|-------|
| Time | O(n * f² + f³) |
| Space | O(f²) |
| WASM impact | ~2KB |

---

## Lasso Regression

**Formula:** Minimize `Σ(yi - ŷi)² + α * Σ|wj|`

Coordinate descent: iteratively optimize each coefficient while holding others fixed.

| Metric | Value |
|--------|-------|
| Time | O(n * f * max_iter) |
| Space | O(f) |
| WASM impact | ~2KB |

---

## Elastic Net

**Formula:** Minimize `Σ(yi - ŷi)² + α * (l1_ratio * Σ|wj| + (1 - l1_ratio) * Σ(wj²))`

Combines L1 (Lasso) and L2 (Ridge) penalties via coordinate descent.

| Metric | Value |
|--------|-------|
| Time | O(n * f * max_iter) |
| Space | O(f) |
| WASM impact | ~3KB |

---

## K-Means

**Algorithm:** Lloyd's algorithm with random or K-Means++ initialization.

1. Initialize k centroids
2. Assign each point to nearest centroid
3. Recompute centroids as cluster means
4. Repeat until convergence

**K-Means++ initialization:** Select first centroid randomly, then each subsequent centroid with probability proportional to squared distance from nearest existing centroid.

| Metric | Value |
|--------|-------|
| Time | O(n * k * f * max_iter) |
| Space | O(n * k + k * f) |
| WASM impact | ~3KB |

---

## DBSCAN

**Algorithm:** Density-based spatial clustering.

1. For each point, count neighbors within radius `eps`
2. Core points have ≥ `minPoints` neighbors
3. Connect core points into clusters
4. Border points assigned to nearest core's cluster
5. Noise points labeled -1

| Metric | Value |
|--------|-------|
| Time | O(n²) worst case, O(n log n) with indexing |
| Space | O(n²) |
| WASM impact | ~4KB |

---

## Decision Tree

**Algorithm:** Greedy top-down recursive partitioning.

1. For each feature, find best split (Gini impurity or variance reduction)
2. Split data at best threshold
3. Recurse on each subset
4. Stop at max_depth or min_samples_split

**Gini impurity:** `G = 1 - Σ(pi²)` where pi is the proportion of class i.

**Variance reduction:** `VR = Var(parent) - weighted_avg(Var(children))`

| Metric | Value |
|--------|-------|
| Time | O(n * f * n * log n) build, O(depth) predict |
| Space | O(n) |
| WASM impact | ~5KB |

---

## Random Forest

**Algorithm:** Bootstrap aggregating of decision trees.

1. Sample n points with replacement (bootstrap)
2. Train a decision tree on the sample
3. Use random subset of features at each split
4. Repeat for n_trees
5. Predict by majority vote (classification) or averaging (regression)

| Metric | Value |
|--------|-------|
| Time | O(n_trees * n * f * log n) build, O(n_trees * depth) predict |
| Space | O(n_trees * n) |
| WASM impact | ~3KB |

---

## Gradient Boosting

**Algorithm:** Sequential additive modeling.

1. Initialize with base prediction (mean)
2. For each tree:
   a. Compute residuals: `ri = yi - ŷi`
   b. Fit decision tree to residuals
   c. Update predictions: `ŷi += learning_rate * tree(xi)`
3. Repeat for n_trees

| Metric | Value |
|--------|-------|
| Time | O(n_trees * n * f * log n) build, O(n_trees * depth) predict |
| Space | O(n_trees * n) |
| WASM impact | ~3KB |

---

## AdaBoost

**Algorithm:** Adaptive Boosting with decision stumps.

1. Initialize uniform sample weights
2. For each iteration:
   a. Fit decision stump (single-feature threshold)
   b. Compute weighted error: `ε = Σ(wi * I(wrong))`
   c. Compute stump weight: `α = 0.5 * ln((1-ε)/ε)`
   d. Update weights: `wi *= exp(α * yi * prediction)`
3. Predict by weighted vote

| Metric | Value |
|--------|-------|
| Time | O(n_estimators * n * f) build, O(n_estimators) predict |
| Space | O(n_estimators) |
| WASM impact | ~3KB |

---

## SVM (Linear)

**Formula:** Minimize `0.5 * ||w||² + C * Σ(max(0, 1 - yi(w·xi + b)))`

Soft-margin SVM via SGD:
- `w = w - lr * (C * ∂loss/∂w + λ * w)` if margin violation
- `w = w - lr * λ * w` otherwise

| Metric | Value |
|--------|-------|
| Time | O(n * f * max_iter) |
| Space | O(f) |
| WASM impact | ~2KB |

---

## Logistic Regression

**Formula:** `P(y=1|x) = 1 / (1 + exp(-(w·x + b)))`

Gradient descent: `w -= lr * (1/n) * Σ(ŷi - yi) * xi`

| Metric | Value |
|--------|-------|
| Time | O(n * f * max_iter) |
| Space | O(f) |
| WASM impact | ~2KB |

---

## Gaussian Naive Bayes

**Formula:** `P(y|x) ∝ P(y) * Π P(xi|y)`

With Gaussian assumption: `P(xi|y) = N(μ_yi, σ²_yi)`

| Metric | Value |
|--------|-------|
| Time | O(n * f) fit, O(n * f * n_classes) predict |
| Space | O(n_classes * f) |
| WASM impact | ~2KB |

---

## PCA

**Algorithm:** Eigendecomposition of covariance matrix.

1. Compute mean of each feature
2. Center data: `X_centered = X - mean`
3. Compute covariance: `C = XᵀX / (n-1)`
4. Eigendecomposition (power iteration)
5. Project onto top k eigenvectors

| Metric | Value |
|--------|-------|
| Time | O(n * f² + f³) |
| Space | O(f²) |
| WASM impact | ~4KB |

---

## Isolation Forest

**Algorithm:** Random partitioning for anomaly detection.

1. Sample subset of data
2. Build tree by randomly selecting feature and split value
3. Shorter average path length = more anomalous
4. Anomaly score: `2^(-E(h)/c(n))` where E(h) is average path length

| Metric | Value |
|--------|-------|
| Time | O(n_trees * sample_size * log(sample_size)) |
| Space | O(n_trees * sample_size) |
| WASM impact | ~4KB |

---

## LOF (Local Outlier Factor)

**Algorithm:** Density-based outlier detection.

1. For each point, find k-nearest neighbors
2. Compute local reachability density (LRD)
3. LOF = average(LRD of neighbors) / LRD of point
4. LOF > 1 indicates outlier

| Metric | Value |
|--------|-------|
| Time | O(n²) |
| Space | O(n * k) |
| WASM impact | ~3KB |

---

## GMM (Gaussian Mixture Models)

**Algorithm:** Expectation-Maximization.

1. Initialize means (k-means), covariances (identity), weights (uniform)
2. E-step: Compute posterior probabilities
3. M-step: Update means, covariances, weights
4. Repeat until convergence

| Metric | Value |
|--------|-------|
| Time | O(n * k * f² * max_iter) |
| Space | O(k * f²) |
| WASM impact | ~5KB |

---

## Spectral Clustering

**Algorithm:** Graph-based clustering via eigendecomposition.

1. Compute similarity matrix: `W_ij = exp(-||xi-xj||² / 2σ²)`
2. Compute degree matrix: `D = diag(W·1)`
3. Compute normalized Laplacian: `L = I - D^(-1/2) W D^(-1/2)`
4. Eigendecompose L
5. K-means on bottom k eigenvectors

| Metric | Value |
|--------|-------|
| Time | O(n² * f + n³) |
| Space | O(n²) |
| WASM impact | ~5KB |

---

## RANSAC

**Algorithm:** Random Sample Consensus for robust regression.

1. Randomly select minimal sample (f+1 points for f features)
2. Fit model to sample
3. Count inliers (points within threshold)
4. If best so far, refit on all inliers
5. Repeat for max_iterations

| Metric | Value |
|--------|-------|
| Time | O(max_iter * n * f) |
| Space | O(f) |
| WASM impact | ~3KB |

---

## Theil-Sen Estimator

**Algorithm:** Median of pairwise slopes.

For each pair of points (i,j): `slope_ij = (yj - yi) / (xj - xi)`
Final slope = median of all pairwise slopes.

| Metric | Value |
|--------|-------|
| Time | O(n²) |
| Space | O(n²) |
| WASM impact | ~2KB |

---

## Weibull Survival Analysis

**Formula:** `S(t) = exp(-(t/λ)^k)`

- **Shape parameter k:** Controls hazard function behavior
  - k < 1: Decreasing hazard (infant mortality)
  - k = 1: Constant hazard (exponential distribution)
  - k > 1: Increasing hazard (aging/wear-out)
- **Scale parameter λ:** Characteristic life scale

**Hazard rate:** `h(t) = (k/λ) * (t/λ)^(k-1)`

**Gamma function:** Lanczos approximation with g=7 coefficients for scale estimation.

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |
| WASM impact | ~3KB |

**Use cases:** Reliability engineering, churn prediction, time-to-event modeling

---

## EWMA Drift Detection

**Formula:** `ewma_t = λ * value + (1-λ) * ewma_{t-1}`

Drift detected when: `|ewma_t - expected| > threshold`

Exponentially Weighted Moving Average (EWMA) gives more weight to recent observations while maintaining smooth trend estimates.

| Metric | Value |
|--------|-------|
| Time | O(n) single-pass |
| Space | O(1) |
| WASM impact | ~2KB |

**Use cases:** Data pipeline monitoring, model performance tracking, concept drift detection

---

## Jaccard Drift Detection

**Formula:** `J(A,B) = |A ∩ B| / |A ∪ B|`

Window-based drift detection for categorical data:
1. Maintain reference window of categorical values
2. Compute Jaccard similarity with current window
3. Signal drift when similarity drops below threshold

| Metric | Value |
|--------|-------|
| Time | O(n * w²) where w is window size |
| Space | O(w * f) |
| WASM impact | ~3KB |

**Use cases:** Categorical data stream monitoring, feature distribution shift detection

---

## NGram Sequence Prediction

**Formula:** `P(next|context) = count(context→next) / count(context)`

Markov chain of order n for sequence prediction:
- **Unigram (n=1):** No context, predict based on global frequency
- **Bigram (n=2):** Predict based on previous item
- **Trigram (n=3):** Predict based on previous 2 items

**Laplace smoothing:** `P_smooth = (count + 1) / (total + V)` where V is vocabulary size.

**Perplexity:** `exp(-1/N * Σ(log(P)))` — lower is better.

| Metric | Value |
|--------|-------|
| Time | O(n * L) where L is sequence length |
| Space | O(V^n) where V is vocabulary size |
| WASM impact | ~4KB |

**Use cases:** Process mining (next activity prediction), text prediction, recommendation systems

---

## Z-Score Drift Detection

**Formula:** `z = (window_mean - baseline_mean) / baseline_std`

Drift detected when: `|z| > threshold`

Statistical method for detecting mean shifts in continuous data streams using sliding window analysis.

| Metric | Value |
|--------|-------|
| Time | O(n * w) |
| Space | O(w) |
| WASM impact | ~2KB |

**Use cases:** Quality control, sensor monitoring, anomaly detection in metrics

---

## Page-Hinkley Test

**Formula:** `PH = max_cumulative - min_cumulative`

Cumulative sum-based change detection:
1. Compute deviations from mean: `d_i = value_i - mean`
2. Track cumulative sum: `S_t = Σ(d_i)`
3. Monitor range: `PH_t = max(S) - min(S)`
4. Signal change when PH exceeds threshold

| Metric | Value |
|--------|-------|
| Time | O(n) single-pass |
| Space | O(1) |
| WASM impact | ~2KB |

**Use cases:** Sudden change detection, monitoring streaming metrics, break point detection

---

## References

- Friedman, J. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
- Freund, Y. & Schapire, R. (1997). A Decision-Theoretic Generalization of On-Line Learning.
- Breiman, L. (2001). Random Forests.
- Ester, M. et al. (1996). A Density-Based Algorithm for Discovering Clusters.
- Liu, F. T. et al. (2008). Isolation Forest.
- Breunig, M. et al. (2000). LOF: Identifying Density-Based Local Outliers.
- Dempster, A. et al. (1977). Maximum Likelihood from Incomplete Data via EM Algorithm.
- Ng, A. et al. (2001). On Spectral Clustering.
- Rousseeuw, P. (1984). Least Median of Squares Regression.
- Fischler, M. & Bolles, R. (1981). Random Sample Consensus.

---

## Feature Engineering (Phase 2)

### Prefix Features

**Algorithm:** One-hot encoding of sequence prefixes.

Extract all unique prefixes up to max_prefix_len from sequences and encode as binary features. Each position in the output represents whether a specific prefix pattern exists.

| Metric | Value |
|--------|-------|
| Time | O(n * L) where L is total sequence length |
| Space | O(V^P) where V is vocabulary size, P is max prefix length |
| WASM impact | ~3KB |

**Use cases:** Process mining trace classification, sequence preprocessing, feature extraction from logs

### Rework Score

**Formula:** `rework = repetitions / sequence_length`

Counts consecutive repetitions of activities as a measure of process inefficiency (rework loops).

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |
| WASM impact | ~1KB |

**Use cases:** Process optimization, identifying inefficiencies, quality metrics

### Activity Counts

**Algorithm:** Frequency encoding of activities.

Counts occurrences of each activity across all sequences and returns sorted by frequency.

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(A) where A is number of unique activities |
| WASM impact | ~2KB |

**Use cases:** Process analysis, activity frequency statistics, bottleneck identification

### Trace Statistics

**Metrics:** Length, unique activities, elapsed time.

Computes basic statistics for each trace/sequence including length, number of unique activities, and time duration.

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |
| WASM impact | ~2KB |

**Use cases:** Process characterization, performance analysis, trace profiling

### Inter-Event Times

**Formula:** `avg_time = Σ(t_{i+1} - t_i) / (n-1)`

Computes average time between consecutive events in each sequence.

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |
| WASM impact | ~1KB |

**Use cases:** Process timing analysis, bottleneck detection, service level monitoring

---

## Ensemble Discovery (Phase 3)

### Quality-Weighted Prediction

**Formula:** `ŷ = Σ(w_i * ŷ_i) / Σ(w_i)`

Combines predictions from multiple models weighted by their quality scores (e.g., R², accuracy).

| Metric | Value |
|--------|-------|
| Time | O(n * m) where m is number of models |
| Space | O(m) |
| WASM impact | ~2KB |

**Use cases:** Model ensembling, robust prediction, automated model selection

### Consensus Scoring

**Formula:** `consensus = agreements / total_comparisons`

Measures agreement between multiple prediction vectors as fraction of identical predictions.

| Metric | Value |
|--------|-------|
| Time | O(n * m²) |
| Space | O(1) |
| WASM impact | ~2KB |

**Use cases:** Model disagreement analysis, ensemble diversity measurement, uncertainty estimation

### Pruned Ensemble

**Algorithm:** Rank by quality and keep top-k members.

1. Sort ensemble members by quality (descending)
2. Keep only top k members
3. Renormalize weights to sum to 1

| Metric | Value |
|--------|-------|
| Time | O(m log m) |
| Space | O(m) |
| WASM impact | ~2KB |

**Use cases:** Model selection, ensemble simplification, computational efficiency

---

## Statistical Distributions (Phase 3)

### Log-Normal Distribution

**Formula:** `f(x) = 1/(xσ√2π) * exp(-0.5 * ((ln(x)-μ)/σ)²)`

Models data where log(x) follows a normal distribution. Useful for skewed positive-valued data.

**Parameters:**
- **μ (mu):** Mean of log-transformed data
- **σ (sigma):** Standard deviation of log-transformed data

**Key functions:**
- PDF: `f(x)` - probability density
- CDF: `F(x)` - cumulative distribution
- Survival: `S(x) = 1 - F(x)`
- Percentile: `x_p = exp(μ + σ * z_p)` where z_p is normal quantile

| Metric | Value |
|--------|-------|
| Time | O(n) for fitting |
| Space | O(1) |
| WASM impact | ~3KB |

**Use cases:** Survival analysis, income distribution, latency modeling, skewed data

### Gamma Distribution

**Formula:** `f(x) = (θ^k / Γ(k)) * x^(k-1) * exp(-θx)`

Two-parameter gamma distribution with shape k and rate θ (1/scale).

**Parameters (Method of Moments):**
- **shape:** `mean² / variance`
- **rate:** `mean / variance`

**Key functions:**
- PDF: `f(x)` - probability density
- CDF: Series approximation of regularized lower incomplete gamma
- Survival: `S(x) = 1 - F(x)`

| Metric | Value |
|--------|-------|
| Time | O(n) for fitting |
| Space | O(1) |
| WASM impact | ~3KB |

**Use cases:** Reliability engineering, queuing theory, rainfall modeling, insurance claims

---

## Statistical Tests (Phase 3)

### Chi-Square Test of Independence

**Formula:** `χ² = Σ((O_ij - E_ij)² / E_ij)`

Tests whether two categorical variables are independent using contingency table analysis.

**Expected value:** `E_ij = (row_total_i * col_total_j) / grand_total`

**Degrees of freedom:** `(rows - 1) * (cols - 1)`

**P-value:** Wilson-Hilferty approximation to chi-square distribution.

| Metric | Value |
|--------|-------|
| Time | O(n * m) for n×m contingency table |
| Space | O(n * m) |
| WASM impact | ~3KB |

**Use cases:** A/B testing, categorical feature analysis, survey data analysis

### Kolmogorov-Smirnov Test

**Formula:** `D = max|CDF1(x) - CDF2(x)| * √(n1*n2/(n1+n2))`

Two-sample test for whether samples come from the same continuous distribution.

**Empirical CDF:** `F_n(x) = count(X ≤ x) / n`

**P-value:** Smirnov formula approximation based on KS statistic λ.

| Metric | Value |
|--------|-------|
| Time | O(n log n) for sorting |
| Space | O(n) |
| WASM impact | ~3KB |

**Use cases:** Distribution comparison, goodness-of-fit testing, data validation

---

## Anomaly Detection (Phase 3)

### Z-Score Anomaly Detection

**Formula:** `z = (x - μ) / σ`

Detects anomalies using Z-score (standard deviations from mean). Anomaly if `|z| > threshold`.

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |
| WASM impact | ~2KB |

**Use cases:** Outlier detection in univariate data, quality control, sensor monitoring

### IQR Anomaly Detection

**Formula:** `outlier if x < Q1 - k*IQR or x > Q3 + k*IQR`

Detects outliers using Interquartile Range method (robust to extreme values).

**Bounds:** `lower = Q1 - k*IQR`, `upper = Q3 + k*IQR` where k=1.5 by default.

| Metric | Value |
|--------|-------|
| Time | O(n log n) for sorting |
| Space | O(n) |
| WASM impact | ~2KB |

**Use cases:** Robust outlier detection, box plot statistics, data cleaning

### Boundary Coverage

**Formula:** `anomaly if x < lower_bound or x > upper_bound`

Checks which values fall outside specified boundaries.

**Score:** Distance from nearest bound for out-of-bounds values.

| Metric | Value |
|--------|-------|
| Time | O(n) |
| Space | O(1) |
| WASM impact | ~2KB |

**Use cases:** Data quality validation, SLA monitoring, constraint checking

### Sequence Anomaly Detection

**Formula:** `anomaly if P(next|context) < threshold`

Detects sequences that deviate from expected patterns using n-gram models.

**Training:** Build n-gram transition counts from sequences
**Detection:** Low probability transitions indicate anomalies

| Metric | Value |
|--------|-------|
| Time | O(n) training, O(L) detection |
| Space | O(V^n) where V is vocabulary size |
| WASM impact | ~4KB |

**Use cases:** Process mining (deviation detection), sequence validation, pattern monitoring

---

## Additional References (Phase 2 & 3)

- van der Aalst, W. (2016). Process Mining: Data Science in Action.
- Johnson, N. (1949). Systems of Frequency Curves Generated by Methods of Random Translation.
- Pearson, K. (1900). On the Criterion that a Given System of Deviations from the Probable in the Case of a Correlated System of Variables is Such that it Can be Reasonably Supposed to Have Arisen from Random Sampling.
- Smirnov, N. (1948). Table for Estimating the Goodness of Fit of Empirical Distributions.
- Wilson, E. & Hilferty, M. (1931). The Distribution of Chi-Square.
- Beasley, J. & Springer, M. (1977). Algorithm AS 111: The Percentage Points of the Normal Distribution.
