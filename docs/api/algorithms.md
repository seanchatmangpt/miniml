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
