# Architecture

How micro-ml works internally: WASM compilation, data representation, and error handling.

## WASM Compilation

micro-ml is written in Rust and compiled to WebAssembly using `wasm-bindgen`:

```
Rust source (crates/micro-ml-core/src/)
  → cargo build --target wasm32-unknown-unknown
  → wasm-bindgen generates JS bindings
  → wasm-pack creates npm package
  → micro_ml_core_bg.wasm + micro_ml_core.js
```

**Key constraint:** Zero external dependencies. All algorithms are implemented from scratch in pure Rust. No BLAS, no LAPACK, no ndarray. This keeps the WASM bundle under 100KB gzipped.

## Data Representation

All matrices are passed as flat `Float64Array` in row-major order:

```
2D matrix:
  [[1, 2, 3],       →    [1, 2, 3, 4, 5, 6]
   [4, 5, 6]]

Layout: row[i] starts at index i * n_features
  element(row=i, col=j) = data[i * n_features + j]
```

This avoids serialization overhead and enables direct WASM memory access.

### TypeScript helpers

The JS wrapper provides `flattenMatrix()` and `unflattenMatrix()` to convert between `number[][]` and `Float64Array`:

```ts
// Input: number[][] → WASM: Float64Array
const { flat, nFeatures } = flattenMatrix([[1, 2], [3, 4]]);
// flat = [1, 2, 3, 4], nFeatures = 2

// Output: Float64Array → number[][]
const rows = unflattenMatrix(flat, nFeatures);
// rows = [[1, 2], [3, 4]]
```

## Module Organization

```
crates/micro-ml-core/src/
├── lib.rs              # Module registry + re-exports
├── error.rs            # MlError type (implements std::error::Error)
├── matrix.rs           # Core utilities: validate_matrix, euclidean_dist_sq, Rng
├──
├── regression/         # Predicting continuous values
│   ├── linear.rs       # Simple linear regression
│   ├── polynomial.rs   # Polynomial curve fitting
│   ├── exponential.rs  # y = a * e^(bx)
│   ├── linear_regression.rs  # Ridge, Lasso (coordinate descent)
│   ├── elastic_net.rs  # L1+L2 combined
│   ├── ransac.rs       # Robust regression (random sample consensus)
│   └── theil_sen.rs    # Median-based robust regression
│
├── classification/     # Predicting categories
│   ├── knn.rs          # K-nearest neighbors
│   ├── logistic.rs     # Logistic regression (gradient descent)
│   ├── svm.rs          # Support vector machine
│   ├── perceptron.rs   # Perceptron (single-layer NN)
│   ├── decision_tree.rs  # Decision tree (classify + regress)
│   ├── naive_bayes.rs  # Gaussian naive Bayes
│   ├── bernoulli_nb.rs # Bernoulli naive Bayes (binary features)
│   ├── multinomial_nb.rs  # Multinomial naive Bayes (count features)
│   ├── sgd_classifier.rs  # SGD with hinge/log/huber loss
│   └── passive_aggressive.rs  # PA-I/PA-II online learning
│
├── clustering/         # Unsupervised grouping
│   ├── kmeans.rs       # K-Means (random init)
│   ├── kmeans_plus.rs  # K-Means++ (smart init)
│   ├── mini_batch_kmeans.rs  # Online K-Means
│   ├── dbscan.rs       # Density-based clustering
│   ├── hierarchical.rs # Single linkage agglomerative
│   ├── agglomerative_complete.rs  # Complete linkage
│   ├── spectral.rs     # Graph-based spectral clustering
│   └── gmm.rs          # Gaussian Mixture Models (EM)
│
├── ensemble/           # Combined models
│   ├── random_forest.rs  # Bagged decision trees
│   ├── gradient_boosting.rs  # Sequential boosting
│   ├── adaboost.rs     # Adaptive boosting
│   ├── extra_trees.rs  # Extremely randomized trees
│   ├── bagging.rs      # Bootstrap aggregating
│   └── voting_classifier.rs  # Hard/soft voting
│
├── preprocessing/      # Data transformation
│   ├── standard_scaler.rs  # Z-score normalization
│   ├── minmax_scaler.rs    # [0,1] scaling
│   ├── robust_scaler.rs    # Median/IQR scaling
│   ├── normalizer.rs       # L2 normalization
│   ├── label_encoder.rs    # Label → integer
│   ├── one_hot_encoder.rs  # Integer → one-hot
│   ├── ordinal_encoder.rs  # Label → ordered integer
│   ├── power_transformer.rs  # Yeo-Johnson
│   ├── imputer.rs          # Mean/median/most_frequent fill
│   ├── pca.rs              # Principal Component Analysis
│   └── pipeline.rs         # Sequential transformations
│
├── metrics/            # Evaluation
│   ├── regression_metrics.rs   # R2, RMSE, MAE
│   ├── classification_metrics.rs  # Precision, recall, F1, MCC, AUC
│   ├── clustering_metrics.rs  # Silhouette score
│   ├── confusion_matrix.rs  # TP, FP, TN, FN
│   └── feature_importance.rs  # Gini importance
│
├── model_selection/    # Hyperparameter tuning
│   ├── cross_validation.rs  # K-fold CV
│   ├── data_split.rs   # Train/test split
│   ├── model_selection.rs  # ROC AUC
│   ├── grid_search.rs  # Grid search
│   ├── rfe.rs          # Recursive Feature Elimination
│   └── permutation_importance.rs  # Model-agnostic importance
│
└── anomaly/            # Outlier detection
    ├── isolation_forest.rs  # Tree-based isolation
    └── lof.rs          # Local Outlier Factor
```

## Error Handling Pattern

Every WASM-exported function returns `Result<T, JsError>`:

```rust
// Internal Rust function
pub fn kmeans_impl(data: &[f64], n_features: usize, k: usize) -> Result<KMeansModel, MlError> {
    let n = validate_matrix(data, n_features)?;  // ← validates input
    if k == 0 || k > n {
        return Err(MlError::new("k must be between 1 and n_samples"));
    }
    // ... algorithm ...
    Ok(KMeansModel { ... })
}

// WASM export
#[wasm_bindgen(js_name = "kmeans")]
pub fn kmeans(data: &[f64], n_features: usize, k: usize, max_iter: usize) -> Result<KMeansModel, JsError> {
    kmeans_impl(data, n_features, k, max_iter)
        .map_err(|e| JsError::new(&e.message))
}
```

`MlError` implements `std::error::Error`, enabling the `?` operator for error propagation.

## Input Validation

`validate_matrix(data, n_features)` is called by every algorithm:

- Checks `data.len() % n_features == 0`
- Returns `n_samples = data.len() / n_features`
- Returns `MlError` if data is empty or misaligned

## Deterministic Randomness

`Rng::from_data(data)` creates a seeded PRNG from the input data itself:

- No system random source needed (WASM constraint)
- Same input always produces same output
- Uses data as seed via simple hash
- Used by K-Means++ initialization and bootstrap sampling

## WASM Export Pattern

Public types are exported with `#[wasm_bindgen]`:

```rust
#[wasm_bindgen]
pub struct KMeansModel {
    k: usize,
    centroids: Vec<f64>,
    assignments: Vec<usize>,
    // ...
}

#[wasm_bindgen]
impl KMeansModel {
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> usize { self.k }

    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> { ... }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String { ... }
}
```

Getters use camelCase via `js_name` attribute. The TypeScript wrapper provides friendly JS interfaces on top.
