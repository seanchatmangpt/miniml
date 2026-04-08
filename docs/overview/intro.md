# micro-ml

A zero-dependency machine learning library compiled to WebAssembly. Run ML algorithms directly in the browser with no external dependencies.

## Why micro-ml?

| | micro-ml | TensorFlow.js | ONNX.js | brain.js |
|--|----------|--------------|---------|----------|
| **Gzipped size** | ~56KB | ~400KB | ~200KB | ~15KB |
| **Dependencies** | Zero | TF runtime | ONNX runtime | None |
| **Languages** | Rust/WASM | JS/C++ | JS/C++ | JS |
| **Algorithms** | 59+ modules | Deep learning | Inference only | Neural nets |
| **Prediction speed** | <1ms | 5-50ms | 10-100ms | 1-10ms |
| **Use case** | Classical ML | Deep learning | Model inference | Simple NN |

## Key Features

- **Zero dependencies** - Pure Rust compiled to WASM, no external runtime
- **Browser-native** - Runs in any modern browser via ES modules
- **Sub-millisecond predictions** - Optimized for real-time inference
- **59 modules** - Regression, classification, clustering, preprocessing, ensemble methods, anomaly detection, and more
- **TypeScript support** - Full type definitions via generated `.d.ts` files
- **Node.js compatible** - Works in both browser and server environments

## Who Is It For?

- **Frontend developers** who need ML without a Python backend
- **Edge computing** where bandwidth is limited
- **Data scientists** who want to deploy models to the browser
- **Educators** teaching ML concepts with interactive demos

## Installation

```bash
npm install micro-ml
```

```ts
import { linearRegression, kmeans, logisticRegression } from 'micro-ml';

// Initialize WASM (automatic on first call, or explicit)
await linearRegression([1, 2, 3], [2, 4, 6]);
```

## Quick Example

```ts
import { linearRegression } from 'micro-ml';

const model = await linearRegression([1, 2, 3, 4], [2, 4, 6, 8]);
console.log(model.slope);     // 2
console.log(model.rSquared);  // 1
console.log(model.predict([5])); // [10]
```

## What's Included

### Regression
Linear, polynomial, exponential, logarithmic, power, Ridge, Lasso, Elastic Net, RANSAC, Theil-Sen

### Classification
kNN, logistic regression, SVM, perceptron, decision tree, naive Bayes (Gaussian, Bernoulli, Multinomial), SGD, passive aggressive

### Clustering
K-Means, K-Means++, DBSCAN, hierarchical (single/complete linkage), spectral, GMM, mini-batch K-Means

### Ensemble Methods
Random forest, gradient boosting, AdaBoost, extra trees, bagging, voting classifier

### Preprocessing
Standard scaler, min-max scaler, robust scaler, normalizer, label encoder, one-hot encoder, ordinal encoder, power transformer, imputer, PCA

### Model Evaluation
Confusion matrix, classification metrics (precision, recall, F1, MCC, AUC), regression metrics (R2, RMSE, MAE), silhouette score, cross-validation, data splitting

### Anomaly Detection
Isolation forest, LOF (Local Outlier Factor)

### Model Selection
Grid search, RFE (Recursive Feature Elimination), permutation importance

### Utilities
Feature importance, pipeline (sequential transformations), moving averages, trend analysis, seasonal decomposition

## License

AGPL-3.0
