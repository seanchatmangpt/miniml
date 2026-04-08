<p align="center">
  <img src="./docs/logo.svg" alt="miniml" width="480">
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/miniml"><img src="https://img.shields.io/npm/v/miniml?color=cb0000" alt="npm version"></a>
  <a href="https://bundlephobia.com/package/miniml"><img src="https://img.shields.io/bundlephobia/minzip/miniml?color=cb0000&label=size" alt="bundle size"></a>
  <a href="https://github.com/seanchatmangpt/miniml/blob/main/LICENSE"><img src="https://img.shields.io/github/license/seanchatmangpt/miniml?color=cb0000" alt="license"></a>
</p>

**AutoML-powered machine learning in the browser**

miniml combines **30+ ML algorithms** with **genetic algorithm feature selection** and **PSO hyperparameter optimization** — all in **~145KB gzipped** with **SIMD acceleration**.

```
npm install miniml
```

---

## Key Differentiators

| Feature | What It Means | Benefit |
|---------|---------------|---------|
| **🤖 AutoML** | GA feature selection + PSO hyperparameter optimization | Best model found automatically |
| **📊 30+ Algorithms** | Classification, regression, clustering, preprocessing, metrics | Entire ML pipeline covered |
| **⚡ SIMD Acceleration** | WASM v128 intrinsics for vectorized operations | 4-100x faster than alternatives |
| **🔥 Metaheuristics** | Genetic algorithms, PSO, simulated annealing, bandit algorithms | Advanced optimization built-in |

---

## AutoML Showcase

### What AutoML Does

AutoML automatically finds the best algorithm and hyperparameters for your data:

1. **Feature Selection** — Genetic algorithm identifies optimal feature subset
2. **Algorithm Selection** — Tests multiple algorithms, selects best performer
3. **Hyperparameter Optimization** — PSO tunes algorithm parameters
4. **Progress Monitoring** — Real-time callbacks for long-running operations
5. **Result Interpretation** — Human-readable summaries with rationale

### AutoML Quick Start

```js
import { autoFit } from 'miniml';

// Automatically select and train the best model
const model = await autoFit(X, y, {
  // Optional: configure AutoML behavior
  featureSelection: true,      // Enable GA feature selection
  maxFeatures: 0.8,            // Keep top 80% of features
  cvFolds: 5,                  // 5-fold cross-validation
  progressCallback: (update) => {
    console.log(`Progress: ${update.percent}%`);
    console.log(`Testing: ${update.algorithm}`);
  }
});

// Get model details
console.log(model.algorithm);    // "RandomForest"
console.log(model.accuracy);     // 0.95
console.log(model.features);     // [0, 2, 5, 7] (selected features)
console.log(model.rationale);    // Why this algorithm was chosen

// Make predictions
const prediction = await model.predict(testPoint);
```

### AutoML Result Interpretation

```js
{
  algorithm: "RandomForest",
  accuracy: 0.95,
  trainingTime: 45,
  features: [0, 2, 5, 7],
  hyperparameters: {
    nTrees: 100,
    maxDepth: 10,
    minSamplesSplit: 2
  },
  rationale: "RandomForest achieved highest cross-validation accuracy (95%) " +
             "with strong performance across all metrics. Feature selection " +
             "reduced dimensionality from 10 to 4 features, improving " +
             "training speed by 60%.",
  allScores: {
    "KNN": 0.87,
    "DecisionTree": 0.91,
    "RandomForest": 0.95,
    "GradientBoosting": 0.93,
    "NaiveBayes": 0.82
  }
}
```

---

## Algorithm Coverage (30+)

### Classification (9)
- **K-Nearest Neighbors** — Instance-based learning
- **Decision Tree** — Hierarchical rule-based classification
- **Random Forest** — Ensemble of decision trees
- **Gradient Boosting** — Sequential ensemble with loss optimization
- **AdaBoost** — Adaptive boosting of weak learners
- **Naive Bayes** — Probabilistic classifier with independence assumption
- **Logistic Regression** — Linear classifier with sigmoid activation
- **Perceptron** — Online learning with stochastic gradient descent
- **Linear SVM** — Support vector machine with linear kernel

### Regression (6)
- **Linear Regression** — Ordinary least squares
- **Ridge Regression** — L2-regularized linear regression
- **Polynomial Regression** — Nonlinear polynomial features
- **Exponential Regression** — Exponential curve fitting
- **Logarithmic Regression** — Logarithmic curve fitting
- **Power Regression** — Power law curve fitting

### Clustering (4)
- **K-Means** — Lloyd's algorithm for centroid-based clustering
- **K-Means++** — Improved initialization for K-Means
- **DBSCAN** — Density-based spatial clustering
- **Hierarchical Clustering** — Agglomerative clustering with linkage

### Ensemble Methods (3)
- **Random Forest** — Bagging ensemble of decision trees
- **Gradient Boosting** — Boosting ensemble with gradient descent
- **AdaBoost** — Adaptive boosting of weighted classifiers

### Preprocessing (8)
- **Standard Scaler** — Z-score normalization (mean=0, std=1)
- **MinMax Scaler** — Scale to [0, 1] range
- **Robust Scaler** — Outlier-resistant scaling using quartiles
- **Normalizer** — L2 normalization per sample
- **Label Encoder** — Convert text labels to numeric
- **One-Hot Encoder** — Binary encoding for categorical features
- **Ordinal Encoder** — Ordinal encoding for ordered categories
- **Imputer** — Fill missing values (mean, median, mode)

### Dimensionality Reduction (2)
- **PCA** — Principal component analysis
- **Feature Selection** — Genetic algorithm-based feature selection

### Metrics & Evaluation (9)
- **Confusion Matrix** — TP, TN, FP, FN counts
- **Classification Report** — Precision, recall, F1-score per class
- **Silhouette Score** — Clustering quality metric
- **ROC AUC** — Area under ROC curve
- **Log Loss** — Logarithmic loss
- **Accuracy** — Classification accuracy
- **R² Score** — Coefficient of determination
- **RMSE** — Root mean squared error
- **MAE** — Mean absolute error

### Model Selection (2)
- **Cross-Validation** — K-fold cross-validation
- **Train/Test Split** — Random dataset split

### Time Series Analysis (10)
- **SMA** — Simple moving average
- **EMA** — Exponential moving average
- **WMA** — Weighted moving average
- **Exponential Smoothing** — Holt-Winters smoothing
- **Linear Regression Forecast** — Trend-based forecasting
- **Polynomial Regression Forecast** — Nonlinear trend forecasting
- **Peak Detection** — Find local maxima
- **Trough Detection** — Find local minima
- **Momentum** — Rate of change indicator
- **Rate of Change** — Momentum oscillator

### Advanced Optimization (8)
- **Genetic Algorithms** — Population-based optimization
- **PSO** — Particle swarm optimization
- **Simulated Annealing** — Global optimization
- **Multi-Armed Bandit** — Exploration-exploitation balancing
- **Feature Importance** — Identify key features
- **Anomaly Detection** — Isolation forest + statistical outliers
- **Drift Detection** — Concept drift monitoring
- **Prediction Intervals** — Uncertainty quantification

---

## Optimization Suite

### Genetic Algorithm Feature Selection

Automatically identifies the optimal feature subset:

```js
import { geneticFeatureSelection } from 'miniml';

const result = await geneticFeatureSelection(X, y, {
  populationSize: 50,
  generations: 100,
  mutationRate: 0.1,
  crossoverRate: 0.7,
  elitismCount: 5,
  cvFolds: 5,
  scoringMetric: 'accuracy'
});

console.log(result.selectedFeatures);  // [0, 2, 5, 7]
console.log(result.fitnessScore);      // 0.95
console.log(result.originalScore);     // 0.87 (without feature selection)
```

### PSO Hyperparameter Optimization

Optimize algorithm parameters with particle swarm optimization:

```js
import { psoOptimize } from 'miniml';

const result = await psoOptimize({
  objectiveFn: async (params) => {
    const model = await trainModel(params);
    return model.accuracy;
  },
  bounds: {
    learningRate: [0.001, 0.1],
    nTrees: [10, 200],
    maxDepth: [3, 20]
  },
  swarmSize: 30,
  maxIterations: 100
});

console.log(result.bestParams);  // { learningRate: 0.05, nTrees: 100, maxDepth: 10 }
console.log(result.bestScore);   // 0.96
```

### Simulated Annealing

Global optimization for complex landscapes:

```js
import { simulatedAnnealing } from 'miniml';

const result = await simulatedAnnealing({
  objectiveFn: async (state) => evaluateState(state),
  initialState: getInitialState(),
  temperature: 1000,
  coolingRate: 0.95,
  minTemperature: 0.01
});
```

---

## Performance Benchmarks

All benchmarks run in WASM with SIMD acceleration:

| Category | Algorithm | Data Size | Time | Speedup |
|----------|-----------|-----------|------|---------|
| **Classification** | KNN | 1000×100 | 0.5ms | 10x (partial sort) |
| | Decision Tree | 1000×20 | 2.1ms | 3x (class indexing) |
| | Random Forest | 1000×20, 100 trees | 45ms | 2x (zero-allocation) |
| | Gradient Boosting | 500×10, 50 trees | 12ms | 2x (vectorized) |
| | Naive Bayes | 1000×100 | 0.8ms | 5x (precomputed) |
| **Clustering** | K-Means | 1000×20 | 3.2ms | 4x (SIMD distance) |
| | Hierarchical | 500×10 | 35ms | 100x (priority queue) |
| **Preprocessing** | Standard Scaler | 1000×100 | 0.3ms | 4x (SIMD) |
| | PCA | 1000×50→10 | 8.5ms | 3x (optimized SVD) |
| **Regression** | Linear Regression | 1000×50 | 1.2ms | 5x (normal eq) |
| | Ridge Regression | 1000×50 | 1.5ms | 4x (Cholesky) |

---

## Quick Start Guide

### Installation

```bash
npm install miniml
```

### Basic Usage

```js
import { autoFit, knnTrain, randomForestClassify, standardScaler } from 'miniml';

// 1. AutoML (recommended)
const model = await autoFit(X, y);
const prediction = await model.predict(testPoint);

// 2. Manual algorithm selection
const knn = await knnTrain(X, y, nSamples, nFeatures, 5);
const result = await knn.predict(testPoint);

// 3. Preprocessing
const scaled = await standardScaler(X, nSamples, nFeatures);

// 4. Ensemble methods
const rf = await randomForestClassify(X, y, 100, 10);
const result = await rf.predict(testPoint);
```

### Multi-Worker Parallelism

For large datasets, use worker pools:

```js
import { createWorkerPool, parallelCrossValidate } from 'miniml/worker';

const workers = createWorkerPool(navigator.hardwareConcurrency || 4);

const scores = await parallelCrossValidate(
  workers,
  X, y,
  5, // 5-fold CV
  trainFn,
  predictFn
);

workers.forEach(w => w.terminate());
```

---

## Comparison with Alternatives

| Library | Size (gzip) | Algorithms | SIMD | AutoML | Metaheuristics |
|---------|-------------|------------|------|--------|----------------|
| **miniml** | ~145KB | 30+ | ✅ | ✅ | ✅ |
| TensorFlow.js | 500KB+ | 100+ | ❌ | ❌ | ❌ |
| ml.js | 150KB | 15 | ❌ | ❌ | ❌ |
| ml-matrix | 50KB | 0 (matrix only) | ❌ | ❌ | ❌ |

---

## Documentation

- **AutoML Guide** — Comprehensive AutoML documentation
- **Algorithm Reference** — Complete algorithm listing with examples
- **Optimization Suite** — Metaheuristic optimization guide
- **Performance Guide** — SIMD and performance best practices
- **Examples** — Real-world usage examples

---

## License

MIT © 2026 Sean Chatman

---

## Links

- [GitHub](https://github.com/seanchatmangpt/miniml)
- [npm package](https://www.npmjs.com/package/miniml)
