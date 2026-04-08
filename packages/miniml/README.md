# miniml

**AutoML-powered machine learning in the browser**

miniml combines **70+ ML algorithms** across **15 algorithm families** with **genetic algorithm feature selection** and **PSO hyperparameter optimization** — all in **~145KB gzipped** with **SIMD acceleration**.

```
npm install miniml
```

---

## Key Differentiators

| Feature | What It Means |
|---------|---------------|
| **🤖 AutoML** | GA feature selection + PSO hyperparameter optimization |
| **📊 70+ Algorithms** | Classification, regression, clustering, preprocessing, time series, probabilistic methods, statistical inference, kernels, Bayesian methods, Gaussian processes, survival analysis, association rules, recommendation systems, graph algorithms |
| **⚡ SIMD Acceleration** | WASM v128 intrinsics for 4-100x speedup |
| **🔥 Metaheuristics** | Genetic algorithms, PSO, simulated annealing built-in |

---

## Quick Start

```js
import { autoFit, knnTrain, randomForestClassify } from 'miniml';

// AutoML automatically selects the best algorithm
const model = await autoFit(X, y);
const prediction = await model.predict(testPoint);

// Manual algorithm selection
const knn = await knnTrain(X, y, nSamples, nFeatures, k);
const result = await knn.predict(testPoint);

// Ensemble methods
const rf = await randomForestClassify(X, y, 100, 10);
const result = await rf.predict(testPoint);
```

---

## AutoML

```js
import { autoFit } from 'miniml';

// Automatically select and train best model
const model = await autoFit(X, y, {
  featureSelection: true,
  cvFolds: 5,
  progressCallback: (update) => {
    console.log(`Testing: ${update.algorithm}`);
  }
});

// Get model details
console.log(model.algorithm);    // "RandomForest"
console.log(model.accuracy);     // 0.95
console.log(model.rationale);    // Why this algorithm was chosen
```

---

## API Reference

### Classification

```js
import {
  knnTrain,
  decisionTreeTrain,
  randomForestClassify,
  gradientBoostingClassify,
  naiveBayesTrain,
  logisticRegression
} from 'miniml';

// k-Nearest Neighbors
const knn = await knnTrain(X, y, nSamples, nFeatures, k);

// Decision Tree
const dt = await decisionTreeTrain(X, y, nSamples, nFeatures, maxDepth);

// Random Forest
const rf = await randomForestClassify(X, y, nTrees, maxDepth);

// Gradient Boosting
const gb = await gradientBoostingClassify(X, y, nEstimators, learningRate, maxDepth);

// Naive Bayes
const nb = await naiveBayesTrain(X, y, nSamples, nFeatures);

// Logistic Regression
const lr = await logisticRegression(X, y, nSamples, nFeatures, maxIterations, learningRate);
```

### Clustering

```js
import {
  kmeans,
  kmeansPlus,
  dbscan,
  hierarchicalClustering
} from 'miniml';

// K-Means
const km = await kmeans(X, nFeatures, nClusters, maxIterations);

// K-Means++ (better initialization)
const kmpp = await kmeansPlus(X, nClusters, maxIterations, nSamples, nFeatures);

// DBSCAN (density-based)
const db = await dbscan(X, nFeatures, eps, minPoints);

// Hierarchical Clustering
const hc = await hierarchicalClustering(X, nFeatures, nClusters);
```

### Regression

```js
import {
  linearRegression,
  ridgeRegression,
  lassoRegression,
  polynomialRegression
} from 'miniml';

// Linear Regression
const lr = await linearRegression(X, y, nSamples, nFeatures);

// Ridge Regression
const rr = await ridgeRegression(X, y, alpha, nSamples, nFeatures);

// Lasso Regression
const lasso = await lassoRegression(X, y, alpha, l1Ratio, nSamples, nFeatures);

// Polynomial Regression
const pr = await polynomialRegression(X, y, nSamples, nFeatures, degree);
```

### Preprocessing

```js
import {
  standardScaler,
  minMaxScaler,
  robustScaler,
  labelEncoder,
  oneHotEncoder
} from 'miniml';

// Standard Scaler (z-score normalization)
const scaled = await standardScaler(X, nSamples, nFeatures);

// MinMax Scaler (0-1 scaling)
const scaled = await minMaxScaler(X, nSamples, nFeatures);

// Robust Scaler (outlier-resistant)
const scaled = await robustScaler(X, nSamples, nFeatures);

// Label Encoder
const encoded = await labelEncoder(y);

// One-Hot Encoder
const oneHot = await oneHotEncoder(y, nClasses);
```

### Probabilistic Methods

```js
import {
  mcIntegrate,
  mcBootstrap,
  computeSteadyState,
  hmmForward,
  metropolisHastings
} from 'miniml';

// Monte Carlo Integration
const integral = await mcIntegrate(fn, a, b, n, seed);

// Bootstrap Confidence Intervals
const ci = await mcBootstrap(data, nBootstrap, 'mean', 0.95, seed);

// Markov Chain Steady State
const steady = await computeSteadyState(transitionMatrix, nStates);

// Hidden Markov Model
const alpha = await hmmForward(initial, transition, emission, obs, nStates, nObs);
```

### Statistical Inference

```js
import {
  tTestOneSample,
  tTestTwoSample,
  mannWhitneyU,
  chiSquareTest,
  oneWayAnova
} from 'miniml';

// t-Test
const t = await tTestOneSample(data, nullHypothesis, alpha);

// ANOVA
const f = await oneWayAnova(groups, groupSizes);
```

### Kernel Methods

```js
import {
  rbfKernelMatrix,
  polynomialKernelMatrix,
  sigmoidKernelMatrix
} from 'miniml';

// RBF Kernel Matrix
const K = await rbfKernelMatrix(data, nSamples, nFeatures, gamma);

// Polynomial Kernel Matrix
const K = await polynomialKernelMatrix(data, nSamples, nFeatures, degree, gamma, coef0);
```

### Bayesian Methods

```js
import {
  bayesianEstimate,
  bayesianLinearRegression
} from 'miniml';

// Bayesian Estimation (MCMC)
const posterior = await bayesianEstimate(logLikelihood, logPrior, nSamples, burnIn, seed, initial, proposalSd);

// Bayesian Linear Regression
const blr = await bayesianLinearRegression(data, nFeatures, targets, priorPrecision, priorAlpha, priorBeta);
```

### Gaussian Processes

```js
import {
  gpFit,
  gpPredict
} from 'miniml';

// Fit Gaussian Process
const model = await gpFit(data, nFeatures, targets, kernelType, kernelParams, noise);

// Predict with Uncertainty
const pred = await gpPredict(model, xTest, nFeatures);
// { mean: [...], std: [...], lower: [...], upper: [...] }
```

### Survival Analysis

```js
import {
  kaplanMeier,
  coxProportionalHazards
} from 'miniml';

// Kaplan-Meier Survival Curve
const km = await kaplanMeier(times, events);

// Cox Proportional Hazards
const cox = await coxProportionalHazards(features, nFeatures, times, events, maxIterations, learningRate);
```

### Metrics

```js
import {
  confusionMatrix,
  classificationReport,
  silhouetteScore
} from 'miniml';

// Confusion Matrix
const cm = await confusionMatrix(yTrue, yPred);

// Classification Report
const report = await classificationReport(yTrue, yPred);

// Silhouette Score
const score = await silhouetteScore(X, labels, nSamples, nFeatures);
```

---

## Performance

| Algorithm | Data Size | Time | Speedup |
|-----------|-----------|------|---------|
| KNN | 1000×100 | 0.5ms | 10x |
| Decision Tree | 1000×20 | 2.1ms | 3x |
| Random Forest | 1000×20, 100 trees | 45ms | 2x |
| Gradient Boosting | 500×10, 50 trees | 12ms | 2x |
| Hierarchical Clustering | 500×10 | 35ms | 100x |
| K-Means | 1000×20 | 3.2ms | 4x |
| PCA | 1000×50→10 | 8.5ms | 3x |
| Standard Scaler | 1000×100 | 0.3ms | 4x |
| MC Integration | 1M samples | 2.4ms | N/A |
| HMM Baum-Welch | 100 obs, 5 states | 8.2ms | N/A |
| ANOVA | 3K×50 | 1.1ms | N/A |
| GP Fit | 200×10 | 15ms | N/A |
| PageRank | 1K nodes | 2.1ms | N/A |

---

## Documentation

- **[AutoML Guide](../docs/automl.md)** — Comprehensive AutoML documentation
- **[Algorithm Reference](../docs/algorithms.md)** — Complete algorithm listing
- **[Optimization Suite](../docs/optimization.md)** — Metaheuristic optimization
- **[Performance Guide](../docs/performance.md)** — SIMD and performance
- **[Examples](../docs/examples.md)** — Real-world usage examples

---

## License

BSL 1.1 — See [LICENSE](../../LICENSE) for details.
