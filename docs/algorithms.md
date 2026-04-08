# Algorithm Reference

Complete reference for all 70+ algorithms in miniml across 15 families.

## Classification Algorithms

### K-Nearest Neighbors (KNN)

Instance-based learning algorithm that classifies new cases based on similarity measure (e.g., distance functions).

```js
import { knnTrain } from 'miniml';

const knn = await knnTrain(X, y, nSamples, nFeatures, k);
const prediction = await knn.predict(testPoint);
```

**Parameters:**
- `X`: Training features (Float64Array, length = nSamples × nFeatures)
- `y`: Training labels (Float64Array, length = nSamples)
- `nSamples`: Number of training samples
- `nFeatures`: Number of features per sample
- `k`: Number of neighbors to consider

**Best for:**
- Small to medium datasets
- Low-dimensional data
- When decision boundaries are irregular

**Time Complexity:** O(n) per prediction

---

### Decision Tree

Hierarchical tree-like model that splits data based on feature values.

```js
import { decisionTreeTrain } from 'miniml';

const dt = await decisionTreeTrain(X, y, nSamples, nFeatures, maxDepth);
const prediction = await dt.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Training labels
- `nSamples`: Number of training samples
- `nFeatures`: Number of features
- `maxDepth`: Maximum tree depth (default: unlimited)

**Best for:**
- Interpretable models
- Mixed feature types
- When feature interactions are important

**Time Complexity:** O(n log n) training, O(log n) prediction

---

### Random Forest

Ensemble of decision trees trained via bagging (bootstrap aggregating).

```js
import { randomForestClassify } from 'miniml';

const rf = await randomForestClassify(X, y, nTrees, maxDepth);
const prediction = await rf.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Training labels
- `nTrees`: Number of trees in forest
- `maxDepth`: Maximum depth per tree

**Best for:**
- High accuracy requirements
- Robustness to overfitting
- Feature importance estimation

**Time Complexity:** O(nTrees × n log n) training, O(nTrees × log n) prediction

---

### Gradient Boosting

Sequential ensemble that builds trees to correct errors of previous trees.

```js
import { gradientBoostingClassify } from 'miniml';

const gb = await gradientBoostingClassify(X, y, nEstimators, maxDepth, learningRate);
const prediction = await gb.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Training labels
- `nEstimators`: Number of boosting rounds
- `maxDepth`: Maximum depth per tree
- `learningRate`: Shrinkage factor (0-1)

**Best for:**
- High predictive accuracy
- Imbalanced datasets
- When training time is acceptable

**Time Complexity:** O(nEstimators × n log n) training

---

### AdaBoost

Adaptive boosting that combines weak learners (typically shallow trees).

```js
import { adaboostClassify } from 'miniml';

const ada = await adaboostClassify(X, y, nEstimators);
const prediction = await ada.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Training labels
- `nEstimators`: Number of weak learners

**Best for:**
- Binary classification
- When weak learners are available
- Quick baseline model

**Time Complexity:** O(nEstimators × n) training

---

### Naive Bayes

Probabilistic classifier based on Bayes' theorem with independence assumption.

```js
import { naiveBayesTrain } from 'miniml';

const nb = await naiveBayesTrain(X, y, nSamples, nFeatures);
const prediction = await nb.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Training labels
- `nSamples`: Number of training samples
- `nFeatures`: Number of features

**Best for:**
- Text classification
- High-dimensional data
- When independence assumption holds

**Time Complexity:** O(n) training, O(1) prediction

---

### Logistic Regression

Linear classifier with sigmoid activation for binary classification.

```js
import { logisticRegressionTrain } from 'miniml';

const lr = await logisticRegressionTrain(X, y, nSamples, nFeatures, maxIterations, learningRate);
const prediction = await lr.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Training labels (binary)
- `nSamples`: Number of training samples
- `nFeatures`: Number of features
- `maxIterations`: Maximum gradient descent iterations
- `learningRate`: Step size for gradient descent

**Best for:**
- Linearly separable data
- Probability estimates needed
- Baseline for comparison

**Time Complexity:** O(maxIterations × n × nFeatures) training

---

### Perceptron

Online learning algorithm with stochastic gradient descent.

```js
import { perceptronTrain } from 'miniml';

const p = await perceptronTrain(X, y, nSamples, nFeatures, maxIterations, learningRate);
const prediction = await p.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Training labels (binary: -1 or 1)
- `nSamples`: Number of training samples
- `nFeatures`: Number of features
- `maxIterations`: Maximum training iterations
- `learningRate`: Step size for weight updates

**Best for:**
- Linearly separable data
- Online learning scenarios
- Simple baseline model

**Time Complexity:** O(maxIterations × n) training

---

### Linear SVM

Support vector machine with linear kernel.

```js
import { linearSvmTrain } from 'miniml';

const svm = await linearSvmTrain(X, y, nSamples, nFeatures, lambda, maxIterations, learningRate);
const prediction = await svm.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Training labels (binary)
- `nSamples`: Number of training samples
- `nFeatures`: Number of features
- `lambda`: Regularization parameter
- `maxIterations`: Maximum Pegasos iterations
- `learningRate`: Step size (eta)

**Best for:**
- High-dimensional data
- When margin maximization is important
- Text classification

**Time Complexity:** O(maxIterations × n) training

---

## Regression Algorithms

### Linear Regression

Ordinary least squares regression.

```js
import { linearRegression } from 'miniml';

const result = await linearRegression(X, y, nSamples, nFeatures);
const prediction = result.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Target values
- `nSamples`: Number of training samples
- `nFeatures`: Number of features

**Best for:**
- Linearly related data
- Quick baseline model
- Interpretable coefficients

**Time Complexity:** O(n × nFeatures² + nFeatures³) training

---

### Ridge Regression

L2-regularized linear regression.

```js
import { ridgeRegression } from 'miniml';

const result = await ridgeRegression(X, y, alpha, nSamples, nFeatures);
const prediction = result.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Target values
- `alpha`: Regularization strength (≥ 0)
- `nSamples`: Number of training samples
- `nFeatures`: Number of features

**Best for:**
- Multicollinear data
- Preventing overfitting
- When feature selection is not needed

**Time Complexity:** O(n × nFeatures² + nFeatures³) training

---

### Lasso Regression

L1-regularized linear regression via coordinate descent.

```js
import { lassoRegression } from 'miniml';

const result = await lassoRegression(X, y, alpha, l1Ratio, maxIterations, tol, nSamples, nFeatures);
const prediction = result.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Target values
- `alpha`: Regularization strength
- `l1Ratio`: L1 penalty ratio (0=Ridge, 1=Lasso)
- `maxIterations`: Maximum iterations
- `tol`: Convergence tolerance

**Best for:**
- Sparse models
- Feature selection via regularization
- High-dimensional data

**Time Complexity:** O(maxIterations × n × nFeatures) training

---

### Polynomial Regression

Nonlinear regression with polynomial features.

```js
import { polynomialRegression } from 'miniml';

const result = await polynomialRegression(X, y, nSamples, nFeatures, degree);
const prediction = result.predict(testPoint);
```

**Parameters:**
- `X`: Training features (single feature)
- `y`: Target values
- `nSamples`: Number of training samples
- `nFeatures`: Number of features (must be 1)
- `degree`: Polynomial degree

**Best for:**
- Capturing nonlinear relationships
- Curve fitting
- When feature is single-dimensional

**Time Complexity:** O(n × degree² + degree³) training

---

### Exponential Regression

Fits exponential curve: y = a × e^(bx)

```js
import { exponentialRegression } from 'miniml';

const result = await exponentialRegression(X, y, nSamples);
const prediction = result.predict(testPoint);
```

**Parameters:**
- `X`: Input values (single feature)
- `y`: Target values
- `nSamples`: Number of samples

**Best for:**
- Growth/decay modeling
- When relationship is exponential

**Time Complexity:** O(n) training

---

### Logarithmic Regression

Fits logarithmic curve: y = a + b × ln(x)

```js
import { logarithmicRegression } from 'miniml';

const result = await logarithmicRegression(X, y, nSamples);
const prediction = result.predict(testPoint);
```

**Parameters:**
- `X`: Input values (single feature, must be > 0)
- `y`: Target values
- `nSamples`: Number of samples

**Best for:**
- Diminishing returns relationships
- When data spans several orders of magnitude

**Time Complexity:** O(n) training

---

### Power Regression

Fits power law: y = a × x^b

```js
import { powerRegression } from 'miniml';

const result = await powerRegression(X, y, nSamples);
const prediction = result.predict(testPoint);
```

**Parameters:**
- `X`: Input values (single feature, must be > 0)
- `y`: Target values (must be > 0)
- `nSamples`: Number of samples

**Best for:**
- Scale-invariant relationships
- Power law distributions
- Physical/natural phenomena

**Time Complexity:** O(n) training

---

### SVR (Support Vector Regression)

Epsilon-SVR via PEGASOS-style subgradient descent.

```js
import { svrFit } from 'miniml';

const model = await svrFit(X, y, nFeatures, targets, epsilon, c, maxIterations, learningRate, seed);
const prediction = await model.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Target values
- `epsilon`: Epsilon tube width
- `c`: Regularization parameter
- `maxIterations`: Maximum iterations
- `learningRate`: Learning rate

**Best for:**
- Nonlinear regression
- Robust to outliers
- High-dimensional data

---

### Quantile Regression

Predicts conditional quantiles via pinball loss + IRLS.

```js
import { quantileRegressionFit } from 'miniml';

const model = await quantileRegressionFit(X, y, nFeatures, targets, quantile, maxIterations, learningRate, tol);
const prediction = await model.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Target values
- `quantile`: Quantile to predict (0-1)
- `maxIterations`: Maximum iterations
- `learningRate`: Learning rate
- `tol`: Convergence tolerance

**Best for:**
- Prediction intervals
- Robust regression
- Heteroscedastic data

---

### Elastic Net

Coordinate descent with combined L1+L2 penalty.

```js
import { elasticNet } from 'miniml';

const model = await elasticNet(X, y, nFeatures, targets, alpha, l1Ratio, maxIterations, tol);
const prediction = await model.predict(testPoint);
```

**Parameters:**
- `X`: Training features
- `y`: Target values
- `alpha`: Overall regularization strength
- `l1Ratio`: L1/L2 mix (0=Ridge, 1=Lasso)
- `maxIterations`: Maximum iterations
- `tol`: Convergence tolerance

**Best for:**
- Sparse models with grouped features
- High-dimensional data
- Feature selection with correlation

---

## Clustering Algorithms

### K-Means

Lloyd's algorithm for centroid-based clustering.

```js
import { kmeans } from 'miniml';

const result = await kmeans(X, nFeatures, nClusters, maxIterations);
const assignments = result.getAssignments();
const centroids = result.getCentroids();
```

**Parameters:**
- `X`: Data points
- `nFeatures`: Number of features
- `nClusters`: Number of clusters (k)
- `maxIterations`: Maximum iterations

**Best for:**
- Large datasets
- Spherical clusters
- When cluster count is known

**Time Complexity:** O(maxIterations × n × nClusters × nFeatures)

---

### K-Means++

K-Means with improved initialization using probabilistic spreading.

```js
import { kmeansPlus } from 'miniml';

const result = await kmeansPlus(X, nClusters, maxIterations, nSamples, nFeatures);
const assignments = result.getAssignments();
```

**Parameters:**
- `X`: Data points
- `nClusters`: Number of clusters
- `maxIterations`: Maximum iterations
- `nSamples`: Number of samples
- `nFeatures`: Number of features

**Best for:**
- Avoiding poor local optima
- When initialization quality matters
- General-purpose clustering

**Time Complexity:** O(n + maxIterations × n × nClusters × nFeatures)

---

### DBSCAN

Density-based spatial clustering with noise detection.

```js
import { dbscan } from 'miniml';

const labels = await dbscan(X, nFeatures, eps, minPoints);
```

**Parameters:**
- `X`: Data points
- `nFeatures`: Number of features
- `eps`: Maximum distance between points in same neighborhood
- `minPoints`: Minimum points to form dense region

**Best for:**
- Arbitrary-shaped clusters
- Noise detection
- When cluster count is unknown

**Time Complexity:** O(n²) naive, O(n log n) with spatial indexing

---

### Hierarchical Clustering

Agglomerative clustering with complete linkage.

```js
import { hierarchicalClustering } from 'miniml';

const labels = await hierarchicalClustering(X, nFeatures, nClusters);
```

**Parameters:**
- `X`: Data points
- `nFeatures`: Number of features
- `nClusters`: Number of clusters to extract

**Best for:**
- Hierarchical data organization
- Dendrogram visualization
- When cluster relationships matter

**Time Complexity:** O(n³) naive, O(n²) with priority queue optimization

---

## Probabilistic Methods

### Monte Carlo Integration

Numerical integration via random sampling.

```js
import { mcIntegrate } from 'miniml';

const integral = await mcIntegrate(fn, a, b, n, seed);
```

**Parameters:**
- `fn`: Function to integrate (takes f64, returns f64)
- `a`: Lower bound
- `b`: Upper bound
- `n`: Number of samples
- `seed`: Random seed

**Best for:**
- High-dimensional integrals
- Complex integrands
- When analytical solutions unavailable

**Time Complexity:** O(n)

---

### Monte Carlo Bootstrap

Bootstrap confidence intervals via resampling.

```js
import { mcBootstrap } from 'miniml';

const ci = await mcBootstrap(data, nBootstrap, statistic, confidence, seed);
```

**Parameters:**
- `data`: Sample data
- `nBootstrap`: Number of bootstrap iterations
- `statistic`: Statistic to compute ('mean', 'median', etc.)
- `confidence`: Confidence level (0-1)
- `seed`: Random seed

**Best for:**
- Confidence interval estimation
- Small sample inference
- Nonparametric statistics

**Time Complexity:** O(nBootstrap × n)

---

### Markov Chain Steady State

Compute steady-state distribution of discrete Markov chain.

```js
import { computeSteadyState } from 'miniml';

const steadyState = await computeSteadyState(transitionMatrix, nStates);
```

**Parameters:**
- `transitionMatrix`: Row-stochastic transition matrix
- `nStates`: Number of states

**Best for:**
- Long-term behavior analysis
- Equilibrium distributions
- PageRank-style algorithms

**Time Complexity:** O(nStates³)

---

### Hidden Markov Models (HMM)

Forward, Viterbi, backward algorithms + Baum-Welch training.

```js
import {
  hmmForward,
  hmmViterbi,
  hmmBackward,
  hmmTrainBaumWelch
} from 'miniml';

// Forward algorithm (likelihood)
const alpha = await hmmForward(initial, transition, emission, obs, nStates, nObs);

// Viterbi decoding (most likely state sequence)
const path = await hmmViterbi(initial, transition, emission, obs, nStates, nObs);

// Backward algorithm
const beta = await hmmBackward(initial, transition, emission, obs, nStates, nObs);

// Baum-Welch training (EM)
const model = await hmmTrainBaumWelch(obs, nStates, nObsSymbols, maxIterations, tol, seed);
```

**Best for:**
- Sequential data modeling
- Speech recognition
- Bioinformatics (sequence alignment)
- Financial time series

---

### MCMC (Metropolis-Hastings)

Bayesian sampling via Markov Chain Monte Carlo.

```js
import { metropolisHastings } from 'miniml';

const samples = await metropolisHastings(logTargetFn, proposalSd, nSamples, burnIn, seed, initial);
```

**Parameters:**
- `logTargetFn`: Log of target distribution (closure or function)
- `proposalSd`: Proposal distribution standard deviation
- `nSamples`: Number of samples to draw
- `burnIn`: Burn-in period
- `seed`: Random seed
- `initial`: Initial value

**Best for:**
- Bayesian inference
- Complex posterior distributions
- When analytical solutions unavailable

---

## Statistical Distributions

### Normal Distribution

PDF, CDF, quantile (PPF), and sampling for normal distribution.

```js
import {
  normalPdf,
  normalCdf,
  normalPpf,
  normalSample
} from 'miniml';

const pdf = normalPdf(x, mean, std);
const cdf = normalCdf(x, mean, std);
const quantile = normalPpf(p, mean, std);
const sample = normalSample(n, mean, std, seed);
```

---

### Binomial Distribution

PMF, CDF, and sampling for binomial distribution.

```js
import {
  binomialPmf,
  binomialCdf,
  binomialSample
} from 'miniml';
```

---

### Poisson Distribution

PMF, CDF, and sampling for Poisson distribution.

```js
import {
  poissonPmf,
  poissonCdf,
  poissonSample
} from 'miniml';
```

---

### Exponential Distribution

PDF, CDF, and sampling for exponential distribution.

```js
import {
  exponentialPdf,
  exponentialCdf,
  exponentialSample
} from 'miniml';
```

---

## Statistical Inference

### t-Tests

One-sample, two-sample, paired, and Welch's t-test.

```js
import {
  tTestOneSample,
  tTestTwoSample,
  tTestPaired,
  welchTTest
} from 'miniml';

const t = await tTestOneSample(data, nullHypothesis, alpha);
const t = await tTestTwoSample(data1, data2, alpha);
const t = await tTestPaired(data1, data2, alpha);
const t = await welchTTest(data1, data2, alpha);
```

**Best for:**
- Comparing means
- Small sample inference
- Normally distributed data

---

### Nonparametric Tests

Mann-Whitney U and Wilcoxon signed-rank tests.

```js
import {
  mannWhitneyU,
  wilcoxonSignedRank
} from 'miniml';

const u = await mannWhitneyU(data1, data2);
const w = await wilcoxonSignedRank(data1, data2);
```

**Best for:**
- Ordinal data
- Non-normal distributions
- Robust alternatives to t-tests

---

### Chi-Square Tests

Goodness of fit and independence tests.

```js
import {
  chiSquareTest,
  chiSquareIndependence
} from 'miniml';

const chi2 = await chiSquareTest(observed, expected);
const chi2 = await chiSquareIndependence(contingencyTable);
```

**Best for:**
- Categorical data analysis
- Test of independence
- Goodness of fit testing

---

### ANOVA

One-way analysis of variance.

```js
import { oneWayAnova } from 'miniml';

const f = await oneWayAnova(groups, groupSizes);
```

**Best for:**
- Comparing multiple group means
- Variance decomposition
- Experimental design analysis

---

## Kernel Methods

### RBF Kernel

Radial basis function kernel for similarity computation.

```js
import { rbfKernel, rbfKernelMatrix } from 'miniml';

const k = await rbfKernel(x, y, gamma);
const K = await rbfKernelMatrix(data, nSamples, nFeatures, gamma);
```

**Parameters:**
- `gamma`: Kernel width parameter (default: 1/nFeatures)

**Best for:**
- SVM kernel trick
- Gaussian processes
- Local similarity

---

### Polynomial Kernel

Polynomial kernel matrix.

```js
import { polynomialKernel, polynomialKernelMatrix } from 'miniml';

const k = await polynomialKernel(x, y, degree, coef0);
const K = await polynomialKernelMatrix(data, nSamples, nFeatures, degree, gamma, coef0);
```

**Best for:**
- Capturing feature interactions
- Polynomial decision boundaries
- Explicit feature maps

---

### Sigmoid Kernel

Hyperbolic tangent kernel.

```js
import { sigmoidKernel, sigmoidKernelMatrix } from 'miniml';

const k = await sigmoidKernel(x, y, gamma, coef0);
const K = await sigmoidKernelMatrix(data, nSamples, nFeatures, gamma, coef0);
```

**Best for:**
- Neural network-like similarity
- Bounded similarity measure

---

## Bayesian Methods

### Bayesian Estimation

MCMC-based parameter estimation.

```js
import { bayesianEstimate } from 'miniml';

const samples = await bayesianEstimate(logLikelihood, logPrior, nSamples, burnIn, seed, initial, proposalSd);
```

**Best for:**
- Parameter uncertainty quantification
- Prior knowledge incorporation
- Complex posteriors

---

### Bayesian Linear Regression

Conjugate prior linear regression.

```js
import { bayesianLinearRegression } from 'miniml';

const model = await bayesianLinearRegression(data, nFeatures, targets, priorPrecision, priorAlpha, priorBeta);
```

**Best for:**
- Uncertainty in regression coefficients
- Small data problems
- Prior information available

---

## Gaussian Processes

### GP Fit

Cholesky-based Gaussian process regression.

```js
import { gpFit } from 'miniml';

const model = await gpFit(data, nFeatures, targets, kernelType, kernelParams, noise);
```

**Parameters:**
- `kernelType`: 'rbf', 'polynomial', or 'sigmoid'
- `kernelParams`: Kernel hyperparameters
- `noise`: Observation noise variance

**Best for:**
- Small datasets
- Uncertainty quantification
- Nonparametric regression

---

### GP Predict

Prediction with uncertainty intervals.

```js
import { gpPredict } from 'miniml';

const pred = await gpPredict(model, xTest, nFeatures);
// { mean: [...], std: [...], lower: [...], upper: [...] }
```

---

## Survival Analysis

### Kaplan-Meier Estimator

Survival curve estimation with confidence intervals.

```js
import { kaplanMeier } from 'miniml';

const km = await kaplanMeier(times, events);
```

**Parameters:**
- `times`: Event/censoring times
- `events`: 1 for event, 0 for censored

**Best for:**
- Patient survival analysis
- Reliability engineering
- Time-to-event modeling

---

### Cox Proportional Hazards

Hazard ratio modeling via partial likelihood.

```js
import { coxProportionalHazards } from 'miniml';

const cox = await coxProportionalHazards(features, nFeatures, times, events, maxIterations, learningRate);
```

**Best for:**
- Survival factor analysis
- Treatment effect estimation
- Risk prediction

---

## Association Rules

### Apriori Algorithm

Frequent itemset mining and association rule discovery.

```js
import { apriori } from 'miniml';

const rules = await apriori(transactions, transactionLengths, minSupport, minConfidence);
```

**Parameters:**
- `transactions`: Flattened transaction array
- `transactionLengths`: Length of each transaction
- `minSupport`: Minimum support threshold
- `minConfidence`: Minimum confidence threshold

**Best for:**
- Market basket analysis
- Recommendation systems
- Pattern discovery

---

## Recommendation Systems

### Matrix Factorization

Collaborative filtering via SGD.

```js
import { matrixFactorization } from 'miniml';

const model = await matrixFactorization(ratings, nUsers, nItems, nFactors, maxIterations, learningRate, regularization, seed);
```

**Best for:**
- Rating prediction
- Collaborative filtering
- Latent factor discovery

---

### User-User Collaborative Filtering

k-NN based collaborative filtering.

```js
import { userUserCollaborative } from 'miniml';

const predictions = await userUserCollaborative(ratings, nUsers, nItems, userId, k);
```

**Best for:**
- User-based recommendations
- Similar user discovery
- Neighborhood methods

---

## Graph Algorithms

### PageRank

Link analysis ranking algorithm.

```js
import { pagerank } from 'miniml';

const ranks = await pagerank(adjacency, nNodes, damping, maxIterations, tol);
```

**Best for:**
- Web page ranking
- Citation network analysis
- Importance scoring

---

### Shortest Path

Dijkstra's algorithm for shortest paths.

```js
import { shortestPath } from 'miniml';

const distances = await shortestPath(adjacency, nNodes, source);
```

**Best for:**
- Route planning
- Network analysis
- Minimum distance computation

---

### Community Detection

Label propagation for community discovery.

```js
import { communityDetection } from 'miniml';

const labels = await communityDetection(adjacency, nNodes);
```

**Best for:**
- Social network analysis
- Graph partitioning
- Cluster discovery in networks

---

## Algorithm Selection Guide

| Data Type | Size | Dimensions | Recommended Algorithm |
|-----------|------|------------|----------------------|
| Tabular | Small (<1K) | Low (<20) | KNN, Decision Tree |
| Tabular | Medium (1K-10K) | Medium (20-100) | Random Forest, Gradient Boosting |
| Tabular | Large (>10K) | High (>100) | Logistic Regression, Linear SVM |
| Text | Any | High | Naive Bayes, Linear SVM |
| Time Series | Any | Low | SMA, EMA, Peak Detection |
| Clustering | Any | Any | K-Means++, DBSCAN, Hierarchical |
| Regression | Small | Low | Linear, Polynomial Regression |
| Regression | Large | High | Ridge Regression, Elastic Net |
| Probabilistic | Any | Any | Monte Carlo, HMM, MCMC |
| Statistical | Any | Any | t-tests, ANOVA, Chi-square |
| Survival | Any | Any | Kaplan-Meier, Cox PH |
| Graph | Any | Any | PageRank, Community Detection |
