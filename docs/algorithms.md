# Algorithm Reference

Complete reference for all 30+ algorithms in miniml.

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

## Preprocessing Algorithms

### Standard Scaler

Z-score normalization: (x - mean) / std

```js
import { standardScaler } from 'miniml';

const scaled = await standardScaler(X, nSamples, nFeatures);
```

**Best for:**
- Gaussian-distributed features
- Algorithms sensitive to feature scale (SVM, KNN)
- When zero-mean is desired

---

### MinMax Scaler

Scale to [0, 1] range: (x - min) / (max - min)

```js
import { minMaxScaler } from 'miniml';

const scaled = await minMaxScaler(X, nSamples, nFeatures);
```

**Best for:**
- Neural networks
- When preserving zero entries is important
- Bounded value requirements

---

### Robust Scaler

Scale using quartiles: (x - median) / IQR

```js
import { robustScaler } from 'miniml';

const scaled = await robustScaler(X, nSamples, nFeatures);
```

**Best for:**
- Data with outliers
- Non-Gaussian distributions
- When median-based scaling is preferred

---

### Normalizer

L2 normalization per sample: x / ||x||₂

```js
import { normalizer } from 'miniml';

const normalized = await normalizer(X, nSamples, nFeatures);
```

**Best for:**
- Text data (TF-IDF)
- Cosine similarity calculations
- When magnitude should be ignored

---

### Label Encoder

Convert text labels to numeric values.

```js
import { labelEncoder } from 'miniml';

const encoded = await labelEncoder(y);
```

**Best for:**
- Converting string labels to integers
- Preparing labels for ML algorithms

---

### One-Hot Encoder

Binary encoding for categorical features.

```js
import { oneHotEncoder } from 'miniml';

const oneHot = await oneHotEncoder(y, nClasses);
```

**Best for:**
- Nominal categorical data
- When no ordinal relationship exists
- Neural network inputs

---

## Dimensionality Reduction

### PCA

Principal component analysis for variance maximization.

```js
import { pca } from 'miniml';

const result = await pca(X, nSamples, nFeatures, nComponents);
const transformed = result.getTransformed();
const explainedVariance = result.getExplainedVarianceRatio();
```

**Parameters:**
- `X`: Data points
- `nSamples`: Number of samples
- `nFeatures`: Original number of features
- `nComponents`: Number of principal components

**Best for:**
- Visualization (2-3 components)
- Noise reduction
- Feature extraction

**Time Complexity:** O(n × nFeatures² + nFeatures³)

---

## Metrics & Evaluation

### Confusion Matrix

```js
import { confusionMatrix } from 'miniml';

const cm = await confusionMatrix(yTrue, yPred);
// Returns: [[TP, FP], [FN, TN]]
```

### Classification Report

```js
import { classificationReport } from 'miniml';

const report = await classificationReport(yTrue, yPred);
// Returns: precision, recall, f1-score per class
```

### Silhouette Score

```js
import { silhouetteScore } from 'miniml';

const score = await silhouetteScore(X, labels, nSamples, nFeatures);
// Range: [-1, 1], higher is better
```

### ROC AUC

```js
import { rocAucScore } from 'miniml';

const auc = await rocAucScore(yTrue, yProbabilities);
// Range: [0, 1], higher is better
```

### R² Score

```js
import { r2Score } from 'miniml';

const r2 = await r2Score(yTrue, yPred);
// Range: (-∞, 1], higher is better
```

### RMSE

```js
import { rmse } from 'miniml';

const error = await rmse(yTrue, yPred);
// Lower is better
```

### MAE

```js
import { mae } from 'miniml';

const error = await mae(yTrue, yPred);
// Lower is better
```

---

## Time Series Algorithms

### Simple Moving Average (SMA)

```js
import { sma } from 'miniml';

const smoothed = await sma(data, window);
```

### Exponential Moving Average (EMA)

```js
import { ema } from 'miniml';

const smoothed = await ema(data, span);
```

### Peak Detection

```js
import { peakDetection } from 'miniml';

const peaks = await peakDetection(data, distance, threshold);
```

### Trough Detection

```js
import { troughDetection } from 'miniml';

const troughs = await troughDetection(data, distance, threshold);
```

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
| Regression | Large | High | Ridge Regression |
