# API Reference

Complete listing of all 59 modules and their exported functions.

## Convention

- **Flat arrays**: All matrix inputs are `Float64Array` (row-major, `[n_samples * n_features]`)
- **Labels**: `Float64Array` of length `n_samples`
- **Returns**: Model objects with `predict()`, `predictProba()` (where applicable), and `toString()`
- **Errors**: `JsError` thrown on invalid input

---

## Regression

### `linearRegression(x, y)` → `LinearModel`
Simple linear regression: `y = slope * x + intercept`
- `x: Float64Array` - Input values
- `y: Float64Array` - Target values
- Returns: `{ slope, intercept, rSquared, n, predict(x) }`

### `linearRegressionSimple(y)` → `LinearModel`
Auto-generates x values as `[0, 1, 2, ...]`

### `polynomialRegression(x, y, degree)` → `PolynomialModel`
Polynomial curve fitting: `y = c0 + c1*x + c2*x^2 + ...`
- `degree: number` - Default: 2
- Returns: `{ degree, rSquared, n, getCoefficients(), predict(x) }`

### `exponentialRegression(x, y)` → `ExponentialModel`
`y = a * e^(b*x)`. All y values must be positive.
- Returns: `{ a, b, rSquared, n, predict(x), doublingTime() }`

### `logarithmicRegression(x, y)` → `LogarithmicModel`
`y = a + b * ln(x)`. All x values must be positive.

### `powerRegression(x, y)` → `PowerModel`
`y = a * x^b`. All x and y must be positive.

### `ridgeRegression(data, nFeatures, labels, alpha)` → `RidgeRegressionModel`
L2-regularized linear regression.
- `alpha: number` - Regularization strength (default: 1.0)
- Returns: `{ coefficients(), predict(data) }`

### `lassoRegression(data, nFeatures, labels, alpha, maxIter)` → `LassoRegressionModel`
L1-regularized via coordinate descent.
- Returns: `{ coefficients(), predict(data) }`

### `elasticNet(data, nFeatures, labels, alpha, l1Ratio, maxIter)` → `ElasticNetModel`
Combined L1+L2 regularization.
- `l1Ratio: number` - Balance between L1 and L2 (default: 0.5)

### `ransacRegression(data, nFeatures, labels, maxIterations, threshold)` → `RansacModel`
Robust regression via random sample consensus.
- Returns: `{ inlierMask(), coefficients(), predict(data) }`

### `theilSenRegression(data, nFeatures, labels)` → `TheilSenModel`
Median-based robust regression.

---

## Classification

### `knnFit(data, nFeatures, labels, k)` → `KnnModel`
K-nearest neighbors classifier.
- `k: number` - Default: 3
- Returns: `{ k, nSamples, predict(data), predictProba(data) }`

### `logisticRegression(data, nFeatures, labels, lr, maxIter, lambda)` → `LogisticModel`
Binary logistic regression via gradient descent.
- Returns: `{ bias, iterations, loss, getWeights(), predict(data), predictProba(data) }`

### `svmClassify(data, nFeatures, labels, lr, maxIter, lambda)` → `SvmModel`
Linear SVM (soft-margin).
- Returns: `{ weights(), bias(), predict(data) }`

### `perceptron(data, nFeatures, labels, lr, maxIter)` → `PerceptronModel`
Single-layer perceptron.
- Returns: `{ bias, iterations, converged, getWeights(), predict(data) }`

### `decisionTreeClassify(data, nFeatures, labels, maxDepth, minSplit)` → `DecisionTreeModel`
Decision tree classifier.
- `maxDepth: number` - Default: 10
- `minSplit: number` - Default: 2
- Returns: `{ depth, nNodes, predict(data), getTree() }`

### `decisionTreeRegress(data, nFeatures, labels, maxDepth, minSplit)` → `DecisionTreeModel`
Decision tree regressor.

### `naiveBayesFit(data, nFeatures, labels)` → `NaiveBayesModel`
Gaussian naive Bayes.
- Returns: `{ nClasses, nFeatures, predict(data), predictProba(data) }`

### `bernoulliNbClassify(data, nFeatures, labels, alpha)` → `BernoulliNbModel`
Bernoulli naive Bayes for binary features.
- `alpha: number` - Laplace smoothing (default: 1.0)

### `multinomialNbClassify(data, nFeatures, labels, alpha)` → `MultinomialNbModel`
Multinomial naive Bayes for count features.

### `sgdClassify(data, nFeatures, labels, lossType, lr, maxIter, lambda)` → `SgdModel`
SGD classifier.
- `lossType: string` - "hinge" (SVM), "log" (logistic), "modified_huber"

### `passiveAggressiveClassify(data, nFeatures, labels, mode, c, maxIter)` → `PassiveAggressiveModel`
- `mode: string` - "pa1" or "pa2"

---

## Clustering

### `kmeans(data, nFeatures, k, maxIter)` → `KMeansModel`
- Returns: `{ k, iterations, inertia, getCentroids(), getAssignments(), predict(data) }`

### `kmeansPlus(data, nFeatures, k, maxIter)` → `Vec<f64>`
K-Means++ with smart centroid initialization. Returns flat array: `[nClusters, assignments..., centroids...]`

### `miniBatchKmeans(data, nFeatures, k, maxIter, batchSize)` → `Vec<f64>`
Online K-Means variant for large datasets.

### `dbscan(data, nFeatures, eps, minPoints)` → `DbscanResult`
- `eps: number` - Neighborhood radius
- `minPoints: number` - Minimum points for core point
- Returns: `{ nClusters, nNoise, getLabels() }`

### `hierarchicalClustering(data, nFeatures, nClusters)` → `Vec<f64>`
Agglomerative clustering with single linkage.

### `agglomerativeComplete(data, nFeatures, nClusters)` → `Vec<f64>`
Agglomerative clustering with complete linkage.

### `spectralClustering(data, nFeatures, nClusters, sigma)` → `Vec<f64>`
Graph-based spectral clustering.

### `gmmFit(data, nFeatures, nClusters, maxIter)` → `GmmModel`
Gaussian Mixture Models via EM algorithm.
- Returns: `{ nClusters, getMeans(), getCovariances(), getWeights(), predict(data), predictProba(data) }`

---

## Ensemble Methods

### `randomForestClassify(data, nFeatures, labels, nTrees, maxDepth)` → `RandomForestModel`
- Returns: `{ nTrees, predict(data), predictProba(data) }`

### `gradientBoostingClassify(data, nFeatures, labels, nTrees, maxDepth, lr)` → `GradientBoostingModel`
- Returns: `{ nTrees, nFeatures, learningRate, predict(data), predictProba(data) }`

### `adaboostClassify(data, nFeatures, labels, nEstimators, learningRate)` → `AdaBoostClassifier`
- Returns: `{ nEstimators, nFeatures, predict(data), predictProba(data) }`

### `extraTreesClassify(data, nFeatures, labels, nTrees, maxDepth)` → `ExtraTreesModel`
Extremely randomized trees.

### `baggingClassify(data, nFeatures, labels, nEstimators, maxDepth, sampleRatio)` → `BaggingModel`
Bootstrap aggregating.

### `votingClassifier(predictions, nModels, weights, votingType, nClasses)` → `VotingClassifier`
- `votingType: string` - "hard" or "soft"
- Returns: `{ nModels, votingType, nClasses, aggregate() }`

---

## Preprocessing

### `standardScalerFit(data, nFeatures)` → `StandardScaler`
Z-score normalization (mean=0, stddev=1).
- Returns: `{ transform(data), inverse(data) }`

### `minmaxScalerFit(data, nFeatures)` → `MinMaxScaler`
Scale to [0, 1] range.

### `robustScalerFit(data, nFeatures)` → `RobustScaler`
Scale using median and IQR.

### `normalizerFit(data, nFeatures)` → `Normalizer`
L2 row normalization.

### `labelEncoderFit(labels)` → `LabelEncoder`
- Returns: `{ transform(labels), inverse(encoded) }`

### `oneHotEncoderFit(data, nFeatures)` → `OneHotEncoder`
- Returns: `{ transform(data), inverse(encoded) }`

### `ordinalEncoderFit(labels)` → `OrdinalEncoder`
- Returns: `{ transform(labels), inverse(encoded) }`

### `powerTransformerFit(data, nFeatures)` → `PowerTransformer`
Yeo-Johnson power transformation for normalizing skewed data.

### `imputerFit(data, nFeatures, strategy)` → `Imputer`
- `strategy: string` - "mean", "median", "most_frequent"
- Returns: `{ transform(data) }`

### `pca(data, nFeatures, nComponents)` → `PcaResult`
- Returns: `{ nComponents, getComponents(), getExplainedVariance(), getExplainedVarianceRatio(), getTransformed(), transform(data) }`

### `pipelineTransform(steps, data, nFeatures)` → `Vec<f64>`
Apply a sequence of named preprocessing steps.

---

## Metrics

### Regression Metrics
- `rSquared(actual, predicted)` → `number`
- `rmse(actual, predicted)` → `number`
- `mae(actual, predicted)` → `number`

### Classification Metrics
- `confusionMatrix(actual, predicted, nClasses)` → `Vec<f64>` (flat NxN matrix)
- `precision(matrix, classIdx)` → `number`
- `recall(matrix, classIdx)` → `number`
- `f1Score(matrix, classIdx)` → `number`
- `mcc(matrix, nClasses)` → `number`
- `rocAuc(yTrue, yScores)` → `number`

### Clustering Metrics
- `silhouetteScore(data, nFeatures, labels)` → `number`

---

## Model Selection

### `kFoldSplit(nSamples, k)` → `Vec<f64>`
Returns fold indices: `[k, fold0_start, fold0_end, fold1_start, ...]`

### `dataSplit(data, labels, testRatio)` → `{ trainData, trainLabels, testData, testLabels }`

### `gridSearch(cvScores, nFolds, nParams)` → `GridSearchResult`
From pre-computed cross-validation scores.

### `rfe(data, nFeatures, labels, nFeaturesToSelect, estimatorType)` → `RfeResult`
Recursive Feature Elimination.

### `permutationImportance(data, nFeatures, labels, nRepeats)` → `Vec<f64>`
Model-agnostic feature importance via permutation.

---

## Anomaly Detection

### `isolationForestFit(data, nFeatures, nTrees, sampleSize)` → `IsolationForestModel`
- Returns: `{ predict(data) }` (returns anomaly scores)

### `lofPredict(data, nFeatures, k)` → `Vec<f64>`
Local Outlier Factor scores. Negative = outlier.

---

## Time Series

### `movingAverage(data, window, type)` → `Vec<f64>`
- `type: MovingAverageType` - SMA, EMA, or WMA

### `seasonalDecompose(data, period)` → `SeasonalDecomposition`
- Returns: `{ period, getTrend(), getSeasonal(), getResidual() }`

### `trendForecast(data, periods)` → `TrendAnalysis`
- Returns: `{ direction, slope, strength, getForecast() }`

### `autocorrelation(data, maxLag)` → `Vec<f64>`

### `detectSeasonality(data)` → `SeasonalityInfo`
- Returns: `{ period, strength }`
