# All Algorithms

Complete list of algorithms available in miniml, organized by family.

## Classification

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| KNN | `knnTrain`, `knnPredict` | K-Nearest Neighbors classifier | `k` neighbors, distance metric |
| Decision Tree | `decisionTreeTrain`, `decisionTreePredict` | CART decision tree | `maxDepth`, `minSamplesSplit` |
| Random Forest | `randomForestTrain`, `randomForestPredict` | Ensemble of decision trees | `nTrees`, `maxDepth` |
| Logistic Regression | `logisticTrain`, `logisticPredict` | Binary classification via sigmoid | `learningRate`, `maxIter`, `lambda` |
| Naive Bayes | `naiveBayesTrain`, `naiveBayesPredict` | Gaussian Naive Bayes classifier | `nClasses` |
| SVM | `svmTrain`, `svmPredict` | Support Vector Machine | `kernel`, `C`, `gamma`, `maxIter` |
| Perceptron | `perceptronTrain`, `perceptronPredict` | Single-layer linear classifier | `learningRate`, `maxIter` |
| Gradient Boosting | `gradientBoostingTrain`, `gradientBoostingPredict` | Sequential ensemble boosting | `nTrees`, `learningRate`, `maxDepth` |
| AdaBoost | `adaboostTrain`, `adaboostPredict` | Adaptive boosting ensemble | `nEstimators`, `learningRate` |
| Neural Network | `neuralTrain`, `neuralPredict` | Feedforward neural network | `hiddenSize`, `learningRate`, `epochs` |
| Stacking | `stackingTrain`, `stackingPredict` | Stacked ensemble meta-learner | Base models, meta-learner |

## Regression

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| Linear Regression | `linearRegressionTrain`, `linearRegressionPredict` | Ordinary least squares | `learningRate`, `maxIter`, `lambda` |
| Ridge Regression | `ridgeRegressionTrain`, `ridgeRegressionPredict` | L2-regularized regression | `lambda` (regularization strength) |
| Polynomial Regression | `polynomialRegressionTrain`, `polynomialRegressionPredict` | Polynomial feature mapping + linear fit | `degree` |
| Elastic Net | `elasticNetTrain`, `elasticNetPredict` | L1+L2 combined regularization | `alpha`, `l1Ratio` |
| SVR | `svrTrain`, `svrPredict` | Support Vector Regression | `kernel`, `C`, `epsilon`, `gamma` |
| Quantile Regression | `quantileRegressionTrain`, `quantileRegressionPredict` | Predict conditional quantiles | `quantile` (0.0-1.0) |
| Bayesian Linear | `bayesianLinearRegression` | Bayesian regression with conjugate prior | `priorPrecision`, `priorAlpha`, `priorBeta` |
| Gaussian Process | `gpFit`, `model.predict()` | Nonparametric GP regression | `kernelType`, `kernelParams`, `noise` |

## Clustering

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| K-Means | `kmeansFit`, `kmeansPredict` | Lloyd's algorithm | `k`, `maxIter` |
| K-Means++ | `kmeansPlusFit`, `kmeansPlusPredict` | Smart initialization | `k`, `maxIter` |
| DBSCAN | `dbscanFit`, `dbscanPredict` | Density-based clustering | `epsilon`, `minPoints` |
| Hierarchical | `hierarchicalFit` | Agglomerative clustering | `nClusters`, `linkage` |

## Preprocessing

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| Standard Scaler | `standardScalerFit`, `standardScalerTransform` | Zero mean, unit variance | None |
| Min-Max Scaler | `minmaxScalerFit`, `minmaxScalerTransform` | Scale to [0, 1] | `featureRange` |
| Robust Scaler | `robustScalerFit`, `robustScalerTransform` | Median and IQR scaling | None |
| Normalizer | `normalizerFit`, `normalizerTransform` | L2 row normalization | None |
| Label Encoder | `labelEncoderFit`, `labelEncoderTransform` | Encode labels as integers | None |
| One-Hot Encoder | `oneHotEncoderFit`, `oneHotEncoderTransform` | Binary column encoding | `nClasses` |
| Ordinal Encoder | `ordinalEncoderFit`, `ordinalEncoderTransform` | Order-preserving integer encoding | None |
| PCA | `pcaFit`, `pcaTransform` | Principal Component Analysis | `nComponents` |
| Imputer | `imputerFit`, `imputerTransform` | Fill missing values | `strategy` ("mean", "median") |

## Statistical Tests

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| One-Sample T-Test | `tTestOneSample` | Test mean vs hypothesized value | `hypothesizedMean`, `alpha` |
| Two-Sample T-Test | `tTestTwoSample` | Independent groups comparison | `alpha` |
| Paired T-Test | `tTestPaired` | Repeated measures comparison | `alpha` |
| Welch's T-Test | `welchTTest` | Unequal variance comparison | `alpha` |
| Mann-Whitney U | `mannWhitneyU` | Nonparametric independent test | None |
| Wilcoxon Signed-Rank | `wilcoxonSignedRank` | Nonparametric paired test | None |
| Chi-Square Test | `chiSquareTest` | Goodness-of-fit test | None |
| Chi-Square Independence | `chiSquareIndependence` | Contingency table test | `nRows`, `nCols` |
| One-Way ANOVA | `oneWayAnova` | Multi-group mean comparison | `groupSizes` |
| KS Test | `ksTest` | Normality test | None |

## Probabilistic & Bayesian

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| Monte Carlo Pi | `mcEstimatePi` | Estimate pi via sampling | `nSamples`, `seed` |
| Bootstrap | `mcBootstrap` | Resampling CI estimation | `nBootstrap`, `statistic`, `confidence` |
| Expected Value | `mcExpectedValue` | MC expected value | `a`, `b`, `nSamples` |
| Markov Chain | `MarkovChain.fromMatrix` | Discrete state transitions | `nStates`, transition matrix |
| Steady State | `chain.steadyState` | Stationary distribution | `maxIter`, `tol` |
| N-Step Probability | `chain.nStepProbability` | Multi-step transitions | `nSteps` |
| HMM Forward | `hmm.forward` | Observation likelihood | `observations` |
| HMM Viterbi | `hmm.viterbi` | Most likely state sequence | `observations` |
| HMM Baum-Welch | `HMM.train` | Learn HMM parameters | `nStates`, `nObsSymbols`, `maxIter` |
| Metropolis-Hastings | `metropolisHastings` | MCMC sampling | `proposalSd`, `nSamples`, `burnIn` |
| Bayesian Estimate | `bayesianEstimate` | Posterior inference | `nSamples`, `burnIn`, `proposalSd` |

## Kernels

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| RBF Kernel | `rbfKernel`, `rbfKernelMatrix` | Gaussian kernel | `gamma` |
| Polynomial Kernel | `polynomialKernel`, `polynomialKernelMatrix` | Polynomial mapping | `degree`, `gamma`, `coef0` |
| Sigmoid Kernel | `sigmoidKernel`, `sigmoidKernelMatrix` | Tanh kernel | `gamma`, `coef0` |

## Survival Analysis

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| Kaplan-Meier | `kaplanMeier` | Survival curve estimation | `times`, `events` |
| Cox PH | `coxProportionalHazards` | Hazard ratio modeling | `maxIter`, `lr` |

## Graph Algorithms

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| PageRank | `pageRank` | Link importance ranking | `damping`, `maxIter`, `tol` |
| Shortest Path | `shortestPath` | Dijkstra's algorithm | `source` |
| Community Detection | `communityDetection` | Label propagation | `maxIter` |

## Optimization

| Algorithm | Function | Description | Key Parameters |
|-----------|----------|-------------|----------------|
| Genetic Algorithm | (internal) | Feature selection | `populationSize`, `generations` |
| PSO | (internal) | Hyperparameter optimization | `particles`, `inertia`, `cognitive`, `social` |
| Simulated Annealing | (internal) | Temperature-based search | `temperature`, `coolingRate` |
| AutoML | `automlFit`, `automlPredict` | Automated pipeline | Algorithm selection, CV folds |

## Metrics

| Metric | Function | Description |
|--------|----------|-------------|
| Accuracy | `accuracy` | Classification accuracy |
| Confusion Matrix | `confusionMatrix` | TP, TN, FP, FN counts |
| Silhouette Score | `silhouetteScore` | Clustering quality |
| R-squared | `r2Score` | Regression goodness of fit |
| MAE, MSE, RMSE | `mae`, `mse`, `rmse` | Regression error metrics |
| Descriptive Stats | `describe` | Mean, median, std, skewness, kurtosis |
