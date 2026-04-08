# Algorithm Selection Guide

Choose the right algorithm for your problem. This guide helps you pick based on data type, problem category, and constraints.

## Decision Tree

```
What is your goal?
|
|-- Predict a number → Regression
|   |-- Linear relationship? → Linear Regression
|   |-- Nonlinear curve? → Polynomial / Exponential / Logarithmic
|   |-- Multiple features? → Ridge / Lasso / Elastic Net
|   |-- Outliers in data? → RANSAC / Theil-Sen
|   |-- Binary target? → Logistic Regression (classification)
|
|-- Predict a category → Classification
|   |-- Small dataset, fast? → kNN / Perceptron
|   |-- High accuracy needed? → Random Forest / Gradient Boosting
|   |-- Text/binary features? → Naive Bayes (Bernoulli/Multinomial)
|   |-- Need probabilities? → Logistic Regression / SVM
|   |-- Streaming data? → SGD / Passive Aggressive
|   |-- Multiple models? → Voting Classifier / AdaBoost
|
|-- Find groups → Clustering
|   |-- Know number of clusters? → K-Means / K-Means++
|   |-- Don't know clusters? → DBSCAN / Hierarchical
|   |-- Overlapping clusters? → GMM
|   |-- Large dataset? → Mini-Batch K-Means
|   |-- Graph-like data? → Spectral Clustering
|
|-- Find anomalies → Anomaly Detection
|   |-- Tree-based? → Isolation Forest
|   |-- Density-based? → LOF
|
|-- Time-to-event analysis → Survival Analysis
|   |-- Equipment reliability? → Weibull Analysis
|   |-- Customer churn? → Weibull / Exponential
|   |-- Need hazard rates? → Weibull (shape-dependent)
|
|-- Detect changes in data → Drift Detection
|   |-- Continuous data? → EWMA / Z-Score
|   |-- Categorical data? → Jaccard Window
|   |-- Sudden changes? → Page-Hinkley
|   |-- Streaming monitoring? → EWMA (low latency)
|
|-- Predict sequences → Sequence Prediction
|   |-- Next activity prediction? → NGram
|   |-- Text prediction? → NGram (bigram/trigram)
|   |-- Need probabilities? → NGram with Laplace smoothing
|
|-- Reduce dimensions → Preprocessing
|   |-- Linear reduction? → PCA
|   |-- Scale features? → Standard / Min-Max / Robust Scaler
|   |-- Encode categories? → Label / One-Hot / Ordinal Encoder
|   |-- Fill missing values? → Imputer
|   |-- Non-normal distribution? → Power Transformer
|   |-- Chain operations? → Pipeline
```

## Algorithm Comparison Tables

### Regression

| Algorithm | Features | Nonlinear | Outlier Robust | Speed | WASM Size |
|-----------|----------|-----------|---------------|-------|-----------|
| Linear | 1 | No | No | Fast | Small |
| Ridge | Multi | No | No | Fast | Small |
| Lasso | Multi | No | No | Fast | Small |
| Elastic Net | Multi | No | No | Medium | Small |
| Polynomial | 1 | Yes | No | Fast | Small |
| Exponential | 1 | Yes | No | Fast | Small |
| Logarithmic | 1 | Yes | No | Fast | Small |
| RANSAC | Multi | Yes | Yes | Medium | Medium |
| Theil-Sen | Multi | No | Yes | Slow | Small |
| Decision Tree (regress) | Multi | Yes | No | Fast | Medium |

### Classification

| Algorithm | Multi-class | Probabilities | Streaming | Accuracy | Speed |
|-----------|-------------|---------------|-----------|----------|-------|
| kNN | Yes | Yes | No | Medium | Slow (inference) |
| Logistic Regression | Binary | Yes | No | Medium | Fast |
| SVM | Binary | No | No | High | Medium |
| Perceptron | Binary | No | Yes | Low | Fast |
| Decision Tree | Yes | No | No | Medium | Fast |
| Random Forest | Yes | Yes | No | High | Medium |
| Gradient Boosting | Binary | Yes | No | High | Medium |
| AdaBoost | Binary | Yes | No | High | Medium |
| Naive Bayes | Yes | Yes | No | Medium | Fast |
| SGD | Binary | No | Yes | Medium | Fast |
| Passive Aggressive | Binary | No | Yes | Medium | Fast |
| Voting Classifier | Yes | Yes | No | High | Varies |
| Extra Trees | Yes | Yes | No | High | Fast |
| Bagging | Yes | No | No | High | Medium |

### Clustering

| Algorithm | K Required | Shape | Outlier Detection | Speed | Scalability |
|-----------|-----------|-------|-------------------|-------|-------------|
| K-Means | Yes | Spherical | No | Fast | Good |
| K-Means++ | Yes | Spherical | No | Fast | Good |
| Mini-Batch K-Means | Yes | Spherical | No | Fast | Excellent |
| DBSCAN | No | Arbitrary | Yes | Medium | Good |
| Hierarchical (single) | Yes | Chain-like | No | Slow | Poor |
| Hierarchical (complete) | Yes | Compact | No | Slow | Poor |
| Spectral | Yes | Complex | No | Slow | Poor |
| GMM | Yes | Ellipsoidal | No | Medium | Good |

### Survival Analysis

| Algorithm | Data Type | Hazard Shape | Output | Use Case |
|-----------|-----------|--------------|--------|----------|
| Weibull | Time-to-event | Variable (k parameter) | Survival prob, hazard rate | Reliability, churn |
| Exponential | Time-to-event | Constant (k=1) | Survival prob, hazard rate | Simple time modeling |

### Drift Detection

| Algorithm | Data Type | Sensitivity | Window Size | Use Case |
|-----------|-----------|-------------|-------------|----------|
| EWMA | Continuous | Tunable (λ) | N/A (online) | Real-time monitoring |
| Z-Score | Continuous | Statistical | Configurable | Mean shift detection |
| Jaccard Window | Categorical | Threshold-based | Configurable | Distribution change |
| Page-Hinkley | Continuous | Cumulative | N/A (online) | Sudden change detection |

### Sequence Prediction

| Algorithm | Context Size | Smoothing | Output | Use Case |
|-----------|-------------|-----------|--------|----------|
| Unigram (n=1) | 0 | Optional | Top-k items | Simple frequency |
| Bigram (n=2) | 1 | Optional | Top-k items | Next activity |
| Trigram (n=3) | 2 | Optional | Top-k items | Context-aware prediction |

## When to Use What

### Small datasets (<100 samples)
- kNN, Decision Tree, Naive Bayes
- Avoid: Gradient Boosting, Random Forest (need more data)

### Large datasets (>10K samples)
- Mini-Batch K-Means, SGD, Passive Aggressive
- DBSCAN (with spatial indexing)
- Avoid: Hierarchical clustering, LOF (O(n^2))

### Real-time inference
- Linear models, Decision Trees, Perceptron
- Pre-computed models: Voting Classifier with stored predictions

### Noisy data
- RANSAC, Theil-Sen (regression)
- Random Forest, Extra Trees (classification)
- Isolation Forest, LOF (anomaly detection)

### Feature selection
- RFE (recursive elimination)
- Permutation Importance (model-agnostic)
- Lasso (built-in L1 regularization)
- Grid Search (hyperparameter tuning)

### Survival analysis (time-to-event)
- Weibull: Equipment failure, customer churn, medical survival
- Exponential: Simple constant-hazard scenarios
- Use hazard rates to identify risk periods
- Use survival probabilities for warranty/cost analysis

### Drift monitoring (data streams)
- EWMA: Real-time metrics, model performance tracking
- Z-Score: Statistical process control, quality monitoring
- Jaccard: Categorical feature distribution changes
- Page-Hinkley: Sudden concept drift, break point detection
- Set appropriate thresholds based on acceptable false positive rate

### Sequence prediction
- NGram (n=1-3): Process mining, text prediction, recommendations
- Use Laplace smoothing for unseen contexts
- Use perplexity to evaluate model quality
- Higher n captures more context but requires more data
