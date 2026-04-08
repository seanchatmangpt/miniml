# miniml Comprehensive Benchmark Suite

Tests all 30+ algorithms across 10 categories with synthetic datasets.

## Run Instructions

```bash
cd /Users/sac/chatmangpt/micro-ml
pnpm install
pnpm build
pnpm bench
```

## Benchmark Categories

### 1. Classification Algorithms (8 algorithms)
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Naive Bayes
- Logistic Regression
- Perceptron

### 2. Regression Algorithms (6 algorithms)
- Linear Regression
- Ridge Regression
- Polynomial Regression
- Exponential Regression
- Logarithmic Regression
- Power Regression

### 3. Clustering Algorithms (4 algorithms)
- K-Means
- K-Means++
- DBSCAN
- Hierarchical Clustering

### 4. Preprocessing (8 methods)
- Standard Scaler
- MinMax Scaler
- Robust Scaler
- Normalizer
- Label Encoder
- One-Hot Encoder
- Ordinal Encoder
- Imputer

### 5. Dimensionality Reduction (2 methods)
- PCA
- Feature Selection

### 6. Metrics & Evaluation (10+ metrics)
- Confusion Matrix
- Classification Report
- Silhouette Score
- ROC AUC
- Log Loss
- Accuracy
- R²
- RMSE
- MAE
- MSE

### 7. Time Series Analysis (10+ methods)
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Weighted Moving Average (WMA)
- Exponential Smoothing
- Linear Regression Forecasting
- Polynomial Regression Forecasting
- Peak Detection
- Trough Detection
- Momentum
- Rate of Change

### 8. Ensemble Methods (3 methods)
- Random Forest (ensemble)
- Gradient Boosting (ensemble)
- AdaBoost (ensemble)
- Stacked Ensemble
- Blended Ensemble
- Voting Ensemble

### 9. Advanced Features (8 modules)
- **AutoML**: Genetic Algorithm Feature Selection + PSO Hyperparameter Optimization
- **Model Persistence**: JSON/Binary save/load, base64 encoding
- **Explainability**: SHAP values, LIME, decision paths, counterfactuals
- **DataFrame Operations**: select, filter, join, sort, aggregate
- **Neural Networks**: Dense layers, activations (ReLU, Sigmoid, Tanh, LeakyReLU), optimizers (SGD, Adam, RMSProp)
- **Causal Inference**: Propensity score matching, instrumental variables, difference-in-differences, uplift modeling
- **Transfer Learning**: ONNX export/import, fine-tuning, feature extraction
- **Data Augmentation**: SMOTE, random oversampling, noise injection, mixup, time series augmentation
- **Advanced Cross-Validation**: Stratified K-fold, group K-fold, time series CV, nested CV, LOOCV, bootstrapping

### 10. Optimization Suite (7 metaheuristics)
- Genetic Algorithms (GA)
- Particle Swarm Optimization (PSO)
- Simulated Annealing
- Multi-Armed Bandits (ε-Greedy, UCB, Thompson Sampling)
- Feature Importance
- Anomaly Detection
- Drift Detection

## Test Datasets

### Classification (1000 samples × 20 features)
- Binary classification (synthetic, 60/40 split)
- Multi-class classification (5 classes, balanced)

### Regression (500 samples × 10 features)
- Linear relationship with noise
- Non-linear relationship

### Clustering (300 samples × 5 features)
- 3 Gaussian clusters
- 2 Elongated clusters

### Time Series (100 points)
- Trend + seasonality
- Random walk

## Performance Metrics

- **Accuracy**: Algorithm correctness on test data
- **Speed**: Execution time (ms)
- **Memory**: Peak memory usage (MB)
- **Scalability**: Performance vs dataset size

## Expected Results

### Fast (<10ms)
- KNN (small k)
- Decision Tree
- Linear/Logistic Regression
- Preprocessing scalers
- Metrics computation

### Medium (10-100ms)
- Random Forest (100 trees)
- Gradient Boosting (50 estimators)
- K-Means clustering
- AutoML (GA feature selection, small population)
- Neural Network forward pass

### Slow (>100ms)
- AutoML (GA + PSO, full optimization)
- DBSCAN (large datasets)
- Ensemble Stacking (multiple base models)
- Neural Network training (multiple epochs)

## Implementation Notes

- All benchmarks run in Node.js environment (Vitest)
- Uses `performance.now()` for timing
- Memory measured via `process.memoryUsage()`
- Results exported to JSON for analysis
