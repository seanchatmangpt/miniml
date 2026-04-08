# Changelog

All notable changes to this project will be documented in this file.

## [26.4.8] - 2026-04-08

### Added
- 70+ ML algorithms across 15 families (classification, regression, clustering, preprocessing, time series, dimensionality reduction, metrics, model selection, feature importance, optimization, distributions, statistical inference, survival analysis, graph algorithms, recommendation)
- AutoML with genetic algorithm feature selection and PSO hyperparameter optimization
- Web Worker support for non-blocking ML operations (`createWorker()`, `createWorkerPool()`)
- SIMD-accelerated matrix operations via WASM v128 intrinsics
- Full TypeScript type definitions (`.d.ts`)

### Changed
- Renamed from micro-ml to miniml
- License changed from MIT to BSL-1.1 (converts to AGPL-3.0 on 2028-04-08)
- Replaced flat-array API with typed model objects (KnnModel, DecisionTreeModel, etc.)
- Scalers and encoders now use factory pattern (create instance, call `fitTransform()`)
- All classification/clustering functions accept `number[][]` (2D arrays)

### Fixed
- All 21+ WASM API mismatches between TypeScript wrapper and Rust WASM exports
- DTS generation enabled and verified
- All 139 tests passing with zero errors

## [1.0.0] - Initial Release

### Added
- Linear, polynomial, exponential, logarithmic, power regression
- KNN, Decision Tree, Naive Bayes, Logistic Regression classifiers
- K-Means, DBSCAN clustering
- Standard Scaler, Min-Max Scaler
- Basic metrics (accuracy, confusion matrix, MSE, R2)
- Simple moving average, exponential moving average
