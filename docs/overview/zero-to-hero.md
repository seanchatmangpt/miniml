# Zero to Hero: micro-ml Complete Guide

A single-page guide that walks you from absolute beginner to advanced practitioner. Every section links to detailed documentation.

---

## Level 0: What Is This?

**micro-ml** is a zero-dependency machine learning library that runs in the browser via WebAssembly. No Python, no server, no heavy frameworks. Just `npm install micro-ml` and you have 59 ML algorithms at your fingertips.

**Start here:** [What is micro-ml?](./intro.md)

**Key facts:**
- ~56KB gzipped WASM bundle
- Zero runtime dependencies
- Sub-millisecond predictions
- Full TypeScript support
- Works in browser and Node.js

---

## Level 1: First Steps (5 minutes)

Install and make your first prediction.

**Read:** [Quick Start Guide](../tutorials/quick-start.md)

```ts
import { linearRegression } from 'micro-ml';
const model = await linearRegression([1, 2, 3], [2, 4, 6]);
console.log(model.predict([4]));  // [8]
```

You've just trained and used a machine learning model. That's it.

**What you learned:**
- How to import and initialize micro-ml
- How to fit a model
- How to make predictions

---

## Level 2: Choosing the Right Algorithm

Not all problems are the same. Learn which algorithm to use when.

**Read:** [Algorithm Selection Guide](./algorithm-guide.md)

The decision tree:
1. **Predicting a number?** → [Regression Tutorial](../tutorials/regression.md)
2. **Predicting a category?** → [Classification Tutorial](../tutorials/classification.md)
3. **Finding groups?** → [Clustering Tutorial](../tutorials/clustering.md)
4. **Finding weird things?** → Anomaly Detection (Isolation Forest, LOF)
5. **Preparing data?** → Preprocessing (scalers, encoders, PCA)

**What you learned:**
- Problem types: regression, classification, clustering, anomaly detection
- How to match algorithms to problems
- When to use simple vs complex models

---

## Level 3: Building Your First Real Model

### Regression: Predicting House Prices

**Read:** [Regression Tutorial](../tutorials/regression.md)

```ts
import { ridgeRegression, rSquared, rmse } from 'micro-ml';

const data = new Float64Array([
  1500, 3, 10,  // sqft, beds, age
  2000, 4, 5,
  1200, 2, 20,
]);
const prices = new Float64Array([300, 450, 200]);

const model = ridgeRegression(data, 3, prices, 1.0);
const prediction = model.predict(new Float64Array([1800, 3, 8]));
```

### Classification: Spam Detection

**Read:** [Classification Tutorial](../tutorials/classification.md)

```ts
import { logisticRegression, confusionMatrix, f1Score } from 'micro-ml';

const features = [[50, 0, 0.1], [200, 1, 0.8], [30, 0, 0.05]];
const labels = [0, 1, 0];

const model = await logisticRegression(features, labels);
const predictions = model.predict([[100, 1, 0.7]]);
```

### Clustering: Customer Segmentation

**Read:** [Clustering Tutorial](../tutorials/clustering.md)

```ts
import { kmeans, silhouetteScore } from 'micro-ml';

const model = await kmeans(customerData, { k: 3 });
console.log(model.getAssignments());
```

**What you learned:**
- Training models with real data
- Multi-feature inputs (flat arrays)
- Making predictions on new data
- Evaluating model quality

---

## Level 4: Data Preprocessing

Raw data needs preparation. Learn the preprocessing pipeline.

**Read:** [API Reference - Preprocessing](../api/reference.md#preprocessing)

```ts
// 1. Scale features
const scaler = standardScalerFit(rawData, nFeatures);
const scaled = scaler.transform(rawData);

// 2. Encode labels
const encoder = labelEncoderFit(labels);
const encoded = encoder.transform(newLabels);

// 3. Dimensionality reduction
const pcaResult = pca(scaled, nFeatures, 5);
const reduced = pcaResult.transform(newData);

// 4. Pipeline (chain operations)
const pipeline = pipelineTransform(steps, rawData, nFeatures);
```

**Available tools:**
- **Scalers:** Standard, Min-Max, Robust, L2 Normalizer
- **Encoders:** Label, One-Hot, Ordinal
- **Transformers:** Power Transformer (Yeo-Johnson), PCA
- **Utilities:** Imputer (fill missing values), Pipeline (chain operations)

**What you learned:**
- Feature scaling and why it matters
- Encoding categorical variables
- Reducing dimensionality with PCA
- Chaining operations with Pipeline

---

## Level 5: Ensemble Methods

Combine multiple models for better accuracy.

**Read:** [API Reference - Ensemble Methods](../api/reference.md#ensemble-methods)

```ts
// Random Forest: many trees, majority vote
const rf = await randomForestClassify(data, nFeatures, labels, 100, 10);

// Gradient Boosting: sequential error correction
const gb = await gradientBoostingClassify(data, nFeatures, labels, 50, 3, 0.1);

// AdaBoost: weighted weak learners
const ab = await adaboostClassify(data, nFeatures, labels, 50, 1.0);

// Voting Classifier: combine different model types
const vc = votingClassifier(allPredictions, 3, weights, 'soft', 2);
const final = vc.aggregate();
```

**What you learned:**
- Why ensembles work better than single models
- Bagging (Random Forest, Extra Trees, Bagging)
- Boosting (Gradient Boosting, AdaBoost)
- Voting (Hard majority vote, Soft probability averaging)

---

## Level 6: Model Evaluation

Don't just train - measure. Learn to evaluate rigorously.

**Read:** [API Reference - Metrics](../api/reference.md#metrics)

### Classification Metrics

```ts
import { confusionMatrix, precision, recall, f1Score, mcc, rocAuc } from 'micro-ml';

const cm = confusionMatrix(actual, predicted, nClasses);
console.log({
  precision: precision(cm, 1),
  recall: recall(cm, 1),
  f1: f1Score(cm, 1),
  mcc: mcc(cm, nClasses),
  auc: rocAuc(actual, scores),
});
```

### Regression Metrics

```ts
import { rSquared, rmse, mae } from 'micro-ml';
console.log({ r2: rSquared(actual, pred), rmse: rmse(actual, pred), mae: mae(actual, pred) });
```

### Clustering Metrics

```ts
import { silhouetteScore } from 'micro-ml';
const score = silhouetteScore(data, nFeatures, labels);
```

### Cross-Validation

```ts
import { kFoldSplit, dataSplit } from 'micro-ml';
const folds = kFoldSplit(nSamples, 5);
const { trainData, testData, trainLabels, testLabels } = dataSplit(data, labels, 0.2);
```

**What you learned:**
- Confusion matrix and derived metrics
- When to use precision vs recall vs F1
- MCC for class imbalance
- AUC for threshold-independent evaluation
- Cross-validation for robust estimates

---

## Level 7: Advanced Topics

### Feature Selection

```ts
import { rfe, permutationImportance } from 'micro-ml';

// Recursive Feature Elimination
const rfeResult = rfe(data, nFeatures, labels, 5, 'tree');

// Model-agnostic permutation importance
const importance = permutationImportance(data, nFeatures, labels, 10);
```

### Hyperparameter Tuning

```ts
import { gridSearch } from 'micro-ml';
// Pre-compute CV scores, then find best params
const best = gridSearch(cvScores, nFolds, nParams);
```

### Anomaly Detection

```ts
import { isolationForestFit, lofPredict } from 'micro-ml';

const ifModel = isolationForestFit(data, nFeatures, 100, 256);
const anomalyScores = ifModel.predict(data);

const lofScores = lofPredict(data, nFeatures, 20);
// Negative scores = outliers
```

### Robust Regression

```ts
import { ransacRegression, theilSenRegression } from 'micro-ml';
// When data has outliers
const ransac = ransacRegression(data, nFeatures, labels, 100, 10.0);
const theil = theilSenRegression(data, nFeatures, labels);
```

**What you learned:**
- Feature selection techniques
- Handling outliers with robust methods
- Anomaly detection algorithms
- Hyperparameter optimization

---

## Level 8: Understanding the Internals

**Read:** [Architecture](./architecture.md)

- How WASM compilation works
- Flat array data representation
- Error handling with `Result<T, JsError>`
- Deterministic randomness via `Rng::from_data()`
- Zero-dependency constraint

**Read:** [Algorithm Details](../api/algorithms.md)

- Mathematical formulas for each algorithm
- Time and space complexity
- Implementation references

---

## Level 9: Contributing

**Read:** [Contributing Guide](../api/contributing.md)

- Code style requirements (`cargo fmt`, `cargo clippy`)
- Module pattern for new algorithms
- Testing conventions
- WASM build process
- PR process

---

## Quick Reference

| I want to... | Function | Read more |
|-------------|----------|-----------|
| Predict a number | `linearRegression`, `polynomialRegression` | [Regression](../tutorials/regression.md) |
| Classify data | `logisticRegression`, `knnClassifier` | [Classification](../tutorials/classification.md) |
| Find groups | `kmeans`, `dbscan` | [Clustering](../tutorials/clustering.md) |
| Scale features | `standardScalerFit`, `minmaxScalerFit` | [API Reference](../api/reference.md) |
| Encode labels | `labelEncoderFit`, `oneHotEncoderFit` | [API Reference](../api/reference.md) |
| Combine models | `randomForestClassify`, `votingClassifier` | [Classification](../tutorials/classification.md) |
| Evaluate | `confusionMatrix`, `rSquared`, `silhouetteScore` | [Classification](../tutorials/classification.md) |
| Find anomalies | `isolationForestFit`, `lofPredict` | [API Reference](../api/reference.md) |
| Select features | `rfe`, `permutationImportance` | [API Reference](../api/reference.md) |
| Reduce dimensions | `pca` | [API Reference](../api/reference.md) |

## All Documentation

| Document | Location |
|----------|----------|
| [What is micro-ml?](./intro.md) | `docs/overview/intro.md` |
| [Algorithm Selection Guide](./algorithm-guide.md) | `docs/overview/algorithm-guide.md` |
| [Architecture](./architecture.md) | `docs/overview/architecture.md` |
| [Zero to Hero (this guide)](./zero-to-hero.md) | `docs/overview/zero-to-hero.md` |
| [API Reference](../api/reference.md) | `docs/api/reference.md` |
| [Algorithm Details](../api/algorithms.md) | `docs/api/algorithms.md` |
| [Contributing Guide](../api/contributing.md) | `docs/api/contributing.md` |
| [Quick Start](../tutorials/quick-start.md) | `docs/tutorials/quick-start.md` |
| [Regression Tutorial](../tutorials/regression.md) | `docs/tutorials/regression.md` |
| [Classification Tutorial](../tutorials/classification.md) | `docs/tutorials/classification.md` |
| [Clustering Tutorial](../tutorials/clustering.md) | `docs/tutorials/clustering.md` |
