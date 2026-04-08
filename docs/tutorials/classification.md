# Classification Tutorial

Building classifiers with micro-ml. Covers binary/multi-class classification, ensemble methods, and evaluation.

## When to Use Classification

- Spam vs not spam
- Disease diagnosis
- Customer churn prediction
- Image/text categorization
- Any problem where the output is a discrete category

## Binary Classification with Logistic Regression

```ts
import { logisticRegression } from 'micro-ml';

// Email spam detection: [wordCount, hasLink, uppercaseRatio]
const features = [
  [50, 0, 0.1],   // not spam
  [200, 1, 0.8],  // spam
  [30, 0, 0.05],  // not spam
  [150, 1, 0.6],  // spam
  [80, 0, 0.2],   // not spam
  [300, 1, 0.9],  // spam
];
const labels = [0, 1, 0, 1, 0, 1];

const model = await logisticRegression(features, labels, {
  learningRate: 0.01,
  maxIterations: 1000,
  lambda: 0.01,  // L2 regularization
});

// Predict
const predictions = model.predict([[100, 1, 0.7]]);
console.log(predictions);  // [1] → spam

// Probabilities
const probas = model.predictProba([[100, 1, 0.7]]);
console.log(probas);  // [0.15, 0.85] → 85% likely spam

console.log(model.getWeights());  // Feature importance
```

## kNN Classifier

No training phase - stores all data and predicts by majority vote of nearest neighbors.

```ts
import { knnClassifier } from 'micro-ml';

const model = await knnClassifier(features, labels, { k: 3 });

const predictions = model.predict([[100, 1, 0.7]]);
const probas = model.predictProba([[100, 1, 0.7]]);
```

## Decision Tree

Interpretable classification with automatic feature selection.

```ts
import { decisionTree } from 'micro-ml';

const model = await decisionTree(features, labels, {
  maxDepth: 5,
  minSamplesSplit: 2,
  mode: 'classify',
});

// Inspect the tree structure
console.log(model.getTree());  // Flat array: [feature, threshold, left, right, ...]
console.log(model.depth);      // Tree depth
console.log(model.nNodes());   // Number of nodes
```

## Ensemble Methods

### Random Forest

Multiple decision trees with bootstrap sampling and feature randomization.

```ts
import { randomForestClassify } from 'micro-ml';

const model = await randomForestClassify(
  new Float64Array(features.flat()),
  3,  // n_features
  new Float64Array(labels),
  100, // n_trees
  10,  // max_depth
);

const predictions = model.predict(new Float64Array([100, 1, 0.7].flat()));
const probas = model.predictProba(new Float64Array([100, 1, 0.7].flat()));
```

### Gradient Boosting

Sequential trees that correct previous errors.

```ts
import { gradientBoostingClassify } from 'micro-ml';

const model = await gradientBoostingClassify(
  new Float64Array(features.flat()),
  3,
  new Float64Array(labels),
  50,   // n_trees
  3,    // max_depth
  0.1,  // learning_rate
);
```

### Voting Classifier

Combine predictions from multiple pre-trained models.

```ts
import { votingClassifier } from 'micro-ml';

// Get predictions from each model separately
const rfPreds = rfModel.predictProba(testData);
const gbPreds = gbModel.predictProba(testData);
const lrPreds = lrModel.predictProba(testData);

// Stack predictions: [n_models, n_samples * n_classes]
const allPreds = new Float64Array([...rfPreds, ...gbPreds, ...lrPreds]);

// Hard voting: majority vote
const vc = votingClassifier(
  allPreds,  // predictions
  3,         // n_models
  new Float64Array([]),  // weights (empty = uniform)
  'hard',    // voting type
  2,         // n_classes
);
const finalPredictions = vc.aggregate();
```

## Evaluating Classifiers

```ts
import { confusionMatrix, precision, recall, f1Score, mcc, rocAuc } from 'micro-ml';

const actual = new Float64Array([0, 1, 0, 1, 1, 0, 1, 0]);
const predicted = new Float64Array([0, 1, 0, 0, 1, 0, 1, 1]);
const scores = new Float64Array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.7, 0.6]);

// Confusion matrix (2x2)
const cm = confusionMatrix(actual, predicted, 2);
// cm = [TN, FP, FN, TP] = [3, 1, 1, 3]

// Per-class metrics
console.log(precision(cm, 1));   // TP/(TP+FP) = 3/4 = 0.75
console.log(recall(cm, 1));      // TP/(TP+FN) = 3/4 = 0.75
console.log(f1Score(cm, 1));     // Harmonic mean = 0.75

// Overall metrics
console.log(mcc(cm, 2));         // Matthews Correlation Coefficient
console.log(rocAuc(actual, scores));  // Area Under ROC Curve
```

### Interpreting Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| Precision | 0-1 | Of predicted positive, how many are correct |
| Recall | 0-1 | Of actual positive, how many were found |
| F1 Score | 0-1 | Balance of precision and recall |
| MCC | -1 to 1 | Overall quality (handles class imbalance) |
| AUC | 0-1 | Probability of ranking positive above negative |

## Cross-Validation

```ts
import { kFoldSplit } from 'micro-ml';

const nSamples = 100;
const folds = kFoldSplit(nSamples, 5);  // 5-fold CV
// folds = [5, start0, end0, start1, end1, ...]

// Use fold indices to split your data and evaluate
```

## Choosing a Classifier

| Situation | Algorithm |
|-----------|-----------|
| Small dataset, fast | kNN, Perceptron |
| Need probabilities | Logistic Regression, Naive Bayes |
| High accuracy | Random Forest, Gradient Boosting |
| Text/binary features | Bernoulli NB, Multinomial NB |
| Streaming data | SGD, Passive Aggressive |
| Multiple models available | Voting Classifier |
| Interpretable | Decision Tree |
| Feature selection | RFE, Permutation Importance |
