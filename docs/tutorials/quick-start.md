# Quick Start

Get micro-ml running in 5 minutes. This guide covers installation, first predictions, and browser usage.

## Installation

```bash
npm install micro-ml
```

Or via CDN:
```html
<script type="module">
  import { linearRegression } from 'https://esm.sh/micro-ml';
</script>
```

## Your First Prediction

```ts
import { linearRegression } from 'micro-ml';

// Fit a model: y = 2x
const model = await linearRegression([1, 2, 3, 4], [2, 4, 6, 8]);

console.log(model.slope);      // 2
console.log(model.intercept);  // 0
console.log(model.rSquared);   // 1

// Predict new values
const predictions = model.predict([5, 6, 7]);
console.log(predictions);  // [10, 12, 14]
```

## Classification Example

```ts
import { logisticRegression } from 'micro-ml';

// Binary classification: spam detection
const features = [
  [1.0, 0.0],  // not spam
  [0.0, 1.0],  // not spam
  [1.0, 1.0],  // spam
  [0.0, 0.0],  // not spam
];
const labels = [0, 0, 1, 0];

const model = await logisticRegression(features, labels, {
  learningRate: 0.1,
  maxIterations: 1000,
});

const predictions = model.predict([[0.9, 0.8]]);
console.log(predictions);  // [1] → spam

const probabilities = model.predictProba([[0.9, 0.8]]);
console.log(probabilities);  // [0.1, 0.9] → 10% not spam, 90% spam
```

## Clustering Example

```ts
import { kmeans } from 'micro-ml';

const data = [
  [0, 0], [0.1, 0.1],    // cluster 1
  [10, 10], [10.1, 10.1], // cluster 2
  [5, 5], [4.9, 5.1],     // cluster 3
];

const model = await kmeans(data, { k: 3 });

console.log(model.getAssignments());  // [0, 0, 1, 1, 2, 2]
console.log(model.getCentroids());    // [[0.05, 0.05], [10.05, 10.05], [4.95, 5.05]]

// Predict cluster for new data
const newClusters = model.predict([[0, 1], [10, 11], [5, 6]]);
console.log(newClusters);  // [0, 1, 2]
```

## Data Preprocessing

```ts
import { standardScalerFit, pca, labelEncoderFit } from 'micro-ml';

// Scale features
const scaler = standardScalerFit(
  new Float64Array([1, 10, 100, 2, 20, 200]),
  2  // 2 features, 3 samples
);
const scaled = scaler.transform(
  new Float64Array([3, 30, 300])
);

// Encode labels
const encoder = labelEncoderFit(new Float64Array(['cat', 'dog', 'cat'].map(s => s.charCodeAt(0))));
// Or with numeric labels:
const enc = labelEncoderFit(new Float64Array([0, 1, 0, 2]));
```

## Evaluating Models

```ts
import { logisticRegression } from 'micro-ml';
import { confusionMatrix, precision, recall, f1Score } from 'micro-ml';

// ... train model, get predictions ...

const actual = new Float64Array([0, 1, 0, 1, 1]);
const predicted = new Float64Array([0, 1, 0, 0, 1]);

const cm = confusionMatrix(actual, predicted, 2);
console.log(precision(cm, 0));  // Precision for class 0
console.log(recall(cm, 1));     // Recall for class 1
console.log(f1Score(cm, 1));    // F1 for class 1
```

## What's Next?

- [Regression tutorial](./regression.md) - Predicting continuous values
- [Classification tutorial](./classification.md) - Building classifiers
- [Clustering tutorial](./clustering.md) - Unsupervised learning
- [Algorithm guide](../overview/algorithm-guide.md) - Choosing the right algorithm
- [API reference](../api/reference.md) - Complete function listing
