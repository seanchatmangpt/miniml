# Your First ML Model

Train a classifier in 5 minutes with miniml. No Python, no servers, no setup beyond `npm install`.

## What You'll Learn

- How to prepare a dataset as `Float64Array` (the format miniml expects)
- Splitting data into training and test sets
- Training a KNN classifier and a Decision Tree
- Evaluating models with accuracy and confusion matrix

## Prerequisites

```bash
npm install @seanchatmangpt/wminml
```

That's it. miniml runs entirely in your browser or Node.js -- no external dependencies.

## Step 1: Prepare Your Data

miniml works with flat `Float64Array` in row-major order. Each row is one sample, each column is one feature.

Let's create a synthetic dataset: two overlapping clusters of points in 2D space.

```typescript
import { init, knnTrain, decisionTreeTrain, accuracy, confusionMatrix, dataSplit } from '@seanchatmangpt/wminml';

await init();

// 20 samples, 2 features each (x, y coordinates)
const X = new Float64Array([
  // Cluster 0: centered around (2, 2)
  1.1, 1.2,  1.5, 1.8,  2.0, 2.1,  1.8, 1.5,  2.3, 2.0,
  1.4, 1.6,  2.2, 1.9,  1.7, 2.3,  2.5, 2.2,  1.3, 1.4,
  // Cluster 1: centered around (5, 5)
  4.8, 5.1,  5.3, 4.9,  5.0, 5.5,  4.7, 5.2,  5.6, 5.0,
  4.9, 4.8,  5.1, 5.4,  5.2, 4.6,  5.5, 5.3,  4.8, 5.0,
]);

// Labels: 0 for first cluster, 1 for second cluster
const y = new Float64Array([
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]);

const nSamples = 40;
const nFeatures = 2;
```

## Step 2: Split Into Train/Test

Never evaluate your model on the same data it trained on. Use `dataSplit` to create a holdout set.

```typescript
// 80% train, 20% test
const split = dataSplit(X, y, nSamples, nFeatures, 0.8);

console.log(`Training samples: ${split.trainX.length / nFeatures}`);
console.log(`Test samples:     ${split.testX.length / nFeatures}`);
```

`dataSplit` shuffles the data first, then returns `trainX`, `trainY`, `testX`, `testY`.

## Step 3: Train a KNN Classifier

KNN (K-Nearest Neighbors) classifies a point by looking at its K closest neighbors and taking a vote. Simple, but surprisingly effective.

```typescript
const trainN = split.trainX.length / nFeatures;
const testN = split.testX.length / nFeatures;

const knnModel = knnTrain(split.trainX, split.trainY, trainN, nFeatures, 3);

// Predict on test data
const knnPreds = new Float64Array(testN);
for (let i = 0; i < testN; i++) {
  const start = i * nFeatures;
  const sample = split.testX.slice(start, start + nFeatures);
  knnPreds[i] = knnModel.predict(sample);
}

console.log('KNN predictions:', knnPreds);
console.log('Actual labels:   ', split.testY);
```

The `.predict()` method takes a single sample (a `Float64Array` of length `nFeatures`) and returns a label.

## Step 4: Evaluate the Model

How well did it do? Use accuracy and the confusion matrix to find out.

```typescript
const acc = accuracy(split.testY, knnPreds);
console.log(`KNN Accuracy: ${(acc * 100).toFixed(1)}%`);

const cm = confusionMatrix(split.testY, knnPreds);
console.log('Confusion Matrix:');
console.log(`  Predicted  0   1`);
console.log(`  Actual 0  ${cm[0][0]}   ${cm[0][1]}`);
console.log(`  Actual 1  ${cm[1][0]}   ${cm[1][1]}`);
```

The confusion matrix shows true positives, true negatives, false positives, and false negatives at a glance. A perfect model has zeros everywhere except the diagonal.

## Step 5: Try Another Algorithm

One model is never enough. Let's train a Decision Tree and compare.

```typescript
const dtModel = decisionTreeTrain(split.trainX, split.trainY, trainN, nFeatures, 5);

const dtPreds = new Float64Array(testN);
for (let i = 0; i < testN; i++) {
  const start = i * nFeatures;
  const sample = split.testX.slice(start, start + nFeatures);
  dtPreds[i] = dtModel.predict(sample);
}

const dtAcc = accuracy(split.testY, dtPreds);
console.log(`Decision Tree Accuracy: ${(dtAcc * 100).toFixed(1)}%`);
console.log(`KNN Accuracy:          ${(acc * 100).toFixed(1)}%`);
```

Which one wins depends on your data. On this clean dataset with well-separated clusters, both should score close to 100%. Real-world data is messier -- that's where algorithm choice matters.

## Summary

You just trained and evaluated two classifiers in under 30 lines of code:

1. **Data**: Flat `Float64Array` in row-major order
2. **Split**: `dataSplit()` for train/test separation
3. **Train**: `knnTrain()` or `decisionTreeTrain()` returns a model object
4. **Predict**: `model.predict(sample)` for single-sample inference
5. **Evaluate**: `accuracy()` and `confusionMatrix()` to measure performance

## Next Steps

- **Tutorial 02**: Let miniml choose the best algorithm for you with [AutoML Quick Start](./02-automl-quickstart.md)
- **How-to**: Explore all classification algorithms in [how_to/classification/](../how_to/classification/)
- **Reference**: Full API details in [packages/miniml/README.md](../../packages/miniml/README.md)
