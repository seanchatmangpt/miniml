# Multi-class Classification

Classify samples into three or more categories.

## Problem

Binary classification (two classes) is the simplest case. Real-world problems often have many categories -- species identification, document categorization, product type classification. You need a model that handles multiple classes natively.

## Solution

Most classifiers in miniml support multi-class classification directly. No one-vs-rest wrapping is needed.

### Step 1: Verify your labels

```typescript
import { labelEncoder } from "miniml";

// String labels need encoding to numeric values
const labels = [
  "cat", "dog", "bird", "cat", "dog", "bird", "cat", "bird", "dog", "cat",
];
const { encoded: y, classes } = labelEncoder(labels);

console.log(`Classes: ${JSON.stringify(Array.from(classes))}`);
// Output: Classes: ["bird","cat","dog"] (sorted)
console.log(`Encoded: ${JSON.stringify(Array.from(y))}`);
// Output: Encoded: [1,2,0,1,2,0,1,0,2,1]
```

### Step 2: Train multi-class models

```typescript
import {
  knnTrain,
  decisionTreeTrain,
  randomForestClassify,
  naiveBayesTrain,
  dataSplit,
  accuracy,
  confusionMatrix,
  standardScaler,
} from "miniml";

const nSamples = X.length / nFeatures;
const { XTrain, XTest, yTrain, yTest } = dataSplit(X, y, 0.2);
const trainN = XTrain.length / nFeatures;
const testN = XTest.length / nFeatures;

const { scaled: XTrainS } = standardScaler(XTrain, trainN, nFeatures);
const { scaled: XTestS } = standardScaler(XTest, testN, nFeatures);

// KNN -- works naturally with multiple classes
const knn = knnTrain(XTrainS, yTrain, trainN, nFeatures, 5);
const knnPred = knn.predict(XTestS);

// Random Forest -- builds separate trees, votes across all classes
const rf = randomForestClassify(XTrainS, yTrain, 15, 6);
const rfPred = rf.predict(XTestS);

// Naive Bayes -- computes per-class priors automatically
const nb = naiveBayesTrain(XTrainS, yTrain, trainN, nFeatures);
const nbPred = nb.predict(XTestS);

// Decision Tree -- splits can create multiple branches
const dt = decisionTreeTrain(XTrainS, yTrain, trainN, nFeatures, 6);
const dtPred = dt.predict(XTestS);
```

### Step 3: Evaluate per-class performance

```typescript
const results = [
  { name: "KNN", preds: knnPred },
  { name: "Random Forest", preds: rfPred },
  { name: "Naive Bayes", preds: nbPred },
  { name: "Decision Tree", preds: dtPred },
];

for (const { name, preds } of results) {
  const acc = accuracy(yTest, preds);
  const cm = confusionMatrix(yTest, preds);
  console.log(`\n${name}: ${(acc * 100).toFixed(1)}%`);

  // Per-class recall from confusion matrix diagonal
  for (let i = 0; i < classes.length; i++) {
    const row = Array.from(cm[i]);
    const total = row.reduce((a, b) => a + b, 0);
    const recall = total > 0 ? row[i] / total : 0;
    console.log(`  Class ${classes[i]}: recall=${(recall * 100).toFixed(1)}%`);
  }
}
```

### Step 4: Handle common multi-class issues

**Too many classes (high cardinality):** Naive Bayes handles this best because it estimates class-conditional probabilities independently. Random Forest also scales well. KNN degrades as classes increase because the nearest-neighbor vote gets diluted.

```typescript
// For high-cardinality problems, increase KNN neighbors
const knnHighCard = knnTrain(XTrainS, yTrain, trainN, nFeatures, 15);

// For Random Forest, increase tree count
const rfHighCard = randomForestClassify(XTrainS, yTrain, 25, 8);
```

**Uneven class counts:** Some classes may have far fewer samples. Check per-class recall and resample if needed (see [Handle Imbalanced Data](imbalanced-data.md)).

## Tips

- Always encode string labels with `labelEncoder` before training.
- Inspect the confusion matrix for every multi-class problem -- overall accuracy hides per-class failures.
- Random Forest is the strongest general-purpose multi-class classifier in miniml.
- Increase `maxDepth` for trees when you have many classes, so splits can separate all of them.

## See Also

- [Train a Classifier](train-model.md) -- model comparison and basics
- [Handle Imbalanced Data](imbalanced-data.md) -- when class counts are uneven
- [Encode Categorical Data](../preprocessing/encoding.md) -- converting string labels to numbers
