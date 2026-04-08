# Train a Classifier

Choose and train the right classification model for your dataset.

## Problem

You have labeled data with known categories and need a model that predicts the class of new, unseen samples. Different classifiers excel under different conditions -- some handle non-linear boundaries, some are fast to train, and some resist overfitting.

## Solution

Start with a simple model, then try more complex ones. Compare their accuracy on a held-out test set.

### Step 1: Prepare the data

```typescript
import {
  dataSplit,
  standardScaler,
  accuracy,
  confusionMatrix,
} from "@seanchatmangpt/wminml";

// Flat arrays: 4 features x 100 samples
const X = new Float64Array([
  /* feature values */
]);
const y = new Float64Array([
  /* class labels: 0, 1, 2, ... */
]);
const nSamples = X.length / 4;
const nFeatures = 4;

const { XTrain, XTest, yTrain, yTest } = dataSplit(X, y, 0.2);

// Scale features for distance-based models
const { scaled: XTrainS, mean, std } = standardScaler(
  XTrain,
  XTrain.length / nFeatures,
  nFeatures
);
const { scaled: XTestS } = standardScaler(
  XTest,
  XTest.length / nFeatures,
  nFeatures
);
// Note: apply the training mean/std to test data for unbiased evaluation
```

### Step 2: Train multiple classifiers

```typescript
import {
  knnTrain,
  decisionTreeTrain,
  randomForestClassify,
  naiveBayesTrain,
  logisticRegression,
} from "@seanchatmangpt/wminml";

// KNN -- good baseline, non-parametric
const knnModel = knnTrain(XTrainS, yTrain, nSamples * 0.8, nFeatures, 5);
const knnPred = knnModel.predict(XTestS);

// Decision Tree -- interpretable, handles mixed features
const dtModel = decisionTreeTrain(XTrainS, yTrain, nSamples * 0.8, nFeatures, 5);
const dtPred = dtModel.predict(XTestS);

// Random Forest -- usually the strongest off-the-shelf choice
const rfModel = randomForestClassify(XTrainS, yTrain, 10, 5);
const rfPred = rfModel.predict(XTestS);

// Naive Bayes -- fast, works well with high-dimensional data
const nbModel = naiveBayesTrain(XTrainS, yTrain, nSamples * 0.8, nFeatures);
const nbPred = nbModel.predict(XTestS);

// Logistic Regression -- fast linear baseline
const lrModel = logisticRegression(XTrainS, yTrain, nSamples * 0.8, nFeatures, 100, 0.01);
const lrPred = lrModel.predict(XTestS);
```

### Step 3: Compare results

```typescript
const models = [
  { name: "KNN", preds: knnPred },
  { name: "Decision Tree", preds: dtPred },
  { name: "Random Forest", preds: rfPred },
  { name: "Naive Bayes", preds: nbPred },
  { name: "Logistic Regression", preds: lrPred },
];

for (const { name, preds } of models) {
  const acc = accuracy(yTest, preds);
  const cm = confusionMatrix(yTest, preds);
  console.log(`${name}: accuracy=${acc.toFixed(3)}`);
  console.log(`  Confusion matrix: ${JSON.stringify(Array.from(cm))}`);
}
```

### Step 4: Pick the right model

| Condition | Best Choice |
|-----------|-------------|
| Small dataset (< 100 samples) | KNN or Naive Bayes |
| Need interpretability | Decision Tree |
| Best accuracy, no tuning | Random Forest |
| Very high-dimensional features | Naive Bayes or Logistic Regression |
| Speed-critical prediction | Logistic Regression |

## Tips

- Always scale features before using KNN or Logistic Regression.
- Random Forest rarely needs tuning; start with `nTrees: 10` and `maxDepth: 5`.
- Decision trees are prone to overfitting -- limit `maxDepth` or use Random Forest instead.
- Use `dataSplit` with `testRatio: 0.2` to hold out data for honest evaluation.

## See Also

- [Handle Imbalanced Data](imbalanced-data.md) -- when classes are not evenly distributed
- [Multi-class Classification](multi-class.md) -- classifying into more than two categories
- [Scale Your Features](../preprocessing/scaling.md) -- preparing numeric features for training
