# Regularization Techniques

Prevent overfitting with Ridge, Lasso, and Elastic Net regression.

## Problem

Your regression model performs well on training data but poorly on test data. It has learned noise instead of signal -- coefficients are too large and the model is too complex for the amount of data you have. You need to constrain the model without losing predictive power.

## Solution

Add a penalty term to the loss function that shrinks coefficients toward zero. Ridge shrinks all coefficients equally; Lasso drives some to exactly zero (automatic feature selection); Elastic Net combines both.

### Step 1: Compare regularized models

```typescript
import {
  ridgeRegression,
  lassoRegression,
  elasticNet,
  linearRegression,
  r2Score,
  rmse,
  standardScaler,
  dataSplit,
} from "@seanchatmangpt/wminml";

const nSamples = X.length / nFeatures;
const { XTrain, XTest, yTrain, yTest } = dataSplit(X, y, 0.2);
const trainN = XTrain.length / nFeatures;
const testN = XTest.length / nFeatures;

// Scale first -- regularization is sensitive to feature scales
const { scaled: XTrainS } = standardScaler(XTrain, trainN, nFeatures);
const { scaled: XTestS } = standardScaler(XTest, testN, nFeatures);

// Unregularized baseline
const lin = linearRegression(XTrainS, yTrain);
console.log(`Linear R2:  ${r2Score(yTest, lin.predict(XTestS)).toFixed(4)}`);

// Ridge (L2) -- shrinks all coefficients, keeps all features
const ridge = ridgeRegression(XTrainS, yTrain, trainN, nFeatures, 1.0);
console.log(`Ridge R2:   ${r2Score(yTest, ridge.predict(XTestS)).toFixed(4)}`);

// Lasso (L1) -- drives some coefficients to zero (feature selection)
const lasso = lassoRegression(XTrainS, yTrain, trainN, nFeatures, 0.5, 100);
console.log(`Lasso R2:   ${r2Score(yTest, lasso.predict(XTestS)).toFixed(4)}`);

// Elastic Net -- blend of L1 and L2
const elastic = elasticNet(XTrainS, yTrain, trainN, nFeatures, 0.5, 0.5, 100);
console.log(`Elastic R2: ${r2Score(yTest, elastic.predict(XTestS)).toFixed(4)}`);
```

### Step 2: Choose the right alpha

The `alpha` parameter controls regularization strength. Higher alpha means stronger regularization (more coefficient shrinkage).

```typescript
function findBestAlpha(
  XTrain: Float64Array,
  yTrain: Float64Array,
  XTest: Float64Array,
  yTest: Float64Array,
  nFeatures: number
): void {
  const alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0];
  const trainN = XTrain.length / nFeatures;
  const testN = XTest.length / nFeatures;

  console.log("Alpha   | Ridge R2  | Lasso R2  | Elastic R2");
  console.log("--------|-----------|-----------|-----------");

  for (const alpha of alphas) {
    const ridge = ridgeRegression(XTrain, yTrain, trainN, nFeatures, alpha);
    const lasso = lassoRegression(XTrain, yTrain, trainN, nFeatures, alpha, 100);
    const elastic = elasticNet(XTrain, yTrain, trainN, nFeatures, alpha, 0.5, 100);

    const ridgeR2 = r2Score(yTest, ridge.predict(XTest));
    const lassoR2 = r2Score(yTest, lasso.predict(XTest));
    const elasticR2 = r2Score(yTest, elastic.predict(XTest));

    console.log(
      `${alpha.toFixed(3).padStart(6)} | ${(ridgeR2 * 100).toFixed(1).padStart(7)}% | ${(lassoR2 * 100).toFixed(1).padStart(7)}% | ${(elasticR2 * 100).toFixed(1).padStart(7)}%`
    );
  }
}
```

### Step 3: Use Lasso for feature selection

Lasso drives irrelevant feature coefficients to exactly zero, acting as automatic feature selection.

```typescript
const lasso = lassoRegression(XTrainS, yTrain, trainN, nFeatures, 1.0, 200);

// Check which features survived
const coeffs = Array.from(lasso.coefficients);
const featureNames = ["age", "income", "score", "rating", "tenure"];

for (let i = 0; i < nFeatures; i++) {
  const status = Math.abs(coeffs[i]) < 1e-6 ? "REMOVED" : `weight=${coeffs[i].toFixed(3)}`;
  console.log(`  ${featureNames[i]}: ${status}`);
}
```

### When to use each

| Technique | Best When | Effect |
|-----------|-----------|--------|
| Ridge (`alpha > 0`) | Many correlated features, all potentially useful | Shrinks all coefficients, none to zero |
| Lasso (`alpha > 0`) | Many features, only some matter | Drives unimportant coefficients to zero |
| Elastic Net (`alpha > 0`, `l1Ratio`) | Correlated groups of features you want to select together | Balances shrinkage and sparsity |

## Tips

- Always scale features before regularization. Unscaled features get penalized unfairly.
- Start with `alpha: 1.0` and search from there.
- For Elastic Net, `l1Ratio: 0.5` is a reasonable default. Set closer to 1.0 for more Lasso-like behavior.
- If all features are important, use Ridge. If you want automatic feature selection, use Lasso.

## See Also

- [Predict Continuous Values](predict-values.md) -- regression model basics
- [Non-linear Relationships](nonlinear.md) -- when the relationship is not linear
- [Scale Your Features](../preprocessing/scaling.md) -- required before regularization
