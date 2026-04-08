# Predict Continuous Values

Train regression models to predict numeric outcomes.

## Problem

You need to predict a continuous quantity -- house prices, temperature, revenue, sensor readings. The target variable is a number, not a category. Different regression models capture different relationships between features and the target.

## Solution

Start with linear regression as a baseline. If the relationship is non-linear, try polynomial or exponential models. Compare using R-squared and RMSE.

### Step 1: Prepare the data

```typescript
import { dataSplit, r2Score, mse, rmse, mae, standardScaler } from "miniml";

// X: feature matrix (flat Float64Array), y: target values
const nSamples = X.length / nFeatures;
const { XTrain, XTest, yTrain, yTest } = dataSplit(X, y, 0.2);
const trainN = XTrain.length / nFeatures;
const testN = XTest.length / nFeatures;

const { scaled: XTrainS } = standardScaler(XTrain, trainN, nFeatures);
const { scaled: XTestS } = standardScaler(XTest, testN, nFeatures);
```

### Step 2: Train multiple regression models

```typescript
import {
  linearRegression,
  polynomialRegression,
  exponentialRegression,
  powerRegression,
  svr,
} from "miniml";

// Linear Regression -- baseline for linear relationships
const lin = linearRegression(XTrain, yTrain);
const linPred = lin.predict(XTest);

// Polynomial Regression -- captures curves in the data
const poly = polynomialRegression(XTrain, yTrain, 3);
const polyPred = poly.predict(XTest);

// Exponential Regression -- for growth/decay patterns
const exp = exponentialRegression(XTrain, yTrain);
const expPred = exp.predict(XTest);

// Power Regression -- for scaling relationships (y = a * x^b)
const pow = powerRegression(XTrain, yTrain);
const powPred = pow.predict(XTest);

// Support Vector Regression -- robust to outliers
const svrModel = svr(XTrainS, yTrain, trainN, nFeatures, 0.1, 1.0, 100, 0.001);
const svrPred = svrModel.predict(XTestS);
```

### Step 3: Compare models with metrics

```typescript
const models = [
  { name: "Linear", preds: linPred },
  { name: "Polynomial (deg 3)", preds: polyPred },
  { name: "Exponential", preds: expPred },
  { name: "Power", preds: powPred },
  { name: "SVR", preds: svrPred },
];

console.log("Model             | R-squared | RMSE   | MAE");
console.log("------------------|-----------|--------|-------");

for (const { name, preds } of models) {
  const r2 = r2Score(yTest, preds);
  const rmseVal = rmse(yTest, preds);
  const maeVal = mae(yTest, preds);
  console.log(
    `${name.padEnd(18)}| ${(r2 * 100).toFixed(1).padStart(7)}% | ${rmseVal.toFixed(2).padStart(6)} | ${maeVal.toFixed(2)}`
  );
}
```

### Step 4: Interpret the results

| Metric | What It Means | Good Value |
|--------|--------------|------------|
| R-squared | Variance explained by the model | Close to 1.0 |
| RMSE | Average prediction error (same units as target) | As low as possible |
| MAE | Median-like error, less sensitive to outliers | As low as possible |

If R-squared is below 0.5, the linear model is a poor fit. Try polynomial or check if important features are missing.

## Tips

- Linear regression works best when the feature-target relationship is approximately linear.
- Inspect coefficients: `lin.coefficients[i]` tells you how much feature `i` affects the prediction.
- SVR with small `epsilon` fits training data closely; larger `epsilon` ignores small deviations.
- Polynomial degree > 5 usually overfits. Start with degree 2 or 3.

## See Also

- [Regularization Techniques](regularization.md) -- preventing overfitting with L1/L2 penalties
- [Non-linear Relationships](nonlinear.md) -- when linear regression is not enough
- [Scale Your Features](../preprocessing/scaling.md) -- essential for SVR and regularized models
