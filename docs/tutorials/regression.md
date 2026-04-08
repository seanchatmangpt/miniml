# Regression Tutorial

Predicting continuous values with micro-ml. Covers linear, polynomial, and robust regression with evaluation.

## When to Use Regression

- Predicting prices, temperatures, sales, or any continuous quantity
- Forecasting time series values
- Understanding relationships between variables

## Linear Regression

The simplest model: `y = slope * x + intercept`.

```ts
import { linearRegression } from 'micro-ml';

// Sales data: month vs revenue (thousands)
const months = [1, 2, 3, 4, 5, 6];
const revenue = [12, 15, 18, 22, 24, 29];

const model = await linearRegression(months, revenue);

console.log(model.slope);      // ~2.97 (growth per month)
console.log(model.intercept);  // ~8.57 (baseline)
console.log(model.rSquared);   // ~0.99 (excellent fit)

// Forecast next 3 months
const forecast = model.predict([7, 8, 9]);
console.log(forecast);  // [~29.3, ~32.3, ~35.2]
```

## Polynomial Regression

When the relationship is nonlinear (curves):

```ts
import { polynomialRegression } from 'micro-ml';

// Quadratic growth: y = x^2
const x = [0, 1, 2, 3, 4];
const y = [0, 1, 4, 9, 16];

const model = await polynomialRegression(x, y, { degree: 2 });
console.log(model.getCoefficients());  // [0, 0, 1]
console.log(model.rSquared);          // 1.0

const prediction = model.predict([5]);
console.log(prediction);  // [25]
```

## Exponential Regression

For growth patterns: `y = a * e^(bx)`.

```ts
import { exponentialRegression } from 'micro-ml';

// User growth
const months = [0, 1, 2, 3, 4, 5];
const users = [100, 150, 225, 337, 506, 759];

const model = await exponentialRegression(months, users);
console.log(model.a);            // ~100
console.log(model.b);            // ~0.405 (40.5% monthly growth)
console.log(model.doublingTime()); // ~1.71 months

const forecast = model.predict([6, 7, 8]);
console.log(forecast);  // [~1138, ~1707, ~2561]
```

## Multi-Feature Regression (Ridge/Lasso)

When you have multiple input features:

```ts
import { ridgeRegression } from 'micro-ml';

// Housing: [sqft, bedrooms, age] → price
const data = new Float64Array([
  1500, 3, 10,  // house 1
  2000, 4, 5,   // house 2
  1200, 2, 20,  // house 3
  2500, 4, 2,   // house 4
]);
const prices = new Float64Array([300, 450, 200, 550]);

const model = ridgeRegression(data, 3, prices, 1.0);
const coefficients = model.coefficients();
console.log(coefficients);  // [sqft_coef, bed_coef, age_coef]

// Predict price for new house
const prediction = model.predict(new Float64Array([1800, 3, 8]));
```

## Robust Regression

When your data has outliers:

```ts
import { ransacRegression } from 'micro-ml';

// Data with outliers
const data = new Float64Array([
  1, 2, 3, 4, 5, 100,  // x values (100 is outlier)
  2, 4, 6, 8, 10, 5,   // y values (5 is outlier)
]);
const labels = new Float64Array([2, 4, 6, 8, 10, 200]);

const model = ransacRegression(data, 1, labels, 100, 10.0);
const inliers = model.inlierMask();
// inliers will be [1,1,1,1,1,0] — last point excluded
```

## Evaluating Regression Models

```ts
import { linearRegression } from 'micro-ml';

// ... train model, get predictions ...

const actual = [2, 4, 6, 8, 10];
const predicted = [2.1, 3.9, 6.2, 7.8, 10.1];

// Built-in R² from model
console.log(model.rSquared);  // 0.999

// Manual metrics
import { rSquared, rmse, mae } from 'micro-ml';

console.log(rSquared(new Float64Array(actual), new Float64Array(predicted)));
console.log(rmse(new Float64Array(actual), new Float64Array(predicted)));
console.log(mae(new Float64Array(actual), new Float64Array(predicted)));
```

### Interpreting Metrics

| Metric | Good | Bad | Meaning |
|--------|------|-----|---------|
| R² | Close to 1 | Close to 0 | Fraction of variance explained |
| RMSE | Close to 0 | Large | Root mean squared error (same units as y) |
| MAE | Close to 0 | Large | Mean absolute error (robust to outliers) |

## Choosing a Regression Model

| Situation | Algorithm |
|-----------|-----------|
| Simple trend | `linearRegression` |
| Curved relationship | `polynomialRegression` (degree 2-3) |
| Growth/decay | `exponentialRegression` |
| Saturation | `logarithmicRegression` |
| Power law | `powerRegression` |
| Multiple features | `ridgeRegression`, `lassoRegression`, `elasticNet` |
| Outliers present | `ransacRegression`, `theilSenRegression` |
| Feature selection needed | `lassoRegression` (L1 sparsity) |
