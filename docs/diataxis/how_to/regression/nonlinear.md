# Non-linear Relationships

Model relationships that are not straight lines.

## Problem

Your data shows curves, saturation effects, or exponential growth. Linear regression produces a poor fit with low R-squared because it can only draw straight lines. You need models that capture curvature.

## Solution

Use polynomial regression for general curves, exponential regression for growth/decay, and power regression for scaling laws.

### Step 1: Detect non-linearity

Fit a linear model first. If R-squared is low and residuals show a pattern, the relationship is non-linear.

```typescript
import { linearRegression, r2Score } from "@seanchatmangpt/wminml";

const lin = linearRegression(X, y);
const r2 = r2Score(y, lin.predict(X));

if (r2 < 0.7) {
  console.log(
    `Linear R-squared is ${(r2 * 100).toFixed(1)}% -- consider non-linear models`
  );
}
```

### Step 2: Try polynomial regression

Polynomial regression fits a curve of any degree. Higher degree captures more complex shapes.

```typescript
import { polynomialRegression, r2Score, dataSplit } from "@seanchatmangpt/wminml";

const { XTrain, XTest, yTrain, yTest } = dataSplit(X, y, 0.2);

// Compare polynomial degrees
for (const degree of [2, 3, 4, 5]) {
  const model = polynomialRegression(XTrain, yTrain, degree);
  const trainR2 = r2Score(yTrain, model.predict(XTrain));
  const testR2 = r2Score(yTest, model.predict(XTest));

  console.log(
    `Degree ${degree}: train R2=${trainR2.toFixed(4)}, test R2=${testR2.toFixed(4)}`
  );
}

// Pick the degree where test R2 is highest (not train R2)
```

### Step 3: Try exponential regression

For data that grows or decays proportionally -- population growth, radioactive decay, compound interest.

```typescript
import { exponentialRegression, r2Score } from "@seanchatmangpt/wminml";

const exp = exponentialRegression(X, y);
const expR2 = r2Score(y, exp.predict(X));

console.log(`Exponential R2: ${(expR2 * 100).toFixed(1)}%`);
console.log(`Model: y = ${exp.a.toFixed(4)} * e^(${exp.b.toFixed(4)} * x)`);

// If R2 is high and the data shows accelerating growth, exponential is a good fit
```

### Step 4: Try power regression

For scaling relationships -- "if x doubles, y quadruples." Common in physics, economics, biology.

```typescript
import { powerRegression, r2Score } from "@seanchatmangpt/wminml";

const pow = powerRegression(X, y);
const powR2 = r2Score(y, pow.predict(X));

console.log(`Power R2: ${(powR2 * 100).toFixed(1)}%`);
console.log(`Model: y = ${pow.a.toFixed(4)} * x^${pow.b.toFixed(4)}`);
```

### Step 5: Pick the right model

```typescript
import { polynomialRegression, exponentialRegression, powerRegression, r2Score } from "@seanchatmangpt/wminml";

const models = [
  { name: "Polynomial (3)", preds: polynomialRegression(X, y, 3).predict(X) },
  { name: "Exponential", preds: exponentialRegression(X, y).predict(X) },
  { name: "Power", preds: powerRegression(X, y).predict(X) },
];

let bestName = "";
let bestR2 = -Infinity;

for (const { name, preds } of models) {
  const r2 = r2Score(y, preds);
  console.log(`${name}: R2=${r2.toFixed(4)}`);
  if (r2 > bestR2) {
    bestR2 = r2;
    bestName = name;
  }
}

console.log(`\nBest model: ${bestName} (R2=${bestR2.toFixed(4)})`);
```

### Choosing the right model

| Data Pattern | Model | Example |
|-------------|-------|---------|
| U-shape or parabola | Polynomial (deg 2) | Cost vs production volume |
| S-curve or multiple bends | Polynomial (deg 3-4) | Dose-response curves |
| Accelerating growth | Exponential | Population, compound interest |
| Decaying signal | Exponential | Radioactive decay, cooling |
| Scaling law | Power | Metabolic rate vs body mass |
| Diminishing returns | Power (exponent < 1) | Experience vs productivity |

## Tips

- Polynomial degree > 5 usually overfits. Watch for test R2 dropping as degree increases.
- Exponential and power regression assume strictly positive values. Transform or filter negatives first.
- Use `dataSplit` to evaluate on held-out data. A high training R2 with low test R2 means overfitting.
- Combine with regularization (see [Regularization Techniques](regularization.md)) for high-degree polynomials.

## See Also

- [Predict Continuous Values](predict-values.md) -- regression basics
- [Regularization Techniques](regularization.md) -- controlling polynomial overfitting
- [Choose K for K-Means](../clustering/choose-k.md) -- another approach to finding structure in data
