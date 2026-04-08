# Troubleshooting

Common issues and solutions when using miniml.

## WASM Initialization

### "WASM initialization failed"

**Cause:** `init()` was not called before using ML functions, or the WASM module failed to load.

**Solution:**
```typescript
import { init, knnTrain } from 'miniml';

// Always call init() first and await it
await init();

// Now safe to use ML functions
const model = knnTrain(X, y, nSamples, nFeatures, k);
```

**Check:** Ensure `init()` is called once at the top of your module before any ML operations. In browsers, the WASM file must be served with correct MIME type (`application/wasm`).

---

## Matrix Dimension Mismatches

### "data length must be divisible by n_features"

**Cause:** The total number of elements in `X` is not evenly divisible by the number of features.

**Solution:** Verify the math:
```typescript
const X = new Float64Array([...]); // e.g., 11 elements
const nFeatures = 2;

// 11 / 2 = 5.5 -- NOT divisible
// Fix: ensure X.length % nFeatures === 0

console.log(X.length / nFeatures);  // must be an integer
```

**Common mistake:** Off-by-one error when constructing the array, or mixing up nFeatures and nSamples.

### "targets length must match number of samples"

**Cause:** The `y` array has a different length than the number of rows in `X`.

**Solution:**
```typescript
const nSamples = X.length / nFeatures;
if (y.length !== nSamples) {
  throw new Error(`Expected ${nSamples} targets, got ${y.length}`);
}
```

---

## Convergence Issues

### "Convergence not reached" (iterative algorithms)

**Cause:** The algorithm hit `maxIter` without converging. Common with SVM, logistic regression, K-Means, and Cox PH.

**Solutions:**

1. **Increase iterations:**
```typescript
// Default maxIter may be too low for your data
const model = logisticTrain(X, y, nSamples, nFeatures, nClasses, {
  maxIter: 1000,  // increase from default
  learningRate: 0.01,
});
```

2. **Adjust learning rate:**
```typescript
// Learning rate too high causes oscillation
// Learning rate too low causes slow convergence
const model = logisticTrain(X, y, nSamples, nFeatures, nClasses, {
  learningRate: 0.001,  // try smaller
  maxIter: 500,
});
```

3. **Scale your features:**
```typescript
import { standardScalerFit, standardScalerTransform } from 'miniml';

const scaler = standardScalerFit(X, nSamples, nFeatures);
const Xscaled = standardScalerTransform(scaler, X, nSamples, nFeatures);
// Now train on Xscaled
```

---

## Insufficient Data

### "Need at least 2 observations for a t-test"

**Cause:** Statistical tests require minimum sample sizes.

| Function | Minimum Requirement |
|----------|-------------------|
| `tTestOneSample` | n >= 2 |
| `tTestTwoSample` | n1 >= 2 AND n2 >= 2 |
| `tTestPaired` | n >= 2 paired observations |
| `oneWayAnova` | >= 2 groups, total n > k |
| `describe` | n >= 1 |
| `ksTest` | n >= 2 |
| `bayesianLinearRegression` | n >= 2 |
| `kaplanMeier` | n >= 1 (with at least 1 event) |

---

## Memory Issues

### "Memory allocation failed"

**Cause:** Large matrices exceed available WASM memory.

**Solutions:**

1. **Reduce data size.** Subsample or use PCA for dimensionality reduction.
2. **Use efficient algorithms.** Linear regression is O(n*d), Gaussian Process is O(n^3).
3. **Increase WASM memory** (if your bundler supports it):
```typescript
// Some bundlers allow configuring initial/maximum WASM memory
import init, { memory } from 'miniml';
```

---

## Numerical Issues

### "Matrix is not positive definite"

**Cause:** Occurs in Bayesian regression, Gaussian Process, or Cholesky-based solvers when the data matrix is singular or near-singular.

**Solutions:**

1. **Add regularization** (increase `priorPrecision` or `noise`):
```typescript
// Bayesian regression: increase prior precision
const model = bayesianLinearRegression(X, nFeatures, y, 1.0, 0.001, 1.0);

// GP: increase noise
const gp = gpFit(X, nFeatures, y, 'rbf', [1.0], 0.1);
```

2. **Remove collinear features.** Check for features that are linear combinations of others.

3. **Standardize features first** so they have similar scales.

### NaN or Infinity in results

**Cause:** Division by zero, log of zero, or numerical overflow.

**Common fixes:**
- Ensure no features have zero variance (all identical values).
- Check for extreme outliers that cause overflow.
- Use the Robust Scaler instead of Standard Scaler for data with outliers.

---

## Performance Tips

### Slow training

- **KNN prediction is slow** for large datasets. Use Decision Tree or Logistic Regression for faster prediction.
- **GP fitting is O(n^3).** Keep training sets under 500-1000 points.
- **SVM training is O(n^2).** Subsample for large datasets.
- **Enable SIMD** by building with `--features simd` flag for faster matrix operations.

### Poor accuracy

- **Scale your features** before training (Standard Scaler or Min-Max Scaler).
- **Try different algorithms** -- no single algorithm is best for all problems.
- **Use AutoML** to automatically find the best algorithm and hyperparameters.
- **Check for data leakage** -- ensure train/test split happens before preprocessing.
