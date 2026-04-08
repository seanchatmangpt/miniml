# AutoML Quick Start

Let miniml find the best algorithm, features, and hyperparameters for your dataset automatically.

## What AutoML Does

Picking the right algorithm and tuning its parameters is hard. miniml's `autoFit` does it for you using two optimization techniques:

- **Genetic Algorithm (GA) feature selection**: Tests combinations of features to find the subset that maximizes accuracy
- **Particle Swarm Optimization (PSO) hyperparameter tuning**: Searches hyperparameter space to find optimal settings for each algorithm

You provide data. AutoML returns the best model it found, plus an explanation of why.

## Basic Usage

```typescript
import { init, autoFit } from 'miniml';

await init();

// 60 samples, 4 features, 3 classes (synthetic iris-like dataset)
const X = new Float64Array(240); // fill with your data
const y = new Float64Array(60);  // class labels: 0, 1, or 2

const result = autoFit(X, y, {
  nSamples: 60,
  nFeatures: 4,
});

console.log(`Best algorithm: ${result.algorithm}`);
console.log(`Accuracy:       ${(result.accuracy * 100).toFixed(1)}%`);
console.log(`Rationale:      ${result.rationale}`);
```

That's it. `autoFit` trains multiple algorithms, selects features, tunes hyperparameters, and returns the winner.

## How It Works Under the Hood

The AutoML pipeline runs in stages:

1. **Feature selection** (optional): A genetic algorithm evaluates subsets of your features across several generations. Features that hurt accuracy get dropped.
2. **Algorithm selection**: Every classifier in miniml gets a chance -- KNN, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, Logistic Regression.
3. **Hyperparameter tuning**: PSO particles explore the hyperparameter space for the top-performing algorithms.
4. **Cross-validation**: Models are evaluated with k-fold cross-validation to avoid overfitting.
5. **Selection**: The model with the highest cross-validated accuracy wins.

## Adding Feature Selection

If your dataset has many features, some may be noise. Feature selection finds the useful ones.

```typescript
const result = autoFit(X, y, {
  nSamples: 60,
  nFeatures: 4,
  featureSelection: {
    enabled: true,
    generations: 15,    // more generations = more thorough (slower)
    populationSize: 20, // candidates per generation
  },
  cvFolds: 5,
});

console.log(`Best algorithm: ${result.algorithm}`);
console.log(`Accuracy:       ${(result.accuracy * 100).toFixed(1)}%`);
```

Feature selection is optional because on small datasets with few features, it can overfit. Use it when you have 10+ features and suspect some are irrelevant.

## Progress Monitoring

AutoML runs can take a while on larger datasets. Use the `progressCallback` to track what's happening.

```typescript
const result = autoFit(X, y, {
  nSamples: 60,
  nFeatures: 4,
  featureSelection: { enabled: true, generations: 10, populationSize: 15 },
  cvFolds: 3,
  progressCallback: (status) => {
    console.log(`[${status.stage}] ${status.message} — best: ${(status.bestAccuracy * 100).toFixed(1)}%`);
  },
});
```

The callback fires at each stage with:
- `stage`: Which phase is running (`feature_selection`, `algorithm_selection`, `hyperparameter_tuning`, `done`)
- `message`: Human-readable description of what's happening
- `bestAccuracy`: Running best accuracy seen so far

## Making Predictions

The result object includes a `predict()` function so you can classify new data immediately.

```typescript
// New sample: 4 features
const newSample = new Float64Array([5.1, 3.5, 1.4, 0.2]);
const prediction = result.predict(newSample);

console.log(`Predicted class: ${prediction}`);
```

No need to retrain or deserialize. The model is ready to use.

## Interpreting the Rationale

The `rationale` field explains why AutoML chose the winning model. For example:

```
"Random Forest achieved highest accuracy (96.7%) with 10 trees,
max depth 5, using features [0, 2, 3]. Feature selection removed
feature 1 (correlation with feature 0: 0.94)."
```

This helps you understand not just *what* won, but *why*.

## AutoML vs Manual Selection

When should you use AutoML versus picking an algorithm yourself?

| Scenario | Recommendation |
|----------|---------------|
| Exploring a new dataset | AutoML -- let it survey the landscape |
| Production pipeline with tight latency | Manual -- you know the best model already |
| Many features, unknown relevance | AutoML with feature selection enabled |
| Benchmarking / research | Manual -- you need control over each variable |
| Quick prototype | AutoML -- fastest path to a working model |

AutoML is a starting point, not an endpoint. Once you know which algorithm works best, you can train it manually with more fine-grained control.

## Summary

1. `autoFit(X, y, options)` tries every algorithm and returns the best
2. Enable `featureSelection` when you have many features
3. Use `progressCallback` to monitor long runs
4. `result.predict(sample)` classifies new data instantly
5. `result.rationale` explains the choice

## Next Steps

- **How-to**: Deep-dive into AutoML configuration in [how_to/automl/](../how_to/automl/)
- **Explanation**: How the genetic algorithm and PSO work in [explanation/automl/](../explanation/automl/)
- **Tutorial 01**: If you skipped it, start with [Your First ML Model](./01-first-model.md)
