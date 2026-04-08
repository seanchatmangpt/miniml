# Hyperparameter Tuning

Use miniml's Particle Swarm Optimization (PSO) to automatically find the best hyperparameters for your model.

## Problem

Model performance depends heavily on hyperparameters (learning rate, regularization strength, tree depth, etc.). Manual tuning is tedious and often suboptimal.

## Solution

`autoFit` uses PSO internally to search the hyperparameter space. You can monitor progress and control the search.

### Basic Usage

```typescript
import { autoFit } from '@seanchatmangpt/wminml';

const X = [
  [1.2, 3.4], [2.1, 5.6], [3.3, 2.1], [4.5, 7.8],
  [5.1, 1.2], [6.7, 4.5], [7.2, 8.9], [8.1, 3.3],
  [0.5, 6.7], [9.0, 2.0],
];
const y = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1];

const result = autoFit(X, y, {
  cvFolds: 5,
  progressCallback: (iteration, bestFitness) => {
    console.log(`PSO iteration ${iteration}: best accuracy = ${bestFitness.toFixed(4)}`);
  },
});

console.log(`\nBest algorithm: ${result.algorithm}`);
console.log(`Best accuracy:  ${result.accuracy.toFixed(4)}`);
console.log(`Rationale:     ${result.rationale}`);
```

### Monitoring Progress

The `progressCallback` fires after each PSO iteration, letting you track convergence in real time:

```typescript
const result = autoFit(X, y, {
  cvFolds: 5,
  progressCallback: (iteration, bestFitness) => {
    if (iteration % 5 === 0 || bestFitness > 0.95) {
      process.stdout.write(`\rIteration ${iteration}: ${bestFitness.toFixed(4)}`);
    }
  },
});
console.log('\nDone.');
```

Typical output:

```
Iteration 0: 0.7000
Iteration 5: 0.8000
Iteration 10: 0.8500
Iteration 15: 0.9000
Iteration 18: 0.9500
```

### Controlling the Search

The PSO explores different algorithms and their hyperparameters simultaneously. Cross-validation folds control evaluation accuracy:

| Setting | Effect |
|---------|--------|
| `cvFolds: 3` | Faster, noisier evaluation |
| `cvFolds: 5` | Good balance (default) |
| `cvFolds: 10` | More accurate evaluation, slower |
| `featureSelection: true` | Also optimizes which features to use |

```typescript
// Thorough search with feature selection
const thorough = autoFit(X, y, {
  featureSelection: true,
  cvFolds: 10,
  progressCallback: (it, fit) => {
    console.log(`[${it}] accuracy=${fit.toFixed(4)}`);
  },
});
```

### Comparing Algorithms

`autoFit` tries multiple algorithms and returns the best one. The `algorithm` field tells you which won:

```typescript
if (result.algorithm.includes('random_forest')) {
  console.log('Ensemble method won -- dataset benefits from bagging');
} else if (result.algorithm.includes('svm')) {
  console.log('SVM won -- dataset has clear margin separation');
} else if (result.algorithm.includes('knn')) {
  console.log('KNN won -- dataset is locally structured');
}
```

## Tips

- Always use cross-validation (`cvFolds >= 3`) to avoid overfitting during hyperparameter search.
- If PSO converges quickly (plateaus after a few iterations), your dataset may be easy -- any reasonable algorithm works.
- If accuracy stays low despite tuning, the problem may be in the features, not the hyperparameters. Try feature selection or feature engineering.
- The `progressCallback` is the best way to decide whether to let the search run longer or stop early.
