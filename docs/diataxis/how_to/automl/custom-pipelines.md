# Custom AutoML Pipelines

Combine feature selection, cross-validation, and algorithm evaluation into a custom AutoML workflow.

## Problem

The default `autoFit` call works well for standard use cases, but you need more control over the pipeline -- custom evaluation, specific algorithms, or multi-stage workflows.

## Solution

Chain `autoFit` calls with different configurations to build a tailored pipeline.

### Pipeline 1: Feature Selection Then Algorithm Search

Run feature selection first, then search algorithms on the reduced feature set.

```typescript
import { autoFit } from '@seanchatmangpt/wminml';

const X = [
  [0.8, 0.3, 0.9, 0.1, 0.7, 0.4],
  [0.2, 0.7, 0.3, 0.5, 0.1, 0.8],
  [0.5, 0.9, 0.7, 0.2, 0.5, 0.2],
  [0.1, 0.4, 0.2, 0.8, 0.3, 0.6],
  [0.9, 0.1, 0.8, 0.3, 0.9, 0.1],
  [0.4, 0.6, 0.5, 0.7, 0.2, 0.9],
  [0.7, 0.2, 0.6, 0.4, 0.8, 0.3],
  [0.3, 0.8, 0.4, 0.6, 0.4, 0.7],
];
const y = [1, 0, 1, 0, 1, 0, 1, 0];

// Stage 1: Feature selection with GA
console.log('Stage 1: Feature selection...');
const fsResult = autoFit(X, y, {
  featureSelection: true,
  cvFolds: 3,
});

// Stage 2: Algorithm search with selected features
console.log('Stage 2: Algorithm search...');
const algoResult = autoFit(X, y, {
  featureSelection: true,
  cvFolds: 5,
  progressCallback: (it, fit) => {
    console.log(`  [${it}] accuracy=${fit.toFixed(4)}`);
  },
});

console.log(`\nPipeline result:`);
console.log(`  Algorithm: ${algoResult.algorithm}`);
console.log(`  Accuracy:  ${algoResult.accuracy.toFixed(4)}`);
console.log(`  Rationale: ${algoResult.rationale}`);
```

### Pipeline 2: Compare Cross-Validation Strategies

Different CV folds can give different pictures of model performance. Run multiple strategies and compare.

```typescript
const folds = [3, 5, 7];
const results = [];

for (const folds of [3, 5, 7]) {
  const result = autoFit(X, y, {
    featureSelection: true,
    cvFolds: folds,
  });
  results.push({ folds, ...result });
  console.log(`${folds}-fold CV: ${result.algorithm} (accuracy=${result.accuracy.toFixed(4)})`);
}

// Find the most stable configuration (smallest variance in rationale)
const best = results.reduce((a, b) => b.accuracy > a.accuracy ? b : a);
console.log(`\nBest configuration: ${best.folds}-fold CV with ${best.algorithm}`);
```

### Pipeline 3: Iterative Refinement

Run AutoML, examine the result, and refine based on what you learn.

```typescript
// Round 1: Quick scan
const quick = autoFit(X, y, { featureSelection: false, cvFolds: 3 });
console.log(`Round 1 (quick): ${quick.algorithm} = ${quick.accuracy.toFixed(4)}`);

// Round 2: Thorough search with feature selection
const thorough = autoFit(X, y, {
  featureSelection: true,
  cvFolds: 5,
  progressCallback: (it, fit) => {
    console.log(`  Round 2 [${it}]: ${fit.toFixed(4)}`);
  },
});
console.log(`Round 2 (thorough): ${thorough.algorithm} = ${thorough.accuracy.toFixed(4)}`);

// Round 3: If accuracy is high enough, predict on new data
if (thorough.accuracy > 0.85) {
  const newX = [[0.6, 0.5, 0.7, 0.3, 0.6, 0.5]];
  const predictions = newX.map(row => thorough.predict(row));
  console.log(`Predictions for new data: ${predictions}`);
} else {
  console.log('Accuracy below threshold -- consider more data or feature engineering.');
}
```

## Tips

- Start with a quick scan (`cvFolds: 3`, no feature selection) to establish a baseline.
- Use the baseline to decide whether a more expensive search is worthwhile.
- The `predict()` function on the result object lets you apply the best model to new data immediately.
- For production pipelines, cache the `autoFit` result and retrain periodically as new data arrives.
