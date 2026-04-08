# AutoML in miniml

## What is AutoML?

AutoML (Automated Machine Learning) automatically finds the best machine learning model for your data. In miniml, AutoML combines three powerful optimization techniques:

1. **Genetic Algorithm Feature Selection** — Identifies the optimal subset of features
2. **PSO Hyperparameter Optimization** — Tunes algorithm parameters automatically
3. **Algorithm Selection** — Tests multiple algorithms, selects the best performer

## Why Use AutoML?

**Manual ML Workflow:**
```
Choose algorithm → Tune hyperparameters → Evaluate → Repeat
```

**AutoML Workflow:**
```
Provide data → Get best model automatically
```

AutoML saves time and often discovers better models than manual tuning because:
- Tests hundreds of feature combinations
- Explores hyperparameter spaces more thoroughly
- Compares multiple algorithms objectively
- Reduces human bias in model selection

## Quick Start

```js
import { autoFit } from '@seanchatmangpt/wminml';

// Basic usage
const model = await autoFit(X, y);

// Make predictions
const prediction = await model.predict(testPoint);

// Get model details
console.log(model.algorithm);    // "RandomForest"
console.log(model.accuracy);     // 0.95
console.log(model.rationale);    // Why this algorithm was chosen
```

## AutoML Configuration

### Basic Configuration

```js
const model = await autoFit(X, y, {
  // Feature selection
  featureSelection: true,      // Enable GA feature selection
  maxFeatures: 0.8,            // Keep top 80% of features

  // Cross-validation
  cvFolds: 5,                  // 5-fold cross-validation

  // Algorithm selection
  algorithms: [
    'KNN',
    'DecisionTree',
    'RandomForest',
    'GradientBoosting',
    'NaiveBayes'
  ],

  // Progress monitoring
  progressCallback: (update) => {
    console.log(`Progress: ${update.percent}%`);
    console.log(`Testing: ${update.algorithm}`);
    console.log(`Accuracy: ${update.accuracy}`);
  }
});
```

### Advanced Configuration

```js
const model = await autoFit(X, y, {
  // Genetic algorithm settings
  featureSelection: true,
  ga: {
    populationSize: 50,
    generations: 100,
    mutationRate: 0.1,
    crossoverRate: 0.7,
    elitismCount: 5
  },

  // PSO settings
  hyperparameterOptimization: true,
  pso: {
    swarmSize: 30,
    maxIterations: 100,
    w: 0.7,      // Inertia weight
    c1: 1.5,     // Cognitive coefficient
    c2: 1.5      // Social coefficient
  },

  // Scoring metric
  scoringMetric: 'accuracy',  // 'accuracy', 'f1', 'roc_auc'

  // Time budget
  maxTimeMs: 60000,  // Stop after 60 seconds

  // Verbose output
  verbose: true
});
```

## Genetic Algorithm Feature Selection

### How It Works

1. **Initialization** — Create random population of feature subsets
2. **Fitness Evaluation** — Score each subset using cross-validation
3. **Selection** — Select best subsets for reproduction
4. **Crossover** — Combine parent subsets to create offspring
5. **Mutation** — Randomly add/remove features
6. **Repeat** — Evolve population over generations

### GA Feature Selection Example

```js
import { geneticFeatureSelection } from '@seanchatmangpt/wminml';

const result = await geneticFeatureSelection(X, y, {
  populationSize: 50,
  generations: 100,
  mutationRate: 0.1,
  crossoverRate: 0.7,
  elitismCount: 5,
  cvFolds: 5,
  scoringMetric: 'accuracy',
  progressCallback: (update) => {
    console.log(`Generation ${update.generation}: ${update.bestScore}`);
  }
});

// Results
console.log(result.selectedFeatures);  // [0, 2, 5, 7]
console.log(result.fitnessScore);      // 0.95
console.log(result.originalScore);     // 0.87 (without feature selection)
console.log(result.reduction);         // 60% (reduced from 10 to 4 features)

// Use selected features
const X_selected = X.map(row =>
  result.selectedFeatures.map(i => row[i])
);
```

### GA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `populationSize` | 50 | Number of feature subsets in population |
| `generations` | 100 | Number of evolution cycles |
| `mutationRate` | 0.1 | Probability of random feature changes |
| `crossoverRate` | 0.7 | Probability of combining parent subsets |
| `elitismCount` | 5 | Number of best subsets carried forward |
| `cvFolds` | 5 | Cross-validation folds for fitness evaluation |
| `scoringMetric` | 'accuracy' | Metric for fitness evaluation |

## PSO Hyperparameter Optimization

### How It Works

1. **Initialization** — Create swarm of particles with random positions
2. **Fitness Evaluation** — Score each particle's position
3. **Velocity Update** — Adjust velocities based on:
   - Personal best (cognitive component)
   - Global best (social component)
4. **Position Update** — Move particles according to velocities
5. **Repeat** — Iterate until convergence

### PSO Optimization Example

```js
import { psoOptimize } from '@seanchatmangpt/wminml';

const result = await psoOptimize({
  // Objective function to minimize
  objectiveFn: async (params) => {
    const model = await trainRandomForest(params);
    const accuracy = await evaluateModel(model);
    return -accuracy;  // PSO minimizes, so negate
  },

  // Parameter bounds
  bounds: {
    nTrees: [10, 200],
    maxDepth: [3, 20],
    minSamplesSplit: [2, 10]
  },

  // PSO parameters
  swarmSize: 30,
  maxIterations: 100,
  w: 0.7,   // Inertia weight
  c1: 1.5,  // Cognitive coefficient
  c2: 1.5,  // Social coefficient

  // Progress monitoring
  progressCallback: (update) => {
    console.log(`Iteration ${update.iteration}: ${update.bestScore}`);
  }
});

// Results
console.log(result.bestParams);  // { nTrees: 100, maxDepth: 10, minSamplesSplit: 2 }
console.log(result.bestScore);   // -0.96 (negated accuracy)
console.log(result.iterations);  // 45 (converged early)
```

### PSO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swarmSize` | 30 | Number of particles in swarm |
| `maxIterations` | 100 | Maximum iterations |
| `w` | 0.7 | Inertia weight (momentum) |
| `c1` | 1.5 | Cognitive coefficient (personal best) |
| `c2` | 1.5 | Social coefficient (global best) |

## Algorithm Selection

### Supported Algorithms

AutoML can select from the following algorithms:

**Classification:**
- KNN (k-Nearest Neighbors)
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Naive Bayes
- Logistic Regression
- Perceptron
- Linear SVM

**Regression:**
- Linear Regression
- Ridge Regression
- Polynomial Regression

### Algorithm Selection Example

```js
import { autoFit } from '@seanchatmangpt/wminml';

const model = await autoFit(X, y, {
  algorithms: [
    'KNN',
    'DecisionTree',
    'RandomForest',
    'GradientBoosting'
  ],
  cvFolds: 5,
  scoringMetric: 'accuracy'
});

// Get all scores
console.log(model.allScores);
// {
//   "KNN": 0.87,
//   "DecisionTree": 0.91,
//   "RandomForest": 0.95,
//   "GradientBoosting": 0.93
// }

// Get winner
console.log(model.algorithm);  // "RandomForest"
console.log(model.accuracy);   // 0.95
```

## Progress Monitoring

AutoML provides real-time progress updates:

```js
const model = await autoFit(X, y, {
  progressCallback: (update) => {
    console.log(`=== Update ===`);
    console.log(`Phase: ${update.phase}`);
    console.log(`Percent: ${update.percent}%`);

    if (update.phase === 'feature_selection') {
      console.log(`Generation: ${update.generation}`);
      console.log(`Best features: ${update.bestFeatures}`);
      console.log(`Fitness: ${update.fitnessScore}`);
    }

    if (update.phase === 'algorithm_selection') {
      console.log(`Testing: ${update.algorithm}`);
      console.log(`Accuracy: ${update.accuracy}`);
    }

    if (update.phase === 'hyperparameter_optimization') {
      console.log(`Iteration: ${update.iteration}`);
      console.log(`Best params: ${update.bestParams}`);
      console.log(`Best score: ${update.bestScore}`);
    }
  }
});
```

### Progress Update Structure

```typescript
interface ProgressUpdate {
  phase: 'feature_selection' | 'algorithm_selection' | 'hyperparameter_optimization' | 'complete';
  percent: number;

  // Feature selection phase
  generation?: number;
  bestFeatures?: number[];
  fitnessScore?: number;

  // Algorithm selection phase
  algorithm?: string;
  accuracy?: number;

  // Hyperparameter optimization phase
  iteration?: number;
  bestParams?: Record<string, number>;
  bestScore?: number;
}
```

## Result Interpretation

### AutoML Result Structure

```typescript
interface AutoMLResult {
  // Best model
  algorithm: string;
  accuracy: number;
  trainingTime: number;

  // Features
  features: number[];
  originalFeatureCount: number;
  selectedFeatureCount: number;

  // Hyperparameters
  hyperparameters: Record<string, number>;

  // All algorithm scores
  allScores: Record<string, number>;

  // Rationale
  rationale: string;

  // Predict method
  predict: (x: number[]) => Promise<number>;
}
```

### Understanding the Rationale

```js
console.log(model.rationale);
```

Example rationale:
```
"RandomForest achieved highest cross-validation accuracy (95%) with strong
performance across all metrics. Feature selection reduced dimensionality
from 10 to 4 features (60% reduction), improving training speed by 60%
while maintaining accuracy. The model shows low variance (std=0.02 across
CV folds), indicating good generalization.

Key features selected: [0, 2, 5, 7]
- Feature 0: Strong correlation with target (0.87)
- Feature 2: High information gain (0.45)
- Feature 5: Low multicollinearity (VIF=1.2)
- Feature 7: Non-linear relationship detected

Hyperparameters optimized via PSO:
- nTrees: 100 (optimal balance of bias-variance)
- maxDepth: 10 (prevents overfitting)
- minSamplesSplit: 2 (allows fine-grained splits)"
```

## Best Practices

### 1. Start Simple

```js
// Basic AutoML (good starting point)
const model = await autoFit(X, y);
```

### 2. Enable Feature Selection for High-Dimensional Data

```js
// Use feature selection when nFeatures > 20
const model = await autoFit(X, y, {
  featureSelection: true,
  maxFeatures: 0.5  // Reduce to 50% of features
});
```

### 3. Set Time Budgets for Large Datasets

```js
// Prevent AutoML from running too long
const model = await autoFit(X, y, {
  maxTimeMs: 30000  // Stop after 30 seconds
});
```

### 4. Use Appropriate Scoring Metrics

```js
// For imbalanced datasets
const model = await autoFit(X, y, {
  scoringMetric: 'f1'  // or 'roc_auc'
});

// For regression
const model = await autoFit(X, y, {
  scoringMetric: 'r2'  // or 'rmse', 'mae'
});
```

### 5. Limit Algorithms for Faster Results

```js
// Test only fast algorithms
const model = await autoFit(X, y, {
  algorithms: ['KNN', 'DecisionTree', 'NaiveBayes']
});
```

## Common Issues

### Issue 1: AutoML Takes Too Long

**Solution:** Reduce search space:
```js
const model = await autoFit(X, y, {
  featureSelection: false,  // Skip feature selection
  hyperparameterOptimization: false,  // Skip PSO
  maxTimeMs: 10000  // Hard time limit
});
```

### Issue 2: All Algorithms Perform Poorly

**Solution:** Check data quality:
```js
// 1. Check for missing values
// 2. Check for class imbalance
// 3. Try feature scaling
import { standardScaler } from '@seanchatmangpt/wminml';
const X_scaled = await standardScaler(X, nSamples, nFeatures);

const model = await autoFit(X_scaled, y);
```

### Issue 3: Memory Issues with Large Datasets

**Solution:** Use incremental feature selection:
```js
// Select features on subset first
const subsetSize = Math.floor(nSamples * 0.1);
const X_subset = X.slice(0, subsetSize);
const y_subset = y.slice(0, subsetSize);

const fsResult = await geneticFeatureSelection(X_subset, y_subset);

// Use selected features on full dataset
const X_selected = X.map(row =>
  fsResult.selectedFeatures.map(i => row[i])
);
```

## Advanced Usage

### Custom Objective Functions

```js
const model = await autoFit(X, y, {
  objectiveFn: async (algorithm, params) => {
    const model = await trainAlgorithm(algorithm, X, y, params);
    const accuracy = await evaluate(model, X, y);
    const f1 = await calculateF1(model, X, y);

    // Custom objective: balance accuracy and F1
    return 0.7 * accuracy + 0.3 * f1;
  }
});
```

### Warm Starting

```js
// Start from known good hyperparameters
const model = await autoFit(X, y, {
  initialParams: {
    'RandomForest': { nTrees: 50, maxDepth: 10 },
    'GradientBoosting': { nEstimators: 50, learningRate: 0.1 }
  }
});
```

### Multi-Objective Optimization

```js
const model = await autoFit(X, y, {
  objectives: [
    { metric: 'accuracy', weight: 0.5 },
    { metric: 'trainingTime', weight: -0.3 },  // Minimize time
    { metric: 'inferenceTime', weight: -0.2 }  // Minimize latency
  ]
});
```
