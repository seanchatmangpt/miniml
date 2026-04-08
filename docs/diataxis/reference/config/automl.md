# AutoML Configuration

Configuration parameters for the AutoML pipeline optimizer, including genetic algorithm feature selection and PSO hyperparameter optimization.

## AutoML Pipeline

AutoML in miniml performs three stages:

1. **Feature Selection** -- Genetic algorithm selects the most informative feature subset
2. **Algorithm Evaluation** -- Tests candidate algorithms with cross-validation
3. **Pipeline Optimization** -- PSO tunes hyperparameters for the best algorithm

## Genetic Algorithm Parameters (Feature Selection)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `populationSize` | 50 | Number of individuals per generation |
| `generations` | 50 | Number of generations to evolve |
| `mutationRate` | 0.1 | Probability of flipping a feature bit per individual |
| `crossoverRate` | 0.7 | Probability of performing crossover between parents |
| `elitismCount` | 2 | Number of top individuals preserved unchanged per generation |
| `tournamentSize` | 3 | Tournament selection pool size |

### Feature Representation

Each individual is a binary vector of length `nFeatures`. A `1` means the feature is included, `0` means excluded.

### Fitness Function

Fitness is the cross-validation score (accuracy for classification, R-squared for regression) of the best algorithm using that feature subset.

## PSO Parameters (Hyperparameter Optimization)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `particles` | 20 | 10-50 | Number of particles in the swarm |
| `inertia` | 0.7 | 0.4-0.9 | Momentum weight (velocity damping) |
| `cognitive` | 1.5 | 1.0-2.0 | Personal best attraction (c1) |
| `social` | 1.5 | 1.0-2.0 | Global best attraction (c2) |
| `maxVelocity` | 0.5 | 0.1-1.0 | Maximum velocity clamping per dimension |
| `maxIterations` | 100 | 50-500 | Maximum iterations before stopping |
| `tolerance` | 1e-6 | -- | Convergence tolerance on fitness improvement |

### Hyperparameter Search Space

PSO optimizes over algorithm-specific hyperparameters:

| Algorithm | Hyperparameters Searched |
|-----------|------------------------|
| KNN | `k` (1-20) |
| Decision Tree | `maxDepth` (3-20), `minSamplesSplit` (2-10) |
| Logistic Regression | `learningRate` (0.001-1.0), `lambda` (0.0-10.0) |
| Naive Bayes | (no hyperparameters) |
| Linear Regression | `learningRate` (0.001-1.0), `lambda` (0.0-10.0) |

## Cross-Validation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `folds` | 5 | Number of CV folds |
| `stratified` | true | Use stratified sampling for classification |

## Algorithm Selection

AutoML evaluates these algorithms:

**Classification:** KNN, Decision Tree, Logistic Regression, Naive Bayes, Perceptron
**Regression:** Linear Regression, Polynomial Regression (degree 2)

The algorithm with the highest CV score is selected for the final pipeline.

## Usage Example

```typescript
import { init, automlFit, automlPredict } from '@seanchatmangpt/wminml';
await init();

// AutoML automatically selects features and tunes hyperparameters
const model = automlFit(X, y, nFeatures, {
  populationSize: 50,
  generations: 50,
  particles: 20,
  maxIterations: 100,
  folds: 5,
});

const predictions = automlPredict(model, XTest);
```

## Tuning Tips

- **More generations/population** = better feature selection but slower.
- **More particles/iterations** = better hyperparameter tuning but slower.
- For small datasets (< 50 samples), reduce `folds` to 3.
- For large datasets (> 1000 samples), reduce `populationSize` to 20 and `generations` to 30.
- Set `tolerance` tighter (1e-8) for higher quality; looser (1e-3) for faster convergence.
