# Optimization Suite

Advanced metaheuristic optimization algorithms in miniml.

## Overview

miniml includes a comprehensive suite of metaheuristic optimization algorithms:

- **Genetic Algorithms (GA)** — Population-based evolutionary optimization
- **Particle Swarm Optimization (PSO)** — Swarm intelligence for global optimization
- **Simulated Annealing** — Probabilistic global optimization
- **Multi-Armed Bandit** — Exploration-exploitation balancing
- **Feature Importance** — Identify key features
- **Anomaly Detection** — Outlier detection methods
- **Drift Detection** — Concept drift monitoring
- **Prediction Intervals** — Uncertainty quantification

---

## Genetic Algorithms (GA)

### Overview

Genetic algorithms are optimization algorithms inspired by natural selection. They evolve a population of candidate solutions through selection, crossover, and mutation.

### Key Concepts

1. **Population** — Set of candidate solutions (chromosomes)
2. **Fitness** — Quality score for each solution
3. **Selection** — Better solutions more likely to reproduce
4. **Crossover** — Combine two parents to create offspring
5. **Mutation** — Random changes to maintain diversity

### Feature Selection with GA

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
    console.log(`Gen ${update.generation}: ${update.bestScore}`);
  }
});

console.log('Selected features:', result.selectedFeatures);
console.log('Fitness score:', result.fitnessScore);
console.log('Original score:', result.originalScore);
```

### GA Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `populationSize` | 50 | 20-200 | Number of chromosomes |
| `generations` | 100 | 50-500 | Evolution cycles |
| `mutationRate` | 0.1 | 0.01-0.5 | Probability of mutation |
| `crossoverRate` | 0.7 | 0.5-0.9 | Probability of crossover |
| `elitismCount` | 5 | 1-20 | Best chromosomes preserved |

### GA Operators

**Selection Methods:**
- **Tournament** — Select best from random subset
- **Roulette Wheel** — Probability proportional to fitness
- **Rank Selection** — Based on fitness rank

**Crossover Methods:**
- **Single Point** — Split at random position
- **Two Point** — Split at two positions
- **Uniform** — Gene-by-gene selection

**Mutation Methods:**
- **Bit Flip** — Flip gene value
- **Swap** — Exchange two genes
- **Scramble** — Shuffle subset of genes

### Custom GA

```js
import { GeneticAlgorithm } from '@seanchatmangpt/wminml';

const ga = new GeneticAlgorithm({
  chromosomeLength: 20,
  populationSize: 50,
  fitnessFn: async (chromosome) => {
    // Custom fitness evaluation
    const features = chromosome.filter(gene => gene === 1);
    const model = await trainModel(features);
    return model.accuracy;
  }
});

const result = await ga.evolve({
  generations: 100,
  mutationRate: 0.1,
  crossoverRate: 0.7
});
```

---

## Particle Swarm Optimization (PSO)

### Overview

PSO is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions (particles) and moving these particles around in the search-space according to simple mathematical formulae.

### Key Concepts

1. **Particles** — Candidate solutions with position and velocity
2. **Personal Best (pbest)** — Best position found by particle
3. **Global Best (gbest)** — Best position found by swarm
4. **Inertia (w)** — Momentum factor
5. **Cognitive (c1)** — Personal best influence
6. **Social (c2)** — Global best influence

### Hyperparameter Optimization with PSO

```js
import { psoOptimize } from '@seanchatmangpt/wminml';

const result = await psoOptimize({
  objectiveFn: async (params) => {
    const model = await trainRandomForest(params);
    const accuracy = await evaluateModel(model);
    return -accuracy;  // PSO minimizes
  },
  bounds: {
    nTrees: [10, 200],
    maxDepth: [3, 20],
    minSamplesSplit: [2, 10]
  },
  swarmSize: 30,
  maxIterations: 100,
  w: 0.7,   // Inertia
  c1: 1.5,  // Cognitive
  c2: 1.5,  // Social
  progressCallback: (update) => {
    console.log(`Iter ${update.iteration}: ${update.bestScore}`);
  }
});

console.log('Best params:', result.bestParams);
console.log('Best score:', result.bestScore);
```

### PSO Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `swarmSize` | 30 | 10-100 | Number of particles |
| `maxIterations` | 100 | 50-500 | Maximum iterations |
| `w` | 0.7 | 0.4-0.9 | Inertia weight |
| `c1` | 1.5 | 0.5-2.0 | Cognitive coefficient |
| `c2` | 1.5 | 0.5-2.0 | Social coefficient |

### PSO Variants

**Standard PSO:**
```
v[i] = w*v[i] + c1*r1*(pbest[i] - x[i]) + c2*r2*(gbest - x[i])
x[i] = x[i] + v[i]
```

**Constriction Coefficient PSO:**
- Prevents explosion
- Better convergence
- χ = 0.729, c1 = c2 = 2.05

**Discrete PSO:**
- For discrete optimization
- Uses sigmoid function for probability

---

## Simulated Annealing

### Overview

Simulated annealing is a probabilistic technique for approximating the global optimum of a given function. It mimics the physical process of annealing in metallurgy.

### Key Concepts

1. **Temperature** — Control parameter for exploration
2. **Cooling Schedule** — Temperature reduction rate
3. **Acceptance Probability** — P(accept worse) = exp(-ΔE/T)
4. **Equilibrium** — Iterations at each temperature

### Optimization Example

```js
import { simulatedAnnealing } from '@seanchatmangpt/wminml';

const result = await simulatedAnnealing({
  objectiveFn: async (state) => {
    // Minimize objective
    const cost = evaluateState(state);
    return cost;
  },
  initialState: getInitialState(),
  neighborFn: (state) => {
    // Generate neighboring state
    return perturbState(state);
  },
  temperature: 1000,
  coolingRate: 0.95,
  minTemperature: 0.01,
  iterationsPerTemp: 100,
  progressCallback: (update) => {
    console.log(`Temp: ${update.temp}, Best: ${update.bestCost}`);
  }
});

console.log('Best state:', result.bestState);
console.log('Best cost:', result.bestCost);
```

### Cooling Schedules

**Geometric Cooling:**
```
T[k+1] = α × T[k]
```
where α ∈ [0.8, 0.99]

**Linear Cooling:**
```
T[k+1] = T[k] - ΔT
```

**Adaptive Cooling:**
```
T[k+1] = T[k] × (1 - acceptanceRate)
```

### SA Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `temperature` | 1000 | 100-10000 | Initial temperature |
| `coolingRate` | 0.95 | 0.8-0.99 | Temperature reduction |
| `minTemperature` | 0.01 | 0.001-0.1 | Stopping criterion |
| `iterationsPerTemp` | 100 | 50-500 | Equilibrium iterations |

---

## Multi-Armed Bandit

### Overview

Multi-armed bandit algorithms balance exploration (trying new options) and exploitation (using known best options).

### Key Concepts

1. **Arms** — Actions/choices available
2. **Rewards** — Feedback from actions
3. **Regret** — Difference from optimal
4. **Exploration vs Exploitation** — Fundamental trade-off

### Epsilon-Greedy

```js
import { epsilonGreedy } from '@seanchatmangpt/wminml';

const bandit = epsilonGreedy({
  nArms: 5,
  epsilon: 0.1,  // 10% exploration
  decay: 0.995   // Decay epsilon over time
});

for (let t = 0; t < 1000; t++) {
  const arm = bandit.selectArm();
  const reward = await pullArm(arm);
  bandit.update(arm, reward);
}

console.log('Best arm:', bandit.getBestArm());
console.log('Arm counts:', bandit.getArmCounts());
console.log('Arm values:', bandit.getArmValues());
```

### UCB (Upper Confidence Bound)

```js
import { ucb } from '@seanchatmangpt/wminml';

const bandit = ucb({
  nArms: 5,
  c: 1.4  // Exploration parameter
});

for (let t = 0; t < 1000; t++) {
  const arm = bandit.selectArm();
  const reward = await pullArm(arm);
  bandit.update(arm, reward);
}
```

### Thompson Sampling

```js
import { thompsonSampling } from '@seanchatmangpt/wminml';

const bandit = thompsonSampling({
  nArms: 5,
  priorAlpha: 1,
  priorBeta: 1
});

for (let t = 0; t < 1000; t++) {
  const arm = bandit.selectArm();
  const reward = await pullArm(arm);
  bandit.update(arm, reward);
}
```

### Bandit Algorithms Comparison

| Algorithm | Exploration | Regret | Use Case |
|-----------|-------------|--------|----------|
| ε-Greedy | Random | High | Simple baseline |
| UCB | Optimism | Medium | Non-stationary |
| Thompson | Probabilistic | Low | Binary rewards |

---

## Feature Importance

### Overview

Feature importance methods identify which features contribute most to model predictions.

### Permutation Importance

```js
import { permutationImportance } from '@seanchatmangpt/wminml';

const importance = await permutationImportance({
  model: trainedModel,
  X: X_test,
  y: y_test,
  scoring: 'accuracy',
  nRepeats: 5
});

console.log('Feature importance:', importance.scores);
console.log('Std deviation:', importance.stds);
```

### Gini Importance

```js
import { giniImportance } from '@seanchatmangpt/wminml';

const importance = await giniImportance(decisionTreeModel);
console.log('Feature importance:', importance);
```

### SHAP-like Values

```js
import { shapValues } from '@seanchatmangpt/wminml';

const shap = await shapValues({
  model: trainedModel,
  X: X_background,
  x: X_instance
});

console.log('SHAP values:', shap.values);
console.log('Base value:', shap.baseValue);
```

---

## Anomaly Detection

### Overview

Anomaly detection identifies data points that deviate significantly from the majority of data.

### Isolation Forest

```js
import { isolationForest } from '@seanchatmangpt/wminml';

const model = await isolationForest(X, {
  nTrees: 100,
  maxSamples: 256,
  contamination: 0.1
});

const predictions = await model.predict(X);
const scores = await model.scoreSamples(X);
```

### Statistical Outlier Detection

```js
import { zScoreOutliers } from '@seanchatmangpt/wminml';

const outliers = await zScoreOutliers(data, {
  threshold: 3,
  method: 'median'  // 'mean' or 'median'
});
```

### IQR Method

```js
import { iqrOutliers } from '@seanchatmangpt/wminml';

const outliers = await iqrOutliers(data, {
  multiplier: 1.5  // Q3 + 1.5×IQR
});
```

---

## Drift Detection

### Overview

Drift detection monitors changes in data distribution over time.

### ADWIN (Adaptive Windowing)

```js
import { adwinDetector } from '@seanchatmangpt/wminml';

const detector = adwinDetector({
  delta: 0.002,
  maxBufferSize: 1000
});

for (const value of dataStream) {
  const driftDetected = detector.update(value);
  if (driftDetected) {
    console.log('Drift detected!');
    // Retrain model
  }
}
```

### DDM (Drift Detection Method)

```js
import { ddmDetector } from '@seanchatmangpt/wminml';

const detector = ddmDetector({
  warningLevel: 2.0,
  driftLevel: 3.0
});

for (const prediction of predictions) {
  const error = prediction.error;
  const status = detector.update(error);

  if (status === 'drift') {
    console.log('Drift detected!');
  } else if (status === 'warning') {
    console.log('Warning: possible drift');
  }
}
```

### Page-Hinkley Test

```js
import { pageHinkleyDetector } from '@seanchatmangpt/wminml';

const detector = pageHinkleyDetector({
  alpha: 0.99,
  lambda: 50,
  minSamples: 30
});

for (const observation of dataStream) {
  const driftDetected = detector.update(observation);
  if (driftDetected) {
    console.log('Change point detected!');
  }
}
```

---

## Prediction Intervals

### Overview

Prediction intervals quantify uncertainty in predictions.

### Bootstrap Intervals

```js
import { bootstrapIntervals } from '@seanchatmangpt/wminml';

const intervals = await bootstrapIntervals({
  model: trainedModel,
  X: X_test,
  nBootstraps: 1000,
  alpha: 0.05  // 95% confidence
});

console.log('Lower bounds:', intervals.lower);
console.log('Upper bounds:', intervals.upper);
```

### Conformal Prediction

```js
import { conformalIntervals } from '@seanchatmangpt/wminml';

const intervals = await conformalIntervals({
  model: trainedModel,
  X: X_cal,
  y: y_cal,
  X_test: X_test,
  alpha: 0.05
});

console.log('Prediction sets:', intervals.sets);
console.log('Coverage:', intervals.coverage);
```

### Quantile Regression

```js
import { quantileRegression } from '@seanchatmangpt/wminml';

const model = await quantileRegression(X, y, {
  quantiles: [0.05, 0.5, 0.95],
  nJobs: 4
});

const intervals = await model.predict(X_test);
```

---

## Optimization Best Practices

### 1. Choose the Right Algorithm

| Problem Type | Recommended Algorithm |
|--------------|---------------------|
| Feature selection | Genetic Algorithm |
| Hyperparameter tuning | PSO |
| Continuous optimization | Simulated Annealing |
| Discrete choices | Multi-Armed Bandit |
| Global optimization | GA + PSO hybrid |

### 2. Set Appropriate Parameters

```js
// Conservative settings (slow but reliable)
{
  populationSize: 100,
  generations: 500,
  mutationRate: 0.05
}

// Aggressive settings (fast but risky)
{
  populationSize: 20,
  generations: 50,
  mutationRate: 0.2
}
```

### 3. Use Warm Starts

```js
// Start from known good solution
const result = await psoOptimize({
  initialSwarm: [knownGoodSolution, ...randomParticles],
  ...
});
```

### 4. Monitor Progress

```js
progressCallback: (update) => {
  console.log(`${update.iteration}: ${update.bestScore}`);
  // Early stopping if converged
  if (update.converged) {
    return true;  // Stop optimization
  }
}
```

### 5. Validate Results

```js
// Cross-validate optimized parameters
const cvScores = await crossValidate(
  X, y,
  optimizedParams,
  5  // 5-fold CV
);
console.log('CV mean:', cvScores.mean);
console.log('CV std:', cvScores.std);
```

---

## Hybrid Approaches

### GA + PSO Hybrid

```js
// GA for global search
const gaResult = await geneticFeatureSelection(X, y, {
  generations: 50  // Coarse search
});

// PSO for local refinement
const psoResult = await psoOptimize({
  initialState: gaResult.selectedFeatures,
  ...  // Fine-tune parameters
});
```

### Simulated Annealing + Local Search

```js
// SA for global exploration
const saResult = await simulatedAnnealing({...});

// Gradient descent for local refinement
const refined = await gradientDescent({
  initialParams: saResult.bestState,
  ...
});
```

### Multi-Objective Optimization

```js
import { nsga2 } from '@seanchatmangpt/wminml';

const paretoFront = await nsga2({
  objectives: [
    { fn: maximizeAccuracy },
    { fn: minimizeTrainingTime },
    { fn: minimizeModelSize }
  ],
  populationSize: 100,
  generations: 200
});
```
