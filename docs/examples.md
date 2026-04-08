# Examples

Real-world usage examples for miniml.

## Table of Contents

1. [AutoML Quick Start](#automl-quick-start)
2. [Classification Pipeline](#classification-pipeline)
3. [Regression with Feature Selection](#regression-with-feature-selection)
4. [Clustering with Optimization](#clustering-with-optimization)
5. [Time Series Forecasting](#time-series-forecasting)
6. [Anomaly Detection](#anomaly-detection)
7. [Parallel Processing with Workers](#parallel-processing-with-workers)

---

## AutoML Quick Start

### Problem: Automatically Find the Best Classifier

```js
import { autoFit } from 'miniml';

// Sample data: iris-like classification
const X = new Float64Array([
  5.1, 3.5, 1.4, 0.2,  // Sepal length, width, Petal length, width
  4.9, 3.0, 1.4, 0.2,
  7.0, 3.2, 4.7, 1.4,
  6.4, 3.2, 4.5, 1.5,
  6.3, 3.3, 6.0, 2.5,
  5.8, 2.7, 5.1, 1.9
]);

const y = new Float64Array([0, 0, 1, 1, 2, 2]);

// AutoML automatically selects best algorithm
const model = await autoFit(X, y, {
  featureSelection: true,
  cvFolds: 3,
  progressCallback: (update) => {
    console.log(`${update.percent}%: Testing ${update.algorithm}`);
  }
});

// Make predictions
const testPoint = new Float64Array([6.0, 3.0, 4.5, 1.5]);
const prediction = await model.predict(testPoint);

console.log('Predicted class:', prediction);
console.log('Algorithm:', model.algorithm);
console.log('Accuracy:', model.accuracy);
console.log('Rationale:', model.rationale);
```

---

## Classification Pipeline

### Problem: Classify Customer Churn

```js
import {
  standardScaler,
  trainTestSplit,
  randomForestClassify,
  classificationReport,
  confusionMatrix
} from 'miniml';

// Sample customer data
// Features: [age, tenure, monthly_charges, total_charges]
const X = new Float64Array([
  25, 12, 50.5, 600.0,
  45, 36, 80.0, 2880.0,
  35, 24, 65.0, 1560.0,
  50, 48, 95.0, 4560.0,
  28, 8, 45.0, 360.0,
  40, 30, 70.0, 2100.0
]);

// Labels: 0 = stayed, 1 = churned
const y = new Float64Array([1, 0, 0, 1, 1, 0]);

const nSamples = 6;
const nFeatures = 4;

// Step 1: Normalize features
const X_scaled = await standardScaler(X, nSamples, nFeatures);

// Step 2: Split data
const split = await trainTestSplit(X_scaled, y, 0.8, 42);

// Step 3: Train classifier
const rf = await randomForestClassify(
  split.X_train,
  split.y_train,
  50,  // nTrees
  10   // maxDepth
);

// Step 4: Evaluate
const y_pred = split.X_test.map(async (x, i) => {
  const row = split.X_test.slice(i * nFeatures, (i + 1) * nFeatures);
  return await rf.predict(row);
});

const report = await classificationReport(split.y_test, y_pred);
const cm = await confusionMatrix(split.y_test, y_pred);

console.log('Classification Report:', report);
console.log('Confusion Matrix:', cm);

// Step 5: Predict new customer
const newCustomer = new Float64Array([38, 20, 60.0, 1200.0]);
const churnProbability = await rf.predict(newCustomer);
console.log('Churn probability:', churnProbability);
```

---

## Regression with Feature Selection

### Problem: Predict House Prices

```js
import {
  geneticFeatureSelection,
  linearRegression,
  r2Score,
  rmse,
  standardScaler
} from 'miniml';

// Sample house data
// Features: [sqft, bedrooms, bathrooms, age, garage_size, lot_size, rooms, floors]
const X = new Float64Array([
  2000, 3, 2, 10, 400, 5000, 7, 2,
  1500, 2, 1, 5, 200, 3000, 5, 1,
  2500, 4, 3, 15, 600, 8000, 9, 2,
  1800, 3, 2, 8, 300, 4000, 6, 1,
  3000, 5, 3, 20, 800, 10000, 12, 2,
  1200, 2, 1, 3, 150, 2500, 4, 1
]);

// Prices (in thousands)
const y = new Float64Array([450, 300, 600, 380, 800, 250]);

const nSamples = 6;
const nFeatures = 8;

// Step 1: Feature selection (find most important features)
const fsResult = await geneticFeatureSelection(X, y, {
  populationSize: 30,
  generations: 50,
  cvFolds: 3,
  scoringMetric: 'r2'
});

console.log('Selected features:', fsResult.selectedFeatures);
// Output: [0, 1, 4, 6] (sqft, bedrooms, garage_size, rooms)

// Step 2: Create dataset with selected features
const X_selected = new Float64Array(nSamples * fsResult.selectedFeatures.length);
for (let i = 0; i < nSamples; i++) {
  for (let j = 0; j < fsResult.selectedFeatures.length; j++) {
    X_selected[i * fsResult.selectedFeatures.length + j] =
      X[i * nFeatures + fsResult.selectedFeatures[j]];
  }
}

// Step 3: Scale features
const X_scaled = await standardScaler(
  X_selected,
  nSamples,
  fsResult.selectedFeatures.length
);

// Step 4: Train regression model
const model = await linearRegression(
  X_scaled,
  y,
  nSamples,
  fsResult.selectedFeatures.length
);

// Step 5: Evaluate
const y_pred = X_scaled.map((_, i) => {
  const row = X_scaled.slice(
    i * fsResult.selectedFeatures.length,
    (i + 1) * fsResult.selectedFeatures.length
  );
  return model.predict(row);
});

const r2 = await r2Score(y, y_pred);
const error = await rmse(y, y_pred);

console.log('R² Score:', r2);        // e.g., 0.95
console.log('RMSE:', error);          // e.g., 25.5 (thousands)

// Step 6: Predict new house
const newHouse = new Float64Array([
  2200,  // sqft
  3,     // bedrooms
  500,   // garage_size
  8      // rooms
]);
const newHouseScaled = await standardScaler(
  newHouse,
  1,
  fsResult.selectedFeatures.length
);
const predictedPrice = model.predict(newHouseScaled);
console.log('Predicted price:', predictedPrice, 'thousands');
```

---

## Clustering with Optimization

### Problem: Customer Segmentation

```js
import {
  kmeansPlus,
  standardScaler,
  silhouetteScore,
  dbscan
} from 'miniml';

// Sample customer data
// Features: [annual_income, spending_score, age, purchase_frequency]
const X = new Float64Array([
  50, 60, 25, 12,
  80, 40, 35, 8,
  30, 80, 22, 15,
  120, 20, 45, 4,
  60, 50, 30, 10,
  40, 70, 28, 14,
  100, 30, 40, 6,
  25, 85, 20, 16
]);

const nSamples = 8;
const nFeatures = 4;

// Step 1: Normalize features
const X_scaled = await standardScaler(X, nSamples, nFeatures);

// Step 2: Find optimal number of clusters
const maxClusters = 5;
const bestScore = { k: 0, score: -1 };

for (let k = 2; k <= maxClusters; k++) {
  const kmeans = await kmeansPlus(X_scaled, k, 100, nSamples, nFeatures);
  const labels = kmeans.getAssignments();
  const score = await silhouetteScore(X_scaled, labels, nSamples, nFeatures);

  console.log(`k=${k}: silhouette score=${score.toFixed(3)}`);

  if (score > bestScore.score) {
    bestScore.k = k;
    bestScore.score = score;
  }
}

console.log('Optimal clusters:', bestScore.k);

// Step 3: Final clustering with optimal k
const finalKmeans = await kmeansPlus(
  X_scaled,
  bestScore.k,
  100,
  nSamples,
  nFeatures
);

const labels = finalKmeans.getAssignments();
const centroids = finalKmeans.getCentroids();

console.log('Cluster assignments:', labels);
console.log('Cluster centroids:', centroids);

// Step 4: Interpret clusters
for (let c = 0; c < bestScore.k; c++) {
  const clusterMembers = labels.filter(l => l === c).length;
  console.log(`Cluster ${c}: ${clusterMembers} customers`);

  // Get centroid values (in original scale)
  const centroid = centroids[c];
  console.log(`  Avg income: ${centroid[0].toFixed(0)}`);
  console.log(`  Avg spending: ${centroid[1].toFixed(0)}`);
  console.log(`  Avg age: ${centroid[2].toFixed(0)}`);
}

// Step 5: Alternative: DBSCAN for density-based clustering
const dbLabels = await dbscan(X_scaled, nFeatures, 0.5, 2);
console.log('DBSCAN labels:', dbLabels);
```

---

## Time Series Forecasting

### Problem: Sales Forecasting

```js
import {
  sma,
  ema,
  peakDetection,
  linearRegression,
  exponentialRegression
} from 'miniml';

// Monthly sales data (12 months)
const sales = new Float64Array([
  100, 110, 105, 120, 125, 130,
  140, 135, 150, 160, 155, 170
]);

// Step 1: Moving average smoothing
const window = 3;
const smoothed = await sma(sales, window);
console.log('Smoothed sales:', smoothed);

// Step 2: Exponential moving average (more weight to recent)
const emaSmoothed = await ema(sales, 3);
console.log('EMA smoothed:', emaSmoothed);

// Step 3: Detect peaks (high sales periods)
const peaks = await peakDetection(sales, 2, 5);
console.log('Peak months:', peaks);  // Indices of peaks

// Step 4: Trend analysis with linear regression
const months = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
const trendModel = await linearRegression(months, sales, 12, 1);

// Forecast next 3 months
const forecast = [];
for (let m = 13; m <= 15; m++) {
  const prediction = trendModel.predict(new Float64Array([m]));
  forecast.push(prediction);
}

console.log('Forecast:', forecast);

// Step 5: Exponential growth fit (if accelerating)
const expModel = await exponentialRegression(months, sales, 12);
const expForecast = [];
for (let m = 13; m <= 15; m++) {
  const prediction = expModel.predict(new Float64Array([m]));
  expForecast.push(prediction);
}

console.log('Exponential forecast:', expForecast);

// Step 6: Compare models
console.log('Linear trend:', trendModel.coefficients);
console.log('Exponential growth:', expModel.coefficients);
```

---

## Anomaly Detection

### Problem: Fraud Detection

```js
import {
  isolationForest,
  zScoreOutliers,
  iqrOutliers,
  standardScaler
} from 'miniml';

// Transaction data
// Features: [amount, frequency, merchant_category, time_of_day, day_of_week]
const X = new Float64Array([
  50, 5, 1, 14, 1,     // Normal transaction
  25, 3, 2, 10, 2,
  75, 8, 1, 15, 1,
  5000, 1, 5, 3, 7,    // Anomaly: very high amount
  45, 6, 1, 13, 1,
  30, 4, 2, 11, 2,
  60, 7, 1, 16, 1,
  10000, 2, 4, 2, 6,  // Anomaly: extremely high amount
  55, 5, 1, 14, 1,
  35, 3, 2, 12, 2
]);

const nSamples = 10;
const nFeatures = 5;

// Step 1: Normalize features
const X_scaled = await standardScaler(X, nSamples, nFeatures);

// Step 2: Isolation Forest for anomaly detection
const isoForest = await isolationForest(X_scaled, {
  nTrees: 100,
  maxSamples: 256,
  contamination: 0.2  // Expect ~20% anomalies
});

const predictions = await isoForest.predict(X_scaled);
const scores = await isoForest.scoreSamples(X_scaled);

console.log('Anomaly predictions:', predictions);  // 1 = normal, -1 = anomaly
console.log('Anomaly scores:', scores);

// Step 3: Identify anomalies
const anomalies = [];
for (let i = 0; i < predictions.length; i++) {
  if (predictions[i] === -1) {
    anomalies.push({ index: i, score: scores[i] });
  }
}

console.log('Detected anomalies:', anomalies);

// Step 4: Statistical outlier detection on amount column
const amounts = X.filter((_, i) => i % nFeatures === 0);
const zScoreOutliersList = await zScoreOutliers(amounts, {
  threshold: 2,  // 2 standard deviations
  method: 'median'
});

console.log('Z-score outliers (amounts):', zScoreOutliersList);

// Step 5: IQR method for robust outlier detection
const iqrOutliersList = await iqrOutliers(amounts, {
  multiplier: 1.5
});

console.log('IQR outliers (amounts):', iqrOutliersList);

// Step 6: Real-time fraud check
async function checkTransaction(transaction) {
  const score = await isoForest.scoreSamples(transaction);
  const isAnomaly = score < 0;  // Negative score = anomaly

  return {
    isAnomaly,
    confidence: Math.abs(score),
    recommendation: isAnomaly ? 'FLAG FOR REVIEW' : 'APPROVE'
  };
}

const newTransaction = new Float64Array([8000, 1, 5, 4, 7]);
const result = await checkTransaction(newTransaction);
console.log('Transaction check:', result);
```

---

## Parallel Processing with Workers

### Problem: Large Dataset Training

```js
import { createWorkerPool, parallelMap, parallelCrossValidate } from 'miniml/worker';

// Create worker pool
const numWorkers = navigator.hardwareConcurrency || 4;
const workers = createWorkerPool(numWorkers);

try {
  // Large dataset
  const nSamples = 10000;
  const nFeatures = 50;
  const X = new Float64Array(nSamples * nFeatures);
  const y = new Float64Array(nSamples);

  // Generate synthetic data
  for (let i = 0; i < nSamples * nFeatures; i++) {
    X[i] = Math.random();
  }
  for (let i = 0; i < nSamples; i++) {
    y[i] = Math.random() > 0.5 ? 1 : 0;
  }

  // Parallel cross-validation
  const cvScores = await parallelCrossValidate(
    workers,
    X, y,
    5,  // 5-fold CV
    async (X_train, y_train) => {
      // Training function
      return await trainRandomForest(X_train, y_train, 50, 10);
    },
    async (model, X_test) => {
      // Prediction function
      return await model.predictBatch(X_test);
    }
  );

  console.log('CV Scores:', cvScores);
  console.log('Mean accuracy:', cvScores.reduce((a, b) => a + b) / cvScores.length);

  // Parallel feature computation
  const features = await parallelMap(
    workers,
    X,
    async (row) => {
      // Compute feature for each row
      return row.reduce((sum, val) => sum + val, 0) / row.length;
    }
  );

  console.log('Computed features:', features.slice(0, 10));

} finally {
  // Always clean up workers
  workers.forEach(w => w.terminate());
}
```

---

## Complete ML Pipeline Example

### Problem: End-to-End Customer Lifetime Value Prediction

```js
import {
  standardScaler,
  geneticFeatureSelection,
  trainTestSplit,
  ridgeRegression,
  r2Score,
  mae,
  autoFit
} from 'miniml';

// Customer data
// Features: [age, tenure, monthly_spend, total_spend, frequency, avg_order_value, days_since_last_purchase, complaints, support_calls, n_products]
const X = new Float64Array([
  35, 24, 150, 3600, 12, 300, 5, 0, 1, 3,
  45, 48, 200, 9600, 24, 400, 2, 1, 3, 5,
  28, 12, 100, 1200, 6, 200, 10, 0, 0, 2,
  52, 60, 250, 15000, 30, 500, 1, 0, 2, 6,
  38, 30, 180, 5400, 15, 360, 3, 0, 1, 4,
  31, 18, 120, 2160, 9, 240, 8, 2, 4, 2
]);

// Target: customer lifetime value (in thousands)
const y = new Float64Array([5.2, 12.8, 2.1, 18.5, 7.3, 3.0]);

const nSamples = 6;
const nFeatures = 10;

async function predictCLV() {
  console.log('=== Customer Lifetime Value Prediction ===\n');

  // Step 1: Feature selection
  console.log('Step 1: Feature Selection');
  const fsResult = await geneticFeatureSelection(X, y, {
    populationSize: 30,
    generations: 50,
    cvFolds: 3,
    scoringMetric: 'r2',
    progressCallback: (update) => {
      console.log(`  Generation ${update.generation}: ${update.bestScore.toFixed(3)}`);
    }
  });

  console.log(`  Selected ${fsResult.selectedFeatures.length}/${nFeatures} features`);
  console.log(`  Original R²: ${fsResult.originalScore.toFixed(3)}`);
  console.log(`  Selected R²: ${fsResult.fitnessScore.toFixed(3)}\n`);

  // Step 2: Create dataset with selected features
  const X_selected = new Float64Array(nSamples * fsResult.selectedFeatures.length);
  for (let i = 0; i < nSamples; i++) {
    for (let j = 0; j < fsResult.selectedFeatures.length; j++) {
      X_selected[i * fsResult.selectedFeatures.length + j] =
        X[i * nFeatures + fsResult.selectedFeatures[j]];
    }
  }

  // Step 3: Split data
  console.log('Step 2: Train/Test Split');
  const split = await trainTestSplit(X_selected, y, 0.67, 42);
  console.log(`  Train: ${split.X_train.length / fsResult.selectedFeatures.length} samples`);
  console.log(`  Test: ${split.X_test.length / fsResult.selectedFeatures.length} samples\n`);

  // Step 4: Scale features
  console.log('Step 3: Feature Scaling');
  const scaler = await standardScaler(
    split.X_train,
    split.X_train.length / fsResult.selectedFeatures.length,
    fsResult.selectedFeatures.length
  );

  // Step 5: Train model
  console.log('Step 4: Model Training');
  const model = await ridgeRegression(
    scaler,
    split.y_train,
    1.0,  // alpha (regularization)
    split.X_train.length / fsResult.selectedFeatures.length,
    fsResult.selectedFeatures.length
  );

  // Step 6: Evaluate
  console.log('\nStep 5: Model Evaluation');
  const y_pred = split.X_test.map((_, i) => {
    const row = split.X_test.slice(
      i * fsResult.selectedFeatures.length,
      (i + 1) * fsResult.selectedFeatures.length
    );
    return model.predict(row);
  });

  const r2 = await r2Score(split.y_test, y_pred);
  const maeValue = await mae(split.y_test, y_pred);

  console.log(`  R² Score: ${r2.toFixed(3)}`);
  console.log(`  MAE: ${maeValue.toFixed(2)} (thousands)\n`);

  // Step 7: Predict new customer
  console.log('Step 6: New Customer Prediction');
  const newCustomer = new Float64Array([
    40,  // age
    36,  // tenure
    175, // monthly_spend
    6300, // total_spend
    18,  // frequency
    350, // avg_order_value
    4,   // days_since_last_purchase
    0,   // complaints
    2,   // support_calls
    4    // n_products
  ]);

  const newCustomerSelected = fsResult.selectedFeatures.map(i => newCustomer[i]);
  const prediction = model.predict(new Float64Array(newCustomerSelected));

  console.log(`  Predicted CLV: ${prediction.toFixed(1)}k`);
  console.log(`  Expected range: ${(prediction - maeValue).toFixed(1)}k - ${(prediction + maeValue).toFixed(1)}k`);

  // Step 8: Feature importance (from coefficients)
  console.log('\nStep 7: Feature Importance');
  const importance = fsResult.selectedFeatures.map((featureIdx, i) => ({
    feature: ['age', 'tenure', 'monthly_spend', 'total_spend', 'frequency',
              'avg_order_value', 'days_since_last_purchase', 'complaints',
              'support_calls', 'n_products'][featureIdx],
    importance: Math.abs(model.coefficients[i])
  })).sort((a, b) => b.importance - a.importance);

  importance.slice(0, 5).forEach((feat, i) => {
    console.log(`  ${i + 1}. ${feat.feature}: ${feat.importance.toFixed(3)}`);
  });

  return {
    model,
    selectedFeatures: fsResult.selectedFeatures,
    r2,
    mae: maeValue
  };
}

// Run the pipeline
predictCLV().then(result => {
  console.log('\n=== Pipeline Complete ===');
  console.log('Model ready for deployment!');
});
```

---

## Tips for Best Results

### 1. Always Scale Features

```js
import { standardScaler } from 'miniml';

const X_scaled = await standardScaler(X, nSamples, nFeatures);
// Most algorithms perform better with scaled data
```

### 2. Use Cross-Validation

```js
import { crossValScore } from 'miniml';

const scores = await crossValScore(model, X, y, 5);
console.log('CV scores:', scores);
console.log('Mean:', scores.reduce((a, b) => a + b) / scores.length);
```

### 3. Try AutoML First

```js
// Let AutoML find the best algorithm
const model = await autoFit(X, y);

// Then tune manually if needed
const customModel = await trainAlgorithm(model.algorithm, X, y, {
  ...model.hyperparameters,
  customParam: value
});
```

### 4. Monitor Long Operations

```js
const model = await autoFit(X, y, {
  progressCallback: (update) => {
    console.log(`Progress: ${update.percent}%`);
    if (update.phase === 'complete') {
      console.log('Done in', update.elapsedTime, 'ms');
    }
  }
});
```

### 5. Handle Errors Gracefully

```js
try {
  const model = await trainModel(X, y);
} catch (error) {
  if (error.message.includes('singular matrix')) {
    console.log('Data has collinear features, try feature selection');
  } else if (error.message.includes('convergence')) {
    console.log('Increase max iterations or learning rate');
  }
}
```
