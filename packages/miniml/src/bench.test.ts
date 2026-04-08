/**
 * Performance Benchmarks for micro-ml
 *
 * Measures execution time for all ML algorithms across various workloads.
 * Target times are based on the optimization plan goals.
 */

import { describe, it, expect } from 'vitest';
import * as ml from './index';

// Utility to measure execution time
function benchmark<T>(fn: () => T): { result: T; elapsed: number } {
  const start = performance.now();
  const result = fn();
  const elapsed = performance.now() - start;
  return { result, elapsed };
}

// Generate test data
function generateData(nSamples: number, nFeatures: number): number[] {
  return Array.from({ length: nSamples * nFeatures }, () => Math.random());
}

function generateLabels(nSamples: number, nClasses: number): number[] {
  return Array.from({ length: nSamples }, () => Math.floor(Math.random() * nClasses));
}

// Store benchmark results
const results: Record<string, number> = {};

describe('Performance Benchmarks', () => {
  describe('Classification Algorithms', () => {
    it('KNN: 1000 samples × 100 features, prediction <10ms', async () => {
      const nSamples = 1000;
      const nFeatures = 100;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 3);

      await ml.knnTrain(x, y, nSamples, nFeatures, 5);

      const testPoint = generateData(1, nFeatures);
      const { elapsed } = benchmark(() => ml.knnPredict(testPoint));

      results['KNN prediction'] = elapsed;
      console.log(`  KNN prediction: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(10);
    });

    it('Logistic Regression: 1000 samples × 50 features, training <100ms', async () => {
      const nSamples = 1000;
      const nFeatures = 50;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 2);

      const { elapsed } = benchmark(() =>
        ml.logisticRegression(x, y, nSamples, nFeatures, 1000, 0.01)
      );

      results['Logistic Regression training'] = elapsed;
      console.log(`  Logistic Regression training: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });

    it('Decision Tree: 1000 samples × 20 features, training <50ms', async () => {
      const nSamples = 1000;
      const nFeatures = 20;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 3);

      const { elapsed } = benchmark(() =>
        ml.decisionTreeClassify(x, y, nFeatures, 10)
      );

      results['Decision Tree training'] = elapsed;
      console.log(`  Decision Tree training: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(50);
    });

    it('Random Forest: 1000 samples × 20 features, 100 trees <500ms', async () => {
      const nSamples = 1000;
      const nFeatures = 20;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 3);

      const { elapsed } = benchmark(() =>
        ml.randomForestClassify(x, y, 100, 10)
      );

      results['Random Forest training'] = elapsed;
      console.log(`  Random Forest training (100 trees): ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(500);
    });

    it('Gradient Boosting: 500 samples × 10 features, 50 trees <200ms', async () => {
      const nSamples = 500;
      const nFeatures = 10;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 2);

      const { elapsed } = benchmark(() =>
        ml.gradientBoostingClassify(x, y, 50, 5, 0.1)
      );

      results['Gradient Boosting training'] = elapsed;
      console.log(`  Gradient Boosting training (50 trees): ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(200);
    });

    it('Naive Bayes: 1000 samples × 100 features, training <20ms', async () => {
      const nSamples = 1000;
      const nFeatures = 100;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 3);

      const { elapsed } = benchmark(() =>
        ml.naiveBayesClassify(x, y, nSamples, nFeatures)
      );

      results['Naive Bayes training'] = elapsed;
      console.log(`  Naive Bayes training: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(20);
    });

    it('AdaBoost: 500 samples × 10 features, 50 estimators <150ms', async () => {
      const nSamples = 500;
      const nFeatures = 10;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 2);

      const { elapsed } = benchmark(() =>
        ml.adaboostClassify(x, y, 50)
      );

      results['AdaBoost training'] = elapsed;
      console.log(`  AdaBoost training (50 estimators): ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(150);
    });

    it('SVM: 500 samples × 20 features, training <100ms', async () => {
      const nSamples = 500;
      const nFeatures = 20;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 2);

      const { elapsed } = benchmark(() =>
        ml.linearSVM(x, y, 0.01, 1000, nSamples, nFeatures)
      );

      results['Linear SVM training'] = elapsed;
      console.log(`  Linear SVM training: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });

    it('Perceptron: 1000 samples × 50 features, training <50ms', async () => {
      const nSamples = 1000;
      const nFeatures = 50;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 2);

      const { elapsed } = benchmark(() =>
        ml.perceptronTrain(x, y, nSamples, nFeatures, 0.01, 1000)
      );

      results['Perceptron training'] = elapsed;
      console.log(`  Perceptron training: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(50);
    });
  });

  describe('Regression Algorithms', () => {
    it('Linear Regression: 1000 samples × 50 features <50ms', async () => {
      const nSamples = 1000;
      const nFeatures = 50;
      const x = generateData(nSamples, nFeatures);
      const y = Array.from({ length: nSamples }, () => Math.random() * 100);

      const { elapsed } = benchmark(() =>
        ml.linearRegression(x, y, nSamples, nFeatures)
      );

      results['Linear Regression training'] = elapsed;
      console.log(`  Linear Regression training: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(50);
    });

    it('Ridge Regression: 1000 samples × 50 features <50ms', async () => {
      const nSamples = 1000;
      const nFeatures = 50;
      const x = generateData(nSamples, nFeatures);
      const y = Array.from({ length: nSamples }, () => Math.random() * 100);

      const { elapsed } = benchmark(() =>
        ml.ridgeRegression(x, y, 1.0, nSamples, nFeatures)
      );

      results['Ridge Regression training'] = elapsed;
      console.log(`  Ridge Regression training: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(50);
    });

    it('Polynomial Regression: 500 samples × 5 features, degree 3 <100ms', async () => {
      const nSamples = 500;
      const nFeatures = 5;
      const x = generateData(nSamples, nFeatures);
      const y = Array.from({ length: nSamples }, () => Math.random() * 100);

      const { elapsed } = benchmark(() =>
        ml.polynomialRegression(x, y, nSamples, nFeatures, 3)
      );

      results['Polynomial Regression training'] = elapsed;
      console.log(`  Polynomial Regression training (degree 3): ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });
  });

  describe('Clustering Algorithms', () => {
    it('K-Means: 1000 samples × 20 features, 10 clusters <100ms', async () => {
      const nSamples = 1000;
      const nFeatures = 20;
      const x = generateData(nSamples, nFeatures);

      const { elapsed } = benchmark(() =>
        ml.kmeans(x, nFeatures, 10, 100, nSamples)
      );

      results['K-Means clustering'] = elapsed;
      console.log(`  K-Means clustering: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });

    it('K-Means++: 1000 samples × 20 features, 10 clusters <150ms', async () => {
      const nSamples = 1000;
      const nFeatures = 20;
      const x = generateData(nSamples, nFeatures);

      const { elapsed } = benchmark(() =>
        ml.kmeansPlus(x, 10, 100, nSamples, nFeatures)
      );

      results['K-Means++ clustering'] = elapsed;
      console.log(`  K-Means++ clustering: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(150);
    });

    it('DBSCAN: 500 samples × 10 features <100ms', async () => {
      const nSamples = 500;
      const nFeatures = 10;
      const x = generateData(nSamples, nFeatures);

      const { elapsed } = benchmark(() =>
        ml.dbscan(x, nFeatures, 0.5, 5)
      );

      results['DBSCAN clustering'] = elapsed;
      console.log(`  DBSCAN clustering: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });

    it('Hierarchical Clustering: 500 samples × 10 features <100ms', async () => {
      const nSamples = 500;
      const nFeatures = 10;
      const x = generateData(nSamples, nFeatures);

      const { elapsed } = benchmark(() =>
        ml.hierarchicalClustering(x, nFeatures, 5)
      );

      results['Hierarchical clustering'] = elapsed;
      console.log(`  Hierarchical clustering: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });
  });

  describe('Dimensionality Reduction', () => {
    it('PCA: 1000 samples × 50 features → 10 components <100ms', async () => {
      const nSamples = 1000;
      const nFeatures = 50;
      const x = generateData(nSamples, nFeatures);

      const { elapsed } = benchmark(() =>
        ml.pca(x, nSamples, nFeatures, 10)
      );

      results['PCA'] = elapsed;
      console.log(`  PCA: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });
  });

  describe('Preprocessing', () => {
    it('Standard Scaler: 1000 samples × 100 features <20ms', async () => {
      const nSamples = 1000;
      const nFeatures = 100;
      const x = generateData(nSamples, nFeatures);

      const { elapsed } = benchmark(() =>
        ml.standardScaler(x, nSamples, nFeatures)
      );

      results['Standard Scaler'] = elapsed;
      console.log(`  Standard Scaler: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(20);
    });

    it('Min-Max Scaler: 1000 samples × 100 features <20ms', async () => {
      const nSamples = 1000;
      const nFeatures = 100;
      const x = generateData(nSamples, nFeatures);

      const { elapsed } = benchmark(() =>
        ml.minMaxScaler(x, nSamples, nFeatures)
      );

      results['Min-Max Scaler'] = elapsed;
      console.log(`  Min-Max Scaler: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(20);
    });

    it('Label Encoder: 1000 samples <10ms', async () => {
      const nSamples = 1000;
      const y = Array.from({ length: nSamples }, () => Math.floor(Math.random() * 10));

      const { elapsed } = benchmark(() =>
        ml.labelEncoder(y)
      );

      results['Label Encoder'] = elapsed;
      console.log(`  Label Encoder: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(10);
    });

    it('One-Hot Encoder: 500 samples, 5 classes <20ms', async () => {
      const nSamples = 500;
      const y = Array.from({ length: nSamples }, () => Math.floor(Math.random() * 5));

      const { elapsed } = benchmark(() =>
        ml.oneHotEncoder(y, 5)
      );

      results['One-Hot Encoder'] = elapsed;
      console.log(`  One-Hot Encoder: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(20);
    });
  });

  describe('Metrics', () => {
    it('Confusion Matrix: 1000 samples <10ms', async () => {
      const nSamples = 1000;
      const yTrue = generateLabels(nSamples, 3);
      const yPred = generateLabels(nSamples, 3);

      const { elapsed } = benchmark(() =>
        ml.confusionMatrix(yTrue, yPred)
      );

      results['Confusion Matrix'] = elapsed;
      console.log(`  Confusion Matrix: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(10);
    });

    it('Silhouette Score: 500 samples × 10 features <50ms', async () => {
      const nSamples = 500;
      const nFeatures = 10;
      const x = generateData(nSamples, nFeatures);
      const labels = generateLabels(nSamples, 3);

      const { elapsed } = benchmark(() =>
        ml.silhouetteScore(x, labels, nSamples, nFeatures)
      );

      results['Silhouette Score'] = elapsed;
      console.log(`  Silhouette Score: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(50);
    });
  });

  describe('Model Selection', () => {
    it('Train-Test Split: 1000 samples <10ms', async () => {
      const nSamples = 1000;
      const nFeatures = 20;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 3);

      const { elapsed } = benchmark(() =>
        ml.trainTestSplit(x, y, 0.2, nSamples, nFeatures)
      );

      results['Train-Test Split'] = elapsed;
      console.log(`  Train-Test Split: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(10);
    });

    it('Cross-Validation: 500 samples × 10 features, 5 folds <100ms', async () => {
      const nSamples = 500;
      const nFeatures = 10;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 2);

      // Use a simple model for CV
      const { result: model } = benchmark(() =>
        ml.logisticRegression(x, y, nSamples, nFeatures, 100, 0.01)
      );

      const { elapsed } = benchmark(() =>
        ml.crossValidateScore(x, y, 5, nSamples, nFeatures)
      );

      results['Cross-Validation (5-fold)'] = elapsed;
      console.log(`  Cross-Validation (5-fold): ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });
  });

  describe('Time Series', () => {
    it('Exponential Smoothing: 500 samples <20ms', async () => {
      const nSamples = 500;
      const data = Array.from({ length: nSamples }, () => Math.random() * 100);

      const { elapsed } = benchmark(() =>
        ml.exponentialSmoothing(data, 0.5)
      );

      results['Exponential Smoothing'] = elapsed;
      console.log(`  Exponential Smoothing: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(20);
    });

    it('Moving Average: 500 samples, window 10 <10ms', async () => {
      const nSamples = 500;
      const data = Array.from({ length: nSamples }, () => Math.random() * 100);

      const { elapsed } = benchmark(() =>
        ml.movingAverage(data, 10)
      );

      results['Moving Average'] = elapsed;
      console.log(`  Moving Average: ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(10);
    });
  });

  describe('Feature Importance', () => {
    it('Permutation Importance: 500 samples × 20 features <100ms', async () => {
      const nSamples = 500;
      const nFeatures = 20;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 2);

      const { result: model } = benchmark(() =>
        ml.randomForestClassify(x, y, 20, 5)
      );

      const { elapsed } = benchmark(() =>
        ml.featureImportanceForest(model, x, y, nSamples, nFeatures)
      );

      results['Feature Importance (Forest)'] = elapsed;
      console.log(`  Feature Importance (Forest): ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(100);
    });
  });

  describe('AutoML', () => {
    it('AutoFit: 500 samples × 10 features <2s', async () => {
      const nSamples = 500;
      const nFeatures = 10;
      const x = generateData(nSamples, nFeatures);
      const y = generateLabels(nSamples, 2);

      const { elapsed } = benchmark(() =>
        ml.autoFit(x, y)
      );

      results['AutoML (autoFit)'] = elapsed;
      console.log(`  AutoML (autoFit): ${elapsed.toFixed(2)}ms`);
      expect(elapsed).toBeLessThan(2000);
    });
  });
});

// Print summary after all tests
describe('Benchmark Summary', () => {
  it('should print all results', () => {
    console.log('\n╔════════════════════════════════════════════════════════╗');
    console.log('║         Performance Benchmark Results                   ║');
    console.log('╚════════════════════════════════════════════════════════╝\n');

    const categories = {
      'Classification': [
        'KNN prediction',
        'Logistic Regression training',
        'Decision Tree training',
        'Random Forest training',
        'Gradient Boosting training',
        'Naive Bayes training',
        'AdaBoost training',
        'Linear SVM training',
        'Perceptron training',
      ],
      'Regression': [
        'Linear Regression training',
        'Ridge Regression training',
        'Polynomial Regression training',
      ],
      'Clustering': [
        'K-Means clustering',
        'K-Means++ clustering',
        'DBSCAN clustering',
        'Hierarchical clustering',
      ],
      'Preprocessing': [
        'Standard Scaler',
        'Min-Max Scaler',
        'Label Encoder',
        'One-Hot Encoder',
      ],
      'Other': [
        'PCA',
        'Confusion Matrix',
        'Silhouette Score',
        'Train-Test Split',
        'Cross-Validation (5-fold)',
        'Exponential Smoothing',
        'Moving Average',
        'Feature Importance (Forest)',
        'AutoML (autoFit)',
      ],
    };

    for (const [category, metrics] of Object.entries(categories)) {
      console.log(`\n${category}:`);
      for (const metric of metrics) {
        if (metric in results) {
          const time = results[metric];
          const status = time < 100 ? '✅' : time < 500 ? '⚠️' : '❌';
          console.log(`  ${status} ${metric}: ${time.toFixed(2)}ms`);
        }
      }
    }

    console.log('\n✅ = Fast (<100ms)');
    console.log('⚠️  = Moderate (100-500ms)');
    console.log('❌ = Slow (>500ms)');
    console.log('');

    const times = Object.values(results);
    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const maxTime = Math.max(...times);
    const minTime = Math.min(...times);

    console.log(`Summary:`);
    console.log(`  Average: ${avgTime.toFixed(2)}ms`);
    console.log(`  Min: ${minTime.toFixed(2)}ms`);
    console.log(`  Max: ${maxTime.toFixed(2)}ms`);
    console.log(`  Total benchmarks: ${times.length}`);

    expect(true).toBe(true); // Always pass, just for summary display
  });
});
