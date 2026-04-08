export * from './types.js';

import type {
  LinearModel,
  PolynomialModel,
  ExponentialModel,
  LogarithmicModel,
  PowerModel,
  TrendAnalysis,
  MovingAverageOptions,
  PolynomialOptions,
  SmoothingOptions,
  ErrorMetrics,
  ResidualsResult,
  NormalizedData,
  NormalizationType,
  KMeansModel,
  KMeansOptions,
  KnnModel,
  KnnOptions,
  LogisticModel,
  LogisticRegressionOptions,
  DbscanResult,
  DbscanOptions,
  NaiveBayesModel,
  DecisionTreeModel,
  DecisionTreeOptions,
  PcaResult,
  PcaOptions,
  PerceptronModel,
  PerceptronOptions,
  SeasonalDecomposition,
  SeasonalityInfo,
  AutoMLResult,
} from './types.js';

// WASM module instance (lazily loaded)
let wasmModule: typeof import('../wasm/miniml_core.js') | null = null;
let initPromise: Promise<void> | null = null;

/**
 * Initialize the WASM module
 * This is called automatically when using any function, but can be called
 * explicitly for eager loading.
 */
export async function init(): Promise<void> {
  if (wasmModule) return;

  if (!initPromise) {
    initPromise = (async () => {
      const mod = await import('../wasm/miniml_core.js');

      // Check if we're in Node.js
      const isNode = typeof globalThis.process !== 'undefined' &&
                     typeof globalThis.process.versions?.node === 'string';

      if (isNode) {
        // Node.js: read WASM file directly
        try {
          const { readFileSync } = await import('fs');
          const { fileURLToPath } = await import('url');
          const { dirname, join } = await import('path');

          const __filename = fileURLToPath(import.meta.url);
          const __dirname = dirname(__filename);
          const { existsSync } = await import('fs');
          // Try dist/ (npm package) then wasm/ (dev)
          const distPath = join(__dirname, 'miniml_core_bg.wasm');
          const devPath = join(__dirname, '..', 'wasm', 'miniml_core_bg.wasm');
          const wasmPath = existsSync(distPath) ? distPath : devPath;

          mod.initSync({ module: readFileSync(wasmPath) });
        } catch {
          await mod.default();
        }
      } else {
        // Browser: use fetch
        await mod.default();
      }

      wasmModule = mod;
    })();
  }

  await initPromise;
}

/**
 * Ensure WASM is loaded before use
 */
async function ensureInit(): Promise<typeof import('../wasm/miniml_core.js')> {
  await init();
  return wasmModule!;
}

// Wrapper to convert WASM model to JS-friendly interface
function wrapLinearModel(wasmModel: any): LinearModel {
  return {
    get slope() { return wasmModel.slope; },
    get intercept() { return wasmModel.intercept; },
    get rSquared() { return wasmModel.rSquared; },
    get n() { return wasmModel.n; },
    predict(x: number[]): number[] {
      return Array.from(wasmModel.predict(new Float64Array(x)));
    },
    toString(): string {
      return wasmModel.toString();
    },
  };
}

function wrapPolynomialModel(wasmModel: any): PolynomialModel {
  return {
    get degree() { return wasmModel.degree; },
    get rSquared() { return wasmModel.rSquared; },
    get n() { return wasmModel.n; },
    getCoefficients(): number[] {
      return Array.from(wasmModel.getCoefficients());
    },
    predict(x: number[]): number[] {
      return Array.from(wasmModel.predict(new Float64Array(x)));
    },
    toString(): string {
      return wasmModel.toString();
    },
  };
}

function wrapExponentialModel(wasmModel: any): ExponentialModel {
  return {
    get a() { return wasmModel.a; },
    get b() { return wasmModel.b; },
    get rSquared() { return wasmModel.rSquared; },
    get n() { return wasmModel.n; },
    predict(x: number[]): number[] {
      return Array.from(wasmModel.predict(new Float64Array(x)));
    },
    toString(): string {
      return wasmModel.toString();
    },
    doublingTime(): number {
      return wasmModel.doublingTime();
    },
  };
}

function wrapLogarithmicModel(wasmModel: any): LogarithmicModel {
  return {
    get a() { return wasmModel.a; },
    get b() { return wasmModel.b; },
    get rSquared() { return wasmModel.rSquared; },
    get n() { return wasmModel.n; },
    predict(x: number[]): number[] {
      return Array.from(wasmModel.predict(new Float64Array(x)));
    },
    toString(): string {
      return wasmModel.toString();
    },
  };
}

function wrapPowerModel(wasmModel: any): PowerModel {
  return {
    get a() { return wasmModel.a; },
    get b() { return wasmModel.b; },
    get rSquared() { return wasmModel.rSquared; },
    get n() { return wasmModel.n; },
    predict(x: number[]): number[] {
      return Array.from(wasmModel.predict(new Float64Array(x)));
    },
    toString(): string {
      return wasmModel.toString();
    },
  };
}

// ============================================================================
// Linear Regression
// ============================================================================

/**
 * Fit a linear regression model: y = slope * x + intercept
 *
 * @example
 * ```ts
 * const model = await linearRegression([1, 2, 3, 4], [2, 4, 6, 8]);
 * console.log(model.slope); // 2
 * console.log(model.intercept); // 0
 * console.log(model.rSquared); // 1
 * const predictions = model.predict([5, 6]); // [10, 12]
 * ```
 */
export async function linearRegression(
  x: number[],
  y: number[]
): Promise<LinearModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.linearRegression(new Float64Array(x), new Float64Array(y));
  return wrapLinearModel(wasmModel);
}

/**
 * Simple linear regression with auto-generated x values (0, 1, 2, ...)
 * Useful for time series data where x is just the index.
 *
 * @example
 * ```ts
 * const model = await linearRegressionSimple([10, 20, 30, 40]);
 * // Equivalent to linearRegression([0, 1, 2, 3], [10, 20, 30, 40])
 * ```
 */
export async function linearRegressionSimple(y: number[]): Promise<LinearModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.linearRegressionSimple(new Float64Array(y));
  return wrapLinearModel(wasmModel);
}

// ============================================================================
// Polynomial Regression
// ============================================================================

/**
 * Fit a polynomial regression model: y = c0 + c1*x + c2*x² + ...
 *
 * @example
 * ```ts
 * const model = await polynomialRegression([0, 1, 2, 3], [1, 2, 5, 10], { degree: 2 });
 * console.log(model.getCoefficients()); // [c0, c1, c2]
 * ```
 */
export async function polynomialRegression(
  x: number[],
  y: number[],
  options: PolynomialOptions = {}
): Promise<PolynomialModel> {
  const wasm = await ensureInit();
  const degree = options.degree ?? 2;
  const wasmModel = wasm.polynomialRegression(new Float64Array(x), new Float64Array(y), degree);
  return wrapPolynomialModel(wasmModel);
}

/**
 * Polynomial regression with auto-generated x values (0, 1, 2, ...)
 */
export async function polynomialRegressionSimple(
  y: number[],
  options: PolynomialOptions = {}
): Promise<PolynomialModel> {
  const wasm = await ensureInit();
  const degree = options.degree ?? 2;
  const wasmModel = wasm.polynomialRegressionSimple(new Float64Array(y), degree);
  return wrapPolynomialModel(wasmModel);
}

// ============================================================================
// Exponential & Logarithmic Regression
// ============================================================================

/**
 * Fit an exponential regression model: y = a * e^(b*x)
 * All y values must be positive.
 *
 * @example
 * ```ts
 * const model = await exponentialRegression([0, 1, 2], [1, 2.7, 7.4]);
 * console.log(model.a); // ~1
 * console.log(model.b); // ~1 (e^1 ≈ 2.718)
 * console.log(model.doublingTime()); // Time to double
 * ```
 */
export async function exponentialRegression(
  x: number[],
  y: number[]
): Promise<ExponentialModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.exponentialRegression(new Float64Array(x), new Float64Array(y));
  return wrapExponentialModel(wasmModel);
}

/**
 * Exponential regression with auto-generated x values (0, 1, 2, ...)
 */
export async function exponentialRegressionSimple(
  y: number[]
): Promise<ExponentialModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.exponentialRegressionSimple(new Float64Array(y));
  return wrapExponentialModel(wasmModel);
}

/**
 * Fit a logarithmic regression model: y = a + b * ln(x)
 * All x values must be positive.
 *
 * @example
 * ```ts
 * const model = await logarithmicRegression([1, 2, 3, 4], [0, 0.69, 1.1, 1.39]);
 * console.log(model.a); // Intercept
 * console.log(model.b); // Coefficient
 * ```
 */
export async function logarithmicRegression(
  x: number[],
  y: number[]
): Promise<LogarithmicModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.logarithmicRegression(new Float64Array(x), new Float64Array(y));
  return wrapLogarithmicModel(wasmModel);
}

/**
 * Fit a power regression model: y = a * x^b
 * All x and y values must be positive.
 *
 * @example
 * ```ts
 * const model = await powerRegression([1, 2, 3, 4], [1, 4, 9, 16]);
 * console.log(model.a); // ~1
 * console.log(model.b); // ~2 (quadratic)
 * ```
 */
export async function powerRegression(
  x: number[],
  y: number[]
): Promise<PowerModel> {
  const wasm = await ensureInit();
  const wasmModel = wasm.powerRegression(new Float64Array(x), new Float64Array(y));
  return wrapPowerModel(wasmModel);
}

// ============================================================================
// Moving Averages
// ============================================================================

/**
 * Calculate a moving average
 *
 * @example
 * ```ts
 * const smoothed = await movingAverage(data, { window: 7, type: 'ema' });
 * ```
 */
export async function movingAverage(
  data: number[],
  options: MovingAverageOptions
): Promise<number[]> {
  const wasm = await ensureInit();
  const type = options.type ?? 'sma';
  const typeEnum = type === 'ema' ? wasm.MovingAverageType.EMA
    : type === 'wma' ? wasm.MovingAverageType.WMA
    : wasm.MovingAverageType.SMA;

  return Array.from(
    wasm.movingAverage(new Float64Array(data), options.window, typeEnum)
  );
}

/**
 * Calculate Simple Moving Average
 */
export async function sma(data: number[], window: number): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.sma(new Float64Array(data), window));
}

/**
 * Calculate Exponential Moving Average
 */
export async function ema(data: number[], window: number): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.ema(new Float64Array(data), window));
}

/**
 * Calculate Weighted Moving Average
 */
export async function wma(data: number[], window: number): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.wma(new Float64Array(data), window));
}

// ============================================================================
// Trend Analysis & Forecasting
// ============================================================================

/**
 * Analyze trend and forecast future values
 *
 * @example
 * ```ts
 * const trend = await trendForecast(data, 10);
 * console.log(trend.direction); // 'up'
 * console.log(trend.slope); // 10
 * console.log(trend.getForecast()); // [50, 60, 70]
 * ```
 */
export async function trendForecast(
  data: number[],
  periods: number
): Promise<TrendAnalysis> {
  const wasm = await ensureInit();
  const result = wasm.trendForecast(new Float64Array(data), periods);

  // Map direction enum to string
  const directionMap: Record<number, 'up' | 'down' | 'flat'> = {
    [wasm.TrendDirection.Up]: 'up',
    [wasm.TrendDirection.Down]: 'down',
    [wasm.TrendDirection.Flat]: 'flat',
  };

  return {
    direction: directionMap[result.direction],
    slope: result.slope,
    strength: result.strength,
    getForecast: () => Array.from(result.getForecast()),
  };
}

/**
 * Calculate rate of change as percentage
 */
export async function rateOfChange(
  data: number[],
  periods: number
): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.rateOfChange(new Float64Array(data), periods));
}

/**
 * Calculate momentum (difference from n periods ago)
 */
export async function momentum(
  data: number[],
  periods: number
): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.momentum(new Float64Array(data), periods));
}

/**
 * Apply exponential smoothing to data
 *
 * @example
 * ```ts
 * const smoothed = await exponentialSmoothing([10, 20, 15, 25], { alpha: 0.3 });
 * ```
 */
export async function exponentialSmoothing(
  data: number[],
  options: SmoothingOptions = {}
): Promise<number[]> {
  const wasm = await ensureInit();
  const alpha = options.alpha ?? 0.3;
  return Array.from(wasm.exponentialSmoothing(new Float64Array(data), alpha));
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Find peaks (local maxima) in data
 * Returns indices of peak values.
 */
export async function findPeaks(data: number[]): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.findPeaks(new Float64Array(data)));
}

/**
 * Find troughs (local minima) in data
 * Returns indices of trough values.
 */
export async function findTroughs(data: number[]): Promise<number[]> {
  const wasm = await ensureInit();
  return Array.from(wasm.findTroughs(new Float64Array(data)));
}

// ============================================================================
// Convenience functions for one-liner usage
// ============================================================================

/**
 * Quick predict: fit model and get predictions in one call
 *
 * @example
 * ```ts
 * const predictions = await predict(xData, yData, [100, 200, 300]);
 * ```
 */
export async function predict(
  xTrain: number[],
  yTrain: number[],
  xPredict: number[]
): Promise<number[]> {
  const model = await linearRegression(xTrain, yTrain);
  return model.predict(xPredict);
}

/**
 * Quick trend line: fit model and extrapolate
 *
 * @example
 * ```ts
 * const future = await trendLine(data, 10); // Next 10 points
 * ```
 */
export async function trendLine(
  data: number[],
  futurePoints: number
): Promise<{ model: LinearModel; trend: number[] }> {
  const model = await linearRegressionSimple(data);
  const futureX = Array.from({ length: futurePoints }, (_, i) => data.length + i);
  const trend = model.predict(futureX);
  return { model, trend };
}

// ============================================================================
// Error Metrics
// ============================================================================

/**
 * Root Mean Squared Error between actual and predicted values
 *
 * @example
 * ```ts
 * const error = rmse([1, 2, 3], [1.1, 2.2, 2.8]); // ~0.173
 * ```
 */
export function rmse(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  const n = actual.length;
  const sumSqErr = actual.reduce(
    (sum, val, i) => sum + (val - predicted[i]) ** 2,
    0
  );
  return Math.sqrt(sumSqErr / n);
}

/**
 * Mean Absolute Error between actual and predicted values
 *
 * @example
 * ```ts
 * const error = mae([1, 2, 3], [1.1, 2.2, 2.8]); // ~0.167
 * ```
 */
export function mae(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  const n = actual.length;
  const sumAbsErr = actual.reduce(
    (sum, val, i) => sum + Math.abs(val - predicted[i]),
    0
  );
  return sumAbsErr / n;
}

/**
 * Mean Absolute Percentage Error between actual and predicted values.
 * Returns value as a percentage (e.g. 5.0 means 5%).
 * Skips data points where actual value is zero.
 *
 * @example
 * ```ts
 * const error = mape([100, 200, 300], [110, 190, 310]); // ~5.56
 * ```
 */
export function mape(actual: number[], predicted: number[]): number {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  let sum = 0;
  let count = 0;
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== 0) {
      sum += Math.abs((actual[i] - predicted[i]) / actual[i]);
      count++;
    }
  }
  if (count === 0) return 0;
  return (sum / count) * 100;
}

/**
 * Compute all error metrics at once
 *
 * @example
 * ```ts
 * const metrics = errorMetrics([1, 2, 3], [1.1, 2.2, 2.8]);
 * console.log(metrics.rmse, metrics.mae, metrics.mape);
 * ```
 */
export function errorMetrics(actual: number[], predicted: number[]): ErrorMetrics {
  return {
    rmse: rmse(actual, predicted),
    mae: mae(actual, predicted),
    mape: mape(actual, predicted),
    n: actual.length,
  };
}

/**
 * Analyze residuals (actual - predicted) for model diagnostics.
 * Useful for checking whether a model's errors are randomly distributed.
 *
 * @example
 * ```ts
 * const result = residuals([1, 2, 3], [1.1, 1.9, 3.2]);
 * console.log(result.mean);         // ~-0.067 (close to 0 = unbiased)
 * console.log(result.standardized); // z-scores of residuals
 * ```
 */
export function residuals(actual: number[], predicted: number[]): ResidualsResult {
  if (actual.length !== predicted.length) {
    throw new Error('Arrays must have the same length');
  }
  const resids = actual.map((val, i) => val - predicted[i]);
  const mean = resids.reduce((a, b) => a + b, 0) / resids.length;
  const variance = resids.reduce((sum, r) => sum + (r - mean) ** 2, 0) / resids.length;
  const stdDev = Math.sqrt(variance);
  const standardized = stdDev === 0
    ? resids.map(() => 0)
    : resids.map(r => (r - mean) / stdDev);

  return { residuals: resids, mean, stdDev, standardized };
}

// ============================================================================
// Data Normalization
// ============================================================================

/**
 * Normalize data using min-max scaling to [0, 1] range.
 * Returns normalized data and an inverse function to restore original scale.
 *
 * @example
 * ```ts
 * const norm = minMaxNormalize([10, 20, 30, 40, 50]);
 * console.log(norm.data); // [0, 0.25, 0.5, 0.75, 1]
 * console.log(norm.inverse([0.5])); // [30]
 * ```
 */
export function minMaxNormalize(data: number[]): NormalizedData {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;

  if (range === 0) {
    return {
      data: data.map(() => 0),
      inverse: (normalized: number[]) => normalized.map(() => min),
    };
  }

  return {
    data: data.map(v => (v - min) / range),
    inverse: (normalized: number[]) => normalized.map(v => v * range + min),
  };
}

/**
 * Normalize data using z-score standardization (mean=0, stddev=1).
 * Returns normalized data and an inverse function to restore original scale.
 *
 * @example
 * ```ts
 * const norm = zScoreNormalize([10, 20, 30, 40, 50]);
 * console.log(norm.data); // [-1.41, -0.71, 0, 0.71, 1.41]
 * ```
 */
export function zScoreNormalize(data: number[]): NormalizedData {
  const mean = data.reduce((a, b) => a + b, 0) / data.length;
  const variance = data.reduce((sum, v) => sum + (v - mean) ** 2, 0) / data.length;
  const stdDev = Math.sqrt(variance);

  if (stdDev === 0) {
    return {
      data: data.map(() => 0),
      inverse: (normalized: number[]) => normalized.map(() => mean),
    };
  }

  return {
    data: data.map(v => (v - mean) / stdDev),
    inverse: (normalized: number[]) => normalized.map(v => v * stdDev + mean),
  };
}

/**
 * Normalize data with the specified method.
 *
 * @example
 * ```ts
 * const norm = normalize([10, 20, 30], 'min-max');
 * const norm2 = normalize([10, 20, 30], 'z-score');
 * ```
 */
export function normalize(data: number[], type: NormalizationType = 'min-max'): NormalizedData {
  return type === 'z-score' ? zScoreNormalize(data) : minMaxNormalize(data);
}

// ============================================================================
// Matrix helpers (internal)
// ============================================================================

function flattenMatrix(matrix: number[][]): { flat: Float64Array; nFeatures: number } {
  const nFeatures = matrix[0].length;
  const flat = new Float64Array(matrix.length * nFeatures);
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < nFeatures; j++) {
      flat[i * nFeatures + j] = matrix[i][j];
    }
  }
  return { flat, nFeatures };
}

function unflattenMatrix(flat: ArrayLike<number>, nFeatures: number): number[][] {
  const n = flat.length / nFeatures;
  const result: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      row.push(flat[i * nFeatures + j]);
    }
    result.push(row);
  }
  return result;
}

// ============================================================================
// K-Means Clustering
// ============================================================================

export async function kmeans(
  data: number[][],
  options: KMeansOptions
): Promise<KMeansModel> {
  const wasm = await ensureInit();
  const { flat, nFeatures } = flattenMatrix(data);
  const maxIter = options.maxIterations ?? 100;
  const model = wasm.kmeans(flat, nFeatures, options.k, maxIter);
  return {
    get k() { return model.k; },
    get iterations() { return model.iterations; },
    get inertia() { return model.inertia; },
    getCentroids() { return unflattenMatrix(model.getCentroids(), model.getNFeatures()); },
    getAssignments() { return Array.from(model.getAssignments()); },
    predict(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      return Array.from(model.predict(f));
    },
    toString() { return model.toString(); },
  };
}

// ============================================================================
// K-Nearest Neighbors
// ============================================================================

export async function knnClassifier(
  data: number[][],
  labels: number[],
  options: KnnOptions = {}
): Promise<KnnModel> {
  const wasm = await ensureInit();
  const { flat, nFeatures } = flattenMatrix(data);
  const k = options.k ?? 3;
  const model = wasm.knnFit(flat, nFeatures, new Float64Array(labels), k);
  return {
    get k() { return model.k; },
    get nSamples() { return model.nSamples; },
    predict(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      return Array.from(model.predict(f));
    },
    predictProba(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      return Array.from(model.predictProba(f));
    },
    toString() { return model.toString(); },
  };
}

// ============================================================================
// Logistic Regression
// ============================================================================

export async function logisticRegression(
  data: number[][],
  labels: number[],
  options: LogisticRegressionOptions = {}
): Promise<LogisticModel> {
  const wasm = await ensureInit();
  const { flat, nFeatures } = flattenMatrix(data);
  const lr = options.learningRate ?? 0.01;
  const maxIter = options.maxIterations ?? 1000;
  const lambda = options.lambda ?? 0.0;
  const model = wasm.logisticRegression(flat, nFeatures, new Float64Array(labels), lr, maxIter, lambda);
  return {
    get bias() { return model.bias; },
    get iterations() { return model.iterations; },
    get loss() { return model.loss; },
    getWeights() { return Array.from(model.getWeights()); },
    predict(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      return Array.from(model.predict(f));
    },
    predictProba(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      return Array.from(model.predictProba(f));
    },
    toString() { return model.toString(); },
  };
}

// ============================================================================
// DBSCAN
// ============================================================================

export async function dbscan(
  data: number[][],
  options: DbscanOptions
): Promise<DbscanResult> {
  const wasm = await ensureInit();
  const { flat, nFeatures } = flattenMatrix(data);
  const minPoints = options.minPoints ?? 5;
  const result = wasm.dbscan(flat, nFeatures, options.eps, minPoints);
  return {
    get nClusters() { return result.nClusters; },
    get nNoise() { return result.nNoise; },
    getLabels() { return Array.from(result.getLabels()); },
    toString() { return result.toString(); },
  };
}

// ============================================================================
// Naive Bayes
// ============================================================================

export async function naiveBayes(
  data: number[][],
  labels: number[]
): Promise<NaiveBayesModel> {
  const wasm = await ensureInit();
  const { flat, nFeatures } = flattenMatrix(data);
  const model = wasm.naiveBayesFit(flat, nFeatures, new Float64Array(labels));
  return {
    get nClasses() { return model.nClasses; },
    get nFeatures() { return model.nFeatures; },
    predict(d: number[][]) {
      const { flat: f, nFeatures: nf } = flattenMatrix(d);
      return Array.from(model.predict(f));
    },
    predictProba(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      const flatProba = Array.from(model.predictProba(f));
      const nClasses = model.nClasses;
      const result: number[][] = [];
      for (let i = 0; i < d.length; i++) {
        result.push(flatProba.slice(i * nClasses, (i + 1) * nClasses));
      }
      return result;
    },
    toString() { return model.toString(); },
  };
}

// ============================================================================
// Decision Tree
// ============================================================================

export async function decisionTree(
  data: number[][],
  targets: number[],
  options: DecisionTreeOptions = {}
): Promise<DecisionTreeModel> {
  const wasm = await ensureInit();
  const { flat, nFeatures } = flattenMatrix(data);
  const maxDepth = options.maxDepth ?? 10;
  const minSplit = options.minSamplesSplit ?? 2;
  const mode = options.mode ?? 'classify';
  const model = mode === 'regress'
    ? wasm.decisionTreeRegress(flat, nFeatures, new Float64Array(targets), maxDepth, minSplit)
    : wasm.decisionTreeClassify(flat, nFeatures, new Float64Array(targets), maxDepth, minSplit);
  return {
    get depth() { return model.depth; },
    get nNodes() { return model.nNodes; },
    predict(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      return Array.from(model.predict(f));
    },
    getTree() { return Array.from(model.getTree()); },
    toString() { return model.toString(); },
  };
}

// ============================================================================
// PCA
// ============================================================================

export async function pca(
  data: number[][],
  options: PcaOptions = {}
): Promise<PcaResult> {
  const wasm = await ensureInit();
  const { flat, nFeatures } = flattenMatrix(data);
  const nComponents = options.nComponents ?? Math.min(nFeatures, data.length, 2);
  const result = wasm.pca(flat, nFeatures, nComponents);
  return {
    get nComponents() { return result.nComponents; },
    getComponents() { return unflattenMatrix(result.getComponents(), nFeatures); },
    getExplainedVariance() { return Array.from(result.getExplainedVariance()); },
    getExplainedVarianceRatio() { return Array.from(result.getExplainedVarianceRatio()); },
    getTransformed() { return unflattenMatrix(result.getTransformed(), nComponents); },
    getMean() { return Array.from(result.getMean()); },
    transform(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      return unflattenMatrix(result.transform(f), nComponents);
    },
    toString() { return result.toString(); },
  };
}

// ============================================================================
// Perceptron
// ============================================================================

export async function perceptron(
  data: number[][],
  labels: number[],
  options: PerceptronOptions = {}
): Promise<PerceptronModel> {
  const wasm = await ensureInit();
  const { flat, nFeatures } = flattenMatrix(data);
  const lr = options.learningRate ?? 0.01;
  const maxIter = options.maxIterations ?? 1000;
  const model = wasm.perceptron(flat, nFeatures, new Float64Array(labels), lr, maxIter);
  return {
    get bias() { return model.bias; },
    get iterations() { return model.iterations; },
    get converged() { return model.converged; },
    getWeights() { return Array.from(model.getWeights()); },
    predict(d: number[][]) {
      const { flat: f } = flattenMatrix(d);
      return Array.from(model.predict(f));
    },
    toString() { return model.toString(); },
  };
}

// ============================================================================
// Seasonality
// ============================================================================

export async function seasonalDecompose(
  data: number[],
  period: number
): Promise<SeasonalDecomposition> {
  const wasm = await ensureInit();
  const result = wasm.seasonalDecompose(new Float64Array(data), period);
  return {
    get period() { return result.period; },
    getTrend() { return Array.from(result.getTrend()); },
    getSeasonal() { return Array.from(result.getSeasonal()); },
    getResidual() { return Array.from(result.getResidual()); },
  };
}

export async function autocorrelation(
  data: number[],
  maxLag?: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const lag = maxLag ?? Math.floor(data.length / 2);
  return Array.from(wasm.autocorrelation(new Float64Array(data), lag));
}

export async function detectSeasonality(
  data: number[]
): Promise<SeasonalityInfo> {
  const wasm = await ensureInit();
  const result = wasm.detectSeasonality(new Float64Array(data));
  return {
    get period() { return result.period; },
    get strength() { return result.strength; },
  };
}

// ============================================================================
// AutoML
// ============================================================================

/**
 * Automated regression: finds best algorithm and optimizes pipeline
 */
export async function autoFitRegression(
  x: number[],
  y: number[],
  nSamples: number,
  nFeatures: number
): Promise<AutoMLResult> {
  const wasm = await ensureInit();
  const result = wasm.auto_fit_regression(new Float64Array(x), new Float64Array(y), nSamples, nFeatures);
  return wrapAutoMLResult(result);
}

/**
 * Automated classification: finds best algorithm and optimizes pipeline
 */
export async function autoFitClassification(
  x: number[],
  y: number[],
  nSamples: number,
  nFeatures: number
): Promise<AutoMLResult> {
  const wasm = await ensureInit();
  const result = wasm.auto_fit_classification(new Float64Array(x), new Float64Array(y), nSamples, nFeatures);
  return wrapAutoMLResult(result);
}

/**
 * Get algorithm recommendation based on data characteristics
 */
export async function recommendAlgorithm(
  nSamples: number,
  nFeatures: number,
  nClasses: number,
  isSparse: boolean
): Promise<string> {
  const wasm = await ensureInit();
  return wasm.recommend_algorithm(nSamples, nFeatures, nClasses, isSparse);
}

// ============================================================================
// Hidden Algorithms - Ensemble Methods
// ============================================================================

/**
 * Random Forest Classification
 * @example
 * ```ts
 * const model = await randomForestClassify(x, y, 100, 10);
 * const predictions = model.predict(testData);
 * ```
 */
export async function randomForestClassify(
  x: number[],
  y: number[],
  nTrees: number,
  maxDepth: number
): Promise<any> {
  const wasm = await ensureInit();
  return wasm.random_forest_classify(new Float64Array(x), new Float64Array(y), nTrees, maxDepth);
}

/**
 * Random Forest Regression
 */
export async function randomForestRegress(
  x: number[],
  y: number[],
  nTrees: number,
  maxDepth: number
): Promise<any> {
  const wasm = await ensureInit();
  return wasm.random_forest_regress(new Float64Array(x), new Float64Array(y), nTrees, maxDepth);
}

/**
 * Gradient Boosting Classification
 */
export async function gradientBoostingClassify(
  x: number[],
  y: number[],
  nEstimators: number,
  learningRate: number,
  maxDepth: number
): Promise<any> {
  const wasm = await ensureInit();
  return wasm.gradient_boosting_classify(
    new Float64Array(x),
    new Float64Array(y),
    nEstimators,
    learningRate,
    maxDepth
  );
}

/**
 * AdaBoost Classification
 */
export async function adaboostClassify(
  x: number[],
  y: number[],
  nEstimators: number
): Promise<any> {
  const wasm = await ensureInit();
  return wasm.adaboost_classify(new Float64Array(x), new Float64Array(y), nEstimators);
}

// ============================================================================
// Hidden Algorithms - Linear Models
// ============================================================================

/**
 * Ridge Regression (L2 regularized)
 */
export async function ridgeRegression(
  x: number[],
  y: number[],
  alpha: number,
  nSamples: number,
  nFeatures: number
): Promise<any> {
  const wasm = await ensureInit();
  return wasm.ridge_regression(new Float64Array(x), new Float64Array(y), alpha, nSamples, nFeatures);
}

/**
 * Lasso Regression (L1 regularized)
 */
export async function lassoRegression(
  x: number[],
  y: number[],
  alpha: number,
  maxIter: number,
  tol: number,
  nSamples: number,
  nFeatures: number
): Promise<any> {
  const wasm = await ensureInit();
  return wasm.lasso_regression(
    new Float64Array(x),
    new Float64Array(y),
    alpha,
    maxIter,
    tol,
    nSamples,
    nFeatures
  );
}

/**
 * Linear SVM (PEGASOS algorithm)
 */
export async function linearSVM(
  x: number[],
  y: number[],
  lambda: number,
  maxIter: number,
  nSamples: number,
  nFeatures: number
): Promise<any> {
  const wasm = await ensureInit();
  return wasm.linear_svm(new Float64Array(x), new Float64Array(y), lambda, maxIter, nSamples, nFeatures);
}

// ============================================================================
// Hidden Algorithms - Clustering
// ============================================================================

/**
 * Hierarchical Clustering (Agglomerative)
 * @example
 * ```ts
 * const labels = await hierarchicalClustering(x, 10, 3);
 * // Returns cluster assignments for 10 samples with 10 features, 3 clusters
 * ```
 */
export async function hierarchicalClustering(
  x: number[],
  nFeatures: number,
  nClusters: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.hierarchical_clustering(new Float64Array(x), nFeatures, nClusters);
  return Array.from(result);
}

/**
 * K-Means++ Clustering (improved initialization)
 */
export async function kmeansPlus(
  x: number[],
  nClusters: number,
  maxIter: number,
  nSamples: number,
  nFeatures: number
): Promise<any> {
  const wasm = await ensureInit();
  return wasm.kmeans_plus(new Float64Array(x), nClusters, maxIter, nSamples, nFeatures);
}

// ============================================================================
// Hidden Algorithms - Preprocessing (Scalers & Encoders)
// ============================================================================

/**
 * Standard Scaler (z-score normalization)
 * @example
 * ```ts
 * const scaled = await standardScaler(x, 100, 5);
 * // Scales to mean=0, std=1
 * ```
 */
export async function standardScaler(
  x: number[],
  nSamples: number,
  nFeatures: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.standard_scaler(new Float64Array(x), nSamples, nFeatures);
  return Array.from(result);
}

/**
 * MinMax Scaler (scales to [0, 1])
 */
export async function minMaxScaler(
  x: number[],
  nSamples: number,
  nFeatures: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.min_max_scaler(new Float64Array(x), nSamples, nFeatures);
  return Array.from(result);
}

/**
 * Robust Scaler (scales using median and IQR, robust to outliers)
 */
export async function robustScaler(
  x: number[],
  nSamples: number,
  nFeatures: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.robust_scaler(new Float64Array(x), nSamples, nFeatures);
  return Array.from(result);
}

/**
 * Normalizer (L2 normalization)
 */
export async function normalizer(x: number[], nSamples: number, nFeatures: number): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.normalizer(new Float64Array(x), nSamples, nFeatures);
  return Array.from(result);
}

/**
 * Label Encoder (converts string labels to integers)
 * @example
 * ```ts
 * const encoded = await labelEncoder(y);
 * // Converts ['cat', 'dog', 'cat'] to [0, 1, 0]
 * ```
 */
export async function labelEncoder(y: number[]): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.label_encoder(new Float64Array(y));
  return Array.from(result);
}

/**
 * One-Hot Encoder (converts integers to binary vectors)
 */
export async function oneHotEncoder(y: number[], nClasses: number): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.one_hot_encoder(new Float64Array(y), nClasses);
  return Array.from(result);
}

/**
 * Ordinal Encoder (converts categories to ordered integers)
 */
export async function ordinalEncoder(y: number[]): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.ordinal_encoder(new Float64Array(y));
  return Array.from(result);
}

/**
 * Simple Imputer (fills missing values with mean/median/mode)
 */
export async function simpleImputer(
  x: number[],
  strategy: 'mean' | 'median' | 'most_frequent',
  nSamples: number,
  nFeatures: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const strategyEnum = strategy === 'mean' ? 0 : strategy === 'median' ? 1 : 2;
  const result = wasm.simple_imputer(new Float64Array(x), strategyEnum, nSamples, nFeatures);
  return Array.from(result);
}

// ============================================================================
// Hidden Algorithms - Metrics
// ============================================================================

/**
 * Confusion Matrix for classification evaluation
 * @example
 * ```ts
 * const matrix = await confusionMatrix(yTrue, yPred);
 * // Returns 2D array [[TN, FP], [FN, TP]] for binary classification
 * ```
 */
export async function confusionMatrix(yTrue: number[], yPred: number[]): Promise<number[][]> {
  const wasm = await ensureInit();
  const result = wasm.confusion_matrix(new Float64Array(yTrue), new Float64Array(yPred));

  // Convert flat array to 2D matrix
  const nClasses = Math.max(...yTrue, ...yPred) + 1;
  const matrix: number[][] = [];
  for (let i = 0; i < nClasses; i++) {
    matrix.push(Array.from(result.slice(i * nClasses, (i + 1) * nClasses)));
  }
  return matrix;
}

/**
 * Classification Report (precision, recall, f1, support)
 */
export async function classificationReport(
  yTrue: number[],
  yPred: number[]
): Promise<{ precision: number[]; recall: number[]; f1: number[]; support: number[] }> {
  const wasm = await ensureInit();
  const result = wasm.classification_report(new Float64Array(yTrue), new Float64Array(yPred));
  return {
    precision: Array.from(result.precision),
    recall: Array.from(result.recall),
    f1: Array.from(result.f1),
    support: Array.from(result.support),
  };
}

/**
 * Silhouette Score for clustering evaluation
 * @example
 * ```ts
 * const score = await silhouetteScore(x, labels, 100, 5);
 * // Returns value between -1 (bad) and 1 (good)
 * ```
 */
export async function silhouetteScore(
  x: number[],
  labels: number[],
  nSamples: number,
  nFeatures: number
): Promise<number> {
  const wasm = await ensureInit();
  return wasm.silhouette_score(new Float64Array(x), new Float64Array(labels), nSamples, nFeatures);
}

/**
 * Calinski-Harabasz Index for clustering evaluation
 */
export async function calinskiHarabaszScore(
  x: number[],
  labels: number[],
  nSamples: number,
  nFeatures: number
): Promise<number> {
  const wasm = await ensureInit();
  return wasm.calinski_harabasz_score(new Float64Array(x), new Float64Array(labels), nSamples, nFeatures);
}

/**
 * Davies-Bouldin Index for clustering evaluation (lower is better)
 */
export async function daviesBouldinScore(
  x: number[],
  labels: number[],
  nSamples: number,
  nFeatures: number
): Promise<number> {
  const wasm = await ensureInit();
  return wasm.davies_bouldin_score(new Float64Array(x), new Float64Array(labels), nSamples, nFeatures);
}

// ============================================================================
// Hidden Algorithms - Model Selection
// ============================================================================

/**
 * Cross-Validation Score
 * @example
 * ```ts
 * const scores = await crossValidateScore(x, y, 5, 100, 10);
 * // Returns 5 CV scores
 * ```
 */
export async function crossValidateScore(
  x: number[],
  y: number[],
  cvFolds: number,
  nSamples: number,
  nFeatures: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.cross_validate_score(new Float64Array(x), new Float64Array(y), cvFolds, nSamples, nFeatures);
  return Array.from(result);
}

/**
 * Train/Test Split
 */
export async function trainTestSplit(
  x: number[],
  y: number[],
  testSize: number,
  nSamples: number,
  nFeatures: number
): Promise<{
  xTrain: number[];
  xTest: number[];
  yTrain: number[];
  yTest: number[];
}> {
  const wasm = await ensureInit();
  const result = wasm.train_test_split(new Float64Array(x), new Float64Array(y), testSize, nSamples, nFeatures);
  return {
    xTrain: Array.from(result.x_train),
    xTest: Array.from(result.x_test),
    yTrain: Array.from(result.y_train),
    yTest: Array.from(result.y_test),
  };
}

// ============================================================================
// Hidden Algorithms - Feature Importance
// ============================================================================

/**
 * Feature Importance (for Decision Tree)
 */
export async function featureImportance(
  x: number[],
  y: number[],
  nSamples: number,
  nFeatures: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.feature_importance(new Float64Array(x), new Float64Array(y), nSamples, nFeatures);
  return Array.from(result);
}

/**
 * Feature Importance for Random Forest
 */
export async function featureImportanceForest(
  x: number[],
  y: number[],
  nTrees: number,
  maxDepth: number,
  nSamples: number,
  nFeatures: number
): Promise<number[]> {
  const wasm = await ensureInit();
  const result = wasm.feature_importance_forest(
    new Float64Array(x),
    new Float64Array(y),
    nTrees,
    maxDepth,
    nSamples,
    nFeatures
  );
  return Array.from(result);
}

// ============================================================================
// Hidden Algorithms - Regression Metrics
// ============================================================================

/**
 * Mean Absolute Error
 */
export async function meanAbsoluteError(yTrue: number[], yPred: number[]): Promise<number> {
  const wasm = await ensureInit();
  return wasm.mean_absolute_error(new Float64Array(yTrue), new Float64Array(yPred));
}

/**
 * Mean Squared Error
 */
export async function meanSquaredError(yTrue: number[], yPred: number[]): Promise<number> {
  const wasm = await ensureInit();
  return wasm.mean_squared_error(new Float64Array(yTrue), new Float64Array(yPred));
}

/**
 * R² Score
 */
export async function r2Score(yTrue: number[], yPred: number[]): Promise<number> {
  const wasm = await ensureInit();
  return wasm.r2_score(new Float64Array(yTrue), new Float64Array(yPred));
}

/**
 * Adjusted R² Score
 */
export async function adjustedR2Score(
  yTrue: number[],
  yPred: number[],
  nFeatures: number
): Promise<number> {
  const wasm = await ensureInit();
  return wasm.adjusted_r2_score(new Float64Array(yTrue), new Float64Array(yPred), nFeatures);
}

/**
 * Median Absolute Error
 */
export async function medianAbsoluteError(yTrue: number[], yPred: number[]): Promise<number> {
  const wasm = await ensureInit();
  return wasm.median_absolute_error(new Float64Array(yTrue), new Float64Array(yPred));
}

/**
 * Explained Variance Score
 */
export async function explainedVarianceScore(yTrue: number[], yPred: number[]): Promise<number> {
  const wasm = await ensureInit();
  return wasm.explained_variance_score(new Float64Array(yTrue), new Float64Array(yPred));
}

/**
 * Wrap WASM AutoMLResult to TypeScript interface
 */
function wrapAutoMLResult(wasmResult: any): AutoMLResult {
  const wrapped: AutoMLResult & { wasmResult: any } = {
    get best_algorithm() { return wasmResult.best_algorithm; },
    get best_score() { return wasmResult.best_score; },
    get evaluations() { return wasmResult.evaluations; },
    get selected_features() { return Array.from(wasmResult.selected_features) as number[]; },
    get algorithm_scores() { return Array.from(wasmResult.algorithm_scores) as string[]; },
    get rationale() { return wasmResult.rationale; },
    get original_features() { return wasmResult.original_features; },
    get feature_selection_performed() { return wasmResult.feature_selection_performed; },
    get problem_type() { return wasmResult.problem_type; },
    wasmResult: wasmResult,

    summary() {
      return wasmResult.summary();
    },

    algorithmScore(algorithmName: string): number | null {
      const score = wasmResult.algorithm_score(algorithmName);
      return score === undefined ? null : score;
    },

    isBetterThan(other: AutoMLResult): boolean {
      // Extract the raw WASM result from the wrapped object
      const otherWasm = (other as any).wasmResult || other;
      return wasmResult.is_better_than(otherWasm);
    },
  };
  return wrapped;
}
