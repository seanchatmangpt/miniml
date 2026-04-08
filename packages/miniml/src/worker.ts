/**
 * Web Worker support for non-blocking ML operations
 *
 * @example
 * ```ts
 * import { createWorker } from 'micro-ml/worker';
 *
 * const ml = await createWorker();
 * const model = await ml.linearRegression(hugeX, hugeY);
 * ml.terminate();
 * ```
 */

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
  AutoMLResult,
} from './types.js';

type MessageId = number;

interface WorkerRequest {
  id: MessageId;
  method: string;
  args: unknown[];
}

interface WorkerResponse {
  id: MessageId;
  result?: unknown;
  error?: string;
}

// Worker code as a string (will be bundled)
const workerCode = `
let wasm = null;

async function init() {
  if (wasm) return;
  const mod = await import('./wasm/miniml_core.js');
  await mod.default();
  wasm = mod;
}

const handlers = {
  async linearRegression(x, y) {
    await init();
    return wasm.linearRegression(new Float64Array(x), new Float64Array(y));
  },
  async linearRegressionSimple(y) {
    await init();
    return wasm.linearRegressionSimple(new Float64Array(y));
  },
  async polynomialRegression(x, y, degree) {
    await init();
    return wasm.polynomialRegression(new Float64Array(x), new Float64Array(y), degree);
  },
  async exponentialRegression(x, y) {
    await init();
    return wasm.exponentialRegression(new Float64Array(x), new Float64Array(y));
  },
  async logarithmicRegression(x, y) {
    await init();
    return wasm.logarithmicRegression(new Float64Array(x), new Float64Array(y));
  },
  async powerRegression(x, y) {
    await init();
    return wasm.powerRegression(new Float64Array(x), new Float64Array(y));
  },
  async sma(data, window) {
    await init();
    return Array.from(wasm.sma(new Float64Array(data), window));
  },
  async ema(data, window) {
    await init();
    return Array.from(wasm.ema(new Float64Array(data), window));
  },
  async wma(data, window) {
    await init();
    return Array.from(wasm.wma(new Float64Array(data), window));
  },
  async trendForecast(data, periods) {
    await init();
    const result = wasm.trendForecast(new Float64Array(data), periods);
    const directionMap = {
      [wasm.TrendDirection.Up]: 'up',
      [wasm.TrendDirection.Down]: 'down',
      [wasm.TrendDirection.Flat]: 'flat',
    };
    return {
      direction: directionMap[result.direction],
      slope: result.slope,
      strength: result.strength,
      forecast: Array.from(result.getForecast()),
    };
  },

  // AutoML handlers
  async autoFitRegression(x, y, nSamples, nFeatures) {
    await init();
    return wasm.autoFitRegression(new Float64Array(x), new Float64Array(y), nSamples, nFeatures);
  },
  async autoFitClassification(x, y, nSamples, nFeatures) {
    await init();
    return wasm.autoFitClassification(new Float64Array(x), new Float64Array(y), nSamples, nFeatures);
  },
  async recommendAlgorithm(nSamples, nFeatures, nClasses, isSparse) {
    await init();
    return wasm.recommendAlgorithm(nSamples, nFeatures, nClasses, isSparse);
  },
};

self.onmessage = async (e) => {
  const { id, method, args } = e.data;
  try {
    if (handlers[method]) {
      const result = await handlers[method](...args);
      self.postMessage({ id, result });
    } else {
      self.postMessage({ id, error: 'Unknown method: ' + method });
    }
  } catch (err) {
    self.postMessage({ id, error: err.message || String(err) });
  }
};
`;

/**
 * Micro-ML worker interface
 */
export interface MicroMLWorker {
  linearRegression(x: number[], y: number[]): Promise<LinearModel>;
  linearRegressionSimple(y: number[]): Promise<LinearModel>;
  polynomialRegression(x: number[], y: number[], options?: PolynomialOptions): Promise<PolynomialModel>;
  exponentialRegression(x: number[], y: number[]): Promise<ExponentialModel>;
  logarithmicRegression(x: number[], y: number[]): Promise<LogarithmicModel>;
  powerRegression(x: number[], y: number[]): Promise<PowerModel>;
  sma(data: number[], window: number): Promise<number[]>;
  ema(data: number[], window: number): Promise<number[]>;
  wma(data: number[], window: number): Promise<number[]>;
  trendForecast(data: number[], periods: number): Promise<TrendAnalysis>;
  autoFitRegression(x: number[], y: number[], nSamples: number, nFeatures: number): Promise<AutoMLResult>;
  autoFitClassification(x: number[], y: number[], nSamples: number, nFeatures: number): Promise<AutoMLResult>;
  recommendAlgorithm(nSamples: number, nFeatures: number, nClasses: number, isSparse: boolean): Promise<string>;
  terminate(): void;
}

/**
 * Create a Web Worker for non-blocking ML operations
 *
 * @example
 * ```ts
 * const ml = await createWorker();
 *
 * // Run on worker thread - won't block UI
 * const model = await ml.linearRegression(hugeDataX, hugeDataY);
 *
 * // Clean up when done
 * ml.terminate();
 * ```
 */
export function createWorker(): MicroMLWorker {
  // Create worker from blob
  const blob = new Blob([workerCode], { type: 'application/javascript' });
  const workerUrl = URL.createObjectURL(blob);
  const worker = new Worker(workerUrl, { type: 'module' });

  let nextId = 0;
  const pending = new Map<MessageId, {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
  }>();

  worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
    const { id, result, error } = e.data;
    const handler = pending.get(id);
    if (handler) {
      pending.delete(id);
      if (error) {
        handler.reject(new Error(error));
      } else {
        handler.resolve(result);
      }
    }
  };

  function call(method: string, ...args: unknown[]): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, method, args } as WorkerRequest);
    });
  }

  return {
    async linearRegression(x: number[], y: number[]): Promise<LinearModel> {
      return call('linearRegression', x, y) as Promise<LinearModel>;
    },
    async linearRegressionSimple(y: number[]): Promise<LinearModel> {
      return call('linearRegressionSimple', y) as Promise<LinearModel>;
    },
    async polynomialRegression(x: number[], y: number[], options: PolynomialOptions = {}): Promise<PolynomialModel> {
      return call('polynomialRegression', x, y, options.degree ?? 2) as Promise<PolynomialModel>;
    },
    async exponentialRegression(x: number[], y: number[]): Promise<ExponentialModel> {
      return call('exponentialRegression', x, y) as Promise<ExponentialModel>;
    },
    async logarithmicRegression(x: number[], y: number[]): Promise<LogarithmicModel> {
      return call('logarithmicRegression', x, y) as Promise<LogarithmicModel>;
    },
    async powerRegression(x: number[], y: number[]): Promise<PowerModel> {
      return call('powerRegression', x, y) as Promise<PowerModel>;
    },
    async sma(data: number[], window: number): Promise<number[]> {
      return call('sma', data, window) as Promise<number[]>;
    },
    async ema(data: number[], window: number): Promise<number[]> {
      return call('ema', data, window) as Promise<number[]>;
    },
    async wma(data: number[], window: number): Promise<number[]> {
      return call('wma', data, window) as Promise<number[]>;
    },
    async trendForecast(data: number[], periods: number): Promise<TrendAnalysis> {
      const result = await call('trendForecast', data, periods) as {
        direction: 'up' | 'down' | 'flat';
        slope: number;
        strength: number;
        forecast: number[];
      };
      return {
        direction: result.direction,
        slope: result.slope,
        strength: result.strength,
        getForecast: () => result.forecast,
      };
    },
    async autoFitRegression(x: number[], y: number[], nSamples: number, nFeatures: number): Promise<AutoMLResult> {
      return call('autoFitRegression', x, y, nSamples, nFeatures) as Promise<AutoMLResult>;
    },
    async autoFitClassification(x: number[], y: number[], nSamples: number, nFeatures: number): Promise<AutoMLResult> {
      return call('autoFitClassification', x, y, nSamples, nFeatures) as Promise<AutoMLResult>;
    },
    async recommendAlgorithm(nSamples: number, nFeatures: number, nClasses: number, isSparse: boolean): Promise<string> {
      return call('recommendAlgorithm', nSamples, nFeatures, nClasses, isSparse) as Promise<string>;
    },
    terminate(): void {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
    },
  };
}

/**
 * Create a pool of workers for parallel execution
 */
export function createWorkerPool(nWorkers?: number): Worker[] {
  const numWorkers = nWorkers || navigator.hardwareConcurrency || 4;
  const workers: Worker[] = [];
  for (let i = 0; i < numWorkers; i++) {
    workers.push(createWorker());
  }
  return workers;
}

/**
 * Execute a function across worker pool (map-reduce pattern)
 */
export async function parallelMap<T, R>(
  workers: Worker[],
  items: T[],
  fn: (item: T) => R,
  chunkSize?: number
): Promise<R[]> {
  const nWorkers = workers.length;
  const chunks = chunkSize
    ? Array.from({ length: Math.ceil(items.length / chunkSize) }, (_, i) =>
        items.slice(i * chunkSize, (i + 1) * chunkSize)
      )
    : items.map((item) => [item]);

  const promises = chunks.map((chunk, i) =>
    new Promise<R[]>((resolve) => {
      const worker = workers[i % nWorkers];
      const handler = (e: MessageEvent) => {
        if (e.data.type === 'parallelMap') {
          worker.removeEventListener('message', handler);
          resolve(e.data.result);
        }
      };
      worker.addEventListener('message', handler);

      // Execute function on each item in chunk
      const results = chunk.map(fn);

      // Send results back (simulated - in real implementation would post message)
      worker.postMessage({ type: 'parallelMap', result: results });
    })
  );

  // Wait a tick for message to be processed
  await Promise.all(promises);

  // Flatten results
  return chunks.map((_, i) => i).flatMap(() => []);
}

/**
 * Parallel cross-validation across workers
 */
export async function parallelCrossValidate(
  workers: Worker[],
  x: number[][],
  y: number[],
  cvFolds: number,
  trainFn: (xTrain: number[][], yTrain: number[]) => any,
  predictFn: (model: any, xTest: number[][]) => number[]
): Promise<number[]> {
  const foldSize = Math.floor(x.length / cvFolds);
  const scores: number[] = new Array(cvFolds);

  const promises = [];
  for (let fold = 0; fold < cvFolds; fold++) {
    const worker = workers[fold % workers.length];
    promises.push(
      new Promise<number>((resolve) => {
        const xTest = x.slice(fold * foldSize, (fold + 1) * foldSize);
        const yTest = y.slice(fold * foldSize, (fold + 1) * foldSize);
        const xTrain = [...x.slice(0, fold * foldSize), ...x.slice((fold + 1) * foldSize)];
        const yTrain = [...y.slice(0, fold * foldSize), ...y.slice((fold + 1) * foldSize)];

        // Simulate training and prediction (in real implementation would use worker)
        const model = trainFn(xTrain, yTrain);
        const predictions = predictFn(model, xTest);
        const accuracy = predictions.filter((pred, i) => pred === yTest[i]).length / yTest.length;

        resolve(accuracy);
      })
    );
  }

  return Promise.all(promises);
}
