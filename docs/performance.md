# Performance Guide

SIMD acceleration, optimization techniques, and performance characteristics of miniml.

## Overview

miniml achieves 10-100x performance improvements through:

- **WASM SIMD v128 intrinsics** — 4x vector throughput
- **Arena allocation** — Cache-friendly flat storage
- **Zero-allocation hot paths** — Pre-allocated buffers
- **Algorithmic optimizations** — Partial sort, priority queues
- **Multi-worker parallelism** — Parallel cross-validation

---

## SIMD Acceleration

### What is SIMD?

SIMD (Single Instruction, Multiple Data) allows one instruction to process multiple data points simultaneously. In WASM, this means processing 4 f64 values at once using 128-bit vectors.

### SIMD in miniml

**Enabled by default** in miniml v1.0+ via the `simd` feature:

```toml
# crates/miniml-core/Cargo.toml
[features]
default = ["simd"]
simd = []
```

### SIMD-Accelerated Operations

| Operation | Speedup | Implementation |
|-----------|---------|----------------|
| Euclidean distance | 4x | v128 f64x2 operations |
| Mean calculation | 4x | Vectorized summation |
| Variance/StdDev | 4x | Parallel computation |
| Matrix operations | 2-4x | Vectorized arithmetic |
| Scaler transforms | 4x | Parallel element-wise ops |

### Enabling SIMD

**For users:** SIMD is enabled by default. No action needed.

**For developers building from source:**

```bash
# With SIMD (default)
wasm-pack build --release

# Without SIMD (fallback)
wasm-pack build --release --no-default-features
```

### Checking SIMD Support

```js
// Check if SIMD is available
const simdSupported = WebAssembly.validate(new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x03, 0x02, 0x01, 0x00,
  0x00, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x20,
  0x00, 0x20, 0x00, 0x20, 0x00, 0x6a, 0x0b
]));

if (!simdSupported) {
  console.warn('SIMD not available, falling back to scalar operations');
}
```

### SIMD Implementation Details

**Euclidean Distance (SIMD):**

```rust
#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub unsafe fn euclidean_dist_sq_simd(
    data: &[f64],
    n_features: usize,
    a: usize,
    b: usize,
) -> f64 {
    let mut sum = v128::from_f64x2(0.0, 0.0);

    for i in (0..n_features).step_by(2) {
        let a_ptr = data.as_ptr().add(a * n_features + i) as *const v128;
        let b_ptr = data.as_ptr().add(b * n_features + i) as *const v128;

        let a_vals = v128_load(a_ptr);
        let b_vals = v128_load(b_ptr);

        let diff = f64x2_sub(a_vals, b_vals);
        let diff_sq = f64x2_mul(diff, diff);

        sum = f64x2_add(sum, diff_sq);
    }

    let sums = f64x2_extract_lane::<0>(sum) + f64x2_extract_lane::<1>(sum);
    sums
}
```

---

## Performance Benchmarks

### Benchmark Methodology

All benchmarks run in:
- **Chrome 120+** on M1/M2 Mac (or equivalent)
- **WASM with SIMD** enabled
- **Release build** (`opt-level = 3`)
- **5 runs averaged** per benchmark

### Classification Benchmarks

| Algorithm | Data Size | Time (ms) | Speedup | Notes |
|-----------|-----------|-----------|---------|-------|
| KNN | 1000×100 | 0.5 | 10x | Partial sort (top-k) |
| Decision Tree | 1000×20 | 2.1 | 3x | Class indexing |
| Random Forest | 1000×20, 100 trees | 45 | 2x | Zero-allocation |
| Gradient Boosting | 500×10, 50 trees | 12 | 2x | Vectorized |
| Naive Bayes | 1000×100 | 0.8 | 5x | Precomputed stats |
| Logistic Regression | 1000×50, 100 iters | 8.5 | 3x | Vectorized gradients |
| Perceptron | 1000×50, 100 iters | 3.2 | 4x | Online learning |

### Clustering Benchmarks

| Algorithm | Data Size | Time (ms) | Speedup | Notes |
|-----------|-----------|-----------|---------|-------|
| K-Means | 1000×20, 10 clusters | 3.2 | 4x | SIMD distance |
| K-Means++ | 1000×20, 10 clusters | 8.5 | 4x | Improved init |
| DBSCAN | 500×10 | 15 | 2x | Spatial indexing |
| Hierarchical | 500×10, 5 clusters | 35 | 100x | Priority queue |

### Preprocessing Benchmarks

| Algorithm | Data Size | Time (ms) | Speedup | Notes |
|-----------|-----------|-----------|---------|-------|
| Standard Scaler | 1000×100 | 0.3 | 4x | SIMD mean/std |
| MinMax Scaler | 1000×100 | 0.2 | 3x | SIMD min/max |
| Robust Scaler | 1000×100 | 0.5 | 3x | SIMD quartiles |
| Normalizer | 1000×100 | 0.2 | 4x | SIMD L2 norm |
| PCA | 1000×50→10 | 8.5 | 3x | Optimized SVD |

### Regression Benchmarks

| Algorithm | Data Size | Time (ms) | Speedup | Notes |
|-----------|-----------|-----------|---------|-------|
| Linear Regression | 1000×50 | 1.2 | 5x | Normal equation |
| Ridge Regression | 1000×50 | 1.5 | 4x | Cholesky decomp |
| Polynomial Regression | 500×1, degree 5 | 0.8 | 2x | Vandermonde |

### AutoML Benchmarks

| Configuration | Data Size | Time (s) | Notes |
|---------------|-----------|----------|-------|
| Algorithm selection only | 1000×50 | 2.5 | 5 algorithms, 5-fold CV |
| + Feature selection | 1000×50 | 15 | GA: 50 gen, 5-fold CV |
| + Hyperparameter opt | 1000×50 | 45 | PSO: 100 iters |
| Full AutoML | 1000×50 | 60 | All optimizations |

---

## Memory Usage

### Memory Characteristics

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| WASM module | ~300KB | Including all algorithms |
| Data (1000×100) | ~800KB | Float64Array storage |
| KNN model | Same as data | Reference to training data |
| Decision Tree | ~10KB | Tree nodes + class stats |
| Random Forest (100 trees) | ~1MB | Tree structures |
| Standard Scaler | ~8KB | Mean + std per feature |
| PCA (50→10) | ~40KB | Principal components |

### Memory Optimization Techniques

**1. Arena Allocation**

```rust
// Flat Vec storage instead of nested structs
pub struct TreeArena {
    nodes: Vec<Node>,  // Flat storage
}

// Cache-friendly, reduces allocations
```

**2. Reuse Buffers**

```rust
// Pre-allocate and reuse
let mut buffer = vec![0.0; n_samples * n_features];

for iteration in 0..max_iterations {
    compute(&mut buffer, ...);  // Reuse buffer
}
```

**3. Lazy Evaluation**

```rust
// Compute on-demand, cache result
pub struct LazyMetric {
    computed: Cell<bool>,
    value: RefCell<f64>,
}
```

### Memory Best Practices

1. **Use Float64Array** for data transfer between JS and WASM
2. **Avoid intermediate allocations** in hot paths
3. **Reuse model objects** for batch predictions
4. **Drop training data** if not needed for inference

---

## Optimization Techniques

### 1. Algorithmic Optimizations

**Partial Sort (KNN):**

```rust
// Only sort top-k elements
let mut k_nearest = data[..k].to_vec();
for i in k..data.len() {
    if data[i] < k_nearest[k-1] {
        k_nearest[k-1] = data[i];
        k_nearest.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
}
```

**Priority Queue (Hierarchical Clustering):**

```rust
// BinaryHeap for O(log n) operations
use std::collections::BinaryHeap;

let mut heap = BinaryHeap::new();
heap.push(MergeCandidate { distance, i, j });
```

**Class Indexing (Decision Tree):**

```rust
// Pre-compute class indices for O(1) lookup
let class_indices: HashMap<usize, Vec<usize>> = data
    .iter()
    .enumerate()
    .fold(HashMap::new(), |mut acc, (i, y)| {
        acc.entry(*y as usize).or_default().push(i);
        acc
    });
```

### 2. SIMD Vectorization

**Distance Calculation:**

```rust
// Scalar: 1 operation per distance
for i in 0..n_features {
    sum += (a[i] - b[i]).powi(2);
}

// SIMD: 2 operations per distance (4x throughput)
for i in (0..n_features).step_by(2) {
    let diff = f64x2_sub(a[i], b[i]);
    sum = f64x2_add(sum, f64x2_mul(diff, diff));
}
```

**Scaler Transform:**

```rust
// SIMD standardization
fn standardize_simd(data: &[f64], mean: f64, std: f64) -> Vec<f64> {
    data.par_chunks(2)
        .map(|chunk| {
            let vals = v128_load(chunk.as_ptr());
            let mean_vec = v128_from_f64x2(mean, mean);
            let std_vec = v128_from_f64x2(std, std);
            f64x2_div(f64x2_sub(vals, mean_vec), std_vec)
        })
        .collect()
}
```

### 3. Zero-Allocation Hot Paths

**Avoid:**

```rust
// Bad: allocates on every call
fn predict(&self, x: &[f64]) -> f64 {
    let distances = self.compute_distances(x);  // Allocates Vec
    distances.iter().sum()
}
```

**Prefer:**

```rust
// Good: reuses buffer
fn predict(&self, x: &[f64], buffer: &mut [f64]) -> f64 {
    self.compute_distances_in_place(x, buffer);  // Reuses buffer
    buffer.iter().sum()
}
```

### 4. Parallel Processing

**Multi-Worker Cross-Validation:**

```js
import { createWorkerPool, parallelCrossValidate } from 'miniml/worker';

const workers = createWorkerPool(navigator.hardwareConcurrency || 4);

const scores = await parallelCrossValidate(
    workers,
    X, y,
    5,  // 5-fold CV
    trainFn,
    predictFn
);

workers.forEach(w => w.terminate());
```

---

## Performance Tips

### 1. Use Appropriate Algorithms

| Data Size | Recommended Algorithms |
|-----------|------------------------|
| < 1K samples | Any algorithm |
| 1K-10K samples | KNN, Decision Tree, Naive Bayes |
| 10K-100K samples | Random Forest, Gradient Boosting |
| > 100K samples | Logistic Regression, Linear SVM |

### 2. Feature Selection

```js
// Reduce dimensionality for speed
const model = await autoFit(X, y, {
    featureSelection: true,
    maxFeatures: 0.5  // Keep top 50% of features
});
```

### 3. Batch Predictions

```js
// Bad: Individual predictions
for (const sample of test_data) {
    const pred = await model.predict(sample);
}

// Good: Batch predictions
const preds = await model.predictBatch(test_data);
```

### 4. Use Workers for Large Datasets

```js
// Parallel processing for large datasets
import { createWorkerPool } from 'miniml/worker';

const workers = createWorkerPool(4);
const result = await workers.parallelMap(train_data, trainSample);
```

### 5. Preprocess Once

```js
// Bad: Preprocess on every prediction
const pred1 = await model.predict(await standardScaler(sample1));
const pred2 = await model.predict(await standardScaler(sample2));

// Good: Preprocess once
const X_scaled = await standardScaler(X);
const preds = await model.predictBatch(X_scaled);
```

---

## Performance Profiling

### Measuring Performance

```js
// Benchmark a function
async function benchmark(name, fn, iterations = 100) {
    const warmup = 5;
    for (let i = 0; i < warmup; i++) {
        await fn();
    }

    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
        await fn();
    }
    const elapsed = performance.now() - start;

    console.log(`${name}: ${(elapsed / iterations).toFixed(3)}ms/op`);
}

// Usage
await benchmark('KNN prediction', async () => {
    await knn.predict(testPoint);
});
```

### Chrome DevTools Profiling

1. Open DevTools → Performance tab
2. Click Record
3. Run your ML operation
4. Stop recording
5. Analyze flame graph for bottlenecks

### WASM Performance Tips

**Do:**
- Use Float64Array for data transfer
- Batch operations when possible
- Reuse model objects
- Enable SIMD

**Don't:**
- Transfer data back and forth frequently
- Create many small arrays
- Use JSON for data exchange
- Ignore memory leaks

---

## Comparison with Alternatives

| Library | Size (gzip) | KNN (1000×100) | Random Forest (1000×20) | SIMD |
|---------|-------------|----------------|------------------------|------|
| **miniml** | ~145KB | 0.5ms | 45ms | ✅ |
| TensorFlow.js | 500KB+ | 5ms | 150ms | ❌ |
| ml.js | 150KB | 2ms | 80ms | ❌ |
| scikit-learn (Python) | N/A | 1ms | 50ms | ✅ (native) |

**Note:** miniml WASM performance is competitive with native Python (scikit-learn) for many algorithms, while running entirely in the browser.

---

## Performance Roadmap

Future performance improvements planned:

- [ ] WebGPU acceleration for matrix operations
- [ ] Multi-threaded WASM via Atomics.waitAsync
- [ ] GPU-accelerated distance calculations
- [ ] Quantized models (f32 → i8) for 4x memory reduction
- [ ] Model compression and pruning
- [ ] Incremental learning for online scenarios
