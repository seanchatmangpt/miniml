# Contributing Guide

How to add new algorithms, fix bugs, and submit changes to micro-ml.

## Prerequisites

- Rust 1.75+ (stable)
- Node.js 18+ (for WASM testing)
- wasm-pack: `cargo install wasm-pack`

## Development Setup

```bash
git clone https://github.com/seanchatmangpt/micro-ml.git
cd micro-ml
cargo build
cargo test --lib
```

## Code Style

Every contribution must pass:

```bash
cargo fmt          # Format code
cargo clippy       # Lint (zero warnings goal)
cargo test --lib   # All tests pass
```

## Module Pattern

Every algorithm module follows this structure:

```rust
// src/my_algorithm.rs

use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::validate_matrix;

/// Public struct returned to JS
#[wasm_bindgen]
pub struct MyModel {
    // Private fields
    weights: Vec<f64>,
    n_features: usize,
}

#[wasm_bindgen]
impl MyModel {
    // Getters use camelCase via js_name
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    /// Predict method
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        // ... prediction logic ...
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("MyModel(features={})", self.n_features)
    }
}

/// Core implementation (no WASM types)
pub fn my_algorithm_impl(
    data: &[f64],
    n_features: usize,
    // ... params ...
) -> Result<MyModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    // Validate params
    // Run algorithm
    // Return model
    Ok(MyModel { weights, n_features })
}

/// WASM export function
#[wasm_bindgen(js_name = "myAlgorithm")]
pub fn my_algorithm(
    data: &[f64],
    n_features: usize,
    // ... params ...
) -> Result<MyModel, JsError> {
    my_algorithm_impl(data, n_features)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Arrange
        let data = vec![1.0, 2.0, 3.0, 4.0];
        // Act
        let result = my_algorithm_impl(&data, 2).unwrap();
        // Assert
        assert!(result.predict(&data).len() > 0);
    }

    #[test]
    fn test_invalid_input() {
        let result = my_algorithm_impl(&[], 1);
        assert!(result.is_err());
    }
}
```

## Key Conventions

### Input Validation
- Always call `validate_matrix(data, n_features)?` first
- Validate parameter ranges (e.g., `n_clusters > 0`)
- Return `MlError::new("description")` for invalid inputs

### Data Layout
- All matrices are flat `&[f64]` in row-major order
- Element at row i, col j: `data[i * n_features + j]`
- `n_samples = data.len() / n_features`

### Randomness
- Use `Rng::from_data(data)` for deterministic PRNG
- Same input always produces same output
- No `rand` crate dependency

### WASM Constraints
- Zero external dependencies (no ndarray, no nalgebra)
- No heap allocations in hot loops where possible
- All public types must implement `#[wasm_bindgen]`

## Testing

### Unit Tests
Every module has a `#[cfg(test)] mod tests` block at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() { /* ... */ }

    #[test]
    fn test_edge_cases() { /* ... */ }

    #[test]
    fn test_invalid_inputs() { /* ... */ }
}
```

### Integration Tests
Cross-module tests live in `tests/ml_validation.rs`:

```rust
use micro_ml_core::linear_regression::linear_regression_impl;
use micro_ml_core::minmax_scaler::standard_scaler_impl;
```

### Running Tests

```bash
cargo test --lib                    # Unit tests only
cargo test --test ml_validation     # Integration tests
cargo test                          # All tests
```

## Adding a New Module

1. Create `src/my_algorithm.rs` following the pattern above
2. Add `pub mod my_algorithm;` to `src/lib.rs`
3. Add `pub use my_algorithm::*;` to `src/lib.rs`
4. Write at least 3 tests (basic, edge case, invalid input)
5. Run `cargo fmt && cargo clippy && cargo test --lib`
6. Optionally add TypeScript wrapper in `packages/micro-ml/src/index.ts`

## Commit Convention

```
type(scope): description

Types: feat, fix, docs, refactor, test, chore
Scope: ml, error, lib, wasm
```

Examples:
- `feat(ml): add isolation forest anomaly detection`
- `fix(ml): correct AUC calculation with Mann-Whitney U`
- `refactor(ml): apply clippy auto-fixes`
- `test(ml): add integration tests for ensemble methods`

## PR Process

1. One algorithm per PR (or logically grouped set)
2. All tests pass
3. `cargo clippy` has zero warnings (or documented exceptions)
4. Conventional commit format
5. Update API reference in `docs/api/reference.md`

## WASM Build

```bash
cd crates/micro-ml-core
wasm-pack build --target web --out-dir ../../../packages/micro-ml/wasm
```

## Bundle Size Budget

Keep the gzipped WASM bundle under 100KB. Each algorithm adds ~2-5KB. If your addition significantly increases bundle size, consider making it feature-flagged.
