use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

/// Theil-Sen Estimator - Robust regression using median of slopes.
///
/// For univariate data (n_features=1), computes all pairwise slopes and takes
/// the median. The intercept is the median of (y_i - slope * x_i).
/// This estimator is resistant to outliers (breakdown point ~29.3%).
#[wasm_bindgen]
pub struct TheilSenModel {
    coefficients: Vec<f64>,
    intercept: f64,
    n_features: usize,
}

#[wasm_bindgen]
impl TheilSenModel {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(getter, js_name = "coefficients")]
    pub fn coefficients_js(&self) -> Vec<f64> { self.coefficients.clone() }

    #[wasm_bindgen(getter, js_name = "intercept")]
    pub fn intercept_js(&self) -> f64 { self.intercept }

    /// Predict target values for the given feature data.
    ///
    /// `data` is a flat row-major array with length divisible by n_features.
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut pred = self.intercept;
            for f in 0..self.n_features {
                pred += self.coefficients[f] * data[i * self.n_features + f];
            }
            result.push(pred);
        }

        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "TheilSenModel(n_features={})",
            self.n_features
        )
    }
}

/// Compute the median of a mutable slice in-place using quickselect.
/// Returns the median value. For even-length slices, returns the average of the two middle values.
fn median_in_place(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return values[0];
    }

    let mid = n / 2;
    if n % 2 == 1 {
        quickselect(values, mid);
        values[mid]
    } else {
        quickselect(values, mid);
        let upper = values[mid];
        let lower = values[mid - 1];
        (upper + lower) / 2.0
    }
}

/// Partial sort: rearrange so that the element at index `k` is in its final sorted position,
/// with all elements before <= values[k] and all elements after >= values[k].
fn quickselect(arr: &mut [f64], k: usize) {
    let n = arr.len();
    if n <= 1 {
        return;
    }

    // Use median-of-three pivot selection for robustness
    let lo = 0;
    let hi = n - 1;
    let mid = lo + (hi - lo) / 2;

    // Sort lo, mid, hi and pick mid as pivot
    if arr[lo] > arr[mid] {
        arr.swap(lo, mid);
    }
    if arr[mid] > arr[hi] {
        arr.swap(mid, hi);
    }
    if arr[lo] > arr[mid] {
        arr.swap(lo, mid);
    }

    let pivot = arr[mid];
    arr.swap(mid, hi); // Move pivot to end

    let mut store = lo;
    for i in lo..hi {
        if arr[i] < pivot {
            arr.swap(store, i);
            store += 1;
        }
    }
    arr.swap(store, hi); // Move pivot to final position

    if k == store {
    } else if k < store {
        quickselect(&mut arr[..store], k);
    } else {
        quickselect(&mut arr[(store + 1)..], k - store - 1);
    }
}

/// Theil-Sen estimator implementation.
///
/// For univariate (n_features=1):
///   1. Compute all pairwise slopes: (y_j - y_i) / (x_j - x_i) for i < j
///   2. Median slope = median of all slopes
///   3. Median intercept = median of (y_i - slope * x_i)
///
/// For multivariate (n_features>1):
///   Iterative approach: for each feature, compute median pairwise slope with target,
///   then use iterative reweighted least squares refinement.
pub fn theil_sen_impl(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
) -> Result<TheilSenModel, MlError> {
    let n_samples = validate_matrix(data, n_features)?;
    if targets.len() != n_samples {
        return Err(MlError::new("targets length must match number of samples"));
    }
    if n_samples < 2 {
        return Err(MlError::new("need at least 2 samples"));
    }

    if n_features == 1 {
        theil_sen_univariate(data, targets, n_samples)
    } else {
        theil_sen_multivariate(data, n_features, targets, n_samples)
    }
}

/// Univariate Theil-Sen: median of all pairwise slopes.
fn theil_sen_univariate(
    data: &[f64],
    targets: &[f64],
    n_samples: usize,
) -> Result<TheilSenModel, MlError> {
    // Compute all pairwise slopes
    let n_slopes = n_samples * (n_samples - 1) / 2;
    let mut slopes = Vec::with_capacity(n_slopes);

    for i in 0..n_samples {
        let xi = mat_get(data, 1, i, 0);
        let yi = targets[i];
        for j in (i + 1)..n_samples {
            let xj = mat_get(data, 1, j, 0);
            let yj = targets[j];
            let dx = xj - xi;
            if dx.abs() > 1e-12 {
                slopes.push((yj - yi) / dx);
            }
        }
    }

    if slopes.is_empty() {
        return Err(MlError::new("cannot compute slopes: all x values are identical"));
    }

    // Median slope
    let median_slope = median_in_place(&mut slopes);

    // Median intercept = median of (y_i - slope * x_i)
    let mut intercepts = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let xi = mat_get(data, 1, i, 0);
        intercepts.push(targets[i] - median_slope * xi);
    }
    let median_intercept = median_in_place(&mut intercepts);

    Ok(TheilSenModel {
        coefficients: vec![median_slope],
        intercept: median_intercept,
        n_features: 1,
    })
}

/// Multivariate Theil-Sen using iterative approach.
///
/// For each feature dimension, compute the median pairwise slope against the target,
/// then refine with iterative reweighted least squares (IRLS).
fn theil_sen_multivariate(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    n_samples: usize,
) -> Result<TheilSenModel, MlError> {
    // Initial coefficient estimates: for each feature, compute median slope vs target
    let mut initial_coef = vec![0.0f64; n_features];

    for f in 0..n_features {
        let n_pairs = n_samples * (n_samples - 1) / 2;
        let mut slopes = Vec::with_capacity(n_pairs);

        for i in 0..n_samples {
            let xi = mat_get(data, n_features, i, f);
            let yi = targets[i];
            for j in (i + 1)..n_samples {
                let xj = mat_get(data, n_features, j, f);
                let yj = targets[j];
                let dx = xj - xi;
                if dx.abs() > 1e-12 {
                    slopes.push((yj - yi) / dx);
                }
            }
        }

        if !slopes.is_empty() {
            initial_coef[f] = median_in_place(&mut slopes);
        }
    }

    // Iterative reweighted least squares refinement
    // Start with initial estimate, compute residuals, weight by inverse absolute residual,
    // refit weighted least squares. Repeat for a few iterations.
    let mut coefficients = initial_coef;
    let n_irls_iterations = 5;

    for _iter in 0..n_irls_iterations {
        // Compute residuals and weights
        let mut residuals = vec![0.0f64; n_samples];
        let mut weights = vec![1.0f64; n_samples];

        for i in 0..n_samples {
            let mut pred = 0.0;
            for f in 0..n_features {
                pred += coefficients[f] * mat_get(data, n_features, i, f);
            }
            residuals[i] = (targets[i] - pred).abs();
        }

        // Use median absolute deviation for robustness
        let mut abs_residuals = residuals.clone();
        let mad = median_in_place(&mut abs_residuals);
        let scale = if mad < 1e-12 { 1.0 } else { mad * 1.4826 }; // 1.4826 converts MAD to std dev

        // Bisquare weight function: w(r) = (1 - (r/s)^2)^2 for |r| < s, else 0
        for i in 0..n_samples {
            let u = residuals[i] / scale;
            if u < 1.0 {
                weights[i] = (1.0 - u * u).powi(2);
            } else {
                weights[i] = 0.0;
            }
        }

        // Weighted least squares: (X'WX)^-1 X'Wy
        let mut xtwx = vec![0.0f64; n_features * n_features];
        let mut xtwy = vec![0.0f64; n_features];
        let mut total_weight = 0.0f64;

        for i in 0..n_samples {
            let w = weights[i];
            if w < 1e-12 {
                continue;
            }
            total_weight += w;

            for f1 in 0..n_features {
                let xf1 = mat_get(data, n_features, i, f1) * w;
                xtwy[f1] += xf1 * targets[i];
                for f2 in 0..n_features {
                    xtwx[f1 * n_features + f2] += xf1 * mat_get(data, n_features, i, f2);
                }
            }
        }

        if total_weight < 1e-12 {
            break;
        }

        // Solve using Gaussian elimination
        match solve_system(&xtwx, n_features, &xtwy) {
            Some(new_coef) => {
                coefficients = new_coef;
            }
            None => break, // Singular matrix, keep previous estimate
        }
    }

    // Compute intercept: median of (y_i - sum(coef * x_i))
    let mut intercepts = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut pred = 0.0;
        for f in 0..n_features {
            pred += coefficients[f] * mat_get(data, n_features, i, f);
        }
        intercepts.push(targets[i] - pred);
    }
    let median_intercept = median_in_place(&mut intercepts);

    Ok(TheilSenModel {
        coefficients,
        intercept: median_intercept,
        n_features,
    })
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_system(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / aug[col * (n + 1) + col];
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    Some(x)
}

#[wasm_bindgen(js_name = "theilSen")]
pub fn theil_sen(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
) -> Result<TheilSenModel, JsError> {
    theil_sen_impl(data, n_features, targets)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_slope() {
        // y = 2x: all pairwise slopes should be 2.0
        let data: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let targets: Vec<f64> = (1..=5).map(|i| 2.0 * i as f64).collect();

        let model = theil_sen_impl(&data, 1, &targets).unwrap();

        assert!(
            (model.coefficients[0] - 2.0).abs() < 0.01,
            "median slope = {}, expected 2.0",
            model.coefficients[0]
        );
        assert!(
            (model.intercept - 0.0).abs() < 0.01,
            "intercept = {}, expected 0.0",
            model.intercept
        );
    }

    #[test]
    fn test_robust_to_outliers() {
        // y = 3x with outliers injected
        let mut data: Vec<f64> = Vec::new();
        let mut targets: Vec<f64> = Vec::new();

        // 17 clean points: y = 3x
        for i in 1..=17 {
            data.push(i as f64);
            targets.push(3.0 * i as f64);
        }

        // 3 outliers
        data.push(5.0); targets.push(100.0);
        data.push(10.0); targets.push(-50.0);
        data.push(15.0); targets.push(200.0);

        let model = theil_sen_impl(&data, 1, &targets).unwrap();

        // Theil-Sen should be robust: slope should be close to 3.0
        // Theil-Sen has ~29.3% breakdown point, so 3/20 = 15% outliers are fine
        assert!(
            (model.coefficients[0] - 3.0).abs() < 0.5,
            "slope = {}, expected ~3.0",
            model.coefficients[0]
        );

        // Predictions on clean data should be reasonable
        let clean_preds = model.predict(&data[..17]);
        for (i, p) in clean_preds.iter().enumerate() {
            let expected = 3.0 * (i + 1) as f64;
            assert!(
                (p - expected).abs() < 2.0,
                "prediction {} far from expected {}",
                p, expected
            );
        }
    }

    #[test]
    fn test_theil_sen_with_intercept() {
        // y = 2x + 5
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let targets: Vec<f64> = (1..=10).map(|i| 2.0 * i as f64 + 5.0).collect();

        let model = theil_sen_impl(&data, 1, &targets).unwrap();

        assert!(
            (model.coefficients[0] - 2.0).abs() < 0.1,
            "slope = {}, expected 2.0",
            model.coefficients[0]
        );
        assert!(
            (model.intercept - 5.0).abs() < 0.5,
            "intercept = {}, expected 5.0",
            model.intercept
        );
    }

    #[test]
    fn test_theil_sen_multivariate() {
        // y = 2*x1 + 3*x2
        let data = vec![
            1.0, 1.0,
            2.0, 1.0,
            1.0, 2.0,
            3.0, 2.0,
            2.0, 3.0,
            4.0, 1.0,
            1.0, 4.0,
            3.0, 3.0,
            5.0, 2.0,
            2.0, 5.0,
        ];
        let targets: Vec<f64> = data.chunks(2)
            .map(|row| 2.0 * row[0] + 3.0 * row[1])
            .collect();

        let model = theil_sen_impl(&data, 2, &targets).unwrap();

        assert_eq!(model.n_features, 2);
        // Multivariate uses IRLS refinement so should be reasonably close
        assert!(
            (model.coefficients[0] - 2.0).abs() < 1.0,
            "coef[0] = {}, expected ~2.0",
            model.coefficients[0]
        );
        assert!(
            (model.coefficients[1] - 3.0).abs() < 1.0,
            "coef[1] = {}, expected ~3.0",
            model.coefficients[1]
        );
    }

    #[test]
    fn test_theil_sen_insufficient_samples() {
        let data = vec![1.0];
        let targets = vec![2.0];
        assert!(theil_sen_impl(&data, 1, &targets).is_err());
    }

    #[test]
    fn test_median_in_place() {
        // Odd length
        let mut v = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let m = median_in_place(&mut v);
        assert!((m - 3.0).abs() < 1e-10);

        // Even length: median = average of middle two
        let mut v2 = vec![4.0, 1.0, 3.0, 2.0];
        let m2 = median_in_place(&mut v2);
        assert!((m2 - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_single_element() {
        let mut v = vec![42.0];
        let m = median_in_place(&mut v);
        assert!((m - 42.0).abs() < 1e-10);
    }
}
