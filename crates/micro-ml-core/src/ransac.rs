use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, Rng};

/// RANSAC (Random Sample Consensus) - Robust regression that ignores outliers.
///
/// Iteratively fits a model to random subsets of data, identifying inliers
/// (points within `residual_threshold` of the model) and keeping the fit with
/// the most inliers. The final model is refit on all inliers of the best iteration.
#[wasm_bindgen]
pub struct RansacModel {
    coefficients: Vec<f64>,
    intercept: f64,
    inlier_mask: Vec<bool>,
    n_inliers: usize,
    n_features: usize,
    n_iterations: usize,
}

#[wasm_bindgen]
impl RansacModel {
    #[wasm_bindgen(getter, js_name = "nInliers")]
    pub fn n_inliers(&self) -> usize { self.n_inliers }

    #[wasm_bindgen(getter, js_name = "nIterations")]
    pub fn n_iterations(&self) -> usize { self.n_iterations }

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

    /// Returns 1.0 for inlier samples, 0.0 for outliers.
    #[wasm_bindgen(js_name = "getInlierMask")]
    pub fn get_inlier_mask(&self) -> Vec<f64> {
        self.inlier_mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "RansacModel(n_features={}, n_inliers={}, n_iterations={})",
            self.n_features, self.n_inliers, self.n_iterations
        )
    }
}

/// Compute residual (absolute difference) between predicted and actual for a single sample.
#[inline]
fn residual_abs(coefficients: &[f64], intercept: f64, features: &[f64], actual: f64) -> f64 {
    let mut pred = intercept;
    for (f, &c) in coefficients.iter().enumerate() {
        pred += c * features[f];
    }
    (pred - actual).abs()
}

/// Fit a line through 2 points (univariate: n_features == 1).
/// Returns (slope, intercept).
fn fit_line_2pt(
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> Option<(f64, f64)> {
    let dx = x2 - x1;
    if dx.abs() < 1e-12 {
        return None; // Degenerate: points are vertically aligned
    }
    let slope = (y2 - y1) / dx;
    let intercept = y1 - slope * x1;
    Some((slope, intercept))
}

/// Solve a small linear system Ax = b via Gaussian elimination with partial pivoting.
/// `a` is an n x n matrix stored row-major, `b` is length n.
fn solve_linear_system(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    // Augmented matrix [A | b]
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
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
            return None; // Singular
        }
        if max_row != col {
            // Swap rows
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / aug[col * (n + 1) + col];
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
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

/// Least squares fit on inlier data (refit step).
/// Fits coefficients and intercept using the normal equations approach.
fn least_squares_fit(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    inlier_indices: &[usize],
) -> Option<(Vec<f64>, f64)> {
    let m = inlier_indices.len();
    if m < n_features + 1 {
        return None;
    }

    // Compute means for centering
    let mut mean_x = vec![0.0f64; n_features];
    let mut mean_y = 0.0f64;
    for &idx in inlier_indices {
        for f in 0..n_features {
            mean_x[f] += data[idx * n_features + f];
        }
        mean_y += targets[idx];
    }
    for f in 0..n_features {
        mean_x[f] /= m as f64;
    }
    mean_y /= m as f64;

    // Compute X'X and X'y on centered data
    let mut xt_x = vec![0.0f64; n_features * n_features];
    let mut xt_y = vec![0.0f64; n_features];

    for &idx in inlier_indices {
        for f1 in 0..n_features {
            let cx1 = data[idx * n_features + f1] - mean_x[f1];
            xt_y[f1] += cx1 * (targets[idx] - mean_y);
            for f2 in 0..n_features {
                let cx2 = data[idx * n_features + f2] - mean_x[f2];
                xt_x[f1 * n_features + f2] += cx1 * cx2;
            }
        }
    }

    let coefficients = solve_linear_system(&xt_x, n_features, &xt_y)?;

    // intercept = mean_y - sum(coef * mean_x)
    let mut intercept = mean_y;
    for f in 0..n_features {
        intercept -= coefficients[f] * mean_x[f];
    }

    Some((coefficients, intercept))
}

/// RANSAC implementation.
///
/// For univariate (n_features=1): randomly picks 2 points, fits a line, evaluates inliers.
/// For multivariate (n_features>1): randomly picks (n_features+1) points, solves linear system.
///
/// Keeps the model with the most inliers, then refits on all inliers.
pub fn ransac_impl(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    max_iterations: usize,
    residual_threshold: f64,
) -> Result<RansacModel, MlError> {
    let n_samples = validate_matrix(data, n_features)?;
    if targets.len() != n_samples {
        return Err(MlError::new("targets length must match number of samples"));
    }
    if max_iterations == 0 {
        return Err(MlError::new("max_iterations must be > 0"));
    }
    if residual_threshold <= 0.0 {
        return Err(MlError::new("residual_threshold must be > 0"));
    }
    if n_features == 1 && n_samples < 2 {
        return Err(MlError::new("need at least 2 samples for univariate RANSAC"));
    }
    if n_features > 1 && n_samples < n_features + 1 {
        return Err(MlError::new(format!(
            "need at least {} samples for {}-feature RANSAC",
            n_features + 1, n_features
        )));
    }

    let mut rng = Rng::from_data(data);
    let min_samples = if n_features == 1 { 2 } else { n_features + 1 };

    let mut best_inlier_count: usize = 0;
    let mut best_coefficients: Vec<f64> = Vec::new();
    let mut best_intercept: f64 = 0.0;
    let mut best_inlier_mask = vec![false; n_samples];

    for _iter in 0..max_iterations {
        // Pick random subset of min_samples points
        let mut indices = Vec::with_capacity(min_samples);
        for _ in 0..min_samples {
            indices.push(rng.next_usize(n_samples));
        }

        // Check for duplicate indices in multivariate case
        if n_features > 1 {
            let has_dup = (1..indices.len()).any(|i| indices[i..].contains(&indices[i - 1]));
            if has_dup {
                continue;
            }
        }

        let trial_result = if n_features == 1 {
            // Univariate: fit line through 2 points
            let x1 = data[indices[0] * n_features];
            let y1 = targets[indices[0]];
            let x2 = data[indices[1] * n_features];
            let y2 = targets[indices[1]];

            match fit_line_2pt(x1, y1, x2, y2) {
                Some((slope, intercept)) => Some((vec![slope], intercept)),
                None => continue,
            }
        } else {
            // Multivariate: solve linear system for (n_features+1) points
            // Build X matrix (min_samples x n_features) and y vector
            let x_mat: Vec<f64> = indices.iter()
                .flat_map(|&idx| {
                    (0..n_features).map(move |f| data[idx * n_features + f])
                })
                .collect();
            let y_vec: Vec<f64> = indices.iter().map(|&idx| targets[idx]).collect();

            // Build normal equations for exact fit: X'X * coef = X'y
            let mut xt_x = vec![0.0f64; n_features * n_features];
            let mut xt_y = vec![0.0f64; n_features];

            for i in 0..min_samples {
                for f1 in 0..n_features {
                    let val1 = x_mat[i * n_features + f1];
                    xt_y[f1] += val1 * y_vec[i];
                    for f2 in 0..n_features {
                        xt_x[f1 * n_features + f2] += val1 * x_mat[i * n_features + f2];
                    }
                }
            }

            match solve_linear_system(&xt_x, n_features, &xt_y) {
                Some(coef) => {
                    // Compute intercept from mean
                    let mut mean_x = vec![0.0f64; n_features];
                    let mut mean_y = 0.0f64;
                    for &idx in &indices {
                        for f in 0..n_features {
                            mean_x[f] += data[idx * n_features + f];
                        }
                        mean_y += targets[idx];
                    }
                    for f in 0..n_features {
                        mean_x[f] /= min_samples as f64;
                    }
                    mean_y /= min_samples as f64;

                    let mut intercept = mean_y;
                    for f in 0..n_features {
                        intercept -= coef[f] * mean_x[f];
                    }
                    Some((coef, intercept))
                }
                None => continue,
            }
        };

        let (trial_coef, trial_intercept) = match trial_result {
            Some(r) => r,
            None => continue,
        };

        // Count inliers
        let mut inlier_mask = vec![false; n_samples];
        let mut inlier_count = 0usize;

        for i in 0..n_samples {
            let features = &data[i * n_features..(i + 1) * n_features];
            let r = residual_abs(&trial_coef, trial_intercept, features, targets[i]);
            if r < residual_threshold {
                inlier_mask[i] = true;
                inlier_count += 1;
            }
        }

        if inlier_count > best_inlier_count {
            best_inlier_count = inlier_count;
            best_coefficients = trial_coef;
            best_intercept = trial_intercept;
            best_inlier_mask = inlier_mask;
        }
    }

    if best_inlier_count < min_samples {
        return Err(MlError::new("RANSAC failed to find a good model"));
    }

    // Refit on all inliers of the best model
    let inlier_indices: Vec<usize> = best_inlier_mask.iter()
        .enumerate()
        .filter(|(_, &is_inlier)| is_inlier)
        .map(|(idx, _)| idx)
        .collect();

    let (final_coef, final_intercept) = least_squares_fit(
        data, n_features, targets, &inlier_indices,
    ).unwrap_or((best_coefficients, best_intercept));

    Ok(RansacModel {
        coefficients: final_coef,
        intercept: final_intercept,
        inlier_mask: best_inlier_mask,
        n_inliers: best_inlier_count,
        n_features,
        n_iterations: max_iterations,
    })
}

#[wasm_bindgen(js_name = "ransac")]
pub fn ransac(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    max_iterations: usize,
    residual_threshold: f64,
) -> Result<RansacModel, JsError> {
    ransac_impl(data, n_features, targets, max_iterations, residual_threshold)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ransac_basic_fit() {
        // y = 2x + 1 with no outliers
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let targets: Vec<f64> = (1..=10).map(|i| 2.0 * i as f64 + 1.0).collect();

        let model = ransac_impl(&data, 1, &targets, 100, 0.5).unwrap();

        assert!((model.coefficients[0] - 2.0).abs() < 0.1, "slope = {}", model.coefficients[0]);
        assert!((model.intercept - 1.0).abs() < 0.5, "intercept = {}", model.intercept);
        assert_eq!(model.n_inliers, 10); // All points are inliers
    }

    #[test]
    fn test_robust_to_outliers() {
        // y = 3x with 3 outliers injected
        let mut data: Vec<f64> = Vec::new();
        let mut targets: Vec<f64> = Vec::new();

        // 17 clean inlier points: y = 3x
        for i in 1..=17 {
            data.push(i as f64);
            targets.push(3.0 * i as f64);
        }

        // 3 outliers far from the line
        data.push(5.0); targets.push(100.0);
        data.push(10.0); targets.push(-50.0);
        data.push(15.0); targets.push(200.0);

        let model = ransac_impl(&data, 1, &targets, 200, 5.0).unwrap();

        // Slope should be close to 3.0 even with outliers
        assert!(
            (model.coefficients[0] - 3.0).abs() < 0.5,
            "slope = {}, expected ~3.0",
            model.coefficients[0]
        );

        // Most points should be inliers (17 clean + possibly 0-1 outliers near the threshold)
        assert!(
            model.n_inliers >= 15,
            "n_inliers = {}, expected >= 15",
            model.n_inliers
        );

        // Outliers should not dominate predictions
        let preds = model.predict(&data[..17]);
        for (i, p) in preds.iter().enumerate() {
            let expected = 3.0 * (i + 1) as f64;
            assert!(
                (p - expected).abs() < 2.0,
                "prediction {} far from expected {}",
                p, expected
            );
        }
    }

    #[test]
    fn test_ransac_multivariate() {
        // y = 2*x1 + 3*x2 + 1
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
            .map(|row| 2.0 * row[0] + 3.0 * row[1] + 1.0)
            .collect();

        let model = ransac_impl(&data, 2, &targets, 200, 0.5).unwrap();

        assert!((model.coefficients[0] - 2.0).abs() < 0.5, "coef[0] = {}", model.coefficients[0]);
        assert!((model.coefficients[1] - 3.0).abs() < 0.5, "coef[1] = {}", model.coefficients[1]);
    }

    #[test]
    fn test_ransac_inlier_mask() {
        let data: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let targets: Vec<f64> = (1..=5).map(|i| 2.0 * i as f64).collect();

        let model = ransac_impl(&data, 1, &targets, 50, 1.0).unwrap();

        let mask = model.get_inlier_mask();
        assert_eq!(mask.len(), 5);
        // All points should be inliers since they lie on the line
        assert!(mask.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_ransac_insufficient_samples() {
        let data = vec![1.0];
        let targets = vec![2.0];
        assert!(ransac_impl(&data, 1, &targets, 10, 1.0).is_err());
    }

    #[test]
    fn test_ransac_invalid_threshold() {
        let data = vec![1.0, 2.0];
        let targets = vec![2.0, 4.0];
        assert!(ransac_impl(&data, 1, &targets, 10, 0.0).is_err());
    }
}
