use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::validate_matrix;

/// Elastic Net Regression - Combined L1+L2 regularized linear regression
///
/// Combines Lasso (L1) and Ridge (L2) penalties controlled by `l1_ratio`:
/// - l1_ratio = 0.0 -> pure Ridge (L2 only)
/// - l1_ratio = 1.0 -> pure Lasso (L1 only)
/// - l1_ratio = 0.5 -> equal mix of both
#[wasm_bindgen]
pub struct ElasticNetModel {
    coefficients: Vec<f64>,
    intercept: f64,
    n_features: usize,
    alpha: f64,
    l1_ratio: f64,
}

#[wasm_bindgen]
impl ElasticNetModel {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(getter, js_name = "coefficients")]
    pub fn coefficients_js(&self) -> Vec<f64> { self.coefficients.clone() }

    #[wasm_bindgen(getter, js_name = "intercept")]
    pub fn intercept_js(&self) -> f64 { self.intercept }

    #[wasm_bindgen(getter, js_name = "alpha")]
    pub fn alpha_js(&self) -> f64 { self.alpha }

    #[wasm_bindgen(getter, js_name = "l1Ratio")]
    pub fn l1_ratio_js(&self) -> f64 { self.l1_ratio }

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
            "ElasticNetModel(n_features={}, alpha={}, l1_ratio={})",
            self.n_features, self.alpha, self.l1_ratio
        )
    }
}

/// Soft threshold operator: S(z, gamma) = sign(z) * max(|z| - gamma, 0)
#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z.abs() <= gamma {
        0.0
    } else {
        z.abs() - gamma
    }.copysign(z)
}

/// Elastic Net regression using coordinate descent.
///
/// Minimizes: (1 / (2n)) * ||y - Xw||^2 + alpha * l1_ratio * ||w||_1
///           + (alpha * (1 - l1_ratio) / 2) * ||w||^2
///
/// Coordinate descent update for feature j:
///   rho_j = (1/n) * sum(x_ij * residual_i)
///   new_coef_j = S(rho_j, alpha * l1_ratio) / (x_j^T x_j / n + alpha * (1 - l1_ratio))
pub fn elastic_net_impl(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ElasticNetModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if targets.len() != n {
        return Err(MlError::new("targets length must match number of samples"));
    }
    if alpha < 0.0 {
        return Err(MlError::new("alpha must be >= 0"));
    }
    if !(0.0..=1.0).contains(&l1_ratio) {
        return Err(MlError::new("l1_ratio must be in [0, 1]"));
    }

    let l1_penalty = alpha * l1_ratio;
    let l2_penalty = alpha * (1.0 - l1_ratio);

    // Center data and targets
    let mut mean_x = vec![0.0f64; n_features];
    for f in 0..n_features {
        for i in 0..n {
            mean_x[f] += data[i * n_features + f];
        }
        mean_x[f] /= n as f64;
    }

    let mut mean_y = 0.0;
    for &t in targets { mean_y += t; }
    mean_y /= n as f64;

    let mut centered_data = vec![0.0f64; data.len()];
    for i in 0..n {
        for f in 0..n_features {
            centered_data[i * n_features + f] = data[i * n_features + f] - mean_x[f];
        }
    }

    let centered_targets: Vec<f64> = targets.iter().map(|&t| t - mean_y).collect();

    // Precompute feature norms squared (x_j^T x_j)
    let mut feature_norm_sq = vec![0.0f64; n_features];
    for f in 0..n_features {
        for i in 0..n {
            feature_norm_sq[f] += centered_data[i * n_features + f] * centered_data[i * n_features + f];
        }
    }

    // Coordinate descent
    let mut coefficients = vec![0.0f64; n_features];
    let mut residuals = centered_targets.clone();

    for _iter in 0..max_iter {
        let mut max_change: f64 = 0.0;

        for f in 0..n_features {
            let norm_sq_n = feature_norm_sq[f] / n as f64;
            if norm_sq_n < 1e-12 {
                continue;
            }

            let old_coef = coefficients[f];

            // rho_j = (1/n) * sum(x_ij * (residual_i + old_coef_j * x_ij))
            // = (1/n) * sum(x_ij * residual_i) + old_coef * (x_j^T x_j / n)
            let mut rho = 0.0;
            for i in 0..n {
                rho += centered_data[i * n_features + f] * residuals[i];
            }
            rho = rho / n as f64 + old_coef * norm_sq_n;

            // Coordinate descent update with both L1 and L2 penalties:
            // new_coef_j = S(rho_j, alpha * l1_ratio) / (x_j^T x_j / n + alpha * (1 - l1_ratio))
            let new_coef = soft_threshold(rho, l1_penalty) / (norm_sq_n + l2_penalty);

            coefficients[f] = new_coef;

            // Update residuals
            let delta = new_coef - old_coef;
            for i in 0..n {
                residuals[i] -= delta * centered_data[i * n_features + f];
            }

            max_change = max_change.max(delta.abs());
        }

        if max_change < tol {
            break;
        }
    }

    // Compute intercept
    let mut intercept = mean_y;
    for f in 0..n_features {
        intercept -= coefficients[f] * mean_x[f];
    }

    Ok(ElasticNetModel {
        coefficients,
        intercept,
        n_features,
        alpha,
        l1_ratio,
    })
}

#[wasm_bindgen(js_name = "elasticNet")]
pub fn elastic_net(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ElasticNetModel, JsError> {
    elastic_net_impl(data, n_features, targets, alpha, l1_ratio, max_iter, tol)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y = 2*x1 + 0*x2 + noise
    /// Elastic net should identify x1 as important and shrink x2 toward zero.
    #[test]
    fn test_elastic_net_fits() {
        let data = vec![
            1.0, 5.0,
            2.0, 3.0,
            3.0, 7.0,
            4.0, 2.0,
            5.0, 8.0,
            6.0, 1.0,
            7.0, 9.0,
            8.0, 4.0,
        ];
        let targets: Vec<f64> = data.iter()
            .step_by(2)
            .map(|&x| 2.0 * x)
            .collect();

        let model = elastic_net_impl(&data, 2, &targets, 0.1, 0.5, 1000, 1e-6).unwrap();
        let preds = model.predict(&data);

        for (p, &t) in preds.iter().zip(&targets) {
            assert!((p - t).abs() < 0.5, "prediction {} far from target {}", p, t);
        }
    }

    /// With l1_ratio=0.0, elastic net reduces to Ridge regression.
    /// Coefficients should be small but non-zero (no sparsity).
    #[test]
    fn test_ridge_like() {
        // x1 is the signal, x2 and x3 are independent noise (no multicollinearity)
        let data = vec![
            1.0, 3.0, 7.0,
            2.0, 1.0, 5.0,
            3.0, 8.0, 2.0,
            4.0, 4.0, 9.0,
            5.0, 6.0, 1.0,
        ];
        // Only first feature matters: y = 3*x1
        let targets: Vec<f64> = (1..=5).map(|i| 3.0 * i as f64).collect();

        let model = elastic_net_impl(&data, 3, &targets, 1.0, 0.0, 1000, 1e-6).unwrap();

        // All coefficients should be non-zero (Ridge does not produce sparsity)
        let non_zero = model.coefficients.iter().filter(|&&c| c.abs() > 1e-10).count();
        assert_eq!(non_zero, 3, "Ridge-like elastic net should keep all coefficients non-zero");

        // First coefficient should be the largest in magnitude (it's the true signal)
        assert!(model.coefficients[0].abs() > model.coefficients[1].abs(),
            "coef[0]={} should dominate coef[1]={}", model.coefficients[0], model.coefficients[1]);
        assert!(model.coefficients[0].abs() > model.coefficients[2].abs(),
            "coef[0]={} should dominate coef[2]={}", model.coefficients[0], model.coefficients[2]);
    }

    /// With l1_ratio=1.0, elastic net reduces to Lasso.
    /// Should produce sparse coefficients.
    #[test]
    fn test_lasso_like() {
        let data = vec![
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0,
            3.0, 0.0, 0.0,
            4.0, 0.0, 0.0,
            5.0, 0.0, 0.0,
        ];
        // y = 2*x1, other features are pure noise
        let targets: Vec<f64> = (1..=5).map(|i| 2.0 * i as f64).collect();

        let model = elastic_net_impl(&data, 3, &targets, 1.0, 1.0, 1000, 1e-6).unwrap();

        // With l1_ratio=1.0 (pure Lasso) and high alpha, coefficients 1 and 2 should be zero
        let non_zero = model.coefficients.iter().filter(|&&c| c.abs() > 1e-10).count();
        assert!(non_zero <= 2, "Lasso-like elastic net should produce sparse coefficients");
    }

    /// With l1_ratio=0.5, elastic net should combine both effects:
    /// some sparsity from L1 + coefficient shrinkage from L2.
    #[test]
    fn test_balanced() {
        // x1 is the signal, x2 and x3 are independent noise
        let data = vec![
            1.0, 3.0, 0.5,
            2.0, 7.0, 1.2,
            3.0, 2.0, 0.8,
            4.0, 9.0, 1.5,
            5.0, 4.0, 0.3,
            6.0, 8.0, 1.1,
            7.0, 1.0, 0.9,
            8.0, 6.0, 1.4,
        ];
        // y = 1*x1 + 0*x2 + 0*x3
        let targets: Vec<f64> = (1..=8).map(|i| i as f64).collect();

        let model = elastic_net_impl(&data, 3, &targets, 0.5, 0.5, 1000, 1e-6).unwrap();

        // First coefficient should be the primary contributor (close to 1.0)
        assert!(
            (model.coefficients[0] - 1.0).abs() < 0.5,
            "coef[0] = {}, expected ~1.0",
            model.coefficients[0]
        );

        // Should produce reasonable predictions
        let preds = model.predict(&data);
        for (p, &t) in preds.iter().zip(&targets) {
            assert!((p - t).abs() < 1.0, "prediction {} far from target {}", p, t);
        }
    }
}
