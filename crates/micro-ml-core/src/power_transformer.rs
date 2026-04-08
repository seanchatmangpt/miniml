use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

/// Yeo-Johnson power transformation for a single value.
///
/// Handles both positive and negative data, unlike Box-Cox which requires x > 0.
///
/// For x >= 0:
///   if lambda != 0: ((x + 1)^lambda - 1) / lambda
///   if lambda == 0: ln(x + 1)
///
/// For x < 0:
///   if lambda != 2: -((-x + 1)^(2 - lambda) - 1) / (2 - lambda)
///   if lambda == 2: -ln(-x + 1)
pub fn yeo_johnson_transform(x: f64, lam: f64) -> f64 {
    if x >= 0.0 {
        if lam.abs() < 1e-10 {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lam) - 1.0) / lam
        }
    } else {
        if (lam - 2.0).abs() < 1e-10 {
            -((-x + 1.0).ln())
        } else {
            -((-x + 1.0).powf(2.0 - lam) - 1.0) / (2.0 - lam)
        }
    }
}

/// Inverse Yeo-Johnson transformation: recover original x from transformed y.
///
/// For y >= 0:
///   if lambda != 0: (y * lambda + 1)^(1/lambda) - 1
///   if lambda == 0: exp(y) - 1
///
/// For y < 0:
///   if lambda != 2: 1 - (1 - (2 - lambda) * y)^(1/(2-lambda))
///   if lambda == 2: 1 - exp(-y)
pub fn yeo_johnson_inverse(y: f64, lam: f64) -> f64 {
    if y >= 0.0 {
        if lam.abs() < 1e-10 {
            y.exp() - 1.0
        } else {
            (y * lam + 1.0).powf(1.0 / lam) - 1.0
        }
    } else {
        if (lam - 2.0).abs() < 1e-10 {
            1.0 - (-y).exp()
        } else {
            1.0 - (1.0 - (2.0 - lam) * y).powf(1.0 / (2.0 - lam))
        }
    }
}

/// Compute skewness of a feature column (unbiased estimator).
fn skewness(values: &[f64], mean: f64, std: f64) -> f64 {
    if std.abs() < 1e-15 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mut sum = 0.0;
    for &v in values {
        let z = (v - mean) / std;
        sum += z * z * z;
    }
    // Unbiased skewness: (n / ((n-1)*(n-2))) * sum
    sum * n / ((n - 1.0) * (n - 2.0))
}

/// Find optimal lambda for one feature column by minimizing absolute skewness.
///
/// Scans lambdas in [-5, 5] with step 0.1 and picks the one whose transformed
/// output has skewness closest to zero. This is a lightweight proxy for MLE
/// since we cannot depend on scipy in a WASM environment.
fn find_optimal_lambda(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 3 {
        return 1.0;
    }

    // Compute mean and std of raw feature (std used only to check variance > 0)
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let var: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let _std = var.sqrt();

    let mut best_lambda = 1.0;
    let mut best_skew = f64::MAX;

    // Scan lambdas from -5.0 to 5.0 with step 0.1
    let steps = 100;
    for i in 0..=steps {
        let lam = -5.0 + (10.0 * i as f64 / steps as f64);

        // Transform values with this lambda
        let transformed: Vec<f64> = values.iter().map(|&x| yeo_johnson_transform(x, lam)).collect();

        let t_mean: f64 = transformed.iter().sum::<f64>() / n as f64;
        let t_var: f64 = transformed.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / n as f64;
        let t_std = t_var.sqrt();

        let sk = skewness(&transformed, t_mean, t_std);
        let abs_sk = sk.abs();

        if abs_sk < best_skew {
            best_skew = abs_sk;
            best_lambda = lam;
        }
    }

    // Fine-tune: scan around best_lambda with step 0.01
    let coarse = best_lambda;
    let fine_steps = 20;
    for i in 0..=fine_steps {
        let lam = (coarse - 0.1) + (0.2 * i as f64 / fine_steps as f64);

        let transformed: Vec<f64> = values.iter().map(|&x| yeo_johnson_transform(x, lam)).collect();

        let t_mean: f64 = transformed.iter().sum::<f64>() / n as f64;
        let t_var: f64 = transformed.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / n as f64;
        let t_std = t_var.sqrt();

        let sk = skewness(&transformed, t_mean, t_std);
        let abs_sk = sk.abs();

        if abs_sk < best_skew {
            best_skew = abs_sk;
            best_lambda = lam;
        }
    }

    best_lambda
}

/// PowerTransformer applies Yeo-Johnson power transformation to make data
/// more Gaussian-like. Each feature gets its own optimal lambda estimated
/// during `fit`.
///
/// Only Yeo-Johnson is supported (not Box-Cox) because Box-Cox requires
/// strictly positive data and Yeo-Johnson works for any real values.
#[wasm_bindgen]
pub struct PowerTransformer {
    lambdas_: Vec<f64>,
    n_features: usize,
    method: String,
}

#[wasm_bindgen]
impl PowerTransformer {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    #[wasm_bindgen(getter)]
    pub fn method(&self) -> String {
        self.method.clone()
    }

    #[wasm_bindgen(js_name = "getLambdas")]
    pub fn get_lambdas(&self) -> Vec<f64> {
        self.lambdas_.clone()
    }

    /// Find optimal lambda per feature using skewness minimization.
    #[wasm_bindgen]
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsError> {
        power_transformer_fit_impl(self, data).map_err(|e| JsError::new(&e.message))
    }

    /// Apply the fitted Yeo-Johnson transformation.
    #[wasm_bindgen]
    pub fn transform(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(data.len());

        for i in 0..n {
            for j in 0..self.n_features {
                let val = mat_get(data, self.n_features, i, j);
                let transformed = yeo_johnson_transform(val, self.lambdas_[j]);
                result.push(transformed);
            }
        }

        result
    }

    /// Reverse the Yeo-Johnson transformation.
    #[wasm_bindgen(js_name = "inverseTransform")]
    pub fn inverse_transform(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(data.len());

        for i in 0..n {
            for j in 0..self.n_features {
                let val = mat_get(data, self.n_features, i, j);
                let original = yeo_johnson_inverse(val, self.lambdas_[j]);
                result.push(original);
            }
        }

        result
    }

    /// Fit and transform in one step.
    #[wasm_bindgen(js_name = "fitTransform")]
    pub fn fit_transform(&mut self, data: &[f64]) -> Result<Vec<f64>, JsError> {
        self.fit(data)?;
        Ok(self.transform(data))
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        let lambdas_str: Vec<String> = self.lambdas_.iter().map(|l| format!("{:.4}", l)).collect();
        format!(
            "PowerTransformer(features={}, method={}, lambdas=[{}])",
            self.n_features,
            self.method,
            lambdas_str.join(", ")
        )
    }
}

fn power_transformer_fit_impl(
    transformer: &mut PowerTransformer,
    data: &[f64],
) -> Result<(), MlError> {
    let n = validate_matrix(data, transformer.n_features)?;

    if transformer.method != "yeo-johnson" {
        return Err(MlError::new(format!(
            "Unsupported method '{}'. Only 'yeo-johnson' is available.",
            transformer.method
        )));
    }

    if n < 3 {
        return Err(MlError::new("Need at least 3 samples to estimate lambda"));
    }

    // Estimate optimal lambda per feature
    for j in 0..transformer.n_features {
        let feature_values: Vec<f64> = (0..n)
            .map(|i| mat_get(data, transformer.n_features, i, j))
            .collect();

        let opt_lambda = find_optimal_lambda(&feature_values);
        transformer.lambdas_[j] = opt_lambda;
    }

    Ok(())
}

/// Create a new PowerTransformer for the given number of features.
///
/// Only "yeo-johnson" is supported. The method parameter is accepted for
/// API compatibility but will produce an error at fit time if not "yeo-johnson".
#[wasm_bindgen(js_name = "powerTransformer")]
pub fn power_transformer(n_features: usize, method: &str) -> PowerTransformer {
    PowerTransformer {
        lambdas_: vec![1.0; n_features],
        n_features,
        method: method.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yeo_johnson_identity_lambda_one() {
        // lambda=1 should give identity-like behavior near zero
        let x = 0.5;
        let y = yeo_johnson_transform(x, 1.0);
        // ((x+1)^1 - 1) / 1 = x
        assert!((y - x).abs() < 1e-10);
    }

    #[test]
    fn test_yeo_johnson_lambda_zero_positive() {
        let x = 1.0;
        let y = yeo_johnson_transform(x, 0.0);
        // ln(x + 1) = ln(2)
        assert!((y - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_yeo_johnson_negative_x_lambda_one() {
        let x = -0.5;
        let y = yeo_johnson_transform(x, 1.0);
        // -((-x+1)^(2-1) - 1) / (2-1) = -((1.5)^1 - 1) = -0.5
        assert!((y - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_yeo_johnson_negative_x_lambda_two() {
        let x = -0.5;
        let y = yeo_johnson_transform(x, 2.0);
        // -ln(-x + 1) = -ln(1.5)
        assert!((y - (-(1.5_f64.ln()))).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_identity_lambda_one() {
        let x = 0.5;
        let y = yeo_johnson_transform(x, 1.0);
        let x_back = yeo_johnson_inverse(y, 1.0);
        assert!((x - x_back).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_lambda_zero() {
        let x = 1.0;
        let y = yeo_johnson_transform(x, 0.0);
        let x_back = yeo_johnson_inverse(y, 0.0);
        assert!((x - x_back).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_negative_value() {
        let x = -0.5;
        let y = yeo_johnson_transform(x, 1.0);
        let x_back = yeo_johnson_inverse(y, 1.0);
        assert!((x - x_back).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_negative_value_lambda_two() {
        let x = -0.3;
        let y = yeo_johnson_transform(x, 2.0);
        let x_back = yeo_johnson_inverse(y, 2.0);
        assert!((x - x_back).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_roundtrip_various_lambdas() {
        let test_values = [-2.0, -0.5, 0.0, 0.5, 1.0, 3.0, 10.0];
        let lambdas = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0];

        for &x in &test_values {
            for &lam in &lambdas {
                let y = yeo_johnson_transform(x, lam);
                let x_back = yeo_johnson_inverse(y, lam);
                assert!(
                    (x - x_back).abs() < 1e-8,
                    "Roundtrip failed: x={}, lambda={}, got back {}",
                    x,
                    lam,
                    x_back
                );
            }
        }
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric data should have skewness near 0
        let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mean = 0.0;
        let std = (10.0 / 5.0_f64).sqrt();
        let sk = skewness(&values, mean, std);
        assert!(sk.abs() < 1e-10, "Symmetric data skewness should be ~0, got {}", sk);
    }

    #[test]
    fn test_skewness_right_skewed() {
        // Right-skewed data should have positive skewness
        let values = vec![1.0, 1.0, 1.0, 2.0, 10.0];
        let mean: f64 = values.iter().sum::<f64>() / 5.0;
        let var: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 5.0;
        let std = var.sqrt();
        let sk = skewness(&values, mean, std);
        assert!(sk > 0.0, "Right-skewed data should have positive skewness, got {}", sk);
    }

    #[test]
    fn test_find_optimal_lambda_reduces_skewness() {
        // Create right-skewed data (exponential-like)
        let values: Vec<f64> = (0..100).map(|i| (i as f64).exp() * 0.01).collect();
        let raw_mean: f64 = values.iter().sum::<f64>() / 100.0;
        let raw_var: f64 = values.iter().map(|v| (v - raw_mean).powi(2)).sum::<f64>() / 100.0;
        let raw_std = raw_var.sqrt();
        let raw_skew = skewness(&values, raw_mean, raw_std).abs();

        let lam = find_optimal_lambda(&values);
        let transformed: Vec<f64> = values.iter().map(|&x| yeo_johnson_transform(x, lam)).collect();
        let t_mean: f64 = transformed.iter().sum::<f64>() / 100.0;
        let t_var: f64 = transformed.iter().map(|v| (v - t_mean).powi(2)).sum::<f64>() / 100.0;
        let t_std = t_var.sqrt();
        let t_skew = skewness(&transformed, t_mean, t_std).abs();

        assert!(
            t_skew < raw_skew,
            "Transformed skewness ({}) should be less than raw ({})",
            t_skew,
            raw_skew
        );
    }

    #[test]
    fn test_constructor() {
        let pt = power_transformer(3, "yeo-johnson");
        assert_eq!(pt.n_features(), 3);
        assert_eq!(pt.method(), "yeo-johnson");
        assert_eq!(pt.get_lambdas(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_fit_fails_empty_data() {
        let mut pt = power_transformer(2, "yeo-johnson");
        assert!(power_transformer_fit_impl(&mut pt, &[]).is_err());
    }

    #[test]
    fn test_fit_fails_wrong_method() {
        let mut pt = power_transformer(2, "box-cox");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(power_transformer_fit_impl(&mut pt, &data).is_err());
    }

    #[test]
    fn test_fit_fails_too_few_samples() {
        let mut pt = power_transformer(2, "yeo-johnson");
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 2 samples < 3 minimum
        assert!(power_transformer_fit_impl(&mut pt, &data).is_err());
    }

    #[test]
    fn test_fit_transform_roundtrip() {
        // Right-skewed data
        let data = vec![
            0.1, 100.0,
            0.2, 200.0,
            0.3, 300.0,
            1.0, 400.0,
            2.0, 500.0,
            5.0, 600.0,
            10.0, 700.0,
            50.0, 800.0,
            100.0, 900.0,
            200.0, 1000.0,
        ];
        let mut pt = power_transformer(2, "yeo-johnson");
        let transformed = pt.fit_transform(&data).unwrap();
        let restored = pt.inverse_transform(&transformed);

        for i in 0..data.len() {
            assert!(
                (data[i] - restored[i]).abs() < 1e-6,
                "Roundtrip failed at index {}: expected {}, got {}",
                i,
                data[i],
                restored[i]
            );
        }
    }

    #[test]
    fn test_fit_sets_lambdas() {
        let data: Vec<f64> = (0..100).flat_map(|i| {
            let v = (i as f64).exp() * 0.01;
            vec![v, -v]
        }).collect();

        let mut pt = power_transformer(2, "yeo-johnson");
        pt.fit(&data).unwrap();

        let lambdas = pt.get_lambdas();
        assert_eq!(lambdas.len(), 2);
        // Lambdas should not all be the default 1.0 after fitting skewed data
        let both_default = (lambdas[0] - 1.0).abs() < 1e-6 && (lambdas[1] - 1.0).abs() < 1e-6;
        assert!(!both_default, "Lambdas should be optimized, got [{}, {}]", lambdas[0], lambdas[1]);
    }

    #[test]
    fn test_transform_without_fit_uses_default_lambda() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pt = power_transformer(2, "yeo-johnson");
        // Without fit, lambda=1.0 for all features
        let transformed = pt.transform(&data);
        // With lambda=1.0: ((x+1)^1 - 1)/1 = x for x >= 0
        for i in 0..data.len() {
            assert!(
                (transformed[i] - data[i]).abs() < 1e-10,
                "Default lambda=1 should be identity for non-negative x"
            );
        }
    }

    #[test]
    fn test_to_string() {
        let pt = power_transformer(2, "yeo-johnson");
        let s = pt.to_string_js();
        assert!(s.contains("PowerTransformer"));
        assert!(s.contains("features=2"));
        assert!(s.contains("yeo-johnson"));
    }

    #[test]
    fn test_negative_data_handling() {
        // Data with negative values should work fine with Yeo-Johnson
        let data = vec![
            -5.0, -1.0,
            -3.0, 0.0,
            -1.0, 1.0,
            0.0, 2.0,
            2.0, 5.0,
            5.0, 10.0,
            10.0, 20.0,
            20.0, 50.0,
            50.0, 100.0,
            100.0, 200.0,
        ];
        let mut pt = power_transformer(2, "yeo-johnson");
        let transformed = pt.fit_transform(&data).unwrap();
        let restored = pt.inverse_transform(&transformed);

        for i in 0..data.len() {
            assert!(
                (data[i] - restored[i]).abs() < 1e-6,
                "Negative data roundtrip failed at index {}: expected {}, got {}",
                i,
                data[i],
                restored[i]
            );
        }
    }
}
