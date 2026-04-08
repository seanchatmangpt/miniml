/// Weibull Survival Analysis
///
/// Provides survival analysis functions for time-to-event modeling:
/// - Weibull distribution fitting (method of moments)
/// - Hazard rate estimation
/// - Survival probability calculation
/// - Gamma function approximation (Lanczos)
///
/// Use cases: Reliability engineering, churn prediction, time-to-event modeling
use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::validate_matrix;

/// Weibull distribution parameters
#[wasm_bindgen]
pub struct WeibullModel {
    shape: f64,  // k: shape parameter (k < 1: decreasing hazard, k > 1: increasing)
    scale: f64,  // λ: scale parameter in same units as data
}

#[wasm_bindgen]
impl WeibullModel {
    #[wasm_bindgen(getter, js_name = "shape")]
    pub fn shape(&self) -> f64 { self.shape }

    #[wasm_bindgen(getter, js_name = "scale")]
    pub fn scale(&self) -> f64 { self.scale }

    /// Hazard rate at time t: h(t) = (k/λ) * (t/λ)^(k-1)
    #[wasm_bindgen(js_name = "hazardRate")]
    pub fn hazard_rate(&self, t: f64) -> f64 {
        if t <= 0.0 || self.scale <= 0.0 {
            return 0.0;
        }
        let t_over_lambda = t / self.scale;
        (self.shape / self.scale) * t_over_lambda.powf(self.shape - 1.0)
    }

    /// Survival probability: S(t) = exp(-(t/λ)^k)
    #[wasm_bindgen(js_name = "survivalProbability")]
    pub fn survival_probability(&self, t: f64) -> f64 {
        if t < 0.0 || self.scale <= 0.0 {
            return 1.0;
        }
        let t_over_lambda = t / self.scale;
        (-t_over_lambda.powf(self.shape)).exp()
    }

    /// Cumulative hazard: H(t) = (t/λ)^k
    #[wasm_bindgen(js_name = "cumulativeHazard")]
    pub fn cumulative_hazard(&self, t: f64) -> f64 {
        if t < 0.0 || self.scale <= 0.0 {
            return 0.0;
        }
        (t / self.scale).powf(self.shape)
    }

    /// Conditional median remaining time given elapsed time t
    /// Solves: P(T > t + r | T > t) = 0.5
    #[wasm_bindgen(js_name = "medianRemaining")]
    pub fn median_remaining(&self, elapsed: f64) -> f64 {
        if elapsed < 0.0 || self.scale <= 0.0 {
            return 0.0;
        }
        let t = elapsed.max(1.0); // avoid t=0 singularity when k < 1
        let cumulative_hazard = (t / self.scale).powf(self.shape);
        let lambda = self.scale;
        let k = self.shape;
        // ((t+r)/λ)^k = (t/λ)^k + ln(2)
        // t+r = λ * ((t/λ)^k + ln(2))^(1/k)
        let median_remaining = lambda * (cumulative_hazard + std::f64::consts::LN_2).powf(1.0 / k) - t;
        median_remaining.max(0.0)
    }

    /// Percentile (p-th quantile) of the distribution
    /// F(t) = 1 - exp(-(t/λ)^k) = p
    /// t = λ * (-ln(1-p))^(1/k)
    #[wasm_bindgen]
    pub fn percentile(&self, p: f64) -> f64 {
        if p < 0.0 || p >= 1.0 || self.scale <= 0.0 {
            return f64::NAN;
        }
        self.scale * (-((1.0 - p).ln())).powf(1.0 / self.shape)
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("WeibullModel(shape={}, scale={})", self.shape, self.scale)
    }
}

/// Fit Weibull distribution to data using method of moments
///
/// # Parameters
/// - `data`: Flat array of time-to-event values (survival times)
/// - `n_samples`: Number of samples (data.len() / 1)
///
/// # Returns
/// WeibullModel with fitted shape and scale parameters
///
/// # Method
/// Uses coefficient of variation (cv = σ/μ) to estimate shape:
///   k ≈ (cv)^(-1.086)
/// Scale: λ = μ / Γ(1 + 1/k)
#[wasm_bindgen(js_name = "weibullFit")]
pub fn weibull_fit(
    data: &[f64],
    n_samples: usize,
) -> Result<WeibullModel, JsError> {
    weibull_fit_impl(data, n_samples)
        .map_err(|e| JsError::new(&e.message))
}

pub fn weibull_fit_impl(
    data: &[f64],
    n_samples: usize,
) -> Result<WeibullModel, MlError> {
    if data.len() != n_samples {
        return Err(MlError::new("data length must equal n_samples"));
    }
    if n_samples == 0 {
        return Err(MlError::new("data must not be empty"));
    }

    // Compute mean and standard deviation
    let n = n_samples as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if mean <= 0.0 {
        return Err(MlError::new("mean must be positive for Weibull fitting"));
    }

    // Coefficient of variation
    let cv = std / mean;

    // Estimate shape parameter from cv
    let shape = weibull_shape_from_cv(cv);

    // Estimate scale parameter
    let scale = weibull_scale(mean, shape);

    if scale <= 0.0 {
        return Err(MlError::new("estimated scale must be positive"));
    }

    Ok(WeibullModel { shape, scale })
}

/// Approximate Weibull shape k from coefficient of variation (cv = σ/μ)
/// Uses closed-form approximation: k ≈ (cv)^(-1.086)
/// Accurate to ~2% for 0.2 ≤ cv ≤ 5
fn weibull_shape_from_cv(cv: f64) -> f64 {
    if cv <= 0.0 || !cv.is_finite() {
        return 1.0; // degenerate → exponential distribution
    }
    cv.powf(-1.086).max(0.1).min(20.0)
}

/// Weibull scale λ from mean and shape: λ = mean / Γ(1 + 1/k)
fn weibull_scale(mean: f64, k: f64) -> f64 {
    let g = gamma_lanczos(1.0 + 1.0 / k);
    if g > 0.0 { mean / g } else { mean }
}

/// Lanczos approximation of Γ(x) for x > 0
///
/// Uses g=7 Lanczos coefficients for high accuracy
pub fn gamma_lanczos(x: f64) -> f64 {
    // Lanczos coefficients (g=7)
    const P: [f64; 8] = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        let pi_x = std::f64::consts::PI * x;
        let sin_pi_x = pi_x.sin();
        if sin_pi_x == 0.0 {
            return f64::INFINITY; // Pole
        }
        std::f64::consts::PI / (sin_pi_x * gamma_lanczos(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = 0.99999999999980993_f64;
        for (i, &p) in P.iter().enumerate() {
            a += p / (x + i as f64 + 1.0);
        }
        let t = x + 7.5; // g + 0.5
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

/// Compute survival probability for multiple time points
#[wasm_bindgen(js_name = "weibullSurvival")]
pub fn weibull_survival(
    data: &[f64],
    n_samples: usize,
    time_points: &[f64],
) -> Result<Vec<f64>, JsError> {
    let model = weibull_fit_impl(data, n_samples)
        .map_err(|e| JsError::new(&e.message))?;

    let result: Vec<f64> = time_points.iter()
        .map(|&t| model.survival_probability(t))
        .collect();

    Ok(result)
}

/// Compute hazard rate for multiple time points
#[wasm_bindgen(js_name = "weibullHazardRates")]
pub fn weibull_hazard_rates(
    data: &[f64],
    n_samples: usize,
    time_points: &[f64],
) -> Result<Vec<f64>, JsError> {
    let model = weibull_fit_impl(data, n_samples)
        .map_err(|e| JsError::new(&e.message))?;

    let result: Vec<f64> = time_points.iter()
        .map(|&t| model.hazard_rate(t))
        .collect();

    Ok(result)
}

/// Compute percentiles of the Weibull distribution
#[wasm_bindgen(js_name = "weibullPercentiles")]
pub fn weibull_percentiles(
    data: &[f64],
    n_samples: usize,
    percentiles: &[f64],
) -> Result<Vec<f64>, JsError> {
    let model = weibull_fit_impl(data, n_samples)
        .map_err(|e| JsError::new(&e.message))?;

    let result: Vec<f64> = percentiles.iter()
        .map(|&p| model.percentile(p))
        .collect();

    Ok(result)
}

/// Exponential distribution fitting (special case of Weibull with k=1)
#[wasm_bindgen(js_name = "exponentialFit")]
pub fn exponential_fit(
    data: &[f64],
    n_samples: usize,
) -> Result<WeibullModel, JsError> {
    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(JsError::new("data length must equal n_samples"));
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    if mean <= 0.0 {
        return Err(JsError::new("mean must be positive for exponential fitting"));
    }

    // Exponential is Weibull with shape=1, scale=mean
    Ok(WeibullModel { shape: 1.0, scale: mean })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weibull_shape_exponential() {
        // cv=1 → k ≈ 1 (exponential)
        let k = weibull_shape_from_cv(1.0);
        assert!((k - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_weibull_shape_increasing_hazard() {
        // cv < 1 → k > 1 (increasing hazard = aging)
        let k = weibull_shape_from_cv(0.5);
        assert!(k > 1.0);
    }

    #[test]
    fn test_weibull_shape_decreasing_hazard() {
        // cv > 1 → k < 1 (decreasing hazard = infant mortality)
        let k = weibull_shape_from_cv(2.0);
        assert!(k < 1.0);
    }

    #[test]
    fn test_gamma_lanczos_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6
        assert!((gamma_lanczos(1.0) - 1.0).abs() < 1e-6);
        assert!((gamma_lanczos(2.0) - 1.0).abs() < 1e-6);
        assert!((gamma_lanczos(3.0) - 2.0).abs() < 1e-6);
        assert!((gamma_lanczos(4.0) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_lanczos_half() {
        // Γ(0.5) = √π
        let result = gamma_lanczos(0.5);
        let expected = std::f64::consts::PI.sqrt();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_fit_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = weibull_fit_impl(&data, 5).unwrap();
        assert!(model.shape > 0.0);
        assert!(model.scale > 0.0);
    }

    #[test]
    fn test_weibull_hazard_rate() {
        let model = WeibullModel { shape: 2.0, scale: 10.0 };
        // h(5) = (2/10) * (5/10)^(2-1) = 0.2 * 0.5 = 0.1
        let h = model.hazard_rate(5.0);
        assert!((h - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_survival_probability() {
        let model = WeibullModel { shape: 2.0, scale: 10.0 };
        // S(5) = exp(-(5/10)^2) = exp(-0.25)
        let s = model.survival_probability(5.0);
        let expected = (-0.25_f64).exp();
        assert!((s - expected).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_cumulative_hazard() {
        let model = WeibullModel { shape: 2.0, scale: 10.0 };
        // H(5) = (5/10)^2 = 0.25
        let h = model.cumulative_hazard(5.0);
        assert!((h - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_weibull_percentile() {
        let model = WeibullModel { shape: 2.0, scale: 10.0 };
        // Median (p=0.5): t = 10 * (-ln(0.5))^(0.5) = 10 * 0.8326 = 8.326
        let p = model.percentile(0.5);
        assert!((p - 8.326).abs() < 0.01);
    }

    #[test]
    fn test_weibull_median_remaining() {
        let model = WeibullModel { shape: 2.0, scale: 10.0 };
        // At t=5, cumulative hazard = (5/10)^2 = 0.25
        // Median remaining = 10 * (0.25 + ln(2))^0.5 - 5
        let m = model.median_remaining(5.0);
        assert!(m > 0.0);
    }

    #[test]
    fn test_exponential_fit() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = exponential_fit(&data, 5).unwrap();
        assert_eq!(model.shape, 1.0); // Exponential has k=1
        assert_eq!(model.scale, 3.0); // Mean of data
    }

    #[test]
    fn test_invalid_empty_data() {
        let result = weibull_fit_impl(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_negative_mean() {
        let result = weibull_fit_impl(&[-1.0, -2.0], 2);
        assert!(result.is_err());
    }
}
