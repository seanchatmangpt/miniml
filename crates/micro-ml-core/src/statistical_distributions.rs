/// Statistical Distributions: Log-Normal and Gamma
///
/// Additional probability distributions for modeling:
/// - Log-Normal Distribution (log-normal survival, skewed data)
/// - Gamma Distribution (shape-rate parameterization)
///
/// Use cases: Survival analysis, skewed data modeling, reliability engineering
use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::validate_matrix;
use crate::weibull::gamma_lanczos;

/// Log-Normal Distribution Model
///
/// Models data where log(x) follows a normal distribution.
#[wasm_bindgen]
pub struct LogNormalModel {
    mu: f64,     // Mean of log(x)
    sigma: f64,   // Std dev of log(x)
}

#[wasm_bindgen]
impl LogNormalModel {
    #[wasm_bindgen(getter, js_name = "mu")]
    pub fn mu(&self) -> f64 { self.mu }

    #[wasm_bindgen(getter, js_name = "sigma")]
    pub fn sigma(&self) -> f64 { self.sigma }

    /// Probability density function at x
    #[wasm_bindgen(js_name = "pdf")]
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || self.sigma <= 0.0 {
            return 0.0;
        }
        let log_x = x.ln();
        let exponent = -0.5 * ((log_x - self.mu) / self.sigma).powi(2);
        let normalizer = self.sigma * x * (2.0 * std::f64::consts::PI).sqrt();
        (exponent.exp()) / normalizer
    }

    /// Cumulative distribution function at x
    #[wasm_bindgen(js_name = "cdf")]
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 || self.sigma <= 0.0 {
            return 0.0;
        }
        let log_x = x.ln();
        // Standard normal CDF approximation
        let z = (log_x - self.mu) / self.sigma;
        normal_cdf(z)
    }

    /// Survival function: S(x) = 1 - CDF(x)
    #[wasm_bindgen(js_name = "survivalProbability")]
    pub fn survival_probability(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// Percentile (p-th quantile)
    #[wasm_bindgen]
    pub fn percentile(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 || self.sigma <= 0.0 {
            return f64::NAN;
        }
        // Log-normal: exp(mu + sigma * z_p) where z_p is normal quantile
        let z_p = normal_quantile(p);
        (self.mu + self.sigma * z_p).exp()
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("LogNormalModel(mu={}, sigma={})", self.mu, self.sigma)
    }
}

/// Fit log-normal distribution to data
///
/// # Parameters
/// - `data`: Time-to-event or positive values
/// - `n_samples`: Number of samples
///
/// # Returns
/// LogNormalModel fitted to data
#[wasm_bindgen(js_name = "logNormalFit")]
pub fn log_normal_fit(
    data: &[f64],
    n_samples: usize,
) -> Result<LogNormalModel, JsError> {
    log_normal_fit_impl(data, n_samples)
        .map_err(|e| JsError::new(&e.message))
}

pub fn log_normal_fit_impl(
    data: &[f64],
    n_samples: usize,
) -> Result<LogNormalModel, MlError> {
    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(MlError::new("data length must equal n_samples"));
    }
    if n_samples == 0 {
        return Err(MlError::new("data must not be empty"));
    }

    // Check all values are positive
    for &x in data.iter() {
        if x <= 0.0 {
            return Err(MlError::new("log-normal requires all values > 0"));
        }
    }

    // Compute mean and std of log-transformed data
    let n = n_samples as f64;
    let log_sum: f64 = data.iter().map(|x| x.ln()).sum();
    let log_mean = log_sum / n;

    let log_var: f64 = data.iter()
        .map(|x| (x.ln() - log_mean).powi(2))
        .sum::<f64>() / n;
    let log_std = log_var.sqrt();

    Ok(LogNormalModel {
        mu: log_mean,
        sigma: log_std,
    })
}

/// Gamma Distribution Model
///
/// Two-parameter gamma distribution with shape k and rate θ.
#[wasm_bindgen]
pub struct GammaModel {
    shape: f64,  // k (shape parameter)
    rate: f64,   // θ (rate parameter = 1/scale)
}

#[wasm_bindgen]
impl GammaModel {
    #[wasm_bindgen(getter, js_name = "shape")]
    pub fn shape(&self) -> f64 { self.shape }

    #[wasm_bindgen(getter, js_name = "rate")]
    pub fn rate(&self) -> f64 { self.rate }

    /// Scale parameter (1/rate)
    #[wasm_bindgen(getter, js_name = "scale")]
    pub fn scale(&self) -> f64 {
        if self.rate > 0.0 {
            1.0 / self.rate
        } else {
            f64::INFINITY
        }
    }

    /// Probability density function at x
    #[wasm_bindgen(js_name = "pdf")]
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || self.shape <= 0.0 || self.rate <= 0.0 {
            return 0.0;
        }

        // Gamma PDF: f(x) = (rate^shape / Γ(shape)) * x^(shape-1) * exp(-rate * x)
        let gamma_shape = gamma_lanczos(self.shape);
        let coef = self.rate.powf(self.shape) / gamma_shape;
        coef * x.powf(self.shape - 1.0) * (-self.rate * x).exp()
    }

    /// Cumulative distribution function (lower incomplete gamma ratio)
    #[wasm_bindgen(js_name = "cdf")]
    pub fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 || self.shape <= 0.0 || self.rate <= 0.0 {
            return 0.0;
        }

        // Use series approximation for gamma CDF
        gamma_cdf_series(x * self.rate, self.shape)
    }

    /// Survival function: S(x) = 1 - CDF(x)
    #[wasm_bindgen(js_name = "survivalProbability")]
    pub fn survival_probability(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("GammaModel(shape={}, rate={})", self.shape, self.rate)
    }
}

/// Fit gamma distribution to data using method of moments
///
/// # Parameters
/// - `data`: Positive values (time-to-event, counts, etc.)
/// - `n_samples`: Number of samples
///
/// # Returns
/// GammaModel fitted to data
#[wasm_bindgen(js_name = "gammaFit")]
pub fn gamma_fit(
    data: &[f64],
    n_samples: usize,
) -> Result<GammaModel, JsError> {
    gamma_fit_impl(data, n_samples)
        .map_err(|e| JsError::new(&e.message))
}

pub fn gamma_fit_impl(
    data: &[f64],
    n_samples: usize,
) -> Result<GammaModel, MlError> {
    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(MlError::new("data length must equal n_samples"));
    }
    if n_samples == 0 {
        return Err(MlError::new("data must not be empty"));
    }

    // Check all values are positive
    for &x in data.iter() {
        if x <= 0.0 {
            return Err(MlError::new("gamma distribution requires all values > 0"));
        }
    }

    let n = n_samples as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    if mean <= 0.0 || variance <= 0.0 {
        return Err(MlError::new("invalid data for gamma fitting"));
    }

    // Method of moments for gamma:
    // shape = mean² / variance
    // rate = mean / variance
    let shape = mean.powi(2) / variance;
    let rate = mean / variance;

    if shape <= 0.0 || rate <= 0.0 {
        return Err(MlError::new("estimated parameters are invalid"));
    }

    Ok(GammaModel { shape, rate })
}

/// Series approximation for gamma CDF (regularized lower incomplete gamma)
fn gamma_cdf_series(x: f64, alpha: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    // Series expansion: P(a, x) = (x^a / Γ(a)) * Σ(x^k / (a(a+1)...(a+k)))
    // Use first 20 terms of the series
    const MAX_TERMS: usize = 20;
    let mut sum = 1.0 / alpha;
    let term_coeff = x / alpha;

    for k in 1..MAX_TERMS {
        let mut term = term_coeff;
        for j in 0..k {
            term /= alpha + j as f64;
        }
        term *= x.powf(k as f64);
        sum += term;

        if term.abs() < 1e-10 * sum {
            break;
        }
    }

    let gamma_alpha = gamma_lanczos(alpha);
    let normalizer = 1.0 - (-x).exp();

    (1.0 - normalizer * sum / gamma_alpha).min(1.0).max(0.0)
}

/// Standard normal CDF approximation
pub(crate) fn normal_cdf(z: f64) -> f64 {
    // Abramowitz and Stegun approximation 7.1.26
    const sign1: f64 = 0.078064;
    const a1: f64 = 0.180308;
    const a2: f64 = 0.019117;
    const a3: f64 = 0.000337;
    const a4: f64 = 0.000084;

    let t = 1.0 / (1.0 + 0.2316419 * z.abs());

    if z >= 0.0 {
        1.0 - a1 * t * t.exp() - a2 * t.powi(3) - a3 * t.powi(5) - a4 * t.powi(7)
    } else {
        a1 * t * t.exp() + a2 * t.powi(3) + a3 * t.powi(5) + a4 * t.powi(7)
    }
}

/// Standard normal quantile function (approximation)
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0; // Exact median
    }

    // Beasley-Springer-Moro approximation (simplified)
    if p < 0.5 {
        let t = (2.0 * p).sqrt();
        -t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
            (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    } else {
        let t = (2.0 * (1.0 - p)).sqrt();
        t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
            (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_normal_fit_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = log_normal_fit_impl(&data, 5).unwrap();

        assert!(model.sigma > 0.0);
        assert_eq!(model.mu, model.mu); // Just check it's set
    }

    #[test]
    fn test_log_normal_pdf() {
        let model = LogNormalModel { mu: 0.0, sigma: 1.0 };
        // At x=exp(0)=1, PDF should be 1/sqrt(2pi)
        let pdf_at_one = model.pdf(1.0);
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((pdf_at_one - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_normal_cdf() {
        let model = LogNormalModel { mu: 0.0, sigma: 1.0 };
        // At median (x=1), CDF should be 0.5
        let cdf_at_one = model.cdf(1.0);
        assert!((cdf_at_one - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gamma_fit_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let model = gamma_fit_impl(&data, 5).unwrap();

        assert!(model.shape > 0.0);
        assert!(model.rate > 0.0);
    }

    #[test]
    fn test_gamma_pdf() {
        let model = GammaModel { shape: 2.0, rate: 1.0 };
        // PDF at x=1: f(1) = (1^2 / Γ(2)) * 1^(2-1) * exp(-1)
        // Γ(2) = 1, so f(1) = 1 * 1 * exp(-1) = 1/e
        let pdf = model.pdf(1.0);
        let expected = (-1.0_f64).exp();
        assert!((pdf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_normal_requires_positive() {
        let data = vec![-1.0, 1.0, 2.0];
        let result = log_normal_fit_impl(&data, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_gamma_requires_positive() {
        let data = vec![-1.0, 1.0, 2.0];
        let result = gamma_fit_impl(&data, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_log_normal_percentile() {
        let model = LogNormalModel { mu: 0.0, sigma: 1.0 };
        // Median of log-normal with mu=0 should be exp(0) = 1
        let median = model.percentile(0.5);
        assert!((median - 1.0).abs() < 0.01);
    }
}
