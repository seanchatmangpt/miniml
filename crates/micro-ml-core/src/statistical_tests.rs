/// Statistical Tests for Data Validation
///
/// Non-parametric statistical tests for hypothesis testing:
/// - Chi-square test (categorical independence)
/// - Kolmogorov-Smirnov test (distribution comparison)
///
/// Use cases: Data quality validation, A/B testing, distribution comparison
use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::validate_matrix;
use crate::statistical_distributions::normal_cdf;

/// Chi-Square Test Result
///
/// Tests independence between two categorical variables.
#[wasm_bindgen]
pub struct ChiSquareResult {
    statistic: f64,
    p_value: f64,
    degrees_of_freedom: usize,
    is_significant: bool,
}

#[wasm_bindgen]
impl ChiSquareResult {
    #[wasm_bindgen(getter, js_name = "statistic")]
    pub fn statistic(&self) -> f64 { self.statistic }

    #[wasm_bindgen(getter, js_name = "pValue")]
    pub fn p_value(&self) -> f64 { self.p_value }

    #[wasm_bindgen(getter, js_name = "degreesOfFreedom")]
    pub fn degrees_of_freedom(&self) -> usize { self.degrees_of_freedom }

    #[wasm_bindgen(getter, js_name = "isSignificant")]
    pub fn is_significant(&self) -> bool { self.is_significant }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("ChiSquareResult(statistic={:.4}, p={:.4}, df={}, significant={})",
                self.statistic, self.p_value, self.degrees_of_freedom, self.is_significant)
    }
}

/// Chi-square test of independence
///
/// Tests whether two categorical variables are independent.
///
/// # Parameters
/// - `observed`: Flat contingency table [n_rows * n_cols]
/// - `n_rows`: Number of rows (categories of variable 1)
/// - `n_cols`: Number of columns (categories of variable 2)
/// - `alpha`: Significance level (default: 0.05)
///
/// # Returns
/// ChiSquareResult with test statistic and p-value
#[wasm_bindgen(js_name = "chiSquareTest")]
pub fn chi_square_test(
    observed: &[f64],
    n_rows: usize,
    n_cols: usize,
    alpha: f64,
) -> Result<ChiSquareResult, JsError> {
    chi_square_test_impl(observed, n_rows, n_cols, alpha)
        .map_err(|e| JsError::new(&e.message))
}

pub fn chi_square_test_impl(
    observed: &[f64],
    n_rows: usize,
    n_cols: usize,
    alpha: f64,
) -> Result<ChiSquareResult, MlError> {
    let n = validate_matrix(observed, n_cols)?;
    if n != n_rows {
        return Err(MlError::new("observed must be n_rows * n_cols"));
    }
    if n_rows < 2 || n_cols < 2 {
        return Err(MlError::new("need at least 2 rows and 2 columns"));
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(MlError::new("alpha must be in (0, 1)"));
    }

    // Compute row and column totals
    let mut row_totals = vec![0.0; n_rows];
    let mut col_totals = vec![0.0; n_cols];
    let grand_total = observed.iter().sum::<f64>();

    for i in 0..n_rows {
        for j in 0..n_cols {
            let val = observed[i * n_cols + j];
            row_totals[i] += val;
            col_totals[j] += val;
        }
    }

    if grand_total == 0.0 {
        return Err(MlError::new("contingency table cannot be all zeros"));
    }

    // Compute chi-square statistic
    let mut statistic = 0.0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let observed_val = observed[i * n_cols + j];
            let expected_val = (row_totals[i] * col_totals[j]) / grand_total;

            if expected_val > 0.0 {
                let contribution = (observed_val - expected_val).powi(2) / expected_val;
                statistic += contribution;
            }
        }
    }

    // Degrees of freedom: (rows - 1) * (cols - 1)
    let df = (n_rows - 1) * (n_cols - 1);

    // P-value using chi-square approximation
    let p_value = chi_square_p_value(statistic, df);

    // Significance test
    let is_significant = p_value < alpha;

    Ok(ChiSquareResult {
        statistic,
        p_value,
        degrees_of_freedom: df,
        is_significant,
    })
}

/// Chi-square CDF approximation (Wilson-Hilferty)
fn chi_square_p_value(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }

    // Wilson-Hilferty approximation
    let h = 2.0 / (9.0 * df as f64);
    let z = (x * (1.0 - h)).powf(1.0 / 3.0) - (1.0 - h);

    // Standard normal CDF
    normal_cdf(z)
}

/// Kolmogorov-Smirnov Test Result
///
/// Tests whether two samples come from the same distribution.
#[wasm_bindgen]
pub struct KsTestResult {
    statistic: f64,
    p_value: f64,
    is_significant: bool,
}

#[wasm_bindgen]
impl KsTestResult {
    #[wasm_bindgen(getter, js_name = "statistic")]
    pub fn statistic(&self) -> f64 { self.statistic }

    #[wasm_bindgen(getter, js_name = "pValue")]
    pub fn p_value(&self) -> f64 { self.p_value }

    #[wasm_bindgen(getter, js_name = "isSignificant")]
    pub fn is_significant(&self) -> bool { self.is_significant }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("KsTestResult(statistic={:.4}, p={:.4}, significant={})",
                self.statistic, self.p_value, self.is_significant)
    }
}

/// Two-sample Kolmogorov-Smirnov test
///
/// Tests whether two samples come from the same continuous distribution.
///
/// # Parameters
/// - `sample1`: First sample
/// - `n1`: Size of first sample
/// - `sample2`: Second sample
/// - `n2`: Size of second sample
/// - `alpha`: Significance level (default: 0.05)
///
/// # Returns
/// KsTestResult with KS statistic and p-value
#[wasm_bindgen(js_name = "ksTest")]
pub fn ks_test(
    sample1: &[f64],
    n1: usize,
    sample2: &[f64],
    n2: usize,
    alpha: f64,
) -> Result<KsTestResult, JsError> {
    ks_test_impl(sample1, n1, sample2, n2, alpha)
        .map_err(|e| JsError::new(&e.message))
}

pub fn ks_test_impl(
    sample1: &[f64],
    n1: usize,
    sample2: &[f64],
    n2: usize,
    alpha: f64,
) -> Result<KsTestResult, MlError> {
    if n1 < 2 || n2 < 2 {
        return Err(MlError::new("need at least 2 samples in each group"));
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(MlError::new("alpha must be in (0, 1)"));
    }

    // Sort both samples
    let mut sorted1 = sample1.to_vec();
    let mut sorted2 = sample2.to_vec();
    sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute empirical CDFs
    let cdf1 = |x: f64| -> f64 {
        sorted1.iter().filter(|&&v| v <= x).count() as f64 / n1 as f64
    };
    let cdf2 = |x: f64| -> f64 {
        sorted2.iter().filter(|&&v| v <= x).count() as f64 / n2 as f64
    };

    // Combine all unique values from both samples
    let mut all_values: Vec<f64> = sorted1.iter().chain(sorted2.iter()).copied().collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_values.dedup();

    // Find maximum absolute difference in CDFs
    let mut max_diff: f64 = 0.0;
    for &val in &all_values {
        let diff = (cdf1(val) - cdf2(val)).abs();
        max_diff = max_diff.max(diff);
    }

    // Kolmogorov-Smirnov statistic
    let statistic = max_diff * ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();

    // P-value approximation (Smirnov formula)
    let effective_n = (n1 * n2) as f64 / (n1 + n2) as f64;
    let lambda = (effective_n.sqrt() + 0.12 + 0.11 / effective_n.sqrt()) * statistic;
    let p_value = ks_p_value_approx(lambda);

    let is_significant = p_value < alpha;

    Ok(KsTestResult {
        statistic,
        p_value,
        is_significant,
    })
}

/// KS test p-value approximation (Smirnov formula)
fn ks_p_value_approx(lambda: f64) -> f64 {
    if lambda > 1.5 {
        return 0.0; // Very large lambda -> very small p-value
    }

    // Approximation from Smirnov (1948)
    // For lambda <= 0.5: use series expansion
    // For lambda > 0.5: use complement

    if lambda <= 0.5 {
        let mut sum = 0.0;
        let mut term = 1.0;
        let mut sign = -1.0f64;

        // Series: 2 * Σ(-1)^k * exp(-2k²π²lambda²)
        for _k in 1..=10 {
            term *= -2.0 * std::f64::consts::PI * std::f64::consts::PI * lambda * lambda;
            sum += sign * term.exp();
            sign *= -1.0;
        }

        1.0 + sum
    } else {
        // For larger lambda, use iterative refinement
        let mut q = 0.0;
        let mut u = lambda;
        let mut v = 0.0;

        // Marsaglia's polar method approximation
        for _ in 0..10 {
            u = u * lambda;
            v = u * u;
            let exp_v = (-v).exp();
            q = 2.0 * exp_v * (1.0 + u + u * u / 3.0 + u * u * u / 6.0);
        }

        q.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi_square_test_basic() {
        // Simple 2x2 contingency table
        let observed = vec![10.0, 20.0, 30.0, 40.0]; // [[10, 20], [30, 40]]
        let result = chi_square_test_impl(&observed, 2, 2, 0.05).unwrap();

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.degrees_of_freedom, 1); // (2-1)*(2-1) = 1
    }

    #[test]
    fn test_chi_square_independence() {
        // Independent data (should have high p-value)
        let observed = vec![25.0, 25.0, 25.0, 25.0];
        let result = chi_square_test_impl(&observed, 2, 2, 0.05).unwrap();

        // With equal frequencies, should NOT be significant
        assert!(!result.is_significant);
    }

    #[test]
    fn test_ks_test_identical() {
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ks_test_impl(&sample, 5, &sample, 5, 0.05).unwrap();

        // Identical samples -> KS statistic = 0 -> not significant
        assert_eq!(result.statistic, 0.0);
        assert!(!result.is_significant);
    }

    #[test]
    fn test_ks_test_different() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0]; // Shifted by 1

        let result = ks_test_impl(&sample1, 5, &sample2, 5, 0.05).unwrap();

        // Different distributions -> some statistic > 0
        assert!(result.statistic > 0.0);
    }

    #[test]
    fn test_chi_square_requires_2x2() {
        let result = chi_square_test_impl(&[1.0], 1, 1, 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn test_ks_test_requires_min_samples() {
        let result = ks_test_impl(&[1.0], 1, &[2.0], 1, 0.05);
        assert!(result.is_err());
    }
}
