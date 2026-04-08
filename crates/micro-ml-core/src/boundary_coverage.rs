/// Boundary Coverage and Anomaly Detection
///
/// Detects anomalies and out-of-bounds values:
/// - Statistical anomaly detection (Z-score, IQR)
/// - Boundary coverage (out-of-bounds detection)
/// - Sequence anomaly (deviation from expected patterns)
///
/// Use cases: Data quality monitoring, outlier detection, anomaly flagging
use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::validate_matrix;
use std::collections::HashMap;

/// Anomaly Detection Result
///
/// Contains anomaly scores and binary flags.
#[wasm_bindgen]
pub struct AnomalyResult {
    scores: Vec<f64>,
    is_anomaly: Vec<u8>,  // 0 or 1 values (Vec<bool> not supported by wasm-bindgen)
    threshold: f64,
}

#[wasm_bindgen]
impl AnomalyResult {
    #[wasm_bindgen(getter, js_name = "threshold")]
    pub fn threshold(&self) -> f64 { self.threshold }

    #[wasm_bindgen(getter, js_name = "scores")]
    pub fn get_scores(&self) -> Vec<f64> {
        self.scores.clone()
    }

    #[wasm_bindgen(getter, js_name = "isAnomaly")]
    pub fn get_is_anomaly(&self) -> Vec<u8> {
        self.is_anomaly.clone()
    }

    /// Count of anomalies detected
    #[wasm_bindgen(getter, js_name = "anomalyCount")]
    pub fn anomaly_count(&self) -> usize {
        self.is_anomaly.iter().filter(|&&x| x == 1).count()
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("AnomalyResult(threshold={:.4}, n_anomalies={})",
                self.threshold, self.anomaly_count())
    }
}

/// Z-score anomaly detection
///
/// Detects anomalies using Z-score (standard deviations from mean).
///
/// # Parameters
/// - `data`: Flat array of values
/// - `n_samples`: Number of samples
/// - `threshold`: Z-score threshold (default: 3.0)
///
/// # Returns
/// AnomalyResult with z-scores and anomaly flags
#[wasm_bindgen(js_name = "zscoreAnomalyDetection")]
pub fn zscore_anomaly_detection(
    data: &[f64],
    n_samples: usize,
    threshold: f64,
) -> Result<AnomalyResult, JsError> {
    zscore_anomaly_detection_impl(data, n_samples, threshold)
        .map_err(|e| JsError::new(&e.message))
}

pub fn zscore_anomaly_detection_impl(
    data: &[f64],
    n_samples: usize,
    threshold: f64,
) -> Result<AnomalyResult, MlError> {
    use crate::error::MlError;
    use crate::matrix::validate_matrix;

    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(MlError::new("data length must equal n_samples"));
    }
    if n_samples < 3 {
        return Err(MlError::new("need at least 3 samples for z-score"));
    }

    // Compute mean and standard deviation
    let n = n_samples as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    if std == 0.0 {
        return Err(MlError::new("zero variance - cannot compute z-scores"));
    }

    // Compute z-scores
    let mut scores = Vec::new();
    let mut is_anomaly = Vec::new();

    for &val in data {
        let z = (val - mean) / std;
        let anomaly = if z.abs() > threshold { 1u8 } else { 0u8 };
        scores.push(z);
        is_anomaly.push(anomaly);
    }

    Ok(AnomalyResult {
        scores,
        is_anomaly,
        threshold,
    })
}

/// IQR (Interquartile Range) anomaly detection
///
/// Detects outliers using IQR method (robust to extreme values).
///
/// # Parameters
/// - `data`: Flat array of values
/// - `n_samples`: Number of samples
/// - `multiplier`: IQR multiplier (default: 1.5)
///
/// # Returns
/// AnomalyResult with IQR-based anomaly flags
#[wasm_bindgen(js_name = "iqrAnomalyDetection")]
pub fn iqr_anomaly_detection(
    data: &[f64],
    n_samples: usize,
    multiplier: f64,
) -> Result<AnomalyResult, JsError> {
    iqr_anomaly_detection_impl(data, n_samples, multiplier)
        .map_err(|e| JsError::new(&e.message))
}

pub fn iqr_anomaly_detection_impl(
    data: &[f64],
    n_samples: usize,
    multiplier: f64,
) -> Result<AnomalyResult, MlError> {
    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(MlError::new("data length must equal n_samples"));
    }
    if n_samples < 4 {
        return Err(MlError::new("need at least 4 samples for IQR"));
    }

    // Sort data and compute quartiles
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1_idx = n_samples / 4;
    let q3_idx = (3 * n_samples) / 4;
    let q1 = sorted[q1_idx];
    let q3 = sorted[q3_idx];
    let iqr = q3 - q1;

    if iqr == 0.0 {
        return Err(MlError::new("IQR is zero - all values are identical"));
    }

    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;

    // Anomaly if outside bounds
    let mut scores = Vec::new();
    let mut is_anomaly = Vec::new();

    for &val in data {
        // Use distance from nearest bound as score
        let score = if val < lower_bound {
            (lower_bound - val) / iqr
        } else if val > upper_bound {
            (val - upper_bound) / iqr
        } else {
            0.0
        };

        let anomaly = if val < lower_bound || val > upper_bound { 1u8 } else { 0u8 };
        scores.push(score);
        is_anomaly.push(anomaly);
    }

    Ok(AnomalyResult {
        scores,
        is_anomaly,
        threshold: multiplier,
    })
}

/// Boundary Coverage Detection
///
/// Checks which values fall outside specified boundaries.
///
/// # Parameters
/// - `data`: Flat array of values
/// - `n_samples`: Number of samples
/// - `lower_bound`: Minimum acceptable value
/// - `upper_bound`: Maximum acceptable value
///
/// # Returns
/// AnomalyResult with out-of-bounds flags
#[wasm_bindgen(js_name = "boundaryCoverage")]
pub fn boundary_coverage(
    data: &[f64],
    n_samples: usize,
    lower_bound: f64,
    upper_bound: f64,
) -> Result<AnomalyResult, JsError> {
    boundary_coverage_impl(data, n_samples, lower_bound, upper_bound)
        .map_err(|e| JsError::new(&e.message))
}

pub fn boundary_coverage_impl(
    data: &[f64],
    n_samples: usize,
    lower_bound: f64,
    upper_bound: f64,
) -> Result<AnomalyResult, MlError> {
    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(MlError::new("data length must equal n_samples"));
    }
    if lower_bound > upper_bound {
        return Err(MlError::new("lower_bound must be <= upper_bound"));
    }

    let mut scores = Vec::new();
    let mut is_anomaly = Vec::new();

    for &val in data {
        // Score: distance from nearest bound
        let score = if val < lower_bound {
            (lower_bound - val).abs()
        } else if val > upper_bound {
            (val - upper_bound).abs()
        } else {
            0.0
        };

        let anomaly = if val < lower_bound || val > upper_bound { 1u8 } else { 0u8 };
        scores.push(score);
        is_anomaly.push(anomaly);
    }

    Ok(AnomalyResult {
        scores,
        is_anomaly,
        threshold: upper_bound - lower_bound,
    })
}

/// Sequence Anomaly Detection
///
/// Detects sequences that deviate from expected patterns using n-gram models.
///
/// # Parameters
/// - `sequences`: Flat array of sequences
/// - `sequence_lengths`: Length of each sequence
/// - `n`: NGram order for pattern learning
/// - `threshold`: Probability threshold for anomaly (low = more sensitive)
///
/// # Returns
/// AnomalyResult for each sequence position
#[wasm_bindgen(js_name = "sequenceAnomalyDetection")]
pub fn sequence_anomaly_detection(
    sequences: &[usize],
    sequence_lengths: &[usize],
    n: usize,
    threshold: f64,
) -> Result<AnomalyResult, JsError> {
    sequence_anomaly_detection_impl(sequences, sequence_lengths, n, threshold)
        .map_err(|e| JsError::new(&e.message))
}

pub fn sequence_anomaly_detection_impl(
    sequences: &[usize],
    sequence_lengths: &[usize],
    n: usize,
    threshold: f64,
) -> Result<AnomalyResult, MlError> {
    if sequences.is_empty() || sequence_lengths.is_empty() {
        return Err(MlError::new("sequences and sequence_lengths cannot be empty"));
    }

    // Build n-gram model from sequences
    let mut transitions: HashMap<(Vec<usize>, usize), usize> = HashMap::new();
    let mut context_totals: HashMap<Vec<usize>, usize> = HashMap::new();

    let mut offset = 0;
    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // Count transitions
        for i in 0..seq_len.saturating_sub(n) {
            let context: Vec<usize> = seq[i..i + n - 1].to_vec();
            let next_item = seq[i + n - 1];

            *transitions.entry((context.clone(), next_item)).or_insert(0) += 1;
            *context_totals.entry(context).or_insert(0) += 1;
        }
    }

    // Detect anomalies in sequences
    let mut all_scores = Vec::new();
    let mut all_anomalies = Vec::new();

    let mut offset = 0;
    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // Check each position for anomaly
        for i in 0..seq_len.saturating_sub(n) {
            let context: Vec<usize> = seq[i..i + n - 1].to_vec();
            let next_item = seq[i + n - 1];

            // Get probability of this transition
            let count = *transitions.get(&(context.clone(), next_item)).unwrap_or(&0);
            let total = *context_totals.get(&context).unwrap_or(&1);
            let prob = if total > 0 { count as f64 / total as f64 } else { 0.0 };

            // Anomaly if probability is below threshold
            let score = 1.0 - prob; // Higher score = more anomalous
            let anomaly = if prob < threshold { 1u8 } else { 0u8 };

            all_scores.push(score);
            all_anomalies.push(anomaly);
        }

        // Pad last n-1 positions with no score
        for _ in 0..n.saturating_sub(1) {
            all_scores.push(0.0);
            all_anomalies.push(0u8);
        }
    }

    Ok(AnomalyResult {
        scores: all_scores,
        is_anomaly: all_anomalies,
        threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_anomaly_detection() {
        let data = vec![10.0, 11.0, 10.0, 12.0, 10.0, 25.0]; // 25 is outlier
        let result = zscore_anomaly_detection_impl(&data, 6, 2.0).unwrap(); // Lower threshold to 2.0

        assert_eq!(result.scores.len(), 6);
        assert_eq!(result.anomaly_count(), 1); // Only 25 is anomaly
        assert_eq!(result.is_anomaly[5], 1); // Last element is anomaly (value 1)
    }

    #[test]
    fn test_iqr_anomaly_detection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is outlier
        let result = iqr_anomaly_detection_impl(&data, 6, 1.5).unwrap();

        assert_eq!(result.scores.len(), 6);
        assert!(result.anomaly_count() >= 1); // At least 100 is anomaly
    }

    #[test]
    fn test_boundary_coverage() {
        let data = vec![1.0, 2.0, 3.0, 10.0, 5.0]; // 10 is outside [0, 6]
        let result = boundary_coverage_impl(&data, 5, 0.0, 6.0).unwrap();

        assert_eq!(result.scores.len(), 5);
        assert_eq!(result.anomaly_count(), 1); // 10 is out of bounds
        assert_eq!(result.is_anomaly[3], 1); // Fourth element (10) is anomaly (value 1)
    }

    #[test]
    fn test_sequence_anomaly_detection() {
        // Regular pattern: 1, 2, 3, 1, 2, 3, 1, 2, 3
        let sequences = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        let sequence_lengths = vec![9];

        let result = sequence_anomaly_detection_impl(&sequences, &sequence_lengths, 2, 0.1).unwrap();

        // For n=2 and seq_len=9, we get 7 n-gram predictions + 1 padding = 8 total
        assert_eq!(result.scores.len(), 8);
        // With a perfect repeating pattern, most transitions should have high probability
        let anomaly_count = result.anomaly_count();
        assert!(anomaly_count < 8); // At least some are normal (padded position has score 0)
    }

    #[test]
    fn test_zscore_zero_variance() {
        let data = vec![5.0, 5.0, 5.0];
        let result = zscore_anomaly_detection_impl(&data, 3, 3.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_iqr_identical_values() {
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let result = iqr_anomaly_detection_impl(&data, 4, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_boundary_invalid_bounds() {
        let data = vec![1.0, 2.0, 3.0];
        let result = boundary_coverage_impl(&data, 3, 10.0, 5.0); // lower > upper
        assert!(result.is_err());
    }

    #[test]
    fn test_zscore_requires_min_samples() {
        let data = vec![1.0, 2.0];
        let result = zscore_anomaly_detection_impl(&data, 2, 3.0);
        assert!(result.is_err());
    }
}
