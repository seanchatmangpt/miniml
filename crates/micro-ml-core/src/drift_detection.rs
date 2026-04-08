/// Drift Detection
///
/// Detects concept drift in data streams using:
/// - EWMA (Exponentially Weighted Moving Average) drift detection
/// - Jaccard window-based drift detection
///
/// Use cases: Model monitoring, data pipeline validation, adaptive systems
use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::validate_matrix;

/// EWMA Drift Detector
///
/// Monitors a statistic using EWMA and signals drift when the value
/// deviates significantly from the expected range.
#[wasm_bindgen]
pub struct EwmaDetector {
    lambda: f64,      // Smoothing factor (0 < lambda ≤ 1)
    expected: f64,   // Target value
    std_err: f64,    // Standard deviation threshold
    current_ewma: f64,
    drift_count: usize,
}

#[wasm_bindgen]
impl EwmaDetector {
    #[wasm_bindgen(getter, js_name = "lambda")]
    pub fn lambda(&self) -> f64 { self.lambda }

    #[wasm_bindgen(getter, js_name = "expected")]
    pub fn expected(&self) -> f64 { self.expected }

    #[wasm_bindgen(getter, js_name = "stdErr")]
    pub fn std_err(&self) -> f64 { self.std_err }

    #[wasm_bindgen(getter, js_name = "currentEwma")]
    pub fn current_ewma(&self) -> f64 { self.current_ewma }

    #[wasm_bindgen(getter, js_name = "driftCount")]
    pub fn drift_count(&self) -> usize { self.drift_count }

    /// Update the detector with a new observation
    /// Returns true if drift is detected
    #[wasm_bindgen]
    pub fn update(&mut self, value: f64) -> bool {
        // EWMA update: ewma_t = lambda * value + (1 - lambda) * ewma_{t-1}
        self.current_ewma = self.lambda * value + (1.0 - self.lambda) * self.current_ewma;

        // Check for drift: |ewma - expected| > std_err
        let deviation = (self.current_ewma - self.expected).abs();
        let is_drift = deviation > self.std_err;

        if is_drift {
            self.drift_count += 1;
        }

        is_drift
    }

    /// Reset the detector (clear drift count and reset EWMA)
    #[wasm_bindgen(js_name = "reset")]
    pub fn reset(&mut self) {
        self.current_ewma = self.expected;
        self.drift_count = 0;
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("EwmaDetector(lambda={}, expected={}, std_err={}, ewma={}, drifts={})",
                self.lambda, self.expected, self.std_err, self.current_ewma, self.drift_count)
    }
}

/// Create a new EWMA drift detector
///
/// # Parameters
/// - `lambda`: Smoothing factor (0 < lambda ≤ 1). Smaller values give more weight to history.
/// - `expected`: Target value (initial EWMA value)
/// - `std_err`: Standard deviation threshold (drift if |ewma - expected| > std_err * std_err)
#[wasm_bindgen(js_name = "ewmaDetector")]
pub fn ewma_detector(
    lambda: f64,
    expected: f64,
    std_err: f64,
) -> Result<EwmaDetector, JsError> {
    ewma_detector_impl(lambda, expected, std_err)
        .map_err(|e| JsError::new(&e.message))
}

pub fn ewma_detector_impl(
    lambda: f64,
    expected: f64,
    std_err: f64,
) -> Result<EwmaDetector, MlError> {
    if lambda <= 0.0 || lambda > 1.0 {
        return Err(MlError::new("lambda must be in (0, 1]"));
    }
    if std_err < 0.0 {
        return Err(MlError::new("std_err must be non-negative"));
    }

    Ok(EwmaDetector {
        lambda,
        expected,
        std_err,
        current_ewma: expected,
        drift_count: 0,
    })
}

/// Batch EWMA drift detection
///
/// Processes a stream of values and returns drift indices
///
/// # Parameters
/// - `data`: Flat array of values
/// - `n_samples`: Number of samples
/// - `lambda`: EWMA smoothing factor
/// - `expected`: Target value
/// - `std_err`: Standard deviation threshold
/// - `burn_in`: Number of samples to skip before detecting drift
///
/// # Returns
/// Array of indices where drift was detected
#[wasm_bindgen(js_name = "ewmaDriftDetection")]
pub fn ewma_drift_detection(
    data: &[f64],
    n_samples: usize,
    lambda: f64,
    expected: f64,
    std_err: f64,
    burn_in: usize,
) -> Result<Vec<f64>, JsError> {
    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(JsError::new("data length must equal n_samples"));
    }

    let mut detector = ewma_detector_impl(lambda, expected, std_err)?;
    let mut drift_indices = Vec::new();

    for (i, &value) in data.iter().enumerate() {
        if i >= burn_in && detector.update(value) {
            drift_indices.push(i as f64);
        }
    }

    Ok(drift_indices)
}

/// Jaccard Window Drift Detector
///
/// Detects drift by comparing Jaccard similarity between consecutive
/// windows of categorical data.
#[wasm_bindgen]
pub struct JaccardDriftDetector {
    window_size: usize,
    threshold: f64,
    reference_window: Vec<Vec<f64>>, // Stored as one-hot encoded vectors
    drift_count: usize,
}

#[wasm_bindgen]
impl JaccardDriftDetector {
    #[wasm_bindgen(getter, js_name = "windowSize")]
    pub fn window_size(&self) -> usize { self.window_size }

    #[wasm_bindgen(getter, js_name = "threshold")]
    pub fn threshold(&self) -> f64 { self.threshold }

    #[wasm_bindgen(getter, js_name = "driftCount")]
    pub fn drift_count(&self) -> usize { self.drift_count }

    /// Update detector with new window (flat array of categorical values)
    /// Returns true if drift is detected
    #[wasm_bindgen]
    pub fn update(&mut self, window_flat: &[f64], n_samples: usize) -> Result<bool, JsError> {
        // Convert flat array to vector of vectors (one-hot encoded)
        let window = to_nested_vec(window_flat, n_samples);

        if self.reference_window.is_empty() {
            // First window becomes reference
            self.reference_window = window;
            return Ok(false);
        }

        // Compute Jaccard similarity with reference
        let similarity = jaccard_similarity_windows(&self.reference_window, &window);

        if similarity < self.threshold {
            self.drift_count += 1;
            // Optionally update reference to adapt to new concept
            // self.reference_window = window;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Reset the detector
    #[wasm_bindgen(js_name = "reset")]
    pub fn reset(&mut self) {
        self.reference_window.clear();
        self.drift_count = 0;
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("JaccardDriftDetector(window_size={}, threshold={}, drifts={})",
                self.window_size, self.threshold, self.drift_count)
    }
}

/// Create a new Jaccard drift detector
///
/// # Parameters
/// - `window_size`: Size of sliding window (number of samples)
/// - `threshold`: Jaccard similarity threshold (0-1). Drift if similarity < threshold.
#[wasm_bindgen(js_name = "jaccardDriftDetector")]
pub fn jaccard_drift_detector(
    window_size: usize,
    threshold: f64,
) -> Result<JaccardDriftDetector, JsError> {
    jaccard_drift_detector_impl(window_size, threshold)
        .map_err(|e| JsError::new(&e.message))
}

pub fn jaccard_drift_detector_impl(
    window_size: usize,
    threshold: f64,
) -> Result<JaccardDriftDetector, MlError> {
    if window_size == 0 {
        return Err(MlError::new("window_size must be > 0"));
    }
    if threshold < 0.0 || threshold > 1.0 {
        return Err(MlError::new("threshold must be in [0, 1]"));
    }

    Ok(JaccardDriftDetector {
        window_size,
        threshold,
        reference_window: Vec::new(),
        drift_count: 0,
    })
}

/// Batch Jaccard drift detection for categorical time series
///
/// # Parameters
/// - `data`: Flat array of categorical values (one-hot encoded per sample)
/// - `n_features`: Number of features (one-hot dimension)
/// - `n_samples`: Number of samples
/// - `window_size`: Size of sliding window
/// - `threshold`: Jaccard similarity threshold
///
/// # Returns
/// Array of indices where drift was detected
#[wasm_bindgen(js_name = "jaccardDriftDetection")]
pub fn jaccard_drift_detection(
    data: &[f64],
    n_features: usize,
    n_samples: usize,
    window_size: usize,
    threshold: f64,
) -> Result<Vec<f64>, JsError> {
    let n = validate_matrix(data, n_features)?;
    if n != n_samples {
        return Err(JsError::new("data length must equal n_samples"));
    }

    if window_size > n_samples {
        return Err(JsError::new("window_size cannot exceed n_samples"));
    }

    let mut detector = jaccard_drift_detector_impl(window_size, threshold)?;
    let mut drift_indices = Vec::new();

    // Slide window through data
    for start in 0..=(n_samples - window_size) {
        let end = start + window_size;
        let window_flat = &data[start * n_features..end * n_features];

        if detector.update(window_flat, window_size)? {
            drift_indices.push(start as f64);
        }
    }

    Ok(drift_indices)
}

/// Compute Jaccard similarity between two sets
///
/// J(A,B) = |A ∩ B| / |A ∪ B|
fn jaccard_similarity(set_a: &[f64], set_b: &[f64]) -> f64 {
    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }

    // For small sets, use O(n*m) comparison (f64 doesn't implement Hash)
    // Since we're dealing with indices cast to f64, exact equality is safe
    let mut intersection = 0usize;
    let mut union_a = 0usize; // Unique elements in A
    let mut union_b = 0usize; // Unique elements in B (not in A)

    // Count unique elements in A and intersection
    let mut seen_a = Vec::new();
    for &a in set_a {
        if !seen_a.contains(&a) {
            seen_a.push(a);
            union_a += 1;
            if set_b.contains(&a) {
                intersection += 1;
            }
        }
    }

    // Count unique elements in B not in A
    let mut seen_b = Vec::new();
    for &b in set_b {
        if !seen_b.contains(&b) && !seen_a.contains(&b) {
            seen_b.push(b);
            union_b += 1;
        }
    }

    let union = union_a + union_b;
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute Jaccard similarity between two windows (nested one-hot vectors)
///
/// Treats each position as a set and computes average Jaccard across positions
fn jaccard_similarity_windows(window_a: &[Vec<f64>], window_b: &[Vec<f64>]) -> f64 {
    if window_a.is_empty() || window_b.is_empty() {
        return 1.0;
    }

    let mut total_similarity = 0.0;
    let mut count = 0;

    for (vec_a, vec_b) in window_a.iter().zip(window_b.iter()) {
        // Convert one-hot to set (non-zero indices)
        let set_a: Vec<f64> = vec_a.iter().enumerate()
            .filter(|(_, &v)| v > 0.5)
            .map(|(i, _)| i as f64)
            .collect();

        let set_b: Vec<f64> = vec_b.iter().enumerate()
            .filter(|(_, &v)| v > 0.5)
            .map(|(i, _)| i as f64)
            .collect();

        let similarity = jaccard_similarity(&set_a, &set_b);
        total_similarity += similarity;
        count += 1;
    }

    if count == 0 {
        1.0
    } else {
        total_similarity / count as f64
    }
}

/// Convert flat array to nested vector (one-hot to categorical)
fn to_nested_vec(flat: &[f64], n_samples: usize) -> Vec<Vec<f64>> {
    flat.chunks(n_samples)
        .map(|chunk| chunk.to_vec())
        .collect()
}

/// Generic drift detection for numeric streams using z-score
///
/// Detects drift when the mean deviates from historical baseline by more than
/// a specified number of standard deviations.
#[wasm_bindgen(js_name = "zscoreDriftDetection")]
pub fn zscore_drift_detection(
    data: &[f64],
    n_samples: usize,
    window_size: usize,
    threshold: f64,
) -> Result<Vec<f64>, JsError> {
    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(JsError::new("data length must equal n_samples"));
    }

    if window_size > n_samples || window_size < 2 {
        return Err(JsError::new("window_size must be in [2, n_samples]"));
    }

    let mut drift_indices = Vec::new();

    // Use first window as baseline
    let baseline_mean: f64 = data[0..window_size].iter().sum::<f64>() / window_size as f64;
    let baseline_var: f64 = data[0..window_size].iter()
        .map(|&x| (x - baseline_mean).powi(2))
        .sum::<f64>() / (window_size - 1) as f64;
    let baseline_std = baseline_var.sqrt();

    if baseline_std == 0.0 {
        return Ok(drift_indices); // No variation, no drift possible
    }

    // Slide window and detect drift
    for start in 1..=(n_samples - window_size) {
        let end = start + window_size;
        let window_mean: f64 = data[start..end].iter().sum::<f64>() / window_size as f64;

        // Z-score: (window_mean - baseline_mean) / baseline_std
        let z_score = (window_mean - baseline_mean) / baseline_std;

        if z_score.abs() > threshold {
            drift_indices.push(start as f64);
        }
    }

    Ok(drift_indices)
}

/// Page-Hinkley test for change point detection
///
/// Cumulative sum-based change detection. Signals change when the
/// cumulative sum exceeds a threshold.
#[wasm_bindgen(js_name = "pageHinkleyDriftDetection")]
pub fn page_hinkley_drift_detection(
    data: &[f64],
    n_samples: usize,
    threshold: f64,
) -> Result<Vec<f64>, JsError> {
    let n = validate_matrix(data, 1)?;
    if n != n_samples {
        return Err(JsError::new("data length must equal n_samples"));
    }

    if n < 2 {
        return Err(JsError::new("need at least 2 samples"));
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;

    // Compute cumulative sum of deviations from mean
    let mut cumulative_sum = 0.0;
    let mut max_cumulative = 0.0;
    let mut min_cumulative = 0.0;
    let mut drift_indices = Vec::new();

    for (i, &value) in data.iter().enumerate() {
        let deviation = value - mean;
        cumulative_sum += deviation;

        // Track min/max
        if cumulative_sum > max_cumulative {
            max_cumulative = cumulative_sum;
        }
        if cumulative_sum < min_cumulative {
            min_cumulative = cumulative_sum;
        }

        // Page-Hinkley statistic: max_cumulative - min_cumulative
        let ph_statistic = max_cumulative - min_cumulative;

        if ph_statistic > threshold {
            drift_indices.push(i as f64);
        }
    }

    Ok(drift_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_detector_creation() {
        let detector = ewma_detector_impl(0.2, 100.0, 10.0).unwrap();
        assert_eq!(detector.lambda(), 0.2);
        assert_eq!(detector.expected(), 100.0);
        assert_eq!(detector.std_err(), 10.0);
        assert_eq!(detector.current_ewma(), 100.0);
    }

    #[test]
    fn test_ewma_detector_invalid_lambda() {
        let result = ewma_detector_impl(0.0, 100.0, 10.0);
        assert!(result.is_err());

        let result = ewma_detector_impl(1.5, 100.0, 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ewma_update_no_drift() {
        let mut detector = ewma_detector_impl(0.5, 100.0, 10.0).unwrap();
        // Value within 1 std: 105 (deviation = 5, std_err = 10)
        assert!(!detector.update(105.0));
        assert_eq!(detector.drift_count(), 0);
    }

    #[test]
    fn test_ewma_update_with_drift() {
        let mut detector = ewma_detector_impl(0.5, 100.0, 10.0).unwrap();
        // Value outside threshold: 125 (deviation = 12.5, std_err = 10)
        assert!(detector.update(125.0));
        assert_eq!(detector.drift_count(), 1);
    }

    #[test]
    fn test_ewma_drift_detection() {
        let mut data = vec![100.0; 100]; // 100 samples at 100
        data[50..100].iter_mut().for_each(|x| *x += 30.0); // Drift at index 50

        let drifts = ewma_drift_detection(&data, 100, 0.3, 100.0, 5.0, 10).unwrap();
        assert!(!drifts.is_empty());
        // First drift should be detected around index 50-60
        assert!(drifts[0] >= 40.0);
    }

    #[test]
    fn test_jaccard_similarity() {
        let set_a = vec![1.0, 2.0, 3.0];
        let set_b = vec![2.0, 3.0, 4.0];
        // Intersection: {2, 3}, Union: {1, 2, 3, 4}
        // Jaccard = 2/4 = 0.5
        let j = jaccard_similarity(&set_a, &set_b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        let set_a = vec![1.0, 2.0, 3.0];
        let set_b = vec![1.0, 2.0, 3.0];
        let j = jaccard_similarity(&set_a, &set_b);
        assert_eq!(j, 1.0);
    }

    #[test]
    fn test_jaccard_similarity_disjoint() {
        let set_a = vec![1.0, 2.0];
        let set_b = vec![3.0, 4.0];
        let j = jaccard_similarity(&set_a, &set_b);
        assert_eq!(j, 0.0);
    }

    #[test]
    fn test_jaccard_drift_detector() {
        let detector = jaccard_drift_detector_impl(3, 0.5).unwrap();
        assert_eq!(detector.window_size(), 3);
        assert_eq!(detector.threshold(), 0.5);
    }

    #[test]
    fn test_jaccard_drift_detector_invalid() {
        let result = jaccard_drift_detector_impl(0, 0.5);
        assert!(result.is_err());

        let result = jaccard_drift_detector_impl(3, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_zscore_drift_detection() {
        let mut data = vec
![1.0; 100]; // 100 samples at 1.0 (with small random variation)
        // Add small noise to create variance
        for i in 0..100 {
            data[i] += (i as f64 * 0.01) % 0.5;
        }
        data[50..100].iter_mut().for_each(|x| *x += 10.0); // Drift at index 50

        let drifts = zscore_drift_detection(&data, 100, 20, 2.0).unwrap();
        assert!(!drifts.is_empty());
        // First drift detected when window first includes the shifted values
    }

    #[test]
    fn test_page_hinkley_drift_detection() {
        let mut data = vec![0.0; 100];
        data[50..100].iter_mut().for_each(|x| *x = 10.0); // Sudden increase

        let drifts = page_hinkley_drift_detection(&data, 100, 50.0).unwrap();
        assert!(!drifts.is_empty());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_invalid_data_length() {
        let result = ewma_drift_detection(&[1.0, 2.0], 3, 0.5, 100.0, 5.0, 0);
        assert!(result.is_err());
    }
}
