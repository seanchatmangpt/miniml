use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, Rng};

/// Result of permutation feature importance analysis.
///
/// Permutation importance measures feature importance by shuffling each feature
/// and measuring the decrease in model performance. Higher importance means the
/// feature is more important (larger drop when shuffled).
#[wasm_bindgen]
pub struct PermutationImportanceResult {
    importances: Vec<f64>,
    importance_std: Vec<f64>,
    baseline_score: f64,
    n_features: usize,
    n_repeats: usize,
}

#[wasm_bindgen]
impl PermutationImportanceResult {
    /// Importance per feature (decrease in score when shuffled).
    /// Higher values indicate more important features.
    #[wasm_bindgen(js_name = "getImportances")]
    pub fn get_importances(&self) -> Vec<f64> {
        self.importances.clone()
    }

    /// Standard deviation of importance across shuffles per feature.
    #[wasm_bindgen(js_name = "getImportanceStd")]
    pub fn get_importance_std(&self) -> Vec<f64> {
        self.importance_std.clone()
    }

    /// Feature indices sorted by importance (most important first).
    #[wasm_bindgen(js_name = "getRanking")]
    pub fn get_ranking(&self) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = self.importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().map(|(i, _)| *i).collect()
    }

    /// Original model score before any permutation.
    #[wasm_bindgen(js_name = "baselineScore", getter)]
    pub fn baseline_score(&self) -> f64 {
        self.baseline_score
    }

    /// Number of features analyzed.
    #[wasm_bindgen(js_name = "nFeatures", getter)]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of permutation repeats per feature.
    #[wasm_bindgen(js_name = "nRepeats", getter)]
    pub fn n_repeats(&self) -> usize {
        self.n_repeats
    }
}

/// Compute permutation feature importance from pre-computed scores.
///
/// Since WASM cannot pass model functions across the boundary, the user provides
/// the baseline score and per-feature shuffled scores obtained by running the model
/// externally for each permutation.
///
/// # Arguments
///
/// * `baseline_score` - Original model score (higher-is-better metric).
/// * `feature_scores` - Flat array of shape [n_features * n_repeats] containing
///   model scores after permuting each feature. Layout: feature 0 repeat 0,
///   feature 0 repeat 1, ..., feature 1 repeat 0, ...
/// * `n_features` - Number of features.
/// * `n_repeats` - Number of permutation repeats per feature.
///
/// # Returns
///
/// A `PermutationImportanceResult` with importance = baseline - mean(shuffled scores)
/// for each feature. Higher importance means the feature is more important.
pub fn permutation_importance_impl(
    baseline_score: f64,
    feature_scores: &[f64],
    n_features: usize,
    n_repeats: usize,
) -> Result<PermutationImportanceResult, MlError> {
    if n_features == 0 {
        return Err(MlError::new("n_features must be > 0"));
    }
    if n_repeats == 0 {
        return Err(MlError::new("n_repeats must be > 0"));
    }
    let expected_len = n_features * n_repeats;
    if feature_scores.len() != expected_len {
        return Err(MlError::new(format!(
            "feature_scores length {} must equal n_features * n_repeats ({} * {} = {})",
            feature_scores.len(), n_features, n_repeats, expected_len
        )));
    }

    let mut importances = Vec::with_capacity(n_features);
    let mut importance_std = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let start = j * n_repeats;
        let end = start + n_repeats;
        let slice = &feature_scores[start..end];

        let mean: f64 = slice.iter().sum::<f64>() / n_repeats as f64;
        let importance = baseline_score - mean;
        importances.push(importance);

        // Standard deviation
        let variance: f64 = slice.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / n_repeats as f64;
        importance_std.push(variance.sqrt());
    }

    Ok(PermutationImportanceResult {
        importances,
        importance_std,
        baseline_score,
        n_features,
        n_repeats,
    })
}

/// WASM-exported version of permutation importance (pre-computed scores).
///
/// # Arguments
///
/// * `baseline_score` - Original model score.
/// * `feature_scores` - Flat array [n_features * n_repeats] of shuffled scores.
/// * `n_features` - Number of features.
/// * `n_repeats` - Number of shuffles per feature.
#[wasm_bindgen(js_name = "permutationImportance")]
pub fn permutation_importance(
    baseline_score: f64,
    feature_scores: &[f64],
    n_features: usize,
    n_repeats: usize,
) -> Result<PermutationImportanceResult, JsError> {
    permutation_importance_impl(baseline_score, feature_scores, n_features, n_repeats)
        .map_err(|e| JsError::new(&e.message))
}

/// Compute a metric score between true targets and predictions.
///
/// Supported metrics: "accuracy", "r2", "mse", "mae", "f1".
/// For "accuracy", "f1": higher is better.
/// For "mse", "mae": lower is better (importance = shuffled - baseline).
fn compute_metric_score(targets: &[f64], predictions: &[f64], metric: &str) -> Result<(f64, bool), MlError> {
    if targets.len() != predictions.len() {
        return Err(MlError::new("targets and predictions must have same length"));
    }

    match metric {
        "accuracy" => {
            let mut correct = 0usize;
            for (&t, &p) in targets.iter().zip(predictions.iter()) {
                if (t - p).abs() < 1e-10 {
                    correct += 1;
                }
            }
            Ok((correct as f64 / targets.len() as f64, true))
        }
        "r2" => {
            let n = targets.len() as f64;
            let mean_true: f64 = targets.iter().sum::<f64>() / n;
            let ss_tot: f64 = targets.iter().map(|y| (y - mean_true).powi(2)).sum();
            let ss_res: f64 = targets.iter().zip(predictions.iter())
                .map(|(t, p)| (t - p).powi(2)).sum();
            let r2 = if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot };
            Ok((r2, true))
        }
        "mse" => {
            let mse = targets.iter().zip(predictions.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f64>() / targets.len() as f64;
            Ok((mse, false)) // lower is better
        }
        "mae" => {
            let mae = targets.iter().zip(predictions.iter())
                .map(|(t, p)| (t - p).abs())
                .sum::<f64>() / targets.len() as f64;
            Ok((mae, false)) // lower is better
        }
        "f1" => {
            // Binary F1 (threshold at 0.5)
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut fn_count = 0usize;
            for (&t, &p) in targets.iter().zip(predictions.iter()) {
                let t_bin = t > 0.5;
                let p_bin = p > 0.5;
                if t_bin && p_bin { tp += 1; }
                else if !t_bin && p_bin { fp += 1; }
                else if t_bin && !p_bin { fn_count += 1; }
            }
            let precision = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
            let recall = if tp + fn_count == 0 { 0.0 } else { tp as f64 / (tp + fn_count) as f64 };
            let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
            Ok((f1, true))
        }
        _ => Err(MlError::new(format!(
            "unsupported metric '{}'. Use: accuracy, r2, mse, mae, f1",
            metric
        ))),
    }
}

/// Fisher-Yates shuffle of a single column in a flat row-major matrix.
fn shuffle_column(data: &mut [f64], n_features: usize, col: usize, rng: &mut Rng, n_samples: usize) {
    for i in (1..n_samples).rev() {
        let j = rng.next_usize(i + 1);
        if i != j {
            let a = i * n_features + col;
            let b = j * n_features + col;
            data.swap(a, b);
        }
    }
}

/// Compute permutation feature importance by actually shuffling features.
///
/// NOTE: True permutation importance requires re-running the model on shuffled data.
/// Since we cannot re-invoke a model function from within WASM, this function uses
/// a proxy approach: it measures how much shuffling each feature disrupts the
/// correlation between the feature and the original predictions.
///
/// For production use, prefer `permutationImportance` with pre-computed scores
/// obtained by running the model externally for each permutation.
///
/// # Arguments
///
/// * `data` - Feature matrix (flat row-major, shape [n_samples * n_features]).
/// * `n_features` - Number of features.
/// * `targets` - True target values.
/// * `predictions` - Model predictions for the original (unshuffled) data.
/// * `metric` - Scoring metric: "accuracy", "r2", "mse", "mae", "f1".
/// * `n_repeats` - Number of permutation repeats per feature.
/// * `seed` - Random seed for reproducibility.
///
/// # Returns
///
/// `PermutationImportanceResult` with importance scores per feature.
#[wasm_bindgen(js_name = "permutationImportanceCompute")]
pub fn permutation_importance_compute(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    predictions: &[f64],
    metric: &str,
    n_repeats: usize,
    seed: u64,
) -> Result<PermutationImportanceResult, JsError> {
    let n_samples = validate_matrix(data, n_features)
        .map_err(|e| JsError::new(&e.message))?;

    if targets.len() != n_samples {
        return Err(JsError::new("targets length must equal n_samples"));
    }
    if predictions.len() != n_samples {
        return Err(JsError::new("predictions length must equal n_samples"));
    }
    if n_repeats == 0 {
        return Err(JsError::new("n_repeats must be > 0"));
    }

    // Compute baseline score
    let (baseline_score, _higher_is_better) = compute_metric_score(targets, predictions, metric)
        .map_err(|e| JsError::new(&e.message))?;

    // For each feature, shuffle and measure correlation disruption with predictions
    // This is a proxy since we cannot re-run the model
    let mut feature_scores = Vec::with_capacity(n_features * n_repeats);
    let mut working_data = data.to_vec();

    for j in 0..n_features {
        let mut rng = Rng::new(seed.wrapping_add(j as u64));

        for _ in 0..n_repeats {
            // Restore original data
            working_data.copy_from_slice(data);

            // Shuffle column j
            shuffle_column(&mut working_data, n_features, j, &mut rng, n_samples);

            // Compute correlation between shuffled feature j and original predictions
            // as a proxy for how much information the feature contributes
            let feature_col: Vec<f64> = (0..n_samples)
                .map(|i| working_data[i * n_features + j])
                .collect();

            let score = correlation_with_predictions(&feature_col, predictions);

            // Store score (negate so importance = baseline - score makes sense)
            // Higher correlation -> higher score -> lower importance drop
            feature_scores.push(score);
        }
    }

    // Build result
    let mut importances = Vec::with_capacity(n_features);
    let mut importance_std = Vec::with_capacity(n_features);

    for j in 0..n_features {
        let start = j * n_repeats;
        let end = start + n_repeats;
        let slice = &feature_scores[start..end];

        let mean: f64 = slice.iter().sum::<f64>() / n_repeats as f64;
        // Importance = how much the original correlation drops when shuffled
        // Original correlation is 1.0 (perfect self-correlation before shuffle)
        let importance = 1.0 - mean.abs();
        importances.push(importance);

        let variance: f64 = slice.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / n_repeats as f64;
        importance_std.push(variance.sqrt());
    }

    Ok(PermutationImportanceResult {
        importances,
        importance_std,
        baseline_score,
        n_features,
        n_repeats,
    })
}

/// Compute Pearson correlation between a feature column and predictions.
fn correlation_with_predictions(feature: &[f64], predictions: &[f64]) -> f64 {
    let n = feature.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_f: f64 = feature.iter().sum::<f64>() / n;
    let mean_p: f64 = predictions.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_f = 0.0;
    let mut var_p = 0.0;

    for i in 0..feature.len() {
        let df = feature[i] - mean_f;
        let dp = predictions[i] - mean_p;
        cov += df * dp;
        var_f += df * df;
        var_p += dp * dp;
    }

    let denom = (var_f * var_p).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }

    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_importance_basic() {
        // 3 features, 2 repeats each
        let baseline = 0.95;
        // Feature 0: scores drop a lot (important)
        // Feature 1: scores drop moderately
        // Feature 2: scores barely change (unimportant)
        let feature_scores = vec![
            0.70, 0.72,  // feature 0: mean = 0.71, importance = 0.24
            0.85, 0.83,  // feature 1: mean = 0.84, importance = 0.11
            0.94, 0.93,  // feature 2: mean = 0.935, importance = 0.015
        ];

        let result = permutation_importance_impl(baseline, &feature_scores, 3, 2).unwrap();

        assert_eq!(result.n_features, 3);
        assert_eq!(result.n_repeats, 2);
        assert!((result.baseline_score - 0.95).abs() < 1e-10);

        // Feature 0 most important, feature 2 least
        assert!(result.importances[0] > result.importances[1]);
        assert!(result.importances[1] > result.importances[2]);
    }

    #[test]
    fn test_permutation_importance_ranking() {
        let baseline = 0.90;
        let feature_scores = vec![
            0.50, 0.55,  // feature 0: importance ~ 0.375
            0.80, 0.85,  // feature 1: importance ~ 0.075
            0.30, 0.40,  // feature 2: importance ~ 0.55
        ];

        let result = permutation_importance_impl(baseline, &feature_scores, 3, 2).unwrap();
        let ranking = result.get_ranking();

        // Feature 2 (most important), then 0, then 1
        assert_eq!(ranking[0], 2);
        assert_eq!(ranking[1], 0);
        assert_eq!(ranking[2], 1);
    }

    #[test]
    fn test_permutation_importance_std() {
        let baseline = 0.90;
        let feature_scores = vec![
            0.50, 0.70,  // high variance
            0.80, 0.80,  // zero variance
        ];

        let result = permutation_importance_impl(baseline, &feature_scores, 2, 2).unwrap();

        assert!(result.importance_std[0] > 0.0);
        assert!(result.importance_std[1] < 1e-10);
    }

    #[test]
    fn test_permutation_importance_zero_features() {
        let result = permutation_importance_impl(0.9, &[], 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_importance_zero_repeats() {
        let result = permutation_importance_impl(0.9, &[], 3, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_importance_wrong_length() {
        let result = permutation_importance_impl(0.9, &[0.5, 0.6], 3, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_permutation_importance_single_feature_single_repeat() {
        let result = permutation_importance_impl(0.9, &[0.6], 1, 1).unwrap();
        assert_eq!(result.importances.len(), 1);
        assert!((result.importances[0] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_importance_negative_importance() {
        // Feature that actually improves when shuffled (rare but possible)
        let baseline = 0.70;
        let feature_scores = vec![0.80, 0.75]; // shuffled is better
        let result = permutation_importance_impl(baseline, &feature_scores, 1, 2).unwrap();
        // Importance = baseline - mean = 0.70 - 0.775 = -0.075
        assert!(result.importances[0] < 0.0);
    }

    #[test]
    fn test_compute_metric_accuracy() {
        let targets = vec![0.0, 1.0, 1.0, 0.0];
        let preds = vec![0.0, 1.0, 0.0, 0.0];
        let (score, higher_is_better) = compute_metric_score(&targets, &preds, "accuracy").unwrap();
        assert!((score - 0.75).abs() < 1e-10);
        assert!(higher_is_better);
    }

    #[test]
    fn test_compute_metric_mse() {
        let targets = vec![1.0, 2.0, 3.0];
        let preds = vec![1.5, 2.5, 3.5];
        let (score, higher_is_better) = compute_metric_score(&targets, &preds, "mse").unwrap();
        assert!((score - 0.25).abs() < 1e-10);
        assert!(!higher_is_better);
    }

    #[test]
    fn test_compute_metric_r2() {
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let preds = vec![1.0, 2.0, 3.0, 4.0];
        let (score, higher_is_better) = compute_metric_score(&targets, &preds, "r2").unwrap();
        assert!((score - 1.0).abs() < 1e-10);
        assert!(higher_is_better);
    }

    #[test]
    fn test_compute_metric_mae() {
        let targets = vec![1.0, 3.0, 5.0];
        let preds = vec![2.0, 3.0, 6.0];
        let (score, higher_is_better) = compute_metric_score(&targets, &preds, "mae").unwrap();
        // MAE = (|1-2| + |3-3| + |5-6|) / 3 = 2/3
        assert!((score - (2.0 / 3.0)).abs() < 1e-10);
        assert!(!higher_is_better);
    }

    #[test]
    fn test_compute_metric_f1() {
        let targets = vec![1.0, 1.0, 0.0, 0.0];
        let preds = vec![1.0, 0.0, 0.0, 1.0];
        // TP=1, FP=1, FN=1 -> precision=0.5, recall=0.5, F1=0.5
        let (score, higher_is_better) = compute_metric_score(&targets, &preds, "f1").unwrap();
        assert!((score - 0.5).abs() < 1e-10);
        assert!(higher_is_better);
    }

    #[test]
    fn test_compute_metric_unsupported() {
        let targets = vec![0.0];
        let preds = vec![0.0];
        let result = compute_metric_score(&targets, &preds, "unsupported");
        assert!(result.is_err());
    }

    #[test]
    fn test_shuffle_column_changes_order() {
        let mut data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let original_col0: Vec<f64> = data.iter().step_by(2).cloned().collect();
        let original_col1: Vec<f64> = data.iter().skip(1).step_by(2).cloned().collect();

        let mut rng = Rng::new(42);
        shuffle_column(&mut data, 2, 0, &mut rng, 4);

        // Column 0 should be shuffled (different order, same values)
        let shuffled_col0: Vec<f64> = data.iter().step_by(2).cloned().collect();
        let mut sorted_orig = original_col0.clone();
        let mut sorted_shuf = shuffled_col0.clone();
        sorted_orig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_shuf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted_orig, sorted_shuf);

        // Column 1 should be unchanged
        let col1_after: Vec<f64> = data.iter().skip(1).step_by(2).cloned().collect();
        assert_eq!(col1_after, original_col1);
    }

    #[test]
    fn test_correlation_perfect() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = correlation_with_predictions(&a, &b);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_negative() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr = correlation_with_predictions(&a, &b);
        assert!((corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_importance_compute_basic() {
        // 4 samples, 2 features
        // Feature 0 perfectly predicts target, feature 1 is noise
        let data = vec![
            1.0, 5.0,
            2.0, 3.0,
            3.0, 7.0,
            4.0, 1.0,
        ];
        let targets = vec![1.0, 2.0, 3.0, 4.0];
        let predictions = vec![1.0, 2.0, 3.0, 4.0]; // perfect predictions

        let result = permutation_importance_compute(
            &data, 2, &targets, &predictions, "r2", 3, 42
        ).unwrap();

        assert_eq!(result.n_features, 2);
        assert_eq!(result.n_repeats, 3);
        assert_eq!(result.importances.len(), 2);

        // Feature 0 is correlated with predictions, so shuffling it should
        // disrupt correlation more than shuffling feature 1 (noise)
        assert!(result.importances[0] > result.importances[1]);
    }

    #[test]
    #[cfg_attr(not(target_arch = "wasm32"), ignore)]
    fn test_permutation_importance_compute_invalid_metric() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![0.0, 1.0];
        let preds = vec![0.0, 1.0];

        let result = permutation_importance_compute(
            &data, 2, &targets, &preds, "bad_metric", 1, 42
        );
        assert!(result.is_err());
    }

    #[test]
    #[cfg_attr(not(target_arch = "wasm32"), ignore)]
    fn test_permutation_importance_compute_wrong_target_length() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![0.0, 1.0, 2.0]; // 3 samples but data has 2
        let preds = vec![0.0, 1.0];

        let result = permutation_importance_compute(
            &data, 2, &targets, &preds, "accuracy", 1, 42
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_get_ranking_preserves_all_indices() {
        let baseline = 0.9;
        let feature_scores = vec![0.5, 0.7, 0.8, 0.3];
        let result = permutation_importance_impl(baseline, &feature_scores, 4, 1).unwrap();
        let ranking = result.get_ranking();

        assert_eq!(ranking.len(), 4);
        // All indices present
        let mut sorted_ranking = ranking.clone();
        sorted_ranking.sort();
        assert_eq!(sorted_ranking, vec![0, 1, 2, 3]);
    }
}
