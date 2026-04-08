use wasm_bindgen::prelude::*;
use crate::error::MlError;

/// Result of grid search cross-validation.
///
/// Given pre-computed CV scores for multiple parameter combinations,
/// holds the best score, its index, and per-combination statistics.
#[wasm_bindgen]
#[derive(Debug)]
pub struct GridSearchResult {
    best_score: f64,
    best_params_idx: usize,
    mean_scores: Vec<f64>,
    std_scores: Vec<f64>,
    n_combinations: usize,
}

#[wasm_bindgen]
impl GridSearchResult {
    /// Returns the best (highest) mean cross-validation score.
    #[wasm_bindgen(js_name = "bestScore")]
    pub fn get_best_score(&self) -> f64 {
        self.best_score
    }

    /// Returns the index (into the original parameter grid) of the best
    /// parameter combination.
    #[wasm_bindgen(js_name = "bestParamsIdx")]
    pub fn get_best_params_idx(&self) -> usize {
        self.best_params_idx
    }

    /// Returns the mean CV score for every parameter combination.
    #[wasm_bindgen(js_name = "meanScores")]
    pub fn get_mean_scores(&self) -> Vec<f64> {
        self.mean_scores.clone()
    }

    /// Returns the standard deviation of CV scores for every parameter
    /// combination.
    #[wasm_bindgen(js_name = "stdScores")]
    pub fn get_std_scores(&self) -> Vec<f64> {
        self.std_scores.clone()
    }

    /// Returns the total number of parameter combinations evaluated.
    #[wasm_bindgen(js_name = "nCombinations")]
    pub fn get_n_combinations(&self) -> usize {
        self.n_combinations
    }

    /// Returns indices of parameter combinations sorted by mean score
    /// (best first).
    #[wasm_bindgen(js_name = "rankResults")]
    pub fn rank_results(&self) -> Vec<usize> {
        let mut indexed: Vec<usize> = (0..self.n_combinations).collect();
        indexed.sort_unstable_by(|&a, &b| {
            self.mean_scores[b]
                .partial_cmp(&self.mean_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed
    }
}

/// Internal implementation: exhaustive grid search over pre-computed
/// cross-validation scores.
///
/// `scores` is a flat array of shape `[n_combinations * n_folds]`.
/// Row `i` (combination `i`) contains `n_folds` consecutive entries
/// starting at offset `i * n_folds`.
///
/// For each combination the mean and standard deviation across folds are
/// computed, and the combination with the highest mean score is selected.
pub fn grid_search_impl(
    scores: &[f64],
    n_folds: usize,
) -> Result<GridSearchResult, MlError> {
    if n_folds < 2 {
        return Err(MlError::new("n_folds must be >= 2"));
    }
    if scores.is_empty() {
        return Err(MlError::new("scores must not be empty"));
    }
    if !scores.len().is_multiple_of(n_folds) {
        return Err(MlError::new(
            "scores length must be a multiple of n_folds",
        ));
    }

    let n_combinations = scores.len() / n_folds;

    let mut mean_scores = Vec::with_capacity(n_combinations);
    let mut std_scores = Vec::with_capacity(n_combinations);

    for i in 0..n_combinations {
        let offset = i * n_folds;
        let fold_scores = &scores[offset..offset + n_folds];

        let mean = fold_scores.iter().sum::<f64>() / n_folds as f64;
        let variance =
            fold_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                / n_folds as f64;
        let std = variance.sqrt();

        mean_scores.push(mean);
        std_scores.push(std);
    }

    // Find combination with highest mean score.
    // On ties, pick the one with the lower index (first encountered).
    let best_params_idx = mean_scores
        .iter()
        .enumerate()
        .max_by(|&(idx_a, a), &(idx_b, b)| {
            a.partial_cmp(b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| idx_b.cmp(&idx_a))
        })
        .map(|(idx, _)| idx)
        .unwrap();

    let best_score = mean_scores[best_params_idx];

    Ok(GridSearchResult {
        best_score,
        best_params_idx,
        mean_scores,
        std_scores,
        n_combinations,
    })
}

/// WASM-exported grid search over pre-computed cross-validation scores.
///
/// `scores` -- flat array of shape `[n_combinations * n_folds]`.
/// `n_folds` -- number of cross-validation folds per combination.
///
/// Returns a `GridSearchResult` with the best score and per-combination
/// statistics.
#[wasm_bindgen(js_name = "gridSearch")]
pub fn grid_search(
    scores: &[f64],
    n_folds: usize,
) -> Result<GridSearchResult, JsError> {
    grid_search_impl(scores, n_folds).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: 3 combinations, 3 folds each
    // Combo 0: [0.8, 0.9, 0.85] -> mean 0.85
    // Combo 1: [0.7, 0.75, 0.8] -> mean 0.75
    // Combo 2: [0.9, 0.95, 0.9] -> mean 0.9167
    fn sample_scores() -> Vec<f64> {
        vec![
            0.8, 0.9, 0.85, // combo 0
            0.7, 0.75, 0.8, // combo 1
            0.9, 0.95, 0.9, // combo 2
        ]
    }

    #[test]
    fn test_best_score_selected() {
        let result = grid_search_impl(&sample_scores(), 3).unwrap();
        assert_eq!(result.best_params_idx, 2);
        // mean of [0.9, 0.95, 0.9] = 2.75 / 3
        let expected_mean = 2.75 / 3.0;
        assert!((result.best_score - expected_mean).abs() < 1e-10);
    }

    #[test]
    fn test_mean_and_std_shapes() {
        let result = grid_search_impl(&sample_scores(), 3).unwrap();
        assert_eq!(result.n_combinations, 3);
        assert_eq!(result.mean_scores.len(), 3);
        assert_eq!(result.std_scores.len(), 3);
    }

    #[test]
    fn test_mean_values() {
        let result = grid_search_impl(&sample_scores(), 3).unwrap();
        assert!((result.mean_scores[0] - 0.85).abs() < 1e-10);
        assert!((result.mean_scores[1] - 0.75).abs() < 1e-10);
        assert!((result.mean_scores[2] - (2.75 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_std_values() {
        let result = grid_search_impl(&sample_scores(), 3).unwrap();
        // Combo 0: [0.8, 0.9, 0.85] mean=0.85
        // var = ((0.05)^2 + (0.05)^2 + 0^2) / 3 = (0.0025+0.0025)/3 = 0.005/3
        let expected_std = (0.005 / 3.0_f64).sqrt();
        assert!((result.std_scores[0] - expected_std).abs() < 1e-10);
    }

    #[test]
    fn test_zero_std_when_constant() {
        // All folds give same score -> std = 0
        let scores = vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.5];
        let result = grid_search_impl(&scores, 3).unwrap();
        assert!((result.std_scores[0]).abs() < 1e-10);
        assert!((result.std_scores[1]).abs() < 1e-10);
        assert_eq!(result.best_params_idx, 0); // mean 1.0 > 0.5
    }

    #[test]
    fn test_rank_results_best_first() {
        let result = grid_search_impl(&sample_scores(), 3).unwrap();
        let ranks = result.rank_results();
        assert_eq!(ranks[0], 2); // best
        assert_eq!(ranks[1], 0);
        assert_eq!(ranks[2], 1); // worst
    }

    #[test]
    fn test_rank_results_single() {
        let scores = vec![0.8, 0.9, 0.85];
        let result = grid_search_impl(&scores, 3).unwrap();
        let ranks = result.rank_results();
        assert_eq!(ranks, vec![0]);
    }

    #[test]
    fn test_tie_break_by_index() {
        // Two combos with identical mean -> first index wins
        let scores = vec![0.8, 0.8, 0.8, 0.8, 0.8, 0.8];
        let result = grid_search_impl(&scores, 3).unwrap();
        assert_eq!(result.best_params_idx, 0);
    }

    #[test]
    fn test_empty_scores_error() {
        let err = grid_search_impl(&[], 3).unwrap_err();
        assert!(err.message.contains("must not be empty"));
    }

    #[test]
    fn test_n_folds_too_small() {
        let err = grid_search_impl(&[0.8, 0.9], 1).unwrap_err();
        assert!(err.message.contains("n_folds must be >= 2"));
    }

    #[test]
    fn test_scores_not_multiple_of_folds() {
        let err = grid_search_impl(&[0.8, 0.9, 0.85, 0.7], 3).unwrap_err();
        assert!(err.message.contains("multiple of n_folds"));
    }

    #[test]
    fn test_single_fold_two() {
        let scores = vec![0.6, 0.8];
        let result = grid_search_impl(&scores, 2).unwrap();
        assert_eq!(result.n_combinations, 1);
        assert!((result.mean_scores[0] - 0.7).abs() < 1e-10);
        assert_eq!(result.best_params_idx, 0);
    }

    #[test]
    fn test_many_combinations() {
        // 10 combinations, 5 folds each
        let scores: Vec<f64> = (0..50).map(|i| (i % 10) as f64 / 10.0).collect();
        let result = grid_search_impl(&scores, 5).unwrap();
        assert_eq!(result.n_combinations, 10);
        assert_eq!(result.rank_results().len(), 10);
        // combo i has scores at indices i*5..i*5+5
        // Combos 1,3,6,9 all have mean 0.7 (tied for best)
        // Tie-break: lowest index wins -> best_params_idx = 1
        assert_eq!(result.best_params_idx, 1);
        assert!((result.best_score - 0.7).abs() < 1e-10);
        let ranks = result.rank_results();
        // Best combo (rank 0) should have mean 0.7
        assert!((result.mean_scores[ranks[0]] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_getters() {
        let result = grid_search_impl(&sample_scores(), 3).unwrap();
        // Verify getters return expected values
        assert_eq!(result.get_best_params_idx(), 2);
        assert_eq!(result.get_n_combinations(), 3);
        let means = result.get_mean_scores();
        assert_eq!(means.len(), 3);
        let stds = result.get_std_scores();
        assert_eq!(stds.len(), 3);
    }

    #[test]
    fn test_get_best_score() {
        let result = grid_search_impl(&sample_scores(), 3).unwrap();
        assert!((result.get_best_score() - (2.75 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_negative_scores() {
        // Some metrics (like neg MSE) can be negative
        let scores = vec![
            -1.0, -0.8, -0.9,  // combo 0: mean -0.9 (better, less negative)
            -2.0, -1.5, -1.8,  // combo 1: mean -1.767
        ];
        let result = grid_search_impl(&scores, 3).unwrap();
        assert_eq!(result.best_params_idx, 0);
        assert!((result.best_score - (-0.9)).abs() < 1e-10);
    }
}
