use wasm_bindgen::prelude::*;
use crate::error::MlError;

/// Result of Recursive Feature Elimination.
///
/// Since WASM cannot pass arbitrary model closures, this simplified RFE works
/// directly with feature importance scores (e.g., from a decision tree's
/// `feature_importance`). Features are ranked by importance and the
/// least-important ones are eliminated.
#[wasm_bindgen]
pub struct RfeResult {
    ranking: Vec<usize>,
    support: Vec<bool>,
    n_features: usize,
    n_features_to_select: usize,
    n_features_selected: usize,
}

#[wasm_bindgen]
impl RfeResult {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    #[wasm_bindgen(getter, js_name = "nFeaturesToSelect")]
    pub fn n_features_to_select(&self) -> usize {
        self.n_features_to_select
    }

    #[wasm_bindgen(getter, js_name = "nFeaturesSelected")]
    pub fn n_features_selected(&self) -> usize {
        self.n_features_selected
    }

    /// Returns 1.0 for selected features, 0.0 for eliminated features.
    #[wasm_bindgen(js_name = "getSupport")]
    pub fn get_support(&self) -> Vec<f64> {
        self.support.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect()
    }

    /// Returns the feature ranking as f64 array.
    /// Rank 1 = most important, higher values = less important.
    #[wasm_bindgen(js_name = "getRanking")]
    pub fn get_ranking(&self) -> Vec<f64> {
        self.ranking.iter().map(|&r| r as f64).collect()
    }

    /// Select columns from a data matrix based on the support mask.
    ///
    /// `data` is a flat array in row-major order with `n_features` columns.
    /// Returns a new flat array with only the selected columns, preserving
    /// row-major order.
    #[wasm_bindgen]
    pub fn transform(&self, data: &[f64], n_features: usize) -> Vec<f64> {
        if data.is_empty() || n_features == 0 {
            return vec![];
        }
        if n_features != self.n_features {
            return vec![];
        }

        let n_samples = data.len() / n_features;
        let selected_count = self.support.iter().filter(|&&s| s).count();
        if selected_count == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n_samples * selected_count);
        for i in 0..n_samples {
            for j in 0..n_features {
                if self.support[j] {
                    result.push(data[i * n_features + j]);
                }
            }
        }
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "RFE(selected={}/{}, features={})",
            self.n_features_selected, self.n_features_to_select, self.n_features
        )
    }
}

/// Recursive Feature Elimination (simplified for WASM).
///
/// Takes a flat array of feature importance scores (one per feature) and
/// returns an `RfeResult` with ranking and support mask.
///
/// Since we cannot retrain models inside WASM (no model closure), this is a
/// single-pass ranking based on the provided importance scores rather than
/// true iterative RFE with refitting.
///
/// # Arguments
/// * `feature_importance` - flat array of importance scores, one per feature
/// * `n_features_to_select` - number of features to keep
///
/// # Errors
/// Returns `MlError` if:
/// - `feature_importance` is empty
/// - `n_features_to_select` is 0 or greater than the number of features
pub fn rfe_impl(
    feature_importance: &[f64],
    n_features_to_select: usize,
) -> Result<RfeResult, MlError> {
    let n_features = feature_importance.len();

    if n_features == 0 {
        return Err(MlError::new("feature_importance must not be empty"));
    }
    if n_features_to_select == 0 {
        return Err(MlError::new("n_features_to_select must be at least 1"));
    }
    if n_features_to_select > n_features {
        return Err(MlError::new(
            "n_features_to_select must not exceed number of features",
        ));
    }

    // Build (index, importance) pairs and sort by importance descending.
    // Features with higher importance get lower rank numbers (rank 1 = best).
    let mut indexed: Vec<(usize, f64)> = feature_importance
        .iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp))
        .collect();

    // Sort by importance descending; ties broken by original index (stable sort).
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Assign rankings: rank 1 = most important, rank n = least important.
    let mut ranking = vec![0usize; n_features];
    for (rank, &(original_index, _)) in indexed.iter().enumerate() {
        ranking[original_index] = rank + 1;
    }

    // Build support mask: features with rank <= n_features_to_select are selected.
    let mut support = vec![false; n_features];
    for &(original_index, _) in indexed.iter().take(n_features_to_select) {
        support[original_index] = true;
    }

    let n_features_selected = support.iter().filter(|&&s| s).count();

    Ok(RfeResult {
        ranking,
        support,
        n_features,
        n_features_to_select,
        n_features_selected,
    })
}

/// WASM-exported Recursive Feature Elimination.
///
/// See [`rfe_impl`] for documentation.
#[wasm_bindgen(js_name = "rfe")]
pub fn rfe(feature_importance: &[f64], n_features_to_select: usize) -> Result<RfeResult, JsError> {
    rfe_impl(feature_importance, n_features_to_select).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ranking() {
        // Feature 0 has importance 0.5, feature 1 has 0.3, feature 2 has 0.2
        let importance = vec![0.5, 0.3, 0.2];
        let result = rfe_impl(&importance, 2).unwrap();

        assert_eq!(result.n_features, 3);
        assert_eq!(result.n_features_to_select, 2);
        assert_eq!(result.n_features_selected, 2);

        // Feature 0 should be rank 1 (most important)
        assert_eq!(result.ranking[0], 1);
        // Feature 1 should be rank 2
        assert_eq!(result.ranking[1], 2);
        // Feature 2 should be rank 3 (least important, eliminated)
        assert_eq!(result.ranking[2], 3);

        // Support should be true for features 0 and 1
        assert!(result.support[0]);
        assert!(result.support[1]);
        assert!(!result.support[2]);
    }

    #[test]
    fn test_select_all_features() {
        let importance = vec![0.1, 0.4, 0.3, 0.2];
        let result = rfe_impl(&importance, 4).unwrap();

        assert_eq!(result.n_features_selected, 4);
        assert!(result.support.iter().all(|&s| s));
    }

    #[test]
    fn test_select_one_feature() {
        let importance = vec![0.1, 0.9, 0.3];
        let result = rfe_impl(&importance, 1).unwrap();

        assert_eq!(result.n_features_selected, 1);
        assert!(result.support[1]); // feature 1 has highest importance
        assert!(!result.support[0]);
        assert!(!result.support[2]);
        assert_eq!(result.ranking[1], 1);
    }

    #[test]
    fn test_empty_importance_errors() {
        let result = rfe_impl(&[], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_select_errors() {
        let result = rfe_impl(&[0.5, 0.3], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_more_than_features_errors() {
        let result = rfe_impl(&[0.5, 0.3], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_tie_breaking_by_index() {
        // Two features with equal importance: lower index should get better rank
        let importance = vec![0.5, 0.5, 0.2];
        let result = rfe_impl(&importance, 2).unwrap();

        assert_eq!(result.ranking[0], 1);
        assert_eq!(result.ranking[1], 2);
        assert_eq!(result.ranking[2], 3);
    }

    #[test]
    fn test_get_support_returns_f64() {
        let importance = vec![0.8, 0.1, 0.1];
        let result = rfe_impl(&importance, 1).unwrap();
        let support = result.get_support();

        assert_eq!(support.len(), 3);
        assert_eq!(support[0], 1.0);
        assert_eq!(support[1], 0.0);
        assert_eq!(support[2], 0.0);
    }

    #[test]
    fn test_get_ranking_returns_f64() {
        let importance = vec![0.4, 0.6];
        let result = rfe_impl(&importance, 1).unwrap();
        let ranking = result.get_ranking();

        assert_eq!(ranking.len(), 2);
        assert_eq!(ranking[0], 2.0); // feature 0 is rank 2 (less important)
        assert_eq!(ranking[1], 1.0); // feature 1 is rank 1 (most important)
    }

    #[test]
    fn test_transform_selects_columns() {
        // 2 features, select only feature 1 (higher importance)
        let importance = vec![0.2, 0.8];
        let result = rfe_impl(&importance, 1).unwrap();

        // Data matrix: 3 samples x 2 features (row-major)
        let data = vec![
            1.0, 10.0,
            2.0, 20.0,
            3.0, 30.0,
        ];
        let transformed = result.transform(&data, 2);

        assert_eq!(transformed.len(), 3); // 3 samples x 1 selected feature
        assert_eq!(transformed[0], 10.0);
        assert_eq!(transformed[1], 20.0);
        assert_eq!(transformed[2], 30.0);
    }

    #[test]
    fn test_transform_selects_multiple_columns() {
        // 4 features, select top 2 (features 0 and 3)
        let importance = vec![0.5, 0.1, 0.1, 0.3];
        let result = rfe_impl(&importance, 2).unwrap();

        // Data matrix: 2 samples x 4 features
        let data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        let transformed = result.transform(&data, 4);

        assert_eq!(transformed.len(), 4); // 2 samples x 2 selected features
        // Selected columns are 0 and 3
        assert_eq!(transformed[0], 1.0);
        assert_eq!(transformed[1], 4.0);
        assert_eq!(transformed[2], 5.0);
        assert_eq!(transformed[3], 8.0);
    }

    #[test]
    fn test_transform_empty_data() {
        let importance = vec![0.5, 0.3];
        let result = rfe_impl(&importance, 1).unwrap();
        let transformed = result.transform(&[], 2);
        assert!(transformed.is_empty());
    }

    #[test]
    fn test_transform_wrong_n_features() {
        let importance = vec![0.5, 0.3, 0.2];
        let result = rfe_impl(&importance, 2).unwrap();
        // Pass wrong n_features
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let transformed = result.transform(&data, 2);
        assert!(transformed.is_empty());
    }

    #[test]
    fn test_single_feature() {
        let importance = vec![1.0];
        let result = rfe_impl(&importance, 1).unwrap();

        assert_eq!(result.n_features, 1);
        assert_eq!(result.n_features_selected, 1);
        assert_eq!(result.ranking[0], 1);
        assert!(result.support[0]);
    }

    #[test]
    fn test_to_string() {
        let importance = vec![0.5, 0.3, 0.2];
        let result = rfe_impl(&importance, 2).unwrap();
        let s = result.to_string_js();
        assert!(s.contains("RFE"));
        assert!(s.contains("selected=2/2"));
    }
}
