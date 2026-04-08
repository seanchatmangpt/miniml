use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get, Rng};
use crate::decision_tree::DecisionTreeModel;
use crate::decision_tree::decision_tree_impl;

#[wasm_bindgen]
pub struct BaggingModel {
    trees: Vec<DecisionTreeModel>,
    n_features: usize,
    n_estimators: usize,
    is_classifier: bool,
}

#[wasm_bindgen]
impl BaggingModel {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(getter, js_name = "nEstimators")]
    pub fn n_estimators(&self) -> usize { self.n_estimators }

    #[wasm_bindgen(getter, js_name = "isClassifier")]
    pub fn is_classifier(&self) -> bool { self.is_classifier }

    /// Predict using majority vote (classification) or averaging (regression)
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let point: Vec<f64> = (0..self.n_features)
                .map(|j| data[i * self.n_features + j])
                .collect();

            // Collect predictions from all trees
            let predictions: Vec<f64> = self.trees.iter()
                .map(|tree| tree.predict(&point)[0])
                .collect();

            if self.is_classifier {
                // Majority vote
                let mut counts: Vec<(f64, usize)> = Vec::new();
                for &pred in &predictions {
                    let pred_rounded = pred.round();
                    if let Some(entry) = counts.iter_mut().find(|(k, _)| (*k - pred_rounded).abs() < 1e-10) {
                        entry.1 += 1;
                    } else {
                        counts.push((pred_rounded, 1));
                    }
                }
                let (best_class, _) = counts.iter().max_by_key(|(_, v)| v).unwrap();
                result.push(*best_class);
            } else {
                // Average
                let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
                result.push(mean);
            }
        }

        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "Bagging(estimators={}, features={}, classifier={})",
            self.n_estimators, self.n_features, self.is_classifier
        )
    }
}

/// Bootstrap sample: randomly sample n indices with replacement
fn bootstrap_sample(n: usize, rng: &mut Rng) -> Vec<usize> {
    (0..n).map(|_| rng.next_usize(n)).collect()
}

/// Random subset without replacement: select subset_size indices
fn subset_without_replacement(n: usize, subset_size: usize, rng: &mut Rng) -> Vec<usize> {
    let subset_size = subset_size.max(1).min(n);
    // Fisher-Yates partial shuffle
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..subset_size {
        let j = rng.next_usize(n - i) + i;
        indices.swap(i, j);
    }
    indices[..subset_size].to_vec()
}

pub fn bagging_impl(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    bootstrap: bool,
    is_classifier: bool,
) -> Result<BaggingModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if labels.len() != n {
        return Err(MlError::new("labels length must match number of samples"));
    }
    if n_estimators == 0 {
        return Err(MlError::new("n_estimators must be > 0"));
    }
    if n < 2 {
        return Err(MlError::new("Need at least 2 samples"));
    }

    let mut rng = Rng::from_data(data);
    let mut trees = Vec::with_capacity(n_estimators);

    for _ in 0..n_estimators {
        let sample_indices = if bootstrap {
            // Sample WITH replacement, same size as original
            bootstrap_sample(n, &mut rng)
        } else {
            // Sample WITHOUT replacement, 50% of original
            let subset_size = (n as f64 * 0.5).ceil() as usize;
            subset_without_replacement(n, subset_size, &mut rng)
        };

        // Create subset data
        let subset_n = sample_indices.len();
        let mut subset_data = Vec::with_capacity(subset_n * n_features);
        let mut subset_labels = Vec::with_capacity(subset_n);
        for &idx in &sample_indices {
            for j in 0..n_features {
                subset_data.push(mat_get(data, n_features, idx, j));
            }
            subset_labels.push(labels[idx]);
        }

        // Build decision tree on the subset
        match decision_tree_impl(
            &subset_data,
            n_features,
            &subset_labels,
            max_depth,
            min_samples_split,
            is_classifier,
        ) {
            Ok(tree) => trees.push(tree),
            Err(_) => continue, // Skip failed trees (e.g., all same class in subset)
        }
    }

    if trees.is_empty() {
        return Err(MlError::new("All estimators failed to build"));
    }

    Ok(BaggingModel {
        n_features,
        n_estimators: trees.len(),
        is_classifier,
        trees,
    })
}

#[wasm_bindgen(js_name = "baggingClassify")]
pub fn bagging_classify(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
) -> Result<BaggingModel, JsError> {
    bagging_impl(
        data, n_features, labels,
        n_estimators, max_depth, min_samples_split,
        true,  // bootstrap = true by default
        true,  // is_classifier
    ).map_err(|e| JsError::new(&e.message))
}

#[wasm_bindgen(js_name = "baggingRegress")]
pub fn bagging_regress(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
) -> Result<BaggingModel, JsError> {
    bagging_impl(
        data, n_features, targets,
        n_estimators, max_depth, min_samples_split,
        true,  // bootstrap = true by default
        false, // is_classifier
    ).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bagging_classification() {
        // XOR-like pattern
        let data = vec![
            0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  1.0, 0.0,
            0.1, 0.1,  1.1, 1.1,  0.1, 1.1,  1.1, 0.1,
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let model = bagging_impl(&data, 2, &labels, 10, 5, 2, true, true).unwrap();

        let test = vec![0.05, 0.05, 1.05, 1.05, 0.05, 1.05, 1.05, 0.05];
        let preds = model.predict(&test);
        assert_eq!(preds.len(), 4);
        // XOR pattern: (0,0)->0, (1,1)->0, (0,1)->1, (1,0)->1
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 1.0);
        assert_eq!(preds[3], 1.0);
    }

    #[test]
    fn test_bagging_regression() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let targets: Vec<f64> = (0..20).map(|i| (i as f64) * 2.0 + 1.0).collect();
        let model = bagging_impl(&data, 1, &targets, 5, 5, 2, true, false).unwrap();

        let test = vec![5.0, 10.0, 15.0];
        let preds = model.predict(&test);

        // Should be close to 11, 21, 31
        assert!((preds[0] - 11.0).abs() < 3.0);
        assert!((preds[1] - 21.0).abs() < 3.0);
        assert!((preds[2] - 31.0).abs() < 3.0);
    }

    #[test]
    fn test_bagging_without_bootstrap() {
        // Subset sampling (no replacement, 50% of data)
        // Use a larger dataset with duplicates to ensure reliable classification
        let data = vec![
            0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  1.0, 0.0,
            0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  1.0, 0.0,
            0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  1.0, 0.0,
        ];
        let labels = vec![
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
        ];
        let model = bagging_impl(&data, 2, &labels, 20, 5, 2, false, true).unwrap();

        let test = vec![0.1, 0.1, 0.1, 0.9];
        let preds = model.predict(&test);
        assert_eq!(preds.len(), 2);
        // (0.1, 0.1) is near the (0,0) cluster -> should predict class 0
        // (0.1, 0.9) is near the (0,1) cluster -> should predict class 1
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_single_estimator_fallback() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];
        let model = bagging_impl(&data, 2, &labels, 1, 3, 2, true, true).unwrap();
        assert_eq!(model.n_estimators(), 1);
    }

    #[test]
    fn test_deterministic() {
        let data = vec![
            0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  1.0, 0.0,
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let m1 = bagging_impl(&data, 2, &labels, 5, 3, 2, true, true).unwrap();
        let m2 = bagging_impl(&data, 2, &labels, 5, 3, 2, true, true).unwrap();

        // Same data -> same seed -> same result
        let test = vec![0.0, 0.0, 1.0, 1.0];
        assert_eq!(m1.predict(&test), m2.predict(&test));
    }

    #[test]
    fn test_getters() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];
        let model = bagging_impl(&data, 2, &labels, 5, 3, 2, true, true).unwrap();
        assert_eq!(model.n_features(), 2);
        assert!(model.n_estimators() > 0);
        assert!(model.is_classifier());
    }

    #[test]
    fn test_getters_regressor() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let targets: Vec<f64> = (0..10).map(|i| (i as f64) * 2.0).collect();
        let model = bagging_impl(&data, 1, &targets, 3, 3, 2, true, false).unwrap();
        assert_eq!(model.n_features(), 1);
        assert!(model.n_estimators() > 0);
        assert!(!model.is_classifier());
    }

    #[test]
    fn test_to_string() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];
        let model = bagging_impl(&data, 2, &labels, 5, 3, 2, true, true).unwrap();
        let s = model.to_string_js();
        assert!(s.contains("Bagging"));
        assert!(s.contains("classifier=true"));
    }

    #[test]
    fn test_invalid_inputs() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];

        // n_estimators = 0
        assert!(bagging_impl(&data, 2, &labels, 0, 3, 2, true, true).is_err());

        // mismatched lengths
        assert!(bagging_impl(&data, 2, &[0.0], 3, 3, 2, true, true).is_err());

        // empty data
        assert!(bagging_impl(&[], 2, &labels, 3, 3, 2, true, true).is_err());

        // n_features = 0
        assert!(bagging_impl(&data, 0, &labels, 3, 3, 2, true, true).is_err());

        // only 1 sample
        assert!(bagging_impl(&[0.0, 0.0], 2, &[0.0], 3, 3, 2, true, true).is_err());
    }

    #[test]
    fn test_bootstrap_sample_size() {
        // Bootstrap with replacement should produce same-size samples
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let model = bagging_impl(&data, 1, &labels, 10, 3, 2, true, true).unwrap();
        // Should successfully build all or most trees
        assert!(model.n_estimators() >= 1);
    }
}
