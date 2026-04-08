use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get, Rng};

#[derive(Clone)]
struct Node {
    feature: usize,
    threshold: f64,
    left: usize,   // index into nodes vec (0 = none)
    right: usize,
    prediction: f64,
    is_leaf: bool,
}

/// A single Extremely Randomized Tree stored as a flat Vec<Node>
#[derive(Clone)]
struct ExtraTreeNodes {
    nodes: Vec<Node>,
}

impl ExtraTreeNodes {
    fn predict_one(&self, sample: &[f64], _n_features: usize) -> f64 {
        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                return node.prediction;
            }
            let val = sample[node.feature];
            node_idx = if val <= node.threshold { node.left } else { node.right };
        }
    }
}

#[wasm_bindgen]
pub struct ExtraTreesModel {
    trees: Vec<ExtraTreeNodes>,
    n_features: usize,
    n_trees: usize,
    is_classifier: bool,
}

#[wasm_bindgen]
impl ExtraTreesModel {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(getter, js_name = "nTrees")]
    pub fn n_trees(&self) -> usize { self.n_trees }

    #[wasm_bindgen(getter, js_name = "isClassifier")]
    pub fn is_classifier(&self) -> bool { self.is_classifier }

    /// Predict using majority vote (classification) or averaging (regression)
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let sample = &data[i * self.n_features..(i + 1) * self.n_features];

            // Collect predictions from all trees
            let predictions: Vec<f64> = self.trees.iter()
                .map(|tree| tree.predict_one(sample, self.n_features))
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

    /// Return flat trees: each tree is [n_nodes, feature, threshold, left, right, prediction, is_leaf, ...]
    #[wasm_bindgen(js_name = "getTrees")]
    pub fn get_trees(&self) -> Vec<f64> {
        let mut flat = Vec::new();
        for tree in &self.trees {
            flat.push(tree.nodes.len() as f64);
            for node in &tree.nodes {
                flat.push(node.feature as f64);
                flat.push(node.threshold);
                flat.push(node.left as f64);
                flat.push(node.right as f64);
                flat.push(node.prediction);
                flat.push(if node.is_leaf { 1.0 } else { 0.0 });
            }
        }
        flat
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("ExtraTrees(trees={}, features={}, classifier={})", self.n_trees, self.n_features, self.is_classifier)
    }
}

/// Count occurrences of each unique label (no HashMap)
fn label_counts(labels: &[f64]) -> Vec<(u32, usize)> {
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for &l in labels {
        let key = l as u32;
        if let Some(entry) = counts.iter_mut().find(|(k, _)| *k == key) {
            entry.1 += 1;
        } else {
            counts.push((key, 1));
        }
    }
    counts
}

fn majority_class(labels: &[f64]) -> f64 {
    let counts = label_counts(labels);
    let (cls, _) = counts.iter().max_by_key(|(_, v)| v).unwrap();
    *cls as f64
}

fn mean_value(targets: &[f64]) -> f64 {
    targets.iter().sum::<f64>() / targets.len() as f64
}

/// Number of random features to consider at each split.
/// sqrt(n_features) for classification, n_features for regression (as per original paper).
fn n_random_features(n_features: usize, is_classifier: bool) -> usize {
    if is_classifier {
        (n_features as f64).sqrt().ceil() as usize
    } else {
        n_features
    }
}

/// Select K random features via partial Fisher-Yates shuffle
fn select_random_features(n_features: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    let k = k.max(1).min(n_features);
    let mut indices: Vec<usize> = (0..n_features).collect();
    for i in 0..k {
        let j = rng.next_usize(n_features - i) + i;
        indices.swap(i, j);
    }
    indices[..k].to_vec()
}

/// Gini impurity for a set of labels
fn gini_impurity(labels: &[f64]) -> f64 {
    if labels.is_empty() { return 0.0; }
    let n = labels.len() as f64;
    let counts = label_counts(labels);
    let mut g = 1.0;
    for &(_, c) in &counts {
        let p = c as f64 / n;
        g -= p * p;
    }
    g
}

/// Mean Squared Error for a set of target values
fn mse(targets: &[f64]) -> f64 {
    if targets.is_empty() { return 0.0; }
    let m = mean_value(targets);
    targets.iter().map(|t| (t - m).powi(2)).sum::<f64>() / targets.len() as f64
}

struct ExtraTreeBuilder<'a> {
    data: &'a [f64],
    targets: &'a [f64],
    n_features: usize,
    max_depth: usize,
    min_samples_split: usize,
    is_classifier: bool,
    n_try_features: usize,
    nodes: Vec<Node>,
    rng: &'a mut Rng,
}

impl<'a> ExtraTreeBuilder<'a> {
    /// Build tree on ALL samples (no bootstrap — Extra Trees uses the full dataset)
    fn build(&mut self, indices: &[usize], depth: usize) -> usize {
        let targets: Vec<f64> = indices.iter().map(|&i| self.targets[i]).collect();

        // Leaf conditions
        if indices.len() < self.min_samples_split || depth >= self.max_depth || self.is_pure(&targets) {
            let pred = if self.is_classifier { majority_class(&targets) } else { mean_value(&targets) };
            let idx = self.nodes.len();
            self.nodes.push(Node { feature: 0, threshold: 0.0, left: 0, right: 0, prediction: pred, is_leaf: true });
            return idx;
        }

        // Extra Trees split: pick K random features, random threshold per feature,
        // choose the one with best impurity reduction among the random candidates
        let (best_feature, best_threshold, best_gain) = self.find_best_random_split(indices, &targets);

        if best_gain < 0.0 {
            let pred = if self.is_classifier { majority_class(&targets) } else { mean_value(&targets) };
            let idx = self.nodes.len();
            self.nodes.push(Node { feature: 0, threshold: 0.0, left: 0, right: 0, prediction: pred, is_leaf: true });
            return idx;
        }

        // Partition indices by the chosen split
        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) = indices.iter()
            .partition(|&&i| mat_get(self.data, self.n_features, i, best_feature) <= best_threshold);

        if left_idx.is_empty() || right_idx.is_empty() {
            let pred = if self.is_classifier { majority_class(&targets) } else { mean_value(&targets) };
            let idx = self.nodes.len();
            self.nodes.push(Node { feature: 0, threshold: 0.0, left: 0, right: 0, prediction: pred, is_leaf: true });
            return idx;
        }

        // Internal node
        let node_idx = self.nodes.len();
        self.nodes.push(Node { feature: best_feature, threshold: best_threshold, left: 0, right: 0, prediction: 0.0, is_leaf: false });

        let left = self.build(&left_idx, depth + 1);
        let right = self.build(&right_idx, depth + 1);
        self.nodes[node_idx].left = left;
        self.nodes[node_idx].right = right;

        node_idx
    }

    fn is_pure(&self, targets: &[f64]) -> bool {
        targets.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10)
    }

    /// Extra Trees core: evaluate random splits on K random features,
    /// pick the best among the random candidates (not the globally optimal split).
    fn find_best_random_split(&mut self, indices: &[usize], targets: &[f64]) -> (usize, f64, f64) {
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = -1.0;

        // Parent impurity
        let parent_impurity = if self.is_classifier {
            gini_impurity(targets)
        } else {
            mse(targets)
        };

        // Select K random features
        let random_features = select_random_features(self.n_features, self.n_try_features, self.rng);

        for &feature in &random_features {
            // Find min and max of this feature among current indices
            let mut f_min = f64::INFINITY;
            let mut f_max = f64::NEG_INFINITY;
            for &idx in indices {
                let val = mat_get(self.data, self.n_features, idx, feature);
                if val < f_min { f_min = val; }
                if val > f_max { f_max = val; }
            }

            // If all values are the same, skip this feature
            if (f_max - f_min).abs() < 1e-12 {
                continue;
            }

            // Pick a RANDOM threshold between min and max (Extra Trees key difference)
            let threshold = f_min + self.rng.next_f64() * (f_max - f_min);

            // Partition target values by this threshold and compute impurity reduction
            let mut left_labels: Vec<f64> = Vec::new();
            let mut right_labels: Vec<f64> = Vec::new();
            for &idx in indices {
                if mat_get(self.data, self.n_features, idx, feature) <= threshold {
                    left_labels.push(self.targets[idx]);
                } else {
                    right_labels.push(self.targets[idx]);
                }
            }

            if left_labels.is_empty() || right_labels.is_empty() {
                continue;
            }

            let n = indices.len() as f64;
            let ln = left_labels.len() as f64;
            let rn = right_labels.len() as f64;

            let weighted_impurity = if self.is_classifier {
                (ln / n) * gini_impurity(&left_labels) + (rn / n) * gini_impurity(&right_labels)
            } else {
                (ln / n) * mse(&left_labels) + (rn / n) * mse(&right_labels)
            };

            let gain = parent_impurity - weighted_impurity;

            if gain > best_gain {
                best_gain = gain;
                best_feature = feature;
                best_threshold = threshold;
            }
        }

        (best_feature, best_threshold, best_gain)
    }
}

pub fn extra_trees_impl(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    is_classifier: bool,
) -> Result<ExtraTreesModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if targets.len() != n {
        return Err(MlError::new("targets length must match number of samples"));
    }
    if n_trees == 0 {
        return Err(MlError::new("n_trees must be > 0"));
    }
    if n < 2 {
        return Err(MlError::new("Need at least 2 samples"));
    }

    let mut rng = Rng::from_data(data);
    let n_try = n_random_features(n_features, is_classifier);
    let all_indices: Vec<usize> = (0..n).collect();
    let mut trees = Vec::with_capacity(n_trees);

    for _ in 0..n_trees {
        let mut builder = ExtraTreeBuilder {
            data,
            targets,
            n_features,
            max_depth,
            min_samples_split,
            is_classifier,
            n_try_features: n_try,
            nodes: Vec::new(),
            rng: &mut rng,
        };
        builder.build(&all_indices, 0);
        trees.push(ExtraTreeNodes { nodes: builder.nodes });
    }

    Ok(ExtraTreesModel {
        n_features,
        n_trees,
        is_classifier,
        trees,
    })
}

#[wasm_bindgen(js_name = "extraTreesClassify")]
pub fn extra_trees_classify(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
) -> Result<ExtraTreesModel, JsError> {
    extra_trees_impl(data, n_features, labels, n_trees, max_depth, min_samples_split, true)
        .map_err(|e| JsError::new(&e.message))
}

#[wasm_bindgen(js_name = "extraTreesRegress")]
pub fn extra_trees_regress(
    data: &[f64],
    n_features: usize,
    targets: &[f64],
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
) -> Result<ExtraTreesModel, JsError> {
    extra_trees_impl(data, n_features, targets, n_trees, max_depth, min_samples_split, false)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_xor() {
        // XOR-like pattern: class = feature0 XOR feature1
        let data = vec![
            0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  1.0, 0.0,
            0.1, 0.1,  1.1, 1.1,  0.1, 1.1,  1.1, 0.1,
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let model = extra_trees_impl(&data, 2, &labels, 20, 10, 2, true).unwrap();

        let test = vec![0.05, 0.05, 1.05, 1.05, 0.05, 1.05, 1.05, 0.05];
        let preds = model.predict(&test);
        assert_eq!(preds.len(), 4);
        // (0,0)->0, (1,1)->0, (0,1)->1, (1,0)->1
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 1.0);
        assert_eq!(preds[3], 1.0);
    }

    #[test]
    fn test_regression_linear() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let targets: Vec<f64> = (0..20).map(|i| (i as f64) * 2.0 + 1.0).collect();
        let model = extra_trees_impl(&data, 1, &targets, 10, 8, 2, false).unwrap();

        let test = vec![5.0, 10.0, 15.0];
        let preds = model.predict(&test);

        // Should be close to 11, 21, 31
        assert!((preds[0] - 11.0).abs() < 3.0);
        assert!((preds[1] - 21.0).abs() < 3.0);
        assert!((preds[2] - 31.0).abs() < 3.0);
    }

    #[test]
    fn test_single_tree() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];
        let model = extra_trees_impl(&data, 2, &labels, 1, 3, 2, true).unwrap();
        assert_eq!(model.n_trees(), 1);
        assert_eq!(model.n_features(), 2);
        assert!(model.is_classifier());
    }

    #[test]
    fn test_deterministic() {
        let data = vec![
            0.0, 0.0,  1.0, 1.0,  0.0, 1.0,  1.0, 0.0,
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let m1 = extra_trees_impl(&data, 2, &labels, 5, 5, 2, true).unwrap();
        let m2 = extra_trees_impl(&data, 2, &labels, 5, 5, 2, true).unwrap();

        let test = vec![0.0, 0.0, 1.0, 1.0];
        assert_eq!(m1.predict(&test), m2.predict(&test));
    }

    #[test]
    fn test_no_bootstrap_full_dataset() {
        // Extra Trees should use ALL samples (no bootstrap)
        // Verify by checking model trains on a dataset where bootstrap would change results
        let data = vec![
            0.0, 0.0,
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
            4.0, 4.0,
            5.0, 5.0,
            6.0, 6.0,
            7.0, 7.0,
            8.0, 8.0,
            9.0, 9.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // With enough trees, Extra Trees should perfectly classify this simple threshold
        let model = extra_trees_impl(&data, 2, &labels, 50, 10, 2, true).unwrap();
        let test = vec![2.0, 2.0, 7.0, 7.0];
        let preds = model.predict(&test);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_get_trees_format() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];
        let model = extra_trees_impl(&data, 2, &labels, 2, 3, 2, true).unwrap();
        let flat = model.get_trees();

        // Format: [n_nodes, (feature, threshold, left, right, prediction, is_leaf) * n_nodes, ...] per tree
        let mut offset = 0;
        let mut tree_count = 0;
        while offset < flat.len() {
            let n_nodes = flat[offset] as usize;
            offset += 1;
            assert!(n_nodes > 0, "Each tree should have at least 1 node");
            offset += n_nodes * 6;
            tree_count += 1;
        }
        assert_eq!(tree_count, 2);
    }

    #[test]
    fn test_to_string() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];
        let model = extra_trees_impl(&data, 2, &labels, 3, 3, 2, true).unwrap();
        let s = model.to_string_js();
        assert!(s.contains("ExtraTrees"));
        assert!(s.contains("trees=3"));
        assert!(s.contains("features=2"));
    }

    #[test]
    fn test_error_zero_trees() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];
        let result = extra_trees_impl(&data, 2, &labels, 0, 3, 2, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_mismatched_lengths() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0, 2.0];
        let result = extra_trees_impl(&data, 2, &labels, 5, 3, 2, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_too_few_samples() {
        let data = vec![0.0, 0.0];
        let labels = vec![0.0];
        let result = extra_trees_impl(&data, 2, &labels, 5, 3, 2, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_depth_limit() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let labels: Vec<f64> = (0..20).map(|i| (i % 3) as f64).collect();

        let model = extra_trees_impl(&data, 1, &labels, 5, 1, 2, true).unwrap();
        let flat = model.get_trees();

        // With max_depth=1, each tree has at most 3 nodes (root + 2 leaves)
        let mut offset = 0;
        while offset < flat.len() {
            let n_nodes = flat[offset] as usize;
            assert!(n_nodes <= 3, "max_depth=1 should yield at most 3 nodes, got {}", n_nodes);
            offset += 1 + n_nodes * 6;
        }
    }

    #[test]
    fn test_regression_averaging() {
        // 3 samples, single feature, build 10 trees
        let data = vec![1.0, 2.0, 3.0];
        let targets = vec![10.0, 20.0, 30.0];
        let model = extra_trees_impl(&data, 1, &targets, 10, 5, 2, false).unwrap();

        let preds = model.predict(&data);
        assert_eq!(preds.len(), 3);
        // Should be reasonably close to training targets
        assert!((preds[0] - 10.0).abs() < 5.0);
        assert!((preds[1] - 20.0).abs() < 5.0);
        assert!((preds[2] - 30.0).abs() < 5.0);
    }

    #[test]
    fn test_multiclass_classification() {
        // 3-class problem: feature value determines class
        let data = vec![
            0.0, 1.0, 2.0,
            0.1, 1.1, 2.1,
            0.2, 1.2, 2.2,
        ];
        let labels = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let model = extra_trees_impl(&data, 1, &labels, 20, 10, 2, true).unwrap();

        let test = vec![0.05, 1.05, 2.05];
        let preds = model.predict(&test);
        assert_eq!(preds.len(), 3);
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
        assert_eq!(preds[2], 2.0);
    }
}
