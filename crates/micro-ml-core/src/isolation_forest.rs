use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get, Rng};

/// A single node in an Isolation Tree
#[derive(Clone)]
struct IsolationNode {
    feature: usize,
    threshold: f64,
    left: usize,
    right: usize,
    is_leaf: bool,
    size: usize,
}

/// A single Isolation Tree stored as a flat Vec<IsolationNode>
#[derive(Clone)]
struct IsolationTree {
    nodes: Vec<IsolationNode>,
}

impl IsolationTree {
    /// Traverse the tree for a single sample, returning the path length (depth of leaf)
    fn path_length(&self, sample: &[f64], _n_features: usize) -> f64 {
        let mut node_idx = 0;
        let mut depth = 0.0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf {
                // If the leaf has more than 1 sample, add the average path length
                // for an unsuccessful BST search on `size` items
                if node.size > 1 {
                    depth += avg_path_length(node.size);
                }
                return depth;
            }
            let val = sample[node.feature];
            node_idx = if val <= node.threshold { node.left } else { node.right };
            depth += 1.0;
        }
    }
}

/// Isolation Forest model for anomaly detection.
///
/// Anomalies are isolated faster (shorter average path length) than normal points.
#[wasm_bindgen]
pub struct IsolationForestModel {
    trees: Vec<IsolationTree>,
    n_features: usize,
    n_trees: usize,
    max_samples: usize,
    contamination: f64,
    threshold_: f64,
}

#[wasm_bindgen]
impl IsolationForestModel {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(getter, js_name = "nTrees")]
    pub fn n_trees(&self) -> usize { self.n_trees }

    #[wasm_bindgen(getter, js_name = "maxSamples")]
    pub fn max_samples(&self) -> usize { self.max_samples }

    #[wasm_bindgen(getter)]
    pub fn contamination(&self) -> f64 { self.contamination }

    #[wasm_bindgen(getter)]
    pub fn threshold_(&self) -> f64 { self.threshold_ }

    /// Predict anomaly labels: 1 for normal, -1 for anomaly
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<i32> {
        let scores = self.score_samples(data);
        scores.iter().map(|&s| if s >= self.threshold_ { -1 } else { 1 }).collect()
    }

    /// Anomaly score in [0.0, 1.0]. Higher = more anomalous.
    ///
    /// Score(x) = 2^(-E(h(x)) / c(n))
    /// where E(h(x)) = average path length across all trees,
    /// c(n) = average path length of unsuccessful search in BST.
    #[wasm_bindgen(js_name = "scoreSamples")]
    pub fn score_samples(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let c_n = avg_path_length(self.max_samples);
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let sample = &data[i * self.n_features..(i + 1) * self.n_features];
            let avg_path: f64 = self.trees.iter()
                .map(|tree| tree.path_length(sample, self.n_features))
                .sum::<f64>() / self.trees.len() as f64;
            // Score in [0, 1]: lower path length = higher anomaly score
            let score = 2.0_f64.powf(-avg_path / c_n);
            result.push(score);
        }

        result
    }

    /// Shifted anomaly score: score_samples - 0.5. Negative = normal, positive = anomaly.
    #[wasm_bindgen(js_name = "decisionFunction")]
    pub fn decision_function(&self, data: &[f64]) -> Vec<f64> {
        self.score_samples(data).iter().map(|&s| s - 0.5).collect()
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "IsolationForest(trees={}, features={}, maxSamples={}, contamination={}, threshold={:.6})",
            self.n_trees, self.n_features, self.max_samples, self.contamination, self.threshold_
        )
    }
}

/// Average path length of unsuccessful search in a Binary Search Tree.
///
/// c(n) = 2 * H(n-1) - 2*(n-1)/n
/// where H(i) = ln(i) + Euler-Mascheroni constant (~0.5772156649)
///
/// For n <= 1, returns 0.0 (no search needed).
/// For n == 2, returns 1.0 (one comparison).
pub fn avg_path_length(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n == 2 {
        return 1.0;
    }
    let n_f = n as f64;
    let euler = 0.5772156649015329; // Euler-Mascheroni constant
    let h = (n_f - 1.0).ln() + euler; // Harmonic number H(n-1)
    2.0 * h - 2.0 * (n_f - 1.0) / n_f
}

/// Build a single Isolation Tree on a subsample.
///
/// Each internal node picks a random feature and a random threshold between
/// the min and max of that feature in the current subset. Splits continue
/// until the node is pure (1 sample) or max depth is reached.
fn build_isolation_tree(
    indices: &[usize],
    data: &[f64],
    n_features: usize,
    max_depth: usize,
    rng: &mut Rng,
) -> IsolationTree {
    let mut nodes = Vec::new();
    build_tree_recursive(indices, data, n_features, 0, max_depth, rng, &mut nodes);
    IsolationTree { nodes }
}

fn build_tree_recursive(
    indices: &[usize],
    data: &[f64],
    n_features: usize,
    depth: usize,
    max_depth: usize,
    rng: &mut Rng,
    nodes: &mut Vec<IsolationNode>,
) -> usize {
    // Leaf: single sample or reached max depth
    if indices.len() <= 1 || depth >= max_depth {
        let idx = nodes.len();
        nodes.push(IsolationNode {
            feature: 0,
            threshold: 0.0,
            left: 0,
            right: 0,
            is_leaf: true,
            size: indices.len(),
        });
        return idx;
    }

    // Pick a random feature
    let feature = rng.next_usize(n_features);

    // Find min and max of the chosen feature among current indices
    let mut f_min = f64::INFINITY;
    let mut f_max = f64::NEG_INFINITY;
    for &idx in indices {
        let val = mat_get(data, n_features, idx, feature);
        if val < f_min { f_min = val; }
        if val > f_max { f_max = val; }
    }

    // If all values are the same, this becomes a leaf
    if (f_max - f_min).abs() < 1e-12 {
        let idx = nodes.len();
        nodes.push(IsolationNode {
            feature: 0,
            threshold: 0.0,
            left: 0,
            right: 0,
            is_leaf: true,
            size: indices.len(),
        });
        return idx;
    }

    // Random threshold between min and max
    let threshold = f_min + rng.next_f64() * (f_max - f_min);

    // Partition indices
    let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices.iter()
        .partition(|&&i| mat_get(data, n_features, i, feature) <= threshold);

    // If partition produced an empty side, make a leaf
    if left_indices.is_empty() || right_indices.is_empty() {
        let idx = nodes.len();
        nodes.push(IsolationNode {
            feature: 0,
            threshold: 0.0,
            left: 0,
            right: 0,
            is_leaf: true,
            size: indices.len(),
        });
        return idx;
    }

    // Internal node placeholder
    let node_idx = nodes.len();
    nodes.push(IsolationNode {
        feature,
        threshold,
        left: 0,
        right: 0,
        is_leaf: false,
        size: 0,
    });

    let left = build_tree_recursive(&left_indices, data, n_features, depth + 1, max_depth, rng, nodes);
    let right = build_tree_recursive(&right_indices, data, n_features, depth + 1, max_depth, rng, nodes);
    nodes[node_idx].left = left;
    nodes[node_idx].right = right;

    node_idx
}

/// Core Isolation Forest implementation.
///
/// Builds `n_trees` isolation trees, each on a random subsample of `max_samples`
/// points. After building, computes the anomaly score threshold based on the
/// `contamination` parameter.
pub fn isolation_forest_impl(
    data: &[f64],
    n_features: usize,
    n_trees: usize,
    max_samples: usize,
    contamination: f64,
) -> Result<IsolationForestModel, MlError> {
    let n_samples = validate_matrix(data, n_features)?;
    if n_trees == 0 {
        return Err(MlError::new("n_trees must be > 0"));
    }
    if n_samples < 2 {
        return Err(MlError::new("Need at least 2 samples"));
    }
    if !(0.0..=0.5).contains(&contamination) {
        return Err(MlError::new("contamination must be in [0.0, 0.5]"));
    }

    // Effective subsample size: use min(max_samples, n_samples)
    let effective_max_samples = max_samples.min(n_samples);

    // Max depth: ceil(log2(effective_max_samples))
    let max_depth = if effective_max_samples <= 1 {
        1
    } else {
        (effective_max_samples as f64).log2().ceil() as usize
    };

    let mut rng = Rng::from_data(data);
    let mut trees = Vec::with_capacity(n_trees);
    let all_indices: Vec<usize> = (0..n_samples).collect();

    for _ in 0..n_trees {
        // Subsample: pick effective_max_samples random indices without replacement
        // via partial Fisher-Yates shuffle
        let mut sample_indices = all_indices.clone();
        for i in 0..effective_max_samples {
            let j = rng.next_usize(n_samples - i) + i;
            sample_indices.swap(i, j);
        }
        let sub_indices = &sample_indices[..effective_max_samples];

        let tree = build_isolation_tree(sub_indices, data, n_features, max_depth, &mut rng);
        trees.push(tree);
    }

    // Compute anomaly scores on training data to determine threshold
    let c_n = avg_path_length(effective_max_samples);
    let mut scores: Vec<f64> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let sample = &data[i * n_features..(i + 1) * n_features];
        let avg_path: f64 = trees.iter()
            .map(|tree| tree.path_length(sample, n_features))
            .sum::<f64>() / trees.len() as f64;
        let score = 2.0_f64.powf(-avg_path / c_n);
        scores.push(score);
    }

    // Determine threshold: the score at the contamination quantile
    // Higher scores = more anomalous, so we want the score at the (1 - contamination) percentile
    let threshold = if contamination > 0.0 && n_samples > 0 {
        // Sort scores ascending; pick the score at position corresponding to contamination
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_anomalies = (contamination * n_samples as f64).ceil() as usize;
        let n_anomalies = n_anomalies.max(1).min(n_samples);
        // The threshold is the score at the boundary: the lowest score that is considered anomalous
        // In sorted ascending order, anomalies are at the end (highest scores)
        scores[n_samples - n_anomalies]
    } else {
        0.5 // Default: equal split
    };

    Ok(IsolationForestModel {
        trees,
        n_features,
        n_trees,
        max_samples: effective_max_samples,
        contamination,
        threshold_: threshold,
    })
}

#[wasm_bindgen(js_name = "isolationForest")]
pub fn isolation_forest(
    data: &[f64],
    n_features: usize,
    n_trees: usize,
    max_samples: usize,
    contamination: f64,
) -> Result<IsolationForestModel, JsError> {
    isolation_forest_impl(data, n_features, n_trees, max_samples, contamination)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avg_path_length() {
        assert!((avg_path_length(0) - 0.0).abs() < 1e-10);
        assert!((avg_path_length(1) - 0.0).abs() < 1e-10);
        assert!((avg_path_length(2) - 1.0).abs() < 1e-10);
        // Verify exact formula: c(n) = 2*H(n-1) - 2*(n-1)/n where H is harmonic number
        // H(n-1) = ln(n-1) + euler
        let c_100 = avg_path_length(100);
        let euler = 0.5772156649015329;
        let h = 99.0_f64.ln() + euler;
        let expected = 2.0 * h - 2.0 * 99.0 / 100.0;
        assert!((c_100 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_basic_construction() {
        // 10 normal points clustered at origin, 2 outlier points far away
        let mut data = Vec::new();
        // 10 normal points
        for _ in 0..10 {
            data.push(0.0);
            data.push(0.0);
        }
        // 2 outliers
        data.push(10.0);
        data.push(10.0);
        data.push(-10.0);
        data.push(-10.0);

        let model = isolation_forest_impl(&data, 2, 100, 50, 0.1).unwrap();
        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_trees(), 100);
        assert_eq!(model.max_samples(), 12); // min(50, 12) = 12
        assert!((model.contamination() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_anomaly_detection_outliers() {
        // 20 normal points near origin, 3 outlier points far away
        let mut data = Vec::new();
        for i in 0..20 {
            let x = (i % 5) as f64 * 0.1;
            let y = (i / 5) as f64 * 0.1;
            data.push(x);
            data.push(y);
        }
        // 3 outliers
        data.push(100.0);
        data.push(100.0);
        data.push(-100.0);
        data.push(-100.0);
        data.push(100.0);
        data.push(-100.0);

        let model = isolation_forest_impl(&data, 2, 200, 23, 0.2).unwrap();
        let predictions = model.predict(&data);

        // Outliers are the last 3 points (indices 20, 21, 22)
        assert_eq!(predictions.len(), 23);
        // Outlier scores should be higher than normal scores
        let scores = model.score_samples(&data);
        let normal_max: f64 = scores[..20].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        for i in 20..23 {
            assert!(
                scores[i] > normal_max,
                "Outlier {} score ({}) should be higher than max normal score ({})",
                i, scores[i], normal_max
            );
        }
        // At least some outliers should be flagged
        let n_flagged = predictions[20..].iter().filter(|&&p| p == -1).count();
        assert!(n_flagged >= 1, "At least 1 outlier should be flagged as anomaly");
    }

    #[test]
    fn test_score_samples_range() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  0.2, 0.2,
            5.0, 5.0,  -5.0, -5.0,
        ];
        let model = isolation_forest_impl(&data, 2, 100, 10, 0.1).unwrap();
        let scores = model.score_samples(&data);

        for &s in &scores {
            assert!(s >= 0.0 && s <= 1.0, "Score should be in [0, 1], got {}", s);
        }

        // Outlier scores should be higher than normal scores
        let normal_avg: f64 = scores[..3].iter().sum::<f64>() / 3.0;
        let outlier_avg: f64 = scores[3..].iter().sum::<f64>() / 2.0;
        assert!(
            outlier_avg > normal_avg,
            "Outlier scores ({:.4}) should be higher than normal scores ({:.4})",
            outlier_avg, normal_avg
        );
    }

    #[test]
    fn test_decision_function_shift() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,
            10.0, 10.0,
        ];
        let model = isolation_forest_impl(&data, 2, 100, 3, 0.1).unwrap();
        let scores = model.score_samples(&data);
        let decisions = model.decision_function(&data);

        assert_eq!(scores.len(), decisions.len());
        for i in 0..scores.len() {
            assert!((decisions[i] - (scores[i] - 0.5)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_predict_labels() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  0.2, 0.2,  0.3, 0.3,
            10.0, 10.0,
        ];
        let model = isolation_forest_impl(&data, 2, 100, 5, 0.1).unwrap();
        let labels = model.predict(&data);

        for &l in &labels {
            assert!(l == 1 || l == -1, "Labels should be 1 (normal) or -1 (anomaly)");
        }
    }

    #[test]
    fn test_deterministic() {
        let data = vec![
            0.0, 0.0,  1.0, 1.0,  2.0, 2.0,
            10.0, 10.0,  -10.0, -10.0,
        ];

        let m1 = isolation_forest_impl(&data, 2, 50, 5, 0.1).unwrap();
        let m2 = isolation_forest_impl(&data, 2, 50, 5, 0.1).unwrap();

        let test = vec![0.0, 0.0, 10.0, 10.0];
        let s1 = m1.score_samples(&test);
        let s2 = m2.score_samples(&test);
        assert_eq!(s1, s2, "Same data should produce same scores (deterministic RNG)");
    }

    #[test]
    fn test_error_zero_trees() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let result = isolation_forest_impl(&data, 2, 0, 2, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_contamination() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        // Too high
        let result = isolation_forest_impl(&data, 2, 10, 2, 0.6);
        assert!(result.is_err());
        // Negative
        let result = isolation_forest_impl(&data, 2, 10, 2, -0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_too_few_samples() {
        let data = vec![0.0, 0.0];
        let result = isolation_forest_impl(&data, 2, 10, 2, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_feature() {
        // 1D data: normal values near 0, outliers at extremes
        let data: Vec<f64> = (0..20).map(|i| (i as f64) * 0.1).collect();
        let mut with_outliers = data.clone();
        with_outliers.push(1000.0);
        with_outliers.push(-1000.0);

        let model = isolation_forest_impl(&with_outliers, 1, 100, 22, 0.1).unwrap();
        assert_eq!(model.n_features(), 1);

        let scores = model.score_samples(&with_outliers);
        // The last two points should have higher anomaly scores
        let normal_max = scores[..20].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(scores[20] > normal_max, "Outlier at index 20 should score higher than all normal points");
        assert!(scores[21] > normal_max, "Outlier at index 21 should score higher than all normal points");
    }

    #[test]
    fn test_max_samples_capped_to_n() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        // Request max_samples=100 but only 3 samples available
        let model = isolation_forest_impl(&data, 2, 10, 100, 0.1).unwrap();
        assert_eq!(model.max_samples(), 3); // Capped to n_samples
    }

    #[test]
    fn test_to_string() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0];
        let model = isolation_forest_impl(&data, 2, 10, 3, 0.1).unwrap();
        let s = model.to_string_js();
        assert!(s.contains("IsolationForest"));
        assert!(s.contains("trees=10"));
        assert!(s.contains("features=2"));
    }

    #[test]
    fn test_threshold_determined_from_contamination() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  0.2, 0.2,
            0.3, 0.3,  0.4, 0.4,  0.5, 0.5,
            0.6, 0.6,  0.7, 0.7,  0.8, 0.8,
            100.0, 100.0,
        ];
        let model = isolation_forest_impl(&data, 2, 100, 10, 0.1).unwrap();
        // With 10% contamination on 10 samples, ~1 sample should be anomalous
        let predictions = model.predict(&data);
        let n_anomalies = predictions.iter().filter(|&&p| p == -1).count();
        assert!(n_anomalies >= 1, "At least 1 sample should be flagged as anomaly with 10% contamination");
    }
}
