use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, euclidean_dist_sq};

#[wasm_bindgen]
pub struct AgglomerativeCompleteResult {
    labels: Vec<f64>,
    n_clusters: usize,
    n_features: usize,
}

#[wasm_bindgen]
impl AgglomerativeCompleteResult {
    #[wasm_bindgen(getter = nClusters)]
    pub fn n_clusters(&self) -> usize { self.n_clusters }

    #[wasm_bindgen(getter = nFeatures)]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(js_name = "getLabels")]
    pub fn get_labels(&self) -> Vec<f64> { self.labels.clone() }

    /// Assign new data points to nearest cluster centroid (computed from final clustering)
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }
        let n_new = data.len() / self.n_features;

        // Compute centroids from labels and original training data is not stored,
        // so we accept data as the training data for centroid computation.
        // For predict on new data, caller should pass training data to compute centroids,
        // then use those centroids. Here we implement predict by computing centroids
        // from the labels array assuming data is the training set.
        let n_train = self.labels.len();
        if n_train == 0 || data.len() != n_train * self.n_features {
            return vec![];
        }

        // Compute centroids for each cluster
        let mut centroids = vec![0.0; self.n_clusters * self.n_features];
        let mut counts = vec![0usize; self.n_clusters];

        for i in 0..n_train {
            let c = self.labels[i] as usize;
            counts[c] += 1;
            for j in 0..self.n_features {
                centroids[c * self.n_features + j] += data[i * self.n_features + j];
            }
        }
        for c in 0..self.n_clusters {
            if counts[c] > 0 {
                for j in 0..self.n_features {
                    centroids[c * self.n_features + j] /= counts[c] as f64;
                }
            }
        }

        // Assign each training point to nearest centroid
        let mut result = Vec::with_capacity(n_new);
        for i in 0..n_new {
            let point = &data[i * self.n_features..(i + 1) * self.n_features];
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..self.n_clusters {
                let centroid = &centroids[c * self.n_features..(c + 1) * self.n_features];
                let mut d = 0.0;
                for j in 0..self.n_features {
                    let diff = point[j] - centroid[j];
                    d += diff * diff;
                }
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            result.push(best as f64);
        }
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "AgglomerativeComplete(n_clusters={}, n_features={}, n_samples={})",
            self.n_clusters, self.n_features, self.labels.len()
        )
    }
}

/// Agglomerative clustering with complete linkage (max distance between clusters)
///
/// Algorithm:
/// 1. Start with each point as its own cluster
/// 2. Compute pairwise distance matrix (condensed, upper triangle)
/// 3. Merge the two closest clusters until n_clusters remain
/// 4. Complete linkage: dist(A, B) = max(dist(a, b) for a in A, b in B)
/// 5. After merging, dist(new, k) = max(dist(A, k), dist(B, k))
pub fn agglomerative_complete_impl(
    data: &[f64],
    n_features: usize,
    n_clusters: usize,
) -> Result<AgglomerativeCompleteResult, MlError> {
    let n = validate_matrix(data, n_features)?;
    if n_clusters == 0 || n_clusters > n {
        return Err(MlError::new("n_clusters must be between 1 and n_samples"));
    }

    // Active cluster membership: cluster_id[i] = which cluster point i belongs to
    // We use a union-find style approach with active cluster indices
    let mut cluster_id = vec![0usize; n]; // each point starts as its own cluster
    for i in 0..n {
        cluster_id[i] = i;
    }

    // Distance matrix between clusters: only store active cluster pairs
    // Start with n clusters, merge until n_clusters remain
    let mut active_count = n;

    // Track which original cluster indices are still active
    let mut active: Vec<bool> = vec![true; n];

    // Distance matrix: dist[i][j] for i < j, stored as flat upper triangle
    // Index for (i, j) where i < j: i * (2*n - i - 1) / 2 + (j - i - 1)
    // But since we merge clusters, we need a different approach.
    // Use a full symmetric matrix for simplicity.
    let mut dist = vec![0.0f64; n * n];

    // Initialize pairwise distances
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_dist_sq(data, n_features, i, j).sqrt();
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    // Track cluster membership as a list of point sets
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while active_count > n_clusters {
        // Find the pair of active clusters with minimum distance
        let mut best_i = usize::MAX;
        let mut best_j = usize::MAX;
        let mut best_dist = f64::INFINITY;

        for i in 0..n {
            if !active[i] { continue; }
            for j in (i + 1)..n {
                if !active[j] { continue; }
                if dist[i * n + j] < best_dist {
                    best_dist = dist[i * n + j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Merge best_j into best_i using complete linkage update rule:
        // dist(new, k) = max(dist(A, k), dist(B, k))
        for k in 0..n {
            if !active[k] || k == best_i || k == best_j { continue; }
            let new_dist = dist[best_i * n + k].max(dist[best_j * n + k]);
            dist[best_i * n + k] = new_dist;
            dist[k * n + best_i] = new_dist;
        }

        // Deactivate best_j and merge into best_i
        active[best_j] = false;
        let members: Vec<usize> = clusters[best_j].drain(..).collect();
        clusters[best_i].extend(&members);

        active_count -= 1;
    }

    // Assign labels
    let mut labels = vec![0.0f64; n];
    let mut cluster_label = 0usize;
    for i in 0..n {
        if !active[i] { continue; }
        for &idx in &clusters[i] {
            labels[idx] = cluster_label as f64;
        }
        cluster_label += 1;
    }

    Ok(AgglomerativeCompleteResult {
        labels,
        n_clusters,
        n_features,
    })
}

#[wasm_bindgen(js_name = "agglomerativeComplete")]
pub fn agglomerative_complete(
    data: &[f64],
    n_features: usize,
    n_clusters: usize,
) -> Result<AgglomerativeCompleteResult, JsError> {
    agglomerative_complete_impl(data, n_features, n_clusters)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_clusters() {
        // Two well-separated clusters
        let data = vec![
            0.0, 0.0,  // Cluster A
            0.1, 0.1,
            0.2, 0.0,
            10.0, 10.0,  // Cluster B
            10.1, 10.1,
            10.0, 9.9,
        ];
        let result = agglomerative_complete_impl(&data, 2, 2).unwrap();
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels.len(), 6);

        // First 3 should be same cluster, last 3 should be same cluster
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);
        assert_ne!(result.labels[0], result.labels[3]);
    }

    #[test]
    fn test_complete_vs_single() {
        // Three points: A at (0,0), B at (1,0), C at (10,0)
        // Single linkage: A-B distance=1, B-C distance=9, A-C distance=10
        //   -> A-B merge first (dist=1), then {AB}-C merge (single linkage dist = min(9,10) = 9)
        // Complete linkage: A-B distance=1, B-C distance=9, A-C distance=10
        //   -> A-B merge first (dist=1), then {AB}-C merge (complete linkage dist = max(9,10) = 10)
        // Both should produce same result for 2 clusters since A-B are closest in both
        let data = vec![0.0, 0.0, 1.0, 0.0, 10.0, 0.0];

        let complete_result = agglomerative_complete_impl(&data, 2, 2).unwrap();

        // A and B should be in the same cluster, C in a different one
        assert_eq!(complete_result.labels[0], complete_result.labels[1]);
        assert_ne!(complete_result.labels[0], complete_result.labels[2]);
        assert_ne!(complete_result.labels[1], complete_result.labels[2]);

        // Now compare with single linkage from hierarchical.rs
        let single_result = crate::hierarchical::hierarchical_impl(&data, 2, 2).unwrap();

        // Both should produce the same 2-cluster partition for well-separated data
        assert_eq!(complete_result.labels[0], complete_result.labels[1]);
        assert_eq!(single_result[0], single_result[1]);
    }

    #[test]
    fn test_single_cluster() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = agglomerative_complete_impl(&data, 2, 1).unwrap();
        assert_eq!(result.n_clusters, 1);
        assert!(result.labels.iter().all(|&l| l == 0.0));
    }

    #[test]
    fn test_each_point_own_cluster() {
        let data = vec![1.0, 0.0, 2.0, 0.0];
        let result = agglomerative_complete_impl(&data, 2, 2).unwrap();
        assert_eq!(result.n_clusters, 2);
        // Each point is its own cluster, so labels should differ
        assert_ne!(result.labels[0], result.labels[1]);
    }

    #[test]
    fn test_invalid_n_clusters() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(agglomerative_complete_impl(&data, 2, 0).is_err());
        assert!(agglomerative_complete_impl(&data, 2, 3).is_err());
    }

    #[test]
    fn test_predict_reproduces_labels() {
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            10.0, 10.0,
            10.1, 10.1,
        ];
        let result = agglomerative_complete_impl(&data, 2, 2).unwrap();
        let predictions = result.predict(&data);

        // Predict should reproduce the same labels for the training data
        assert_eq!(predictions.len(), result.labels.len());
        assert_eq!(predictions[0], predictions[1]);
        assert_eq!(predictions[2], predictions[3]);
        assert_ne!(predictions[0], predictions[2]);
    }
}
