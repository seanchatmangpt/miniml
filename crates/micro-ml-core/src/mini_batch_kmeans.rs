use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get, Rng};

#[wasm_bindgen]
pub struct MiniBatchKMeansModel {
    centroids: Vec<f64>,        // [n_clusters * n_features]
    cluster_counts: Vec<f64>,   // running count per cluster (for weighted updates)
    n_clusters: usize,
    n_features: usize,
    n_iter: usize,
}

#[wasm_bindgen]
impl MiniBatchKMeansModel {
    #[wasm_bindgen(getter)]
    pub fn n_clusters(&self) -> usize { self.n_clusters }

    #[wasm_bindgen(getter = nFeatures)]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(getter = nIter)]
    pub fn n_iter(&self) -> usize { self.n_iter }

    #[wasm_bindgen(js_name = "getCentroids")]
    pub fn get_centroids(&self) -> Vec<f64> { self.centroids.clone() }

    /// Assign new data points to nearest centroid
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![];
        }
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let point = &data[i * self.n_features..(i + 1) * self.n_features];
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..self.n_clusters {
                let centroid = &self.centroids[c * self.n_features..(c + 1) * self.n_features];
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
            "MiniBatchKMeans(n_clusters={}, n_features={}, n_iter={})",
            self.n_clusters, self.n_features, self.n_iter
        )
    }
}

/// Mini-Batch K-Means implementation
///
/// Online variant of K-Means that processes data in small batches.
/// Uses per-center learning rate: centroid[c] += (1/count[c]) * (x - centroid[c])
pub fn mini_batch_kmeans_impl(
    data: &[f64],
    n_features: usize,
    n_clusters: usize,
    batch_size: usize,
    max_iter: usize,
) -> Result<MiniBatchKMeansModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if n_clusters == 0 || n_clusters > n {
        return Err(MlError::new("n_clusters must be between 1 and number of samples"));
    }
    if batch_size == 0 {
        return Err(MlError::new("batch_size must be > 0"));
    }

    let mut rng = Rng::from_data(data);
    let mut centroids = vec![0.0; n_clusters * n_features];

    // K-Means++ style initialization
    let first = rng.next_usize(n);
    centroids[..n_features].copy_from_slice(&data[first * n_features..(first + 1) * n_features]);

    let mut dists = vec![f64::INFINITY; n];
    for c in 1..n_clusters {
        // Update distances to nearest centroid so far
        for i in 0..n {
            let mut d = 0.0;
            for j in 0..n_features {
                let diff = mat_get(data, n_features, i, j) - centroids[(c - 1) * n_features + j];
                d += diff * diff;
            }
            if d < dists[i] {
                dists[i] = d;
            }
        }
        // Weighted random selection proportional to squared distance
        let total: f64 = dists.iter().copied().sum();
        let mut target = rng.next_f64() * total;
        let mut chosen = 0;
        for i in 0..n {
            target -= dists[i];
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids[c * n_features..(c + 1) * n_features]
            .copy_from_slice(&data[chosen * n_features..(chosen + 1) * n_features]);
    }

    let mut cluster_counts = vec![0.0; n_clusters];
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Sample batch_size random points (without replacement, wrapping around if needed)
        let effective_batch = batch_size.min(n);
        let mut batch_indices = Vec::with_capacity(effective_batch);
        for _ in 0..effective_batch {
            batch_indices.push(rng.next_usize(n));
        }

        // Assign each batch point to nearest centroid
        for &idx in &batch_indices {
            let point = &data[idx * n_features..(idx + 1) * n_features];
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..n_clusters {
                let centroid = &centroids[c * n_features..(c + 1) * n_features];
                let mut d = 0.0;
                for j in 0..n_features {
                    let diff = point[j] - centroid[j];
                    d += diff * diff;
                }
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }

            // Incremental update: per-center learning rate
            cluster_counts[best] += 1.0;
            let step = 1.0 / cluster_counts[best];
            for j in 0..n_features {
                centroids[best * n_features + j] += step * (point[j] - centroids[best * n_features + j]);
            }
        }
    }

    Ok(MiniBatchKMeansModel {
        centroids,
        cluster_counts,
        n_clusters,
        n_features,
        n_iter,
    })
}

/// Convenience: fit and return predictions in one call
pub fn mini_batch_kmeans_fit_predict(
    data: &[f64],
    n_features: usize,
    n_clusters: usize,
    batch_size: usize,
    max_iter: usize,
) -> Result<Vec<f64>, MlError> {
    let model = mini_batch_kmeans_impl(data, n_features, n_clusters, batch_size, max_iter)?;
    Ok(model.predict(data))
}

#[wasm_bindgen(js_name = "miniBatchKMeans")]
pub fn mini_batch_kmeans(
    data: &[f64],
    n_features: usize,
    n_clusters: usize,
    batch_size: usize,
    max_iter: usize,
) -> Result<MiniBatchKMeansModel, JsError> {
    mini_batch_kmeans_impl(data, n_features, n_clusters, batch_size, max_iter)
        .map_err(|e| JsError::new(&e.message))
}

#[wasm_bindgen(js_name = "miniBatchKMeansFitPredict")]
pub fn mini_batch_kmeans_fit_predict_js(
    data: &[f64],
    n_features: usize,
    n_clusters: usize,
    batch_size: usize,
    max_iter: usize,
) -> Result<Vec<f64>, JsError> {
    mini_batch_kmeans_fit_predict(data, n_features, n_clusters, batch_size, max_iter)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converges() {
        // Two well-separated clusters: cluster A around origin, cluster B around (10, 10)
        let mut data = Vec::new();
        for _ in 0..20 {
            data.extend_from_slice(&[0.0, 0.0]);
        }
        for _ in 0..20 {
            data.extend_from_slice(&[10.0, 10.0]);
        }

        let model = mini_batch_kmeans_impl(&data, 2, 2, 10, 100).unwrap();
        assert_eq!(model.n_clusters, 2);
        assert_eq!(model.n_features, 2);
        assert_eq!(model.centroids.len(), 4);

        // Centroids should be near (0, 0) and (10, 10)
        let c0 = &model.centroids[0..2];
        let c1 = &model.centroids[2..4];

        let near_origin = c0[0].abs() < 2.0 && c0[1].abs() < 2.0;
        let near_ten = c1[0].abs() - 10.0 < 2.0 && c1[1].abs() - 10.0 < 2.0;

        // Either assignment works
        assert!(
            (near_origin && near_ten)
                || (c1[0].abs() < 2.0 && c1[1].abs() < 2.0
                    && c0[0].abs() - 10.0 < 2.0
                    && c0[1].abs() - 10.0 < 2.0),
            "Centroids should be near (0,0) and (10,10): c0={:?}, c1={:?}",
            c0,
            c1
        );
    }

    #[test]
    fn test_batch_assignment() {
        // Three clear clusters
        let mut data = Vec::new();
        for _ in 0..5 {
            data.extend_from_slice(&[0.0, 0.0]);
        }
        for _ in 0..5 {
            data.extend_from_slice(&[5.0, 5.0]);
        }
        for _ in 0..5 {
            data.extend_from_slice(&[10.0, 0.0]);
        }

        let model = mini_batch_kmeans_impl(&data, 2, 3, 5, 200).unwrap();
        let predictions = model.predict(&data);

        assert_eq!(predictions.len(), 15);

        // First 5 points should share a cluster, middle 5 share another, last 5 share third
        assert_eq!(predictions[0], predictions[1]);
        assert_eq!(predictions[1], predictions[2]);
        assert_eq!(predictions[2], predictions[3]);
        assert_eq!(predictions[3], predictions[4]);

        assert_eq!(predictions[5], predictions[6]);
        assert_eq!(predictions[6], predictions[7]);
        assert_eq!(predictions[7], predictions[8]);
        assert_eq!(predictions[8], predictions[9]);

        assert_eq!(predictions[10], predictions[11]);
        assert_eq!(predictions[11], predictions[12]);
        assert_eq!(predictions[12], predictions[13]);
        assert_eq!(predictions[13], predictions[14]);

        // All three groups should be in different clusters
        assert_ne!(predictions[0], predictions[5]);
        assert_ne!(predictions[5], predictions[10]);
        assert_ne!(predictions[0], predictions[10]);
    }

    #[test]
    fn test_fit_predict_convenience() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  0.2, 0.0,
            5.0, 5.0,  5.1, 5.1,  4.9, 5.0,
        ];
        let preds = mini_batch_kmeans_fit_predict(&data, 2, 2, 3, 100).unwrap();
        assert_eq!(preds.len(), 6);

        // First 3 and last 3 should be in different clusters
        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[0], preds[2]);
        assert_eq!(preds[3], preds[4]);
        assert_eq!(preds[3], preds[5]);
        assert_ne!(preds[0], preds[3]);
    }

    #[test]
    fn test_invalid_inputs() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(mini_batch_kmeans_impl(&data, 2, 0, 1, 10).is_err());
        assert!(mini_batch_kmeans_impl(&data, 2, 3, 1, 10).is_err());
        assert!(mini_batch_kmeans_impl(&data, 2, 1, 0, 10).is_err());
    }

    #[test]
    fn test_single_cluster() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5];
        let model = mini_batch_kmeans_impl(&data, 2, 1, 2, 50).unwrap();
        let preds = model.predict(&data);
        assert!(preds.iter().all(|&p| p == 0.0));
    }
}
