use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, euclidean_dist_sq};

/// Local Outlier Factor result.
///
/// `scores` are negative LOF values (scikit-learn convention: higher = more normal).
/// A score of ~-1.0 means normal; much less than -1.0 means potential outlier.
#[wasm_bindgen]
pub struct LofResult {
    scores: Vec<f64>,
    n_neighbors: usize,
    n_features: usize,
    contamination: f64,
}

#[wasm_bindgen]
impl LofResult {
    #[wasm_bindgen(getter, js_name = "nNeighbors")]
    pub fn n_neighbors(&self) -> usize { self.n_neighbors }

    #[wasm_bindgen(getter)]
    pub fn contamination(&self) -> f64 { self.contamination }

    #[wasm_bindgen(js_name = "getScores")]
    pub fn get_scores(&self) -> Vec<f64> { self.scores.clone() }

    /// Predict anomaly labels: 1 = normal, -1 = anomaly.
    /// Uses the contamination fraction to determine the LOF threshold.
    #[wasm_bindgen]
    pub fn predict(&self, _data: &[f64]) -> Vec<i32> {
        if self.scores.is_empty() {
            return vec![];
        }

        // Compute threshold: the (1 - contamination) quantile of LOF scores.
        // Sort scores ascending (most negative / most outlier first).
        let mut indexed: Vec<(usize, f64)> = self.scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n = self.scores.len();
        let n_anomalies = (self.contamination * n as f64).round() as usize;
        let n_anomalies = n_anomalies.max(0).min(n.saturating_sub(1));

        let anomaly_set: std::collections::HashSet<usize> =
            indexed[..n_anomalies].iter().map(|&(i, _)| i).collect();

        (0..n).map(|i| if anomaly_set.contains(&i) { -1 } else { 1 }).collect()
    }

    /// Returns the negative LOF scores. Higher = more normal, lower = more anomalous.
    #[wasm_bindgen(js_name = "scoreSamples")]
    pub fn score_samples(&self) -> Vec<f64> { self.scores.clone() }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "LOF(n_neighbors={}, n_features={}, contamination={})",
            self.n_neighbors, self.n_features, self.contamination
        )
    }
}

/// Compute the k-nearest neighbors (indices and Euclidean distances) for every point.
///
/// Returns a vector of length `n`, each entry is a vector of `(neighbor_index, distance)`
/// sorted by ascending distance. The vector contains `k+1` entries (the point itself at
/// distance 0 plus its k nearest neighbors).
fn knn_distances(data: &[f64], n_features: usize, k: usize) -> Vec<Vec<(usize, f64)>> {
    let n = data.len() / n_features;

    (0..n).map(|i| {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .map(|j| {
                let d = euclidean_dist_sq(data, n_features, i, j).sqrt();
                (j, d)
            })
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        // Return k+1 entries: self + k nearest neighbors
        dists.truncate(k + 1);
        dists
    }).collect()
}

/// Reachability distance: max(k-distance of o, distance from p to o).
///
/// This smooths out distance variations within dense regions.
#[inline]
fn reachability_distance(dist_to_o: f64, k_dist_o: f64) -> f64 {
    dist_to_o.max(k_dist_o)
}

/// Local Outlier Factor implementation.
///
/// # Arguments
/// * `data` - Flat row-major matrix (n_samples * n_features)
/// * `n_features` - Number of features per sample
/// * `n_neighbors` - Number of neighbors (k). Must be >= 2.
/// * `contamination` - Expected proportion of outliers in [0, 1]. Used for predict().
///
/// # Algorithm
/// For each point p:
/// 1. Find k-nearest neighbors
/// 2. k-distance(p) = distance to the k-th nearest neighbor
/// 3. reach_dist(p, o) = max(k-distance(o), dist(p, o))
/// 4. lrd(p) = 1 / (mean reachability distance to k-neighbors)
/// 5. LOF(p) = mean(lrd(neighbor) / lrd(p)) for each neighbor
///
/// Returns negative LOF scores: -LOF(p). Higher = more normal (scikit-learn convention).
pub fn lof_impl(
    data: &[f64],
    n_features: usize,
    n_neighbors: usize,
    contamination: f64,
) -> Result<LofResult, MlError> {
    let n = validate_matrix(data, n_features)?;
    if n_neighbors < 2 {
        return Err(MlError::new("n_neighbors must be >= 2"));
    }
    if n_neighbors >= n {
        return Err(MlError::new("n_neighbors must be < number of samples"));
    }
    if !(0.0..=1.0).contains(&contamination) {
        return Err(MlError::new("contamination must be in [0, 1]"));
    }

    let k = n_neighbors;

    // Step 1: Compute k-nearest neighbors for all points
    let neighbors = knn_distances(data, n_features, k);

    // Step 2: Compute k-distance for each point
    // k-distance(p) = distance to the k-th nearest neighbor (index k in the sorted list,
    // since index 0 is the point itself at distance 0)
    let k_distances: Vec<f64> = neighbors.iter().map(|nb| nb[k].1).collect();

    // Step 3: Compute local reachability density for each point
    let lrd = compute_lrd(&neighbors, &k_distances, k);

    // Step 4: Compute LOF for each point
    let lof_scores: Vec<f64> = (0..n).map(|i| {
        let lrd_p = lrd[i];
        if lrd_p == 0.0 {
            return f64::INFINITY;
        }

        let sum: f64 = (1..=k)
            .map(|j| {
                let neighbor_idx = neighbors[i][j].0;
                let lrd_o = lrd[neighbor_idx];
                lrd_o / lrd_p
            })
            .sum();

        sum / k as f64
    }).collect();

    // Step 5: Convert to negative LOF scores (higher = more normal)
    let scores: Vec<f64> = lof_scores.iter().map(|&s| -s).collect();

    Ok(LofResult {
        scores,
        n_neighbors: k,
        n_features,
        contamination,
    })
}

/// Compute Local Reachability Density for all points.
fn compute_lrd(
    neighbors: &[Vec<(usize, f64)>],
    k_distances: &[f64],
    k: usize,
) -> Vec<f64> {
    neighbors.iter().map(|nbs| {
        let avg_reach_dist: f64 = (1..=k)
            .map(|j| {
                let neighbor_idx = nbs[j].0;
                let dist_p_to_o = nbs[j].1;
                let k_dist_o = k_distances[neighbor_idx];
                reachability_distance(dist_p_to_o, k_dist_o)
            })
            .sum::<f64>() / k as f64;

        if avg_reach_dist == 0.0 {
            f64::INFINITY
        } else {
            1.0 / avg_reach_dist
        }
    }).collect()
}

#[wasm_bindgen(js_name = "lof")]
pub fn lof(
    data: &[f64],
    n_features: usize,
    n_neighbors: usize,
    contamination: f64,
) -> Result<LofResult, JsError> {
    lof_impl(data, n_features, n_neighbors, contamination)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_points_clustered() {
        // A tight cluster of points — all should have LOF ~ 1 (score ~ -1)
        let mut rng = crate::matrix::Rng::new(42);
        let data: Vec<f64> = (0..20)
            .flat_map(|_| {
                let x = 5.0 + (rng.next_f64() - 0.5) * 0.4;
                let y = 5.0 + (rng.next_f64() - 0.5) * 0.4;
                vec![x, y]
            })
            .collect();

        let result = lof_impl(&data, 2, 5, 0.05).unwrap();
        assert_eq!(result.scores.len(), 20);

        // All clustered points should have LOF close to 1 (score close to -1)
        for &score in &result.scores {
            assert!(
                score > -2.0,
                "Clustered point should have LOF ~ 1, got score = {}",
                score
            );
        }
    }

    #[test]
    fn test_isolated_outlier() {
        // 15 tight cluster points near origin + 1 isolated outlier far away
        let mut data: Vec<f64> = (0..15)
            .flat_map(|i| {
                let angle = i as f64 * 0.42;
                vec![angle.cos() * 0.3, angle.sin() * 0.3]
            })
            .collect();

        // Add an isolated outlier at (10, 10)
        data.push(10.0);
        data.push(10.0);

        let result = lof_impl(&data, 2, 5, 0.05).unwrap();
        assert_eq!(result.scores.len(), 16);

        // The outlier (last point) should have a much more negative score
        let outlier_score = result.scores[15];
        let normal_scores: Vec<f64> = result.scores[..15].to_vec();

        let avg_normal = normal_scores.iter().sum::<f64>() / normal_scores.len() as f64;

        // Outlier score should be significantly more negative than average normal score
        assert!(
            outlier_score < avg_normal - 1.0,
            "Outlier score ({}) should be much lower than avg normal score ({})",
            outlier_score,
            avg_normal
        );

        // Normal points should have LOF close to 1 (score close to -1)
        for &score in &normal_scores {
            assert!(
                score > -2.0,
                "Normal clustered point should have LOF ~ 1, got score = {}",
                score
            );
        }

        // Outlier should have LOF > 1 (score < -1)
        assert!(
            outlier_score < -1.0,
            "Outlier should have LOF > 1, got score = {}",
            outlier_score
        );
    }

    #[test]
    fn test_predict_labels() {
        // 18 cluster points + 2 outliers
        let mut data: Vec<f64> = (0..18)
            .flat_map(|i| {
                let angle = i as f64 * 0.35;
                vec![angle.cos() * 0.2, angle.sin() * 0.2]
            })
            .collect();

        // Two outliers far away
        data.extend_from_slice(&[50.0, 50.0]);
        data.extend_from_slice(&[-50.0, -50.0]);

        let result = lof_impl(&data, 2, 5, 0.1).unwrap(); // 10% contamination = ~2 outliers
        let labels = result.predict(&data);

        assert_eq!(labels.len(), 20);

        // Outliers should be labeled -1
        assert_eq!(labels[18], -1, "Point at (50,50) should be anomaly");
        assert_eq!(labels[19], -1, "Point at (-50,-50) should be anomaly");
    }

    #[test]
    fn test_validation_errors() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];

        // n_neighbors < 2
        assert!(lof_impl(&data, 2, 1, 0.1).is_err());

        // n_neighbors >= n_samples
        assert!(lof_impl(&data, 2, 4, 0.1).is_err());

        // contamination out of range
        assert!(lof_impl(&data, 2, 2, -0.1).is_err());
        assert!(lof_impl(&data, 2, 2, 1.5).is_err());

        // invalid matrix
        assert!(lof_impl(&data, 3, 2, 0.1).is_err());
    }

    #[test]
    fn test_score_samples_matches() {
        let data = vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0];
        let result = lof_impl(&data, 2, 2, 0.1).unwrap();
        let scores = result.score_samples();
        assert_eq!(scores, result.scores);
    }

}
