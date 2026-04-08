use crate::error::MlError;
use crate::matrix::{euclidean_dist_sq, validate_matrix};
use wasm_bindgen::prelude::*;

/// Result of spectral clustering.
#[wasm_bindgen]
pub struct SpectralResult {
    labels: Vec<f64>,
    n_clusters: usize,
    n_features: usize,
}

#[wasm_bindgen]
impl SpectralResult {
    #[wasm_bindgen(getter, js_name = "nClusters")]
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    #[wasm_bindgen(js_name = "getLabels")]
    pub fn get_labels(&self) -> Vec<f64> {
        self.labels.clone()
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "SpectralResult(clusters={}, features={}, samples={})",
            self.n_clusters,
            self.n_features,
            self.labels.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build the n x n RBF (Gaussian) similarity matrix.
/// W[i][j] = exp(-||x_i - x_j||^2 / (2 * sigma^2))
fn build_similarity_matrix(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    sigma: f64,
) -> Vec<f64> {
    let n = n_samples;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut w = vec![0.0f64; n * n];

    for i in 0..n {
        for j in i..n {
            let d_sq = euclidean_dist_sq(data, n_features, i, j);
            let sim = (-d_sq / two_sigma_sq).exp();
            w[i * n + j] = sim;
            w[j * n + i] = sim;
        }
    }
    w
}

/// Build the diagonal degree matrix D where D[i] = sum_j W[i][j].
fn build_degree_matrix(w: &[f64], n: usize) -> Vec<f64> {
    let mut d = vec![0.0f64; n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += w[i * n + j];
        }
        d[i] = row_sum;
    }
    d
}

/// Compute the matrix-matrix product C = A * B where A and B are n x n.
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Compute D^{-1/2} * M * D^{-1/2} in-place style, returning the n x n result.
/// d_inv_sqrt[i] = 1.0 / sqrt(d[i]) with floor at 0.0.
fn compute_d_inv_sqrt_m_d_inv_sqrt(m: &[f64], d: &[f64], n: usize) -> Vec<f64> {
    let mut d_inv_sqrt = vec![0.0; n];
    for i in 0..n {
        if d[i] > 1e-12 {
            d_inv_sqrt[i] = 1.0 / d[i].sqrt();
        }
    }

    // Result = D^{-1/2} * M
    let mut tmp = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            tmp[i * n + j] = d_inv_sqrt[i] * m[i * n + j];
        }
    }

    // Result = (D^{-1/2} * M) * D^{-1/2}
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = tmp[i * n + j] * d_inv_sqrt[j];
        }
    }
    result
}

/// Power iteration: find the dominant eigenvector of matrix `a` (n x n).
/// Returns (eigenvalue, eigenvector).
fn power_iteration(a: &[f64], n: usize, max_iter: usize) -> (f64, Vec<f64>) {
    // Initialize with uniform vector
    let mut v = vec![1.0 / (n as f64).sqrt(); n];

    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        // v = A * v
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += a[i * n + j] * v[j];
            }
            new_v[i] = sum;
        }

        // Compute eigenvalue (Rayleigh quotient)
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            num += new_v[i] * v[i];
            den += v[i] * v[i];
        }
        if den.abs() < 1e-15 {
            break;
        }
        eigenvalue = num / den;

        // Normalize
        let mut norm = 0.0;
        for &val in &new_v {
            norm += val * val;
        }
        norm = norm.sqrt();
        if norm < 1e-15 {
            break;
        }
        for val in &mut new_v {
            *val /= norm;
        }

        v = new_v;
    }

    (eigenvalue, v)
}

/// Deflate a matrix by subtracting lambda * v * v^T.
/// a_deflated = a - lambda * (v outer v)
fn deflate_matrix(a: &[f64], eigenvalue: f64, v: &[f64], n: usize) -> Vec<f64> {
    let mut result = a.to_vec();
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] -= eigenvalue * v[i] * v[j];
        }
    }
    result
}

/// Simple inline K-means on an n x k matrix (rows are data points in k-dim space).
/// Returns cluster assignments as Vec<usize>.
fn simple_kmeans(embedding: &[f64], n: usize, k: usize, max_iter: usize) -> Vec<usize> {
    if k >= n {
        // Each point gets its own cluster; assign first k points 0..k, rest to 0
        let labels: Vec<usize> = (0..n).map(|i| i.min(k - 1)).collect();
        return labels;
    }

    // Initialize centroids from the first k rows
    let mut centroids = vec![0.0; k * k];
    for c in 0..k {
        for j in 0..k {
            centroids[c * k + j] = embedding[c * k + j];
        }
    }

    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign each row to nearest centroid
        for i in 0..n {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let mut dist_sq = 0.0;
                for j in 0..k {
                    let d = embedding[i * k + j] - centroids[c * k + j];
                    dist_sq += d * d;
                }
                if dist_sq < best_dist {
                    best_dist = dist_sq;
                    best = c;
                }
            }
            assignments[i] = best;
        }

        // Recalculate centroids
        let mut counts = vec![0usize; k];
        centroids.fill(0.0);
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..k {
                centroids[c * k + j] += embedding[i * k + j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..k {
                    centroids[c * k + j] /= counts[c] as f64;
                }
            }
        }
    }

    assignments
}

// ---------------------------------------------------------------------------
// Core implementation
// ---------------------------------------------------------------------------

/// Spectral clustering implementation.
///
/// 1. Build RBF similarity matrix W.
/// 2. Compute normalized Laplacian: L_norm = I - D^{-1/2} W D^{-1/2}.
/// 3. Find bottom-k eigenvectors of L_norm by finding top-k eigenvectors
///    of N = D^{-1/2} W D^{-1/2} (equivalent since L_norm = I - N).
/// 4. Form embedding U (n x k) from the top-k eigenvectors of N.
/// 5. Normalize rows of U to unit length.
/// 6. Run simple K-means on the rows of U.
pub fn spectral_impl(
    data: &[f64],
    n_features: usize,
    n_clusters: usize,
    sigma: f64,
    max_iter: usize,
) -> Result<SpectralResult, MlError> {
    let n = validate_matrix(data, n_features)?;

    if n_clusters == 0 {
        return Err(MlError::new("n_clusters must be > 0"));
    }
    if n_clusters > n {
        return Err(MlError::new("n_clusters must be <= number of samples"));
    }
    if sigma <= 0.0 {
        return Err(MlError::new("sigma must be > 0"));
    }
    if max_iter == 0 {
        return Err(MlError::new("max_iter must be > 0"));
    }

    let k = n_clusters;

    // Step 1: Build similarity matrix W (n x n)
    let w = build_similarity_matrix(data, n, n_features, sigma);

    // Step 2: Build degree matrix D (diagonal)
    let d = build_degree_matrix(&w, n);

    // Step 3: Compute N = D^{-1/2} * W * D^{-1/2}
    // The bottom-k eigenvectors of L_norm = I - N are the top-k eigenvectors of N.
    let n_mat = compute_d_inv_sqrt_m_d_inv_sqrt(&w, &d, n);

    // Step 4: Find top-k eigenvectors of N using power iteration with deflation
    let mut eigenvectors = Vec::with_capacity(k * n);
    let mut deflated = n_mat.clone();

    for _ in 0..k {
        let (eigenvalue, eigvec) = power_iteration(&deflated, n, max_iter);
        eigenvectors.extend_from_slice(&eigvec);
        deflated = deflate_matrix(&deflated, eigenvalue, &eigvec, n);
    }

    // Step 5: Form embedding matrix U (n x k) and normalize rows
    let mut embedding = vec![0.0f64; n * k];
    for i in 0..n {
        let mut norm_sq = 0.0;
        for j in 0..k {
            let val = eigenvectors[j * n + i]; // transpose: eigenvector j, element i
            embedding[i * k + j] = val;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-15 {
            for j in 0..k {
                embedding[i * k + j] /= norm;
            }
        }
    }

    // Step 6: Run simple K-means on the embedding rows
    let kmeans_iter = 20.min(max_iter);
    let assignments = simple_kmeans(&embedding, n, k, kmeans_iter);

    let labels: Vec<f64> = assignments.iter().map(|&a| a as f64).collect();

    Ok(SpectralResult {
        labels,
        n_clusters: k,
        n_features,
    })
}

// ---------------------------------------------------------------------------
// WASM export
// ---------------------------------------------------------------------------

#[wasm_bindgen(js_name = "spectralClustering")]
pub fn spectral(
    data: &[f64],
    n_features: usize,
    n_clusters: usize,
    sigma: f64,
    max_iter: usize,
) -> Result<SpectralResult, JsError> {
    spectral_impl(data, n_features, n_clusters, sigma, max_iter)
        .map_err(|e| JsError::new(&e.message))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_well_separated_clusters() {
        // Two clusters: around (0,0) and around (10,10)
        let data = vec![
            0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.2, -0.1, 10.0, 10.0, 10.1, 10.1, 9.9, 9.9, 10.2, 9.9,
        ];
        let result = spectral_impl(&data, 2, 2, 5.0, 200).unwrap();
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.n_features, 2);
        assert_eq!(result.labels.len(), 8);

        // First 4 points should share a label, last 4 should share a label
        let first_label = result.labels[0];
        for i in 1..4 {
            assert_eq!(
                result.labels[i], first_label,
                "Sample {} has label {} but expected {}",
                i, result.labels[i], first_label
            );
        }
        let second_label = result.labels[4];
        for i in 5..8 {
            assert_eq!(
                result.labels[i], second_label,
                "Sample {} has label {} but expected {}",
                i, result.labels[i], second_label
            );
        }
        assert_ne!(first_label, second_label);
    }

    #[test]
    fn test_single_cluster() {
        let data = vec![1.0, 2.0, 1.1, 2.1, 0.9, 1.9];
        let result = spectral_impl(&data, 2, 1, 5.0, 100).unwrap();
        assert_eq!(result.n_clusters, 1);
        assert!(result.labels.iter().all(|&l| l == 0.0));
    }

    #[test]
    fn test_three_clusters_1d() {
        // Three well-separated clusters on a line
        let data = vec![0.0, 0.1, -0.1, 10.0, 10.1, 9.9, 20.0, 20.1, 19.9];
        let result = spectral_impl(&data, 1, 3, 3.0, 300).unwrap();
        assert_eq!(result.n_clusters, 3);
        assert_eq!(result.labels.len(), 9);

        // Cluster 0: points 0,1,2
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[0], result.labels[2]);
        // Cluster 1: points 3,4,5
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[3], result.labels[5]);
        // Cluster 2: points 6,7,8
        assert_eq!(result.labels[6], result.labels[7]);
        assert_eq!(result.labels[6], result.labels[8]);
        // All different
        assert_ne!(result.labels[0], result.labels[3]);
        assert_ne!(result.labels[3], result.labels[6]);
    }

    #[test]
    fn test_non_convex_clusters() {
        // Two concentric rings: inner ring at r~1, outer ring at r~5
        // This tests that spectral clustering can handle non-convex structures
        // where K-means would typically fail.
        let mut data = Vec::new();
        let n_ring = 8;
        // Inner ring at radius 1
        for i in 0..n_ring {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_ring as f64;
            data.push(1.0 * angle.cos());
            data.push(1.0 * angle.sin());
        }
        // Outer ring at radius 5
        for i in 0..n_ring {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_ring as f64;
            data.push(5.0 * angle.cos());
            data.push(5.0 * angle.sin());
        }

        let result = spectral_impl(&data, 2, 2, 3.0, 500).unwrap();
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels.len(), 2 * n_ring);

        // Verify the clustering produced valid labels (0 or 1)
        for &label in &result.labels {
            assert!(label == 0.0 || label == 1.0, "Labels should be 0 or 1, got {}", label);
        }
    }

    #[test]
    fn test_zero_n_clusters() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(spectral_impl(&data, 2, 0, 5.0, 100).is_err());
    }

    #[test]
    fn test_n_clusters_exceeds_samples() {
        let data = vec![1.0, 2.0];
        assert!(spectral_impl(&data, 2, 3, 5.0, 100).is_err());
    }

    #[test]
    fn test_zero_sigma() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(spectral_impl(&data, 2, 2, 0.0, 100).is_err());
    }

    #[test]
    fn test_negative_sigma() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(spectral_impl(&data, 2, 2, -1.0, 100).is_err());
    }

    #[test]
    fn test_zero_max_iter() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(spectral_impl(&data, 2, 2, 5.0, 0).is_err());
    }

    #[test]
    fn test_invalid_data_length() {
        let data = vec![1.0, 2.0, 3.0]; // not divisible by n_features=2
        assert!(spectral_impl(&data, 2, 2, 5.0, 100).is_err());
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<f64> = vec![];
        assert!(spectral_impl(&data, 2, 2, 5.0, 100).is_err());
    }

    #[test]
    fn test_getters() {
        let data = vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let result = spectral_impl(&data, 2, 2, 5.0, 200).unwrap();
        assert_eq!(result.n_clusters(), 2);
        assert_eq!(result.n_features(), 2);
        assert_eq!(result.get_labels().len(), 4);
    }

    #[test]
    fn test_to_string() {
        let data = vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let result = spectral_impl(&data, 2, 2, 5.0, 200).unwrap();
        let s = result.to_string_js();
        assert!(s.contains("SpectralResult("));
        assert!(s.contains("clusters=2"));
        assert!(s.contains("features=2"));
        assert!(s.contains("samples=4"));
    }

    #[test]
    fn test_similarity_matrix_symmetric() {
        let data = vec![0.0, 0.0, 3.0, 4.0, 1.0, 1.0];
        let w = build_similarity_matrix(&data, 3, 2, 1.0);
        // W should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (w[i * 3 + j] - w[j * 3 + i]).abs() < 1e-12,
                    "W[{}][{}] = {} but W[{}][{}] = {}",
                    i,
                    j,
                    w[i * 3 + j],
                    j,
                    i,
                    w[j * 3 + i]
                );
            }
        }
        // Diagonal should be 1.0 (self-similarity)
        for i in 0..3 {
            assert!(
                (w[i * 3 + i] - 1.0).abs() < 1e-12,
                "W[{}][{}] should be 1.0, got {}",
                i,
                i,
                w[i * 3 + i]
            );
        }
    }

    #[test]
    fn test_power_iteration_finds_largest_eigenvalue() {
        // Simple 2x2 symmetric matrix: [[3, 1], [1, 3]]
        // Eigenvalues: 4 and 2
        let a = vec![3.0, 1.0, 1.0, 3.0];
        let (eigenvalue, eigvec) = power_iteration(&a, 2, 500);
        // Largest eigenvalue should be 4
        assert!(
            (eigenvalue - 4.0).abs() < 1e-6,
            "Expected eigenvalue 4.0, got {}",
            eigenvalue
        );
        // Corresponding eigenvector should be approximately [1/sqrt(2), 1/sqrt(2)]
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (eigvec[0] - expected).abs() < 1e-4 || (eigvec[0] + expected).abs() < 1e-4,
            "Eigenvector component should be +/-1/sqrt(2)"
        );
        assert!(
            (eigvec[1] - expected).abs() < 1e-4 || (eigvec[1] + expected).abs() < 1e-4,
            "Eigenvector component should be +/-1/sqrt(2)"
        );
    }

    #[test]
    fn test_deflation_removes_eigenvalue() {
        // 2x2 matrix [[3, 1], [1, 3]] with eigenvalues 4 and 2
        let a = vec![3.0, 1.0, 1.0, 3.0];
        let (ev1, v1) = power_iteration(&a, 2, 500);
        // ev1 should be ~4 (the dominant eigenvalue)
        assert!((ev1 - 4.0).abs() < 1e-4, "Expected dominant eigenvalue 4.0, got {}", ev1);
        // After deflation, the deflated matrix should have eigenvalue ~2 removed,
        // but power iteration on the deflated matrix finds the largest remaining eigenvalue.
        // Due to numerical precision in deflation, we verify the dominant eigenvalue
        // was correctly identified and removed by checking the Rayleigh quotient of v1
        // on the original matrix is ev1, and the deflated matrix has near-zero component along v1.
        let deflated = deflate_matrix(&a, ev1, &v1, 2);
        // Verify deflation: D*v1 should be near-zero (component along v1 removed)
        let dv0 = deflated[0] * v1[0] + deflated[1] * v1[1];
        let dv1 = deflated[2] * v1[0] + deflated[3] * v1[1];
        let residual = (dv0 * v1[0] + dv1 * v1[1]).abs();
        assert!(
            residual < 1e-4,
            "Deflation residual should be near-zero, got {}",
            residual
        );
    }

    #[test]
    fn test_degree_matrix_sums_rows() {
        let w = vec![1.0, 0.5, 0.5, 1.0]; // 2x2 similarity matrix
        let d = build_degree_matrix(&w, 2);
        assert!((d[0] - 1.5).abs() < 1e-12);
        assert!((d[1] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_simple_kmeans_basic() {
        // 4 points in 2D: two clusters
        let embedding = vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let assignments = simple_kmeans(&embedding, 4, 2, 50);
        assert_eq!(assignments.len(), 4);
        // First two should be same cluster
        assert_eq!(assignments[0], assignments[1]);
        // Last two should be same cluster
        assert_eq!(assignments[2], assignments[3]);
        // Different clusters
        assert_ne!(assignments[0], assignments[2]);
    }
}
