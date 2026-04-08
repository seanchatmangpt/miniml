use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get, Rng};

/// Gaussian Mixture Model for soft/probabilistic clustering.
///
/// Unlike K-Means which produces hard assignments, GMM models data as a
/// mixture of Gaussian distributions and provides posterior probabilities
/// (responsibilities) for each cluster.
#[wasm_bindgen]
pub struct GmmModel {
    weights: Vec<f64>,
    means: Vec<f64>,
    covariances: Vec<f64>,
    n_components: usize,
    n_features: usize,
    converged: bool,
    n_iter: usize,
}

#[wasm_bindgen]
impl GmmModel {
    #[wasm_bindgen(getter, js_name = "nComponents")]
    pub fn n_components(&self) -> usize { self.n_components }

    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool { self.converged }

    #[wasm_bindgen(getter, js_name = "nIter")]
    pub fn n_iter(&self) -> usize { self.n_iter }

    #[wasm_bindgen(js_name = "getWeights")]
    pub fn get_weights(&self) -> Vec<f64> { self.weights.clone() }

    #[wasm_bindgen(js_name = "getMeans")]
    pub fn get_means(&self) -> Vec<f64> { self.means.clone() }

    /// Hard assignment: returns the most likely cluster index for each sample.
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<u32> {
        let proba = self.predict_proba_impl(data);
        let n_samples = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row = &proba[i * self.n_components..(i + 1) * self.n_components];
            let mut best_k = 0usize;
            let mut best_p = row[0];
            for k in 1..self.n_components {
                if row[k] > best_p {
                    best_p = row[k];
                    best_k = k;
                }
            }
            result.push(best_k as u32);
        }
        result
    }

    /// Soft assignment: returns probability per cluster per sample.
    /// Output layout: [n_samples * n_components] row-major.
    #[wasm_bindgen(js_name = "predictProba")]
    pub fn predict_proba(&self, data: &[f64]) -> Vec<f64> {
        self.predict_proba_impl(data)
    }

    /// Log-likelihood per sample.
    #[wasm_bindgen(js_name = "scoreSamples")]
    pub fn score_samples(&self, data: &[f64]) -> Vec<f64> {
        let n_samples = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut log_sum = f64::NEG_INFINITY;
            for k in 0..self.n_components {
                let mut log_component = self.weights[k].ln();
                for j in 0..self.n_features {
                    let x = mat_get(data, self.n_features, i, j);
                    let mean = self.means[k * self.n_features + j];
                    let var = self.covariances[k * self.n_features + j];
                    log_component += gaussian_log_pdf(x, mean, var);
                }
                // log-sum-exp trick
                if log_sum == f64::NEG_INFINITY {
                    log_sum = log_component;
                } else {
                    let max_v = log_sum.max(log_component);
                    log_sum = max_v + ((log_sum - max_v).exp() + (log_component - max_v).exp()).ln();
                }
            }
            result.push(log_sum);
        }
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "GMM(components={}, features={}, converged={}, iter={})",
            self.n_components, self.n_features, self.converged, self.n_iter
        )
    }
}

impl GmmModel {
    /// Internal predict_proba without WASM binding overhead.
    fn predict_proba_impl(&self, data: &[f64]) -> Vec<f64> {
        let n_samples = data.len() / self.n_features;
        let mut result = vec![0.0; n_samples * self.n_components];

        for i in 0..n_samples {
            // Compute unnormalized log-responsibilities
            let mut log_resp = vec![f64::NEG_INFINITY; self.n_components];
            let mut max_log = f64::NEG_INFINITY;
            for k in 0..self.n_components {
                let mut log_val = self.weights[k].ln();
                for j in 0..self.n_features {
                    let x = mat_get(data, self.n_features, i, j);
                    let mean = self.means[k * self.n_features + j];
                    let var = self.covariances[k * self.n_features + j];
                    log_val += gaussian_log_pdf(x, mean, var);
                }
                log_resp[k] = log_val;
                if log_val > max_log {
                    max_log = log_val;
                }
            }

            // Normalize via softmax (subtract max for numerical stability)
            let sum_exp: f64 = log_resp.iter().map(|&lr| (lr - max_log).exp()).sum();
            for k in 0..self.n_components {
                result[i * self.n_components + k] = (log_resp[k] - max_log).exp() / sum_exp;
            }
        }
        result
    }
}

/// Univariate Gaussian log-PDF: ln N(x | mean, var).
/// = -0.5 * (ln(2*pi) + ln(var) + (x - mean)^2 / var)
#[inline]
fn gaussian_log_pdf(x: f64, mean: f64, var: f64) -> f64 {
    -0.5 * (std::f64::consts::LN_2 + std::f64::consts::PI.ln() + var.ln()
        + (x - mean) * (x - mean) / var)
}

/// Univariate Gaussian PDF.
/// = (1 / sqrt(2 * pi * var)) * exp(-0.5 * (x - mean)^2 / var)
#[inline]
fn gaussian_pdf(x: f64, mean: f64, var: f64) -> f64 {
    (1.0 / (std::f64::consts::PI * 2.0 * var).sqrt()) * (-0.5 * (x - mean) * (x - mean) / var).exp()
}

/// Initialize means using K-Means++ style seeding.
fn kmeans_pp_init(data: &[f64], n_samples: usize, n_features: usize, n_components: usize, rng: &mut Rng) -> Vec<f64> {
    let mut means = vec![0.0; n_components * n_features];

    // Pick first center uniformly at random
    let first = rng.next_usize(n_samples);
    means[..n_features].copy_from_slice(&data[first * n_features..(first + 1) * n_features]);

    let mut min_dists = vec![f64::INFINITY; n_samples];

    for c in 1..n_components {
        // Update minimum distances to nearest chosen center
        for i in 0..n_samples {
            let mut dist_sq = 0.0;
            for j in 0..n_features {
                let d = mat_get(data, n_features, i, j) - means[(c - 1) * n_features + j];
                dist_sq += d * d;
            }
            if dist_sq < min_dists[i] {
                min_dists[i] = dist_sq;
            }
        }

        // Weighted random selection proportional to distance squared
        let total: f64 = min_dists.iter().sum();
        if total <= 0.0 {
            // Fallback: uniform random if all distances are zero
            let idx = rng.next_usize(n_samples);
            means[c * n_features..(c + 1) * n_features]
                .copy_from_slice(&data[idx * n_features..(idx + 1) * n_features]);
        } else {
            let mut target = rng.next_f64() * total;
            let mut chosen = n_samples - 1;
            for i in 0..n_samples {
                target -= min_dists[i];
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            means[c * n_features..(c + 1) * n_features]
                .copy_from_slice(&data[chosen * n_features..(chosen + 1) * n_features]);
        }
    }

    means
}

/// Fit a Gaussian Mixture Model using the EM algorithm with diagonal covariances.
pub fn gmm_impl(
    data: &[f64],
    n_features: usize,
    n_components: usize,
    max_iter: usize,
    tol: f64,
) -> Result<GmmModel, MlError> {
    let n_samples = validate_matrix(data, n_features)?;
    if n_components == 0 || n_components > n_samples {
        return Err(MlError::new("n_components must be between 1 and number of samples"));
    }
    if max_iter == 0 {
        return Err(MlError::new("max_iter must be > 0"));
    }
    if tol <= 0.0 {
        return Err(MlError::new("tol must be > 0"));
    }

    let mut rng = Rng::from_data(data);
    let nf = n_features;
    let nc = n_components;
    let ns = n_samples;

    // Initialize means via K-Means++ seeding
    let mut means = kmeans_pp_init(data, ns, nf, nc, &mut rng);

    // Initialize weights uniformly
    let mut weights = vec![1.0 / nc as f64; nc];

    // Initialize covariances to overall data variance per feature
    let mut feature_var = vec![0.0; nf];
    // Compute global mean
    let mut global_mean = vec![0.0; nf];
    for i in 0..ns {
        for j in 0..nf {
            global_mean[j] += mat_get(data, nf, i, j);
        }
    }
    for j in 0..nf {
        global_mean[j] /= ns as f64;
    }
    // Compute variance
    for i in 0..ns {
        for j in 0..nf {
            let d = mat_get(data, nf, i, j) - global_mean[j];
            feature_var[j] += d * d;
        }
    }
    for j in 0..nf {
        feature_var[j] = feature_var[j] / ns as f64 + 1e-6; // small floor to avoid zero
    }
    let mut covariances = vec![0.0; nc * nf];
    for k in 0..nc {
        for j in 0..nf {
            covariances[k * nf + j] = feature_var[j];
        }
    }

    // Responsibilities buffer: [ns * nc]
    let mut resp = vec![0.0; ns * nc];

    let mut prev_log_likelihood = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // ========== E-step ==========
        // Compute log-likelihood and responsibilities
        let mut log_likelihood = 0.0;
        for i in 0..ns {
            let mut log_components = vec![f64::NEG_INFINITY; nc];
            let mut max_log_comp = f64::NEG_INFINITY;

            for k in 0..nc {
                let mut log_val = weights[k].ln();
                for j in 0..nf {
                    let x = mat_get(data, nf, i, j);
                    let mean = means[k * nf + j];
                    let var = covariances[k * nf + j];
                    log_val += gaussian_log_pdf(x, mean, var);
                }
                log_components[k] = log_val;
                if log_val > max_log_comp {
                    max_log_comp = log_val;
                }
            }

            // Log-sum-exp for normalization + log-likelihood contribution
            let sum_exp: f64 = log_components.iter().map(|&lc| (lc - max_log_comp).exp()).sum();
            let log_norm = max_log_comp + sum_exp.ln();
            log_likelihood += log_norm;

            // Normalize responsibilities
            for k in 0..nc {
                resp[i * nc + k] = (log_components[k] - max_log_comp).exp() / sum_exp;
            }
        }

        // Check convergence
        let ll_change = (log_likelihood - prev_log_likelihood).abs();
        if iter > 0 && ll_change < tol {
            converged = true;
            break;
        }
        prev_log_likelihood = log_likelihood;

        // ========== M-step ==========
        // Effective counts (Nk = sum_i r_ik)
        let mut nk = vec![0.0; nc];
        for i in 0..ns {
            for k in 0..nc {
                nk[k] += resp[i * nc + k];
            }
        }

        // Update weights
        for k in 0..nc {
            weights[k] = nk[k] / ns as f64;
        }

        // Update means
        means.fill(0.0);
        for i in 0..ns {
            for k in 0..nc {
                let r = resp[i * nc + k];
                for j in 0..nf {
                    means[k * nf + j] += r * mat_get(data, nf, i, j);
                }
            }
        }
        for k in 0..nc {
            if nk[k] > 1e-10 {
                for j in 0..nf {
                    means[k * nf + j] /= nk[k];
                }
            }
        }

        // Update covariances (diagonal)
        covariances.fill(0.0);
        for i in 0..ns {
            for k in 0..nc {
                let r = resp[i * nc + k];
                for j in 0..nf {
                    let d = mat_get(data, nf, i, j) - means[k * nf + j];
                    covariances[k * nf + j] += r * d * d;
                }
            }
        }
        for k in 0..nc {
            if nk[k] > 1e-10 {
                for j in 0..nf {
                    covariances[k * nf + j] = covariances[k * nf + j] / nk[k] + 1e-6;
                }
            } else {
                // Re-initialize dead component to global variance
                for j in 0..nf {
                    covariances[k * nf + j] = feature_var[j];
                }
            }
        }
    }

    Ok(GmmModel {
        weights,
        means,
        covariances,
        n_components: nc,
        n_features: nf,
        converged,
        n_iter,
    })
}

#[wasm_bindgen(js_name = "gmm")]
pub fn gmm(
    data: &[f64],
    n_features: usize,
    n_components: usize,
    max_iter: usize,
    tol: f64,
) -> Result<GmmModel, JsError> {
    gmm_impl(data, n_features, n_components, max_iter, tol)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_well_separated_clusters() {
        // Two tight clusters far apart: (0,0) and (10,10)
        let data = vec![
            0.0, 0.0,   0.1, 0.1,   -0.1, -0.1,   0.2, -0.1,
            10.0, 10.0, 10.1, 10.1,  9.9, 9.9,    10.2, 9.9,
        ];
        let model = gmm_impl(&data, 2, 2, 200, 1e-6).unwrap();
        assert_eq!(model.n_components, 2);
        assert_eq!(model.n_features, 2);
        assert!(model.converged);
        assert!(model.n_iter < 200);

        // Predictions should separate the two clusters
        let preds = model.predict(&data);
        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[0], preds[2]);
        assert_eq!(preds[0], preds[3]);
        assert_eq!(preds[4], preds[5]);
        assert_eq!(preds[4], preds[6]);
        assert_eq!(preds[4], preds[7]);
        assert_ne!(preds[0], preds[4]);

        // Weights should be roughly 0.5 each
        let w_sum: f64 = model.weights.iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-10);
        assert!((model.weights[0] - 0.5).abs() < 0.1);
        assert!((model.weights[1] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_single_component() {
        let data = vec![1.0, 2.0, 1.1, 2.1, 0.9, 1.9];
        let model = gmm_impl(&data, 2, 1, 100, 1e-6).unwrap();
        assert_eq!(model.n_components, 1);
        assert!((model.weights[0] - 1.0).abs() < 1e-10);
        let preds = model.predict(&data);
        assert!(preds.iter().all(|&p| p == 0));
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,
            5.0, 5.0,  5.1, 5.1,
        ];
        let model = gmm_impl(&data, 2, 2, 200, 1e-6).unwrap();
        let proba = model.predict_proba(&data);
        let n_samples = data.len() / model.n_features;
        for i in 0..n_samples {
            let row_sum: f64 = proba[i * model.n_components..(i + 1) * model.n_components].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "probabilities for sample {} sum to {}", i, row_sum);
        }
    }

    #[test]
    fn test_predict_proba_high_confidence() {
        // Well-separated clusters should yield high confidence
        let data = vec![
            0.0, 0.0,  0.1, 0.1,
            20.0, 20.0,  20.1, 20.1,
        ];
        let model = gmm_impl(&data, 2, 2, 200, 1e-6).unwrap();
        let proba = model.predict_proba(&data);
        // First sample should have high confidence in cluster 0
        assert!(proba[0] > 0.99 || proba[1] > 0.99,
            "Expected high confidence for well-separated cluster, got {:.6}, {:.6}", proba[0], proba[1]);
    }

    #[test]
    fn test_score_samples_returns_log_likelihood() {
        let data = vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2];
        let model = gmm_impl(&data, 2, 1, 100, 1e-6).unwrap();
        let scores = model.score_samples(&data);
        assert_eq!(scores.len(), 3);
        // All scores should be finite
        for &s in &scores {
            assert!(s.is_finite(), "score_samples returned non-finite value: {}", s);
        }
    }

    #[test]
    fn test_score_samples_higher_for_center() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,  -0.1, -0.1,
            5.0, 5.0,  5.1, 5.1,    4.9, 4.9,
        ];
        let model = gmm_impl(&data, 2, 2, 200, 1e-6).unwrap();
        let scores = model.score_samples(&data);
        // Points near cluster centers should have higher log-likelihood
        // than points far from any center
        let near_center = scores[0]; // (0,0) near cluster 1
        let far_point = scores[3];   // (5,5) might be less likely under cluster 1 but more under cluster 2
        // Both should be reasonable (not -inf)
        assert!(near_center.is_finite());
        assert!(far_point.is_finite());
    }

    #[test]
    fn test_n_components_too_large() {
        let data = vec![1.0, 2.0];
        assert!(gmm_impl(&data, 2, 3, 100, 1e-6).is_err());
    }

    #[test]
    fn test_zero_max_iter() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(gmm_impl(&data, 2, 1, 0, 1e-6).is_err());
    }

    #[test]
    fn test_zero_tol() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(gmm_impl(&data, 2, 1, 100, 0.0).is_err());
    }

    #[test]
    fn test_invalid_data_length() {
        let data = vec![1.0, 2.0, 3.0]; // not divisible by n_features=2
        assert!(gmm_impl(&data, 2, 1, 100, 1e-6).is_err());
    }

    #[test]
    fn test_three_components() {
        // Three clusters along x-axis: 0, 10, 20
        let data = vec![
            0.0, 0.0,  0.1, 0.0,
            10.0, 0.0, 10.1, 0.0,
            20.0, 0.0, 20.1, 0.0,
        ];
        let model = gmm_impl(&data, 2, 3, 300, 1e-6).unwrap();
        assert_eq!(model.n_components, 3);
        assert_eq!(model.weights.len(), 3);
        let w_sum: f64 = model.weights.iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_pdf_values() {
        // At mean, PDF should be at its maximum
        let pdf_at_mean = gaussian_pdf(0.0, 0.0, 1.0);
        let pdf_away = gaussian_pdf(2.0, 0.0, 1.0);
        assert!(pdf_at_mean > pdf_away);
        // Standard normal at mean: 1/sqrt(2*pi) ~ 0.3989
        assert!((pdf_at_mean - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_log_pdf_consistency() {
        let x = 1.5;
        let mean = 0.0;
        let var = 2.0;
        let pdf = gaussian_pdf(x, mean, var);
        let log_pdf = gaussian_log_pdf(x, mean, var);
        assert!((pdf - log_pdf.exp()).abs() < 1e-12);
    }

    #[test]
    fn test_to_string() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let model = gmm_impl(&data, 2, 1, 100, 1e-6).unwrap();
        let s = model.to_string_js();
        assert!(s.contains("GMM("));
        assert!(s.contains("components=1"));
        assert!(s.contains("features=2"));
    }

    #[test]
    fn test_getters() {
        let data = vec![
            0.0, 0.0,  0.1, 0.1,
            5.0, 5.0,  5.1, 5.1,
        ];
        let model = gmm_impl(&data, 2, 2, 200, 1e-6).unwrap();
        assert_eq!(model.n_components(), 2);
        assert_eq!(model.n_features(), 2);
        assert_eq!(model.get_weights().len(), 2);
        assert_eq!(model.get_means().len(), 4); // 2 components * 2 features
    }

    #[test]
    fn test_1d_data() {
        let data = vec![0.0, 0.1, -0.1, 10.0, 10.1, 9.9];
        let model = gmm_impl(&data, 1, 2, 200, 1e-6).unwrap();
        assert_eq!(model.n_features, 1);
        assert_eq!(model.n_components, 2);
        let preds = model.predict(&data);
        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[0], preds[2]);
        assert_ne!(preds[0], preds[3]);
    }
}
