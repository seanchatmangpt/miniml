use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

/// Bernoulli Naive Bayes model for binary/categorical features.
///
/// Suitable for text classification with binary feature vectors
/// (e.g., bag-of-words presence/absence).
#[wasm_bindgen]
pub struct BernoulliNBModel {
    log_prior: Vec<f64>,
    log_likelihood: Vec<f64>,
    n_classes: usize,
    n_features: usize,
    classes: Vec<u32>,
}

#[wasm_bindgen]
impl BernoulliNBModel {
    #[wasm_bindgen(getter, js_name = "nClasses")]
    pub fn n_classes(&self) -> usize { self.n_classes }

    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(js_name = "getClasses")]
    pub fn classes_js(&self) -> Vec<u32> { self.classes.clone() }

    /// Predict class labels for binary feature vectors.
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<u32> {
        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let mut best_class = 0u32;
            let mut best_log_prob = f64::NEG_INFINITY;

            for c in 0..self.n_classes {
                let mut log_prob = self.log_prior[c];
                for j in 0..self.n_features {
                    let x = if data[i * self.n_features + j] > 0.5 { 1.0 } else { 0.0 };
                    let ll = self.log_likelihood[c * self.n_features + j];
                    // log P(x_j|y=c) = x_j * log(p) + (1-x_j) * log(1-p)
                    // where log(1-p) = log(1 - exp(ll))
                    let one_minus_p_log = (1.0 - ll.exp()).ln();
                    log_prob += x * ll + (1.0 - x) * one_minus_p_log;
                }
                if log_prob > best_log_prob {
                    best_log_prob = log_prob;
                    best_class = self.classes[c];
                }
            }
            result.push(best_class);
        }
        result
    }

    /// Predict class probabilities for binary feature vectors (softmax over log-probs).
    #[wasm_bindgen(js_name = "predictProba")]
    pub fn predict_proba(&self, data: &[f64]) -> Vec<f64> {
        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test * self.n_classes);

        for i in 0..n_test {
            let mut log_probs = Vec::with_capacity(self.n_classes);
            for c in 0..self.n_classes {
                let mut log_prob = self.log_prior[c];
                for j in 0..self.n_features {
                    let x = if data[i * self.n_features + j] > 0.5 { 1.0 } else { 0.0 };
                    let ll = self.log_likelihood[c * self.n_features + j];
                    let one_minus_p_log = (1.0 - ll.exp()).ln();
                    log_prob += x * ll + (1.0 - x) * one_minus_p_log;
                }
                log_probs.push(log_prob);
            }
            // Softmax to convert log-probs to probabilities
            let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = log_probs.iter().map(|&lp| (lp - max_lp).exp()).sum();
            for lp in &log_probs {
                result.push(((lp - max_lp).exp()) / sum_exp);
            }
        }
        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("BernoulliNB(classes={}, features={})", self.n_classes, self.n_features)
    }
}

/// Fit a Bernoulli Naive Bayes model.
///
/// * `data` - flat row-major matrix of binary features (values > 0.5 treated as 1)
/// * `n_features` - number of features per sample
/// * `labels` - class labels for each sample
/// * `alpha` - Laplace smoothing parameter (default 1.0)
pub fn bernoulli_nb_impl(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    alpha: f64,
) -> Result<BernoulliNBModel, MlError> {
    let n = validate_matrix(data, n_features)?;
    if labels.len() != n {
        return Err(MlError::new("labels length must match number of samples"));
    }

    // Find unique classes
    let mut classes: Vec<u32> = labels.iter().map(|&v| v as u32).collect();
    classes.sort_unstable();
    classes.dedup();
    let n_classes = classes.len();

    if n_classes == 0 {
        return Err(MlError::new("no classes found in labels"));
    }

    // Count samples per class and feature-1 counts per class
    let mut class_counts = vec![0usize; n_classes];
    let mut feature_counts = vec![0usize; n_classes * n_features]; // count(x_j=1, y=c)

    for i in 0..n {
        let c = classes.iter().position(|&cls| cls == labels[i] as u32).unwrap();
        class_counts[c] += 1;
        for j in 0..n_features {
            if mat_get(data, n_features, i, j) > 0.5 {
                feature_counts[c * n_features + j] += 1;
            }
        }
    }

    // Compute log priors
    let log_prior: Vec<f64> = class_counts
        .iter()
        .map(|&cnt| (cnt as f64 / n as f64).ln())
        .collect();

    // Compute log likelihoods with Laplace smoothing
    // log P(x_j=1|y=c) = log((count(x_j=1, y=c) + alpha) / (count(y=c) + 2*alpha))
    let mut log_likelihood = vec![0.0; n_classes * n_features];
    for c in 0..n_classes {
        let denom = (class_counts[c] as f64 + 2.0 * alpha).ln();
        for j in 0..n_features {
            let numer = (feature_counts[c * n_features + j] as f64 + alpha).ln();
            log_likelihood[c * n_features + j] = numer - denom;
        }
    }

    Ok(BernoulliNBModel {
        log_prior,
        log_likelihood,
        n_classes,
        n_features,
        classes,
    })
}

#[wasm_bindgen(js_name = "bernoulliNB")]
pub fn bernoulli_nb_fit(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    alpha: f64,
) -> Result<BernoulliNBModel, JsError> {
    bernoulli_nb_impl(data, n_features, labels, alpha).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_features() {
        // 4 samples, 3 binary features, 2 classes
        // Class 0: feature 0 tends to be 1, features 1,2 tend to be 0
        // Class 1: feature 2 tends to be 1, features 0,1 tend to be 0
        let data = vec![
            1.0, 0.0, 0.0,  // class 0
            1.0, 0.0, 0.0,  // class 0
            0.0, 0.0, 1.0,  // class 1
            0.0, 0.0, 1.0,  // class 1
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let model = bernoulli_nb_impl(&data, 3, &labels, 1.0).unwrap();

        assert_eq!(model.n_classes, 2);
        assert_eq!(model.n_features, 3);
        assert_eq!(model.classes, vec![0, 1]);

        // Predict training data
        let preds = model.predict(&data);
        assert_eq!(preds, vec![0, 0, 1, 1]);

        // Predict new samples
        let test = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let preds = model.predict(&test);
        assert_eq!(preds, vec![0, 1]);

        // Probabilities sum to 1 for each sample
        let proba = model.predict_proba(&data);
        for i in 0..4 {
            let sum: f64 = proba[i * 2..(i + 1) * 2].iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "probabilities must sum to 1");
        }
    }

    #[test]
    fn test_binary_threshold() {
        // Values > 0.5 should be treated as 1
        let data = vec![
            0.9, 0.1, 0.3,  // class 0
            0.8, 0.2, 0.1,  // class 0
            0.1, 0.2, 0.9,  // class 1
        ];
        let labels = vec![0.0, 0.0, 1.0];
        let model = bernoulli_nb_impl(&data, 3, &labels, 1.0).unwrap();

        // Predict with values right at threshold
        let test = vec![0.6, 0.0, 0.0]; // feature 0 = 1 (above 0.5), should be class 0
        let preds = model.predict(&test);
        assert_eq!(preds, vec![0]);
    }

    #[test]
    fn test_invalid_inputs() {
        let data = vec![1.0, 0.0];
        let labels = vec![0.0, 1.0]; // mismatch
        let result = bernoulli_nb_impl(&data, 2, &labels, 1.0);
        assert!(result.is_err());
    }
}
