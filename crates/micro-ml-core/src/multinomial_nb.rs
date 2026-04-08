use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

/// Multinomial Naive Bayes model for count/frequency features.
///
/// Suitable for text classification with word counts or TF-IDF features.
#[wasm_bindgen]
pub struct MultinomialNBModel {
    log_prior: Vec<f64>,
    log_likelihood: Vec<f64>,
    class_log_count: Vec<f64>,
    n_classes: usize,
    n_features: usize,
    classes: Vec<u32>,
}

#[wasm_bindgen]
impl MultinomialNBModel {
    #[wasm_bindgen(getter, js_name = "nClasses")]
    pub fn n_classes(&self) -> usize { self.n_classes }

    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(js_name = "getClasses")]
    pub fn classes_js(&self) -> Vec<u32> { self.classes.clone() }

    /// Predict class labels for count/frequency feature vectors.
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
                    let x = data[i * self.n_features + j];
                    log_prob += x * self.log_likelihood[c * self.n_features + j];
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

    /// Predict class probabilities for count/frequency feature vectors (softmax over log-probs).
    #[wasm_bindgen(js_name = "predictProba")]
    pub fn predict_proba(&self, data: &[f64]) -> Vec<f64> {
        let n_test = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n_test * self.n_classes);

        for i in 0..n_test {
            let mut log_probs = Vec::with_capacity(self.n_classes);
            for c in 0..self.n_classes {
                let mut log_prob = self.log_prior[c];
                for j in 0..self.n_features {
                    let x = data[i * self.n_features + j];
                    log_prob += x * self.log_likelihood[c * self.n_features + j];
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
        format!("MultinomialNB(classes={}, features={})", self.n_classes, self.n_features)
    }
}

/// Fit a Multinomial Naive Bayes model.
///
/// * `data` - flat row-major matrix of non-negative counts/frequencies
/// * `n_features` - number of features per sample
/// * `labels` - class labels for each sample
/// * `alpha` - Laplace smoothing parameter (default 1.0)
pub fn multinomial_nb_impl(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    alpha: f64,
) -> Result<MultinomialNBModel, MlError> {
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

    // Count samples per class and sum of feature counts per class
    let mut class_counts = vec![0usize; n_classes];
    let mut class_feature_sums = vec![0.0f64; n_classes * n_features]; // sum(x_j, y=c)

    for i in 0..n {
        let c = classes.iter().position(|&cls| cls == labels[i] as u32).unwrap();
        class_counts[c] += 1;
        for j in 0..n_features {
            class_feature_sums[c * n_features + j] += mat_get(data, n_features, i, j).max(0.0);
        }
    }

    // Compute log priors
    let log_prior: Vec<f64> = class_counts
        .iter()
        .map(|&cnt| (cnt as f64 / n as f64).ln())
        .collect();

    // Compute log sum of feature counts per class (for reference)
    let mut class_log_count = vec![0.0; n_classes];
    for c in 0..n_classes {
        let total: f64 = class_feature_sums[c * n_features..(c + 1) * n_features].iter().sum();
        class_log_count[c] = total.ln();
    }

    // Compute log likelihoods with Laplace smoothing
    // log P(x_j|y=c) = log((sum(x_j, y=c) + alpha) / (total_count(y=c) + alpha * n_features))
    let mut log_likelihood = vec![0.0; n_classes * n_features];
    for c in 0..n_classes {
        let total_count: f64 = class_feature_sums[c * n_features..(c + 1) * n_features].iter().sum();
        let denom = (total_count + alpha * n_features as f64).ln();
        for j in 0..n_features {
            let numer = (class_feature_sums[c * n_features + j] + alpha).ln();
            log_likelihood[c * n_features + j] = numer - denom;
        }
    }

    Ok(MultinomialNBModel {
        log_prior,
        log_likelihood,
        class_log_count,
        n_classes,
        n_features,
        classes,
    })
}

#[wasm_bindgen(js_name = "multinomialNB")]
pub fn multinomial_nb_fit(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    alpha: f64,
) -> Result<MultinomialNBModel, JsError> {
    multinomial_nb_impl(data, n_features, labels, alpha).map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_features() {
        // 4 samples, 3 word-count features, 2 classes
        // Class 0 (sports): word "goal" (feature 0) frequent, others low
        // Class 1 (tech):   word "code" (feature 2) frequent, others low
        let data = vec![
            5.0, 0.0, 0.0,  // class 0 (sports)
            4.0, 1.0, 0.0,  // class 0 (sports)
            0.0, 1.0, 6.0,  // class 1 (tech)
            0.0, 0.0, 5.0,  // class 1 (tech)
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let model = multinomial_nb_impl(&data, 3, &labels, 1.0).unwrap();

        assert_eq!(model.n_classes, 2);
        assert_eq!(model.n_features, 3);
        assert_eq!(model.classes, vec![0, 1]);

        // Predict training data
        let preds = model.predict(&data);
        assert_eq!(preds, vec![0, 0, 1, 1]);

        // Predict new samples
        let test = vec![5.0, 0.0, 0.0, 0.0, 0.0, 5.0];
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
    fn test_three_classes() {
        // 3 classes with distinct count patterns
        let data = vec![
            4.0, 0.0, 0.0,  // class 0
            3.0, 1.0, 0.0,  // class 0
            0.0, 5.0, 0.0,  // class 1
            0.0, 4.0, 1.0,  // class 1
            0.0, 0.0, 6.0,  // class 2
            1.0, 0.0, 5.0,  // class 2
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let model = multinomial_nb_impl(&data, 3, &labels, 1.0).unwrap();

        assert_eq!(model.n_classes, 3);

        // Predict training data
        let preds = model.predict(&data);
        assert_eq!(preds, vec![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn test_negative_values_clamped() {
        // Negative values should be clamped to 0
        let data = vec![
            3.0, -1.0, 0.0,  // class 0
            0.0, 0.0, 4.0,   // class 1
        ];
        let labels = vec![0.0, 1.0];
        let model = multinomial_nb_impl(&data, 3, &labels, 1.0).unwrap();
        assert_eq!(model.n_classes, 2);
    }

    #[test]
    fn test_invalid_inputs() {
        let data = vec![1.0, 0.0];
        let labels = vec![0.0, 1.0]; // mismatch
        let result = multinomial_nb_impl(&data, 2, &labels, 1.0);
        assert!(result.is_err());
    }
}
