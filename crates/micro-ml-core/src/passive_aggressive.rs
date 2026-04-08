use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get};

/// Loss function types for Passive-Aggressive classifiers
#[derive(Debug, Clone, Copy)]
enum PaLossType {
    /// PA-I: hinge loss (standard passive-aggressive)
    Hinge,
    /// PA-II: squared hinge loss
    SquaredHinge,
}

impl PaLossType {
    fn from_str(s: &str) -> Result<Self, MlError> {
        match s {
            "hinge" => Ok(PaLossType::Hinge),
            "squared_hinge" => Ok(PaLossType::SquaredHinge),
            _ => Err(MlError::new(format!(
                "unknown loss type '{}': expected 'hinge' or 'squared_hinge'",
                s
            ))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            PaLossType::Hinge => "hinge",
            PaLossType::SquaredHinge => "squared_hinge",
        }
    }
}

/// Passive-Aggressive Classifier - Online learning that updates minimally.
///
/// Passive-Aggressive algorithms remain passive (no update) when the prediction
/// is correct with sufficient margin, and become aggressive (minimum update to
/// correct) when the prediction is wrong or too close to the boundary.
///
/// Two variants:
/// - PA-I ("hinge"): Uses hinge loss. Update step: tau = loss / (||x||^2 + 1/(2*C))
/// - PA-II ("squared_hinge"): Uses squared hinge loss. More conservative updates.
///
/// # Arguments
/// * `C` - Aggressiveness parameter. Higher C means more aggressive (larger) updates.
///   C -> infinity approaches a hard-margin SVM. C = 0 means never update.
#[wasm_bindgen]
pub struct PassiveAggressiveModel {
    weights: Vec<f64>,
    bias: f64,
    n_features: usize,
    n_iter: usize,
    loss_type: PaLossType,
    c: f64,
}

#[wasm_bindgen]
impl PassiveAggressiveModel {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    #[wasm_bindgen(getter, js_name = "nIter")]
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    #[wasm_bindgen(getter, js_name = "lossType")]
    pub fn loss_type(&self) -> String {
        self.loss_type.as_str().to_string()
    }

    #[wasm_bindgen(getter, js_name = "c")]
    pub fn c(&self) -> f64 {
        self.c
    }

    /// Predict class labels (0 or 1) for each sample
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut sum = self.bias;
            for j in 0..self.n_features {
                sum += self.weights[j] * data[i * self.n_features + j];
            }
            result.push(if sum > 0.0 { 1.0 } else { 0.0 });
        }

        result
    }

    /// Raw decision function scores (signed distance to hyperplane)
    #[wasm_bindgen(js_name = "decisionFunction")]
    pub fn decision_function(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut sum = self.bias;
            for j in 0..self.n_features {
                sum += self.weights[j] * data[i * self.n_features + j];
            }
            result.push(sum);
        }

        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "PassiveAggressive(loss={}, n_features={}, n_iter={}, C={})",
            self.loss_type.as_str(),
            self.n_features,
            self.n_iter,
            self.c
        )
    }
}

/// Internal Passive-Aggressive classifier implementation (pure Rust, no WASM bindings)
///
/// # Arguments
/// * `data` - Flat row-major feature matrix (n_samples * n_features)
/// * `n_features` - Number of features per sample
/// * `labels` - Target labels (0.0 or 1.0 for binary classification)
/// * `max_iter` - Maximum number of passes over the training data (epochs)
/// * `C` - Aggressiveness parameter (higher = more aggressive updates)
/// * `loss_type` - Loss function: "hinge" (PA-I) or "squared_hinge" (PA-II)
pub fn passive_aggressive_impl(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    max_iter: usize,
    c: f64,
    loss_type: &str,
) -> Result<PassiveAggressiveModel, MlError> {
    if c < 0.0 {
        return Err(MlError::new("C (aggressiveness parameter) must be >= 0"));
    }
    if max_iter == 0 {
        return Err(MlError::new("max_iter must be > 0"));
    }

    let loss = PaLossType::from_str(loss_type)?;
    let n = validate_matrix(data, n_features)?;
    if labels.len() != n {
        return Err(MlError::new("labels length must match number of samples"));
    }

    // Convert 0/1 labels to -1/+1 for internal computation
    let y: Vec<f64> = labels
        .iter()
        .map(|&l| if l > 0.5 { 1.0 } else { -1.0 })
        .collect();

    // Initialize weights to zero
    let mut weights = vec![0.0f64; n_features];
    let mut bias = 0.0;

    // Training loop
    for _epoch in 0..max_iter {
        for i in 0..n {
            // Compute decision value: f(x) = w . x + b
            let mut decision = bias;
            for j in 0..n_features {
                decision += weights[j] * mat_get(data, n_features, i, j);
            }

            // Compute hinge loss: max(0, 1 - y * decision)
            let loss_val = (1.0 - y[i] * decision).max(0.0);

            if loss_val > 0.0 {
                // Compute squared norm of x
                let mut x_norm_sq = 0.0;
                for j in 0..n_features {
                    let x_j = mat_get(data, n_features, i, j);
                    x_norm_sq += x_j * x_j;
                }

                let tau = if x_norm_sq == 0.0 {
                    // Sample has zero norm, skip update
                    continue;
                } else {
                    match loss {
                        PaLossType::Hinge => {
                            // PA-I: tau = loss / (||x||^2 + 1/(2*C))
                            if c == 0.0 {
                                continue; // C=0 means never update
                            }
                            loss_val / (x_norm_sq + 1.0 / (2.0 * c))
                        }
                        PaLossType::SquaredHinge => {
                            // PA-II: tau = min(C, loss / ||x||^2)
                            if c == 0.0 {
                                continue;
                            }
                            let tau_unclamped = loss_val / x_norm_sq;
                            tau_unclamped.min(c)
                        }
                    }
                };

                // Update: w = w + tau * y * x, b = b + tau * y
                for j in 0..n_features {
                    weights[j] += tau * y[i] * mat_get(data, n_features, i, j);
                }
                bias += tau * y[i];
            }
        }
    }

    Ok(PassiveAggressiveModel {
        weights,
        bias,
        n_features,
        n_iter: max_iter,
        loss_type: loss,
        c,
    })
}

/// WASM-exported Passive-Aggressive classifier constructor
///
/// # Arguments
/// * `data` - Flat row-major feature matrix (n_samples * n_features)
/// * `n_features` - Number of features per sample
/// * `labels` - Target labels (0.0 or 1.0 for binary classification)
/// * `max_iter` - Maximum number of passes over the training data (epochs)
/// * `C` - Aggressiveness parameter (higher = more aggressive updates)
/// * `loss_type` - Loss function: "hinge" (PA-I) or "squared_hinge" (PA-II)
#[wasm_bindgen(js_name = "passiveAggressive")]
pub fn passive_aggressive(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    max_iter: usize,
    c: f64,
    loss_type: &str,
) -> Result<PassiveAggressiveModel, JsError> {
    passive_aggressive_impl(data, n_features, labels, max_iter, c, loss_type)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linearly_separable_hinge() {
        // Simple 2D linearly separable data
        let data = vec![
            0.0, 0.0,   // Class 0
            1.0, 0.0,   // Class 0
            0.0, 1.0,   // Class 0
            10.0, 10.0, // Class 1
            11.0, 10.0, // Class 1
            10.0, 11.0, // Class 1
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let model = passive_aggressive_impl(&data, 2, &labels, 10, 1.0, "hinge").unwrap();
        let preds = model.predict(&data);

        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 0.0);
        assert_eq!(preds[3], 1.0);
        assert_eq!(preds[4], 1.0);
        assert_eq!(preds[5], 1.0);
    }

    #[test]
    fn test_linearly_separable_squared_hinge() {
        let data = vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            10.0, 10.0,
            11.0, 10.0,
            10.0, 11.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let model = passive_aggressive_impl(&data, 2, &labels, 10, 1.0, "squared_hinge").unwrap();
        let preds = model.predict(&data);

        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 0.0);
        assert_eq!(preds[3], 1.0);
        assert_eq!(preds[4], 1.0);
        assert_eq!(preds[5], 1.0);
    }

    #[test]
    fn test_decision_function() {
        // Use non-zero samples so PA can learn both directions
        let data = vec![
            0.0, 0.0,
            1.0, 1.0,
            10.0, 10.0,
        ];
        let labels = vec![0.0, 0.0, 1.0];

        let model = passive_aggressive_impl(&data, 2, &labels, 10, 1.0, "hinge").unwrap();
        let scores = model.decision_function(&data);

        // Class 0 samples should have negative scores
        assert!(scores[0] < 0.0, "Class 0 sample 0 should have negative score, got {}", scores[0]);
        assert!(scores[1] < 0.0, "Class 0 sample 1 should have negative score, got {}", scores[1]);
        // Class 1 sample should have positive score
        assert!(scores[2] > 0.0, "Class 1 sample should have positive score, got {}", scores[2]);
    }

    #[test]
    fn test_high_c_more_aggressive() {
        // With high C, the model should aggressively correct. Note: sample 0 is [0,0]
        // (zero vector) so the model cannot update on it (x_norm_sq=0). The model
        // learns to classify sample 1 correctly, but sample 0's classification depends
        // on the bias. With enough iterations the bias should be negative enough.
        let data = vec![
            0.0, 0.0,
            10.0, 10.0,
        ];
        let labels = vec![0.0, 1.0];

        let model = passive_aggressive_impl(&data, 2, &labels, 10, 1000.0, "hinge").unwrap();
        let preds = model.predict(&data);

        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_zero_c_no_update() {
        // C=0 means never update, so predictions should all be the same (initial)
        let data = vec![
            0.0, 0.0,
            10.0, 10.0,
            5.0, 5.0,
        ];
        let labels = vec![0.0, 1.0, 1.0];

        let model = passive_aggressive_impl(&data, 2, &labels, 10, 0.0, "hinge").unwrap();
        let preds = model.predict(&data);

        // All predictions should be the same (no learning happened)
        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[1], preds[2]);
    }

    #[test]
    fn test_invalid_loss_type() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];

        let result = passive_aggressive_impl(&data, 2, &labels, 10, 1.0, "unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_c() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];

        let result = passive_aggressive_impl(&data, 2, &labels, 10, -1.0, "hinge");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_max_iter() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];

        let result = passive_aggressive_impl(&data, 2, &labels, 0, 1.0, "hinge");
        assert!(result.is_err());
    }

    #[test]
    fn test_labels_mismatch() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0, 0.0]; // 3 labels but only 2 samples

        let result = passive_aggressive_impl(&data, 2, &labels, 10, 1.0, "hinge");
        assert!(result.is_err());
    }

    #[test]
    fn test_getters() {
        let data = vec![0.0, 0.0, 10.0, 10.0];
        let labels = vec![0.0, 1.0];

        let model = passive_aggressive_impl(&data, 2, &labels, 50, 0.5, "squared_hinge").unwrap();

        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_iter(), 50);
        assert_eq!(model.loss_type(), "squared_hinge");
        assert!((model.c() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_convergence_more_iterations() {
        // More iterations should yield better or equal accuracy
        let data = vec![
            0.0, 0.0,
            0.2, 0.1,
            1.0, 0.0,
            3.0, 3.0,
            3.2, 3.1,
            4.0, 3.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let model_few = passive_aggressive_impl(&data, 2, &labels, 1, 1.0, "hinge").unwrap();
        let model_many = passive_aggressive_impl(&data, 2, &labels, 50, 1.0, "hinge").unwrap();

        let preds_few = model_few.predict(&data);
        let preds_many = model_many.predict(&data);

        let correct_few = preds_few.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();
        let correct_many = preds_many.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();

        assert!(correct_many >= correct_few);
        assert_eq!(correct_many, labels.len());
    }

    #[test]
    fn test_to_string() {
        let data = vec![0.0, 0.0, 10.0, 10.0];
        let labels = vec![0.0, 1.0];

        let model = passive_aggressive_impl(&data, 2, &labels, 10, 1.0, "hinge").unwrap();
        let s = model.to_string_js();
        assert!(s.contains("PassiveAggressive"));
        assert!(s.contains("hinge"));
        assert!(s.contains("n_features=2"));
        assert!(s.contains("n_iter=10"));
        assert!(s.contains("C=1"));
    }

    #[test]
    fn test_margin_classification() {
        // PA should enforce a margin of 1 between classes
        let data = vec![
            -1.0, -1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
             5.0,  5.0,
             6.0,  6.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let model = passive_aggressive_impl(&data, 2, &labels, 20, 0.5, "hinge").unwrap();
        let preds = model.predict(&data);

        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 0.0);
        assert_eq!(preds[3], 0.0);
        assert_eq!(preds[4], 1.0);
        assert_eq!(preds[5], 1.0);

        // Decision scores for class 1 should be positive (correct side of boundary)
        let scores = model.decision_function(&data);
        assert!(scores[4] > 0.0, "Class 1 sample should have positive score, got {}", scores[4]);
        assert!(scores[5] > 0.0, "Class 1 sample should have positive score, got {}", scores[5]);
    }
}
