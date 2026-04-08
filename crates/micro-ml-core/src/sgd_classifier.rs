use wasm_bindgen::prelude::*;
use crate::error::MlError;
use crate::matrix::{validate_matrix, mat_get, Rng};

/// Loss function types for SGD Classifier
#[derive(Debug, Clone, Copy)]
enum LossType {
    /// Linear SVM (hinge loss): max(0, 1 - y * f(x))
    Hinge,
    /// Logistic regression (log loss): log(1 + exp(-y * f(x)))
    Log,
    /// Smooth hinge (modified Huber): quadratic near 0, linear far away
    ModifiedHuber,
}

impl LossType {
    fn from_str(s: &str) -> Result<Self, MlError> {
        match s {
            "hinge" => Ok(LossType::Hinge),
            "log" => Ok(LossType::Log),
            "modified_huber" => Ok(LossType::ModifiedHuber),
            _ => Err(MlError::new(format!(
                "unknown loss type '{}': expected 'hinge', 'log', or 'modified_huber'",
                s
            ))),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            LossType::Hinge => "hinge",
            LossType::Log => "log",
            LossType::ModifiedHuber => "modified_huber",
        }
    }

    /// Compute loss value for a single sample
    fn loss(&self, y: f64, f: f64) -> f64 {
        let z = y * f;
        match self {
            LossType::Hinge => (1.0 - z).max(0.0),
            LossType::Log => {
                // log(1 + exp(-z)) with numerical stability
                if z > 18.0 {
                    (-z).exp()
                } else if z < -18.0 {
                    -z
                } else {
                    (1.0 + (-z).exp()).ln()
                }
            }
            LossType::ModifiedHuber => {
                if z >= 1.0 {
                    0.0
                } else if z >= -1.0 {
                    0.5 * (1.0 - z) * (1.0 - z)
                } else {
                    -(1.0 + z)
                }
            }
        }
    }

    /// Compute subgradient with respect to the decision function f(x)
    /// Returns d(loss)/d(f), to be multiplied by x when updating weights
    fn dloss(&self, y: f64, f: f64) -> f64 {
        let z = y * f;
        match self {
            LossType::Hinge => {
                if z < 1.0 { -y } else { 0.0 }
            }
            LossType::Log => {
                // d/dz log(1+exp(-z)) = -exp(-z) / (1+exp(-z)) = -sigmoid(-z)
                // chain rule: d/df = d/dz * dz/df = d/dz * y
                if z > 18.0 {
                    -y * (-z).exp()
                } else {
                    -y / (1.0 + z.exp())
                }
            }
            LossType::ModifiedHuber => {
                if z >= 1.0 {
                    0.0
                } else if z >= -1.0 {
                    y * (z - 1.0) // derivative of 0.5*(1-z)^2 w.r.t. f, times chain rule dz/df=y
                } else {
                    -y
                }
            }
        }
    }
}

/// SGD Classifier - Stochastic Gradient Descent for classification.
///
/// Supports multiple loss functions:
/// - "hinge": Linear SVM (hinge loss)
/// - "log": Logistic regression (log loss)
/// - "modified_huber": Smooth hinge loss (quadratic near zero, linear far away)
///
/// Uses L2 regularization via weight decay and supports two learning rate schedules:
/// - "constant": fixed learning rate eta = eta0
/// - "inverse": eta = eta0 / (1 + eta0 * alpha * t)
#[wasm_bindgen]
pub struct SgdClassifier {
    weights: Vec<f64>,
    bias: f64,
    n_features: usize,
    loss_type: LossType,
    n_iter: usize,
    classes: Vec<f64>,
}

#[wasm_bindgen]
impl SgdClassifier {
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize { self.n_features }

    #[wasm_bindgen(getter, js_name = "nIter")]
    pub fn n_iter(&self) -> usize { self.n_iter }

    #[wasm_bindgen(getter, js_name = "lossType")]
    pub fn loss_type(&self) -> String { self.loss_type.as_str().to_string() }

    /// Predict class labels (0.0 or 1.0) for each sample
    #[wasm_bindgen]
    pub fn predict(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len() / self.n_features;
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let mut sum = self.bias;
            for f in 0..self.n_features {
                sum += self.weights[f] * data[i * self.n_features + f];
            }
            // Classes are stored as [0.0, 1.0] for binary classification
            result.push(if sum > 0.0 { self.classes[1] } else { self.classes[0] });
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
            for f in 0..self.n_features {
                sum += self.weights[f] * data[i * self.n_features + f];
            }
            result.push(sum);
        }

        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!(
            "SgdClassifier(loss={}, n_features={}, n_iter={})",
            self.loss_type.as_str(),
            self.n_features,
            self.n_iter
        )
    }
}

/// Internal SGD Classifier implementation (pure Rust, no WASM bindings)
///
/// # Arguments
/// * `data` - Flat row-major feature matrix (n_samples * n_features)
/// * `n_features` - Number of features per sample
/// * `labels` - Target labels (0.0 or 1.0 for binary classification)
/// * `loss` - Loss function: "hinge", "log", or "modified_huber"
/// * `max_iter` - Maximum number of passes over the training data (epochs)
/// * `eta0` - Initial learning rate
/// * `alpha` - L2 regularization strength (weight decay)
/// * `learning_rate` - Learning rate schedule: "constant" or "inverse"
pub fn sgd_classifier_impl(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    loss: &str,
    max_iter: usize,
    eta0: f64,
    alpha: f64,
    learning_rate: &str,
) -> Result<SgdClassifier, MlError> {
    if eta0 <= 0.0 {
        return Err(MlError::new("eta0 (initial learning rate) must be > 0"));
    }
    if alpha < 0.0 {
        return Err(MlError::new("alpha (regularization strength) must be >= 0"));
    }
    if max_iter == 0 {
        return Err(MlError::new("max_iter must be > 0"));
    }
    if !matches!(learning_rate, "constant" | "inverse") {
        return Err(MlError::new(
            "learning_rate must be 'constant' or 'inverse'",
        ));
    }

    let loss_type = LossType::from_str(loss)?;
    let n = validate_matrix(data, n_features)?;
    if labels.len() != n {
        return Err(MlError::new("labels length must match number of samples"));
    }

    // Determine unique classes (binary: expect exactly 2 distinct values)
    let mut classes: Vec<f64> = labels.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup();
    if classes.len() != 2 {
        return Err(MlError::new(format!(
            "expected exactly 2 classes for binary classification, found {}",
            classes.len()
        )));
    }

    // Map labels to -1/+1 for internal computation
    let _class_neg = classes[0];
    let class_pos = classes[1];
    let y: Vec<f64> = labels
        .iter()
        .map(|&l| if l == class_pos { 1.0 } else { -1.0 })
        .collect();

    // Initialize weights to zero
    let mut weights = vec![0.0f64; n_features];
    let mut bias = 0.0;

    // Create deterministic RNG for shuffling
    let mut rng = Rng::from_data(data);

    // SGD training loop
    let mut t = 0usize; // global step counter for learning rate schedule
    for _epoch in 0..max_iter {
        // Shuffle training indices each epoch
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.next_usize(i + 1);
            indices.swap(i, j);
        }

        for idx in &indices {
            t += 1;

            // Compute current learning rate
            let eta = match learning_rate {
                "constant" => eta0,
                "inverse" => eta0 / (1.0 + eta0 * alpha * t as f64),
                _ => unreachable!(),
            };

            // Compute decision value f(x) = w . x + b
            let mut f_val = bias;
            for f in 0..n_features {
                f_val += weights[f] * mat_get(data, n_features, *idx, f);
            }

            // Compute subgradient of loss w.r.t. f(x)
            let dloss = loss_type.dloss(y[*idx], f_val);

            // Update weights: w -= eta * (dloss * x + alpha * w)
            // Simplified: w *= (1 - eta * alpha), then w -= eta * dloss * x
            let decay = 1.0 - eta * alpha;
            for f in 0..n_features {
                weights[f] = weights[f] * decay - eta * dloss * mat_get(data, n_features, *idx, f);
            }
            bias -= eta * dloss;
        }
    }

    Ok(SgdClassifier {
        weights,
        bias,
        n_features,
        loss_type,
        n_iter: max_iter,
        classes,
    })
}

/// WASM-exported SGD Classifier constructor
///
/// # Arguments
/// * `data` - Flat row-major feature matrix (n_samples * n_features)
/// * `n_features` - Number of features per sample
/// * `labels` - Target labels (0.0 or 1.0 for binary classification)
/// * `loss` - Loss function: "hinge", "log", or "modified_huber"
/// * `max_iter` - Maximum number of passes over the training data (epochs)
/// * `eta0` - Initial learning rate
/// * `alpha` - L2 regularization strength (weight decay)
/// * `learning_rate` - Learning rate schedule: "constant" or "inverse"
#[wasm_bindgen(js_name = "sgdClassifier")]
pub fn sgd_classifier(
    data: &[f64],
    n_features: usize,
    labels: &[f64],
    loss: &str,
    max_iter: usize,
    eta0: f64,
    alpha: f64,
    learning_rate: &str,
) -> Result<SgdClassifier, JsError> {
    sgd_classifier_impl(data, n_features, labels, loss, max_iter, eta0, alpha, learning_rate)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linearly_separable() {
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

        let model = sgd_classifier_impl(&data, 2, &labels, "hinge", 100, 0.01, 0.0001, "constant").unwrap();
        let preds = model.predict(&data);

        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 0.0);
        assert_eq!(preds[3], 1.0);
        assert_eq!(preds[4], 1.0);
        assert_eq!(preds[5], 1.0);
    }

    #[test]
    fn test_hinge_loss() {
        // SVM-style classification with hinge loss
        let data = vec![
            -1.0, -1.0,
            -1.0,  1.0,
            -0.5,  0.0,
             1.0,  1.0,
             1.0, -1.0,
             0.5,  0.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let model = sgd_classifier_impl(&data, 2, &labels, "hinge", 200, 0.05, 0.0001, "constant").unwrap();
        let preds = model.predict(&data);

        // All class-0 samples should be predicted as 0
        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 0.0);
        // All class-1 samples should be predicted as 1
        assert_eq!(preds[3], 1.0);
        assert_eq!(preds[4], 1.0);
        assert_eq!(preds[5], 1.0);
    }

    #[test]
    fn test_log_loss() {
        // Logistic regression style with log loss
        let data = vec![
            0.0, 0.0,
            0.5, 0.5,
            1.0, 0.0,
            5.0, 5.0,
            5.5, 5.5,
            6.0, 5.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let model = sgd_classifier_impl(&data, 2, &labels, "log", 200, 0.05, 0.0001, "constant").unwrap();
        let preds = model.predict(&data);

        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 0.0);
        assert_eq!(preds[3], 1.0);
        assert_eq!(preds[4], 1.0);
        assert_eq!(preds[5], 1.0);

        // Verify loss type getter
        assert_eq!(model.loss_type(), "log");
    }

    #[test]
    fn test_convergence() {
        // More iterations should yield better (or equal) accuracy
        let data = vec![
            0.0, 0.0,
            0.2, 0.1,
            1.0, 0.0,
            3.0, 3.0,
            3.2, 3.1,
            4.0, 3.0,
        ];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let model_few = sgd_classifier_impl(&data, 2, &labels, "log", 10, 0.1, 0.0001, "constant").unwrap();
        let model_many = sgd_classifier_impl(&data, 2, &labels, "log", 500, 0.1, 0.0001, "constant").unwrap();

        let preds_few = model_few.predict(&data);
        let preds_many = model_many.predict(&data);

        // Count correct predictions for each
        let correct_few = preds_few.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();
        let correct_many = preds_many.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();

        // More iterations should not do worse
        assert!(correct_many >= correct_few);

        // With enough iterations, should get all correct on separable data
        assert_eq!(correct_many, labels.len());
    }

    #[test]
    fn test_modified_huber_loss() {
        // Modified Huber should also work on separable data
        let data = vec![
            0.0, 0.0,
            10.0, 10.0,
        ];
        let labels = vec![0.0, 1.0];

        let model = sgd_classifier_impl(&data, 2, &labels, "modified_huber", 200, 0.05, 0.0001, "constant").unwrap();
        let preds = model.predict(&data);

        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 1.0);
    }

    #[test]
    fn test_inverse_learning_rate() {
        // Inverse scaling learning rate schedule should converge
        let data = vec![
            0.0, 0.0,
            1.0, 1.0,
            10.0, 10.0,
        ];
        let labels = vec![0.0, 0.0, 1.0];

        let model = sgd_classifier_impl(&data, 2, &labels, "hinge", 100, 0.1, 0.001, "inverse").unwrap();
        let preds = model.predict(&data);

        assert_eq!(preds[0], 0.0);
        assert_eq!(preds[1], 0.0);
        assert_eq!(preds[2], 1.0);
    }

    #[test]
    fn test_decision_function() {
        let data = vec![
            0.0, 0.0,
            10.0, 10.0,
        ];
        let labels = vec![0.0, 1.0];

        let model = sgd_classifier_impl(&data, 2, &labels, "hinge", 200, 0.05, 0.0001, "constant").unwrap();
        let scores = model.decision_function(&data);

        // Class 0 sample should have negative score
        assert!(scores[0] < 0.0);
        // Class 1 sample should have positive score
        assert!(scores[1] > 0.0);
    }

    #[test]
    fn test_invalid_loss_type() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];

        let result = sgd_classifier_impl(&data, 2, &labels, "unknown_loss", 100, 0.01, 0.0001, "constant");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_learning_rate() {
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![0.0, 1.0];

        let result = sgd_classifier_impl(&data, 2, &labels, "hinge", 100, 0.01, 0.0001, "bad_schedule");
        assert!(result.is_err());
    }

    #[test]
    fn test_getters() {
        let data = vec![0.0, 0.0, 10.0, 10.0];
        let labels = vec![0.0, 1.0];

        let model = sgd_classifier_impl(&data, 2, &labels, "hinge", 50, 0.01, 0.0001, "constant").unwrap();

        assert_eq!(model.n_features(), 2);
        assert_eq!(model.n_iter(), 50);
        assert_eq!(model.loss_type(), "hinge");
    }
}
