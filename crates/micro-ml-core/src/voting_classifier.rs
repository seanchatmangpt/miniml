use wasm_bindgen::prelude::*;
use crate::error::MlError;

/// Voting Classifier - combines predictions from multiple pre-trained models.
///
/// Since WASM cannot pass heterogeneous model objects, the VotingClassifier
/// stores pre-computed predictions from each model. The user predicts with
/// each model separately, then passes all predictions to the voting classifier
/// for aggregation.
///
/// Supports two voting strategies:
/// - "hard": majority vote per sample (mode of class predictions)
/// - "soft": weighted average of probabilities per class, pick highest
#[wasm_bindgen]
pub struct VotingClassifier {
    n_samples: usize,
    n_classes: usize,
    predictions: Vec<Vec<f64>>,
    weights: Vec<f64>,
    voting_type: String,
    classes: Vec<f64>,
}

#[wasm_bindgen]
impl VotingClassifier {
    #[wasm_bindgen(getter, js_name = "nModels")]
    pub fn n_models(&self) -> usize { self.predictions.len() }

    #[wasm_bindgen(getter, js_name = "votingType")]
    pub fn voting_type(&self) -> String { self.voting_type.clone() }

    #[wasm_bindgen(getter, js_name = "nClasses")]
    pub fn n_classes(&self) -> usize { self.n_classes }

    /// Returns the unique class labels discovered during construction.
    #[wasm_bindgen(js_name = "getClasses")]
    pub fn get_classes(&self) -> Vec<f64> { self.classes.clone() }

    /// Aggregate predictions from all models into final predictions.
    ///
    /// - "hard" voting: for each sample, count votes per class and pick the class
    ///   with the most votes. Ties broken by lowest class label.
    /// - "soft" voting: for each sample, average the probability vectors across
    ///   all models (weighted if weights provided), then pick the class with
    ///   the highest average probability.
    #[wasm_bindgen]
    pub fn aggregate(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.n_samples);

        for i in 0..self.n_samples {
            if self.voting_type == "soft" {
                result.push(self.aggregate_soft(i));
            } else {
                result.push(self.aggregate_hard(i));
            }
        }

        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("VotingClassifier(models={}, samples={}, classes={}, voting={})",
                self.predictions.len(), self.n_samples, self.n_classes, self.voting_type)
    }
}

impl VotingClassifier {
    /// Hard voting: count votes per class per sample, pick class with most votes.
    /// Ties broken by lowest class label value.
    fn aggregate_hard(&self, sample_idx: usize) -> f64 {
        let mut vote_counts = vec![0.0f64; self.n_classes];

        for (model_idx, preds) in self.predictions.iter().enumerate() {
            let weight = if self.weights.is_empty() { 1.0 } else { self.weights[model_idx] };
            let pred = preds[sample_idx];
            // Find the class index for this prediction
            if let Some(class_idx) = self.classes.iter().position(|&c| (c - pred).abs() < 1e-9) {
                vote_counts[class_idx] += weight;
            }
        }

        // Find class with most votes, break ties by lowest class label
        let mut best_class_idx = 0;
        let mut best_count = vote_counts[0];
        for idx in 1..self.n_classes {
            if vote_counts[idx] > best_count {
                best_count = vote_counts[idx];
                best_class_idx = idx;
            }
        }

        self.classes[best_class_idx]
    }

    /// Soft voting: weighted average of probability vectors, pick class with highest avg.
    /// Each model's predictions for a sample should be a probability vector of length n_classes,
    /// stored contiguously. The `predictions` field is expected to be pre-shaped so that
    /// preds[sample_idx] accesses the correct probability slice.
    fn aggregate_soft(&self, sample_idx: usize) -> f64 {
        let mut avg_probs = vec![0.0f64; self.n_classes];
        let mut total_weight = 0.0;

        for (model_idx, preds) in self.predictions.iter().enumerate() {
            let weight = if self.weights.is_empty() { 1.0 } else { self.weights[model_idx] };
            total_weight += weight;

            // In soft voting, each model's predictions for this sample
            // are assumed to be a probability vector of length n_classes.
            // We index into the flat array at offset sample_idx * n_classes.
            for c in 0..self.n_classes {
                let prob_idx = sample_idx * self.n_classes + c;
                if prob_idx < preds.len() {
                    avg_probs[c] += weight * preds[prob_idx];
                }
            }
        }

        if total_weight > 0.0 {
            for prob in &mut avg_probs {
                *prob /= total_weight;
            }
        }

        // Pick class with highest average probability
        let mut best_class_idx = 0;
        let mut best_prob = avg_probs[0];
        for idx in 1..self.n_classes {
            if avg_probs[idx] > best_prob {
                best_prob = avg_probs[idx];
                best_class_idx = idx;
            }
        }

        self.classes[best_class_idx]
    }
}

/// Build a VotingClassifier from pre-computed predictions.
///
/// # Arguments
/// * `predictions` - Flat array of shape [n_models, n_samples] for hard voting,
///   or [n_models, n_samples * n_classes] for soft voting. Each model's predictions
///   are stored sequentially.
/// * `n_models` - Number of models whose predictions are included.
/// * `weights` - Optional per-model weights. Empty slice means uniform weights.
/// * `voting_type` - "hard" for majority vote, "soft" for weighted probability averaging.
/// * `n_classes` - Number of unique classes (required for soft voting to shape probabilities,
///   and for hard voting to resolve class indices).
pub fn voting_classifier_impl(
    predictions: &[f64],
    n_models: usize,
    weights: &[f64],
    voting_type: &str,
    n_classes: usize,
) -> Result<VotingClassifier, MlError> {
    if n_models == 0 {
        return Err(MlError::new("n_models must be > 0"));
    }
    if predictions.is_empty() {
        return Err(MlError::new("predictions must not be empty"));
    }
    if !weights.is_empty() && weights.len() != n_models {
        return Err(MlError::new("weights length must equal n_models"));
    }
    if voting_type != "hard" && voting_type != "soft" {
        return Err(MlError::new("voting_type must be 'hard' or 'soft'"));
    }
    if n_classes == 0 {
        return Err(MlError::new("n_classes must be > 0"));
    }

    // Determine n_samples based on voting type
    let preds_per_model = predictions.len() / n_models;
    let n_samples = if voting_type == "soft" {
        if !preds_per_model.is_multiple_of(n_classes) {
            return Err(MlError::new(
                "for soft voting, predictions per model must be divisible by n_classes"
            ));
        }
        preds_per_model / n_classes
    } else {
        preds_per_model
    };

    if n_samples == 0 {
        return Err(MlError::new("no samples found in predictions"));
    }

    // Split flat predictions into per-model vectors
    let model_preds: Vec<Vec<f64>> = (0..n_models)
        .map(|m| {
            let start = m * preds_per_model;
            predictions[start..start + preds_per_model].to_vec()
        })
        .collect();

    // Discover unique classes from hard voting predictions, or use 0..n_classes for soft
    let classes: Vec<f64> = if voting_type == "hard" {
        let mut unique: Vec<f64> = predictions.to_vec();
        unique.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup();
        if unique.len() != n_classes {
            return Err(MlError::new(format!(
                "declared n_classes ({}) does not match unique values found in predictions ({})",
                n_classes, unique.len()
            )));
        }
        unique
    } else {
        (0..n_classes).map(|i| i as f64).collect()
    };

    Ok(VotingClassifier {
        n_samples,
        n_classes,
        predictions: model_preds,
        weights: weights.to_vec(),
        voting_type: voting_type.to_string(),
        classes,
    })
}

#[wasm_bindgen(js_name = "votingClassifier")]
pub fn voting_classifier(
    predictions: &[f64],
    n_models: usize,
    weights: &[f64],
    voting_type: &str,
    n_classes: usize,
) -> Result<VotingClassifier, JsError> {
    voting_classifier_impl(predictions, n_models, weights, voting_type, n_classes)
        .map_err(|e| JsError::new(&e.message))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Hard voting tests --

    #[test]
    fn test_hard_voting_unanimous() {
        // 3 models, 2 samples, all models agree
        // Model 1: [0.0, 1.0], Model 2: [0.0, 1.0], Model 3: [0.0, 1.0]
        let predictions = vec![
            0.0, 1.0,  // model 1
            0.0, 1.0,  // model 2
            0.0, 1.0,  // model 3
        ];
        let vc = voting_classifier_impl(&predictions, 3, &[], "hard", 2).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![0.0, 1.0]);
    }

    #[test]
    fn test_hard_voting_majority() {
        // 3 models, 1 sample. Models vote: 0, 0, 1 -> majority is 0
        let predictions = vec![0.0, 0.0, 1.0];
        let vc = voting_classifier_impl(&predictions, 3, &[], "hard", 2).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn test_hard_voting_three_classes() {
        // 4 models, 1 sample. Votes: 0, 2, 2, 1 -> majority is 2
        // All 3 classes (0, 1, 2) present to satisfy n_classes validation
        let predictions = vec![0.0, 2.0, 2.0, 1.0];
        let vc = voting_classifier_impl(&predictions, 4, &[], "hard", 3).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![2.0]);
    }

    #[test]
    fn test_hard_voting_tie_breaks_lowest() {
        // 3 models, 1 sample. Votes: 1, 2, 0 -> class 1 and 2 tie at 1 vote each,
        // class 0 has 1 vote. Three-way tie, pick lowest (0).
        // All 3 classes (0, 1, 2) present to satisfy n_classes validation
        let predictions = vec![1.0, 2.0, 0.0];
        let vc = voting_classifier_impl(&predictions, 3, &[], "hard", 3).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![0.0]);
    }

    // -- Soft voting tests --

    #[test]
    fn test_soft_voting() {
        // 2 models, 1 sample, 2 classes
        // Model 1 probas: [0.9, 0.1], Model 2 probas: [0.7, 0.3]
        // Average: [0.8, 0.2] -> class 0 wins
        let predictions = vec![
            0.9, 0.1,  // model 1
            0.7, 0.3,  // model 2
        ];
        let vc = voting_classifier_impl(&predictions, 2, &[], "soft", 2).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn test_soft_voting_reversal() {
        // 2 models, 1 sample, 2 classes
        // Model 1 probas: [0.1, 0.9], Model 2 probas: [0.3, 0.7]
        // Average: [0.2, 0.8] -> class 1 wins
        let predictions = vec![
            0.1, 0.9,
            0.3, 0.7,
        ];
        let vc = voting_classifier_impl(&predictions, 2, &[], "soft", 2).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_soft_voting_three_classes() {
        // 2 models, 1 sample, 3 classes
        // Model 1: [0.1, 0.3, 0.6], Model 2: [0.2, 0.5, 0.3]
        // Average: [0.15, 0.4, 0.45] -> class 2 wins
        let predictions = vec![
            0.1, 0.3, 0.6,
            0.2, 0.5, 0.3,
        ];
        let vc = voting_classifier_impl(&predictions, 2, &[], "soft", 3).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![2.0]);
    }

    // -- Weighted voting tests --

    #[test]
    fn test_weighted_voting_hard() {
        // 3 models, 1 sample. Votes: 0, 0, 1 with weights [1, 1, 3]
        // Weighted counts: class 0 = 1+1 = 2, class 1 = 3 -> class 1 wins
        let predictions = vec![0.0, 0.0, 1.0];
        let weights = vec![1.0, 1.0, 3.0];
        let vc = voting_classifier_impl(&predictions, 3, &weights, "hard", 2).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_weighted_voting_soft() {
        // 2 models, 1 sample, 2 classes
        // Model 1 (weight 3): [0.1, 0.9], Model 2 (weight 1): [0.9, 0.1]
        // Weighted avg: class0 = (3*0.1 + 1*0.9)/4 = 1.2/4 = 0.3
        //               class1 = (3*0.9 + 1*0.1)/4 = 2.8/4 = 0.7
        // -> class 1 wins
        let predictions = vec![
            0.1, 0.9,
            0.9, 0.1,
        ];
        let weights = vec![3.0, 1.0];
        let vc = voting_classifier_impl(&predictions, 2, &weights, "soft", 2).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_weighted_voting_soft_flip() {
        // 2 models, 1 sample, 2 classes
        // Model 1 (weight 1): [0.1, 0.9], Model 2 (weight 3): [0.9, 0.1]
        // Weighted avg: class0 = (1*0.1 + 3*0.9)/4 = 2.8/4 = 0.7
        //               class1 = (1*0.9 + 3*0.1)/4 = 1.2/4 = 0.3
        // -> class 0 wins (reversed from unweighted)
        let predictions = vec![
            0.1, 0.9,
            0.9, 0.1,
        ];
        let weights = vec![1.0, 3.0];
        let vc = voting_classifier_impl(&predictions, 2, &weights, "soft", 2).unwrap();
        let result = vc.aggregate();
        assert_eq!(result, vec![0.0]);
    }

    // -- Error handling tests --

    #[test]
    fn test_zero_models_rejected() {
        let result = voting_classifier_impl(&[0.0, 1.0], 0, &[], "hard", 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_predictions_rejected() {
        let result = voting_classifier_impl(&[], 2, &[], "hard", 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_weights_rejected() {
        let result = voting_classifier_impl(&[0.0, 1.0, 0.0], 3, &[1.0, 2.0], "hard", 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_voting_type_rejected() {
        let result = voting_classifier_impl(&[0.0, 1.0], 1, &[], "ranked", 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_soft_voting_misaligned_n_classes_rejected() {
        // 2 models, predictions per model = 3, n_classes = 2 -> 3 % 2 != 0
        let predictions = vec![0.1, 0.9, 0.5, 0.8, 0.2, 0.6];
        let result = voting_classifier_impl(&predictions, 2, &[], "soft", 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_n_classes_mismatch_hard_rejected() {
        // Declared 3 classes but only 2 unique values present
        let predictions = vec![0.0, 1.0];
        let result = voting_classifier_impl(&predictions, 2, &[], "hard", 3);
        assert!(result.is_err());
    }

    // -- Getter tests --

    #[test]
    fn test_getters() {
        let predictions = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let vc = voting_classifier_impl(&predictions, 3, &[], "hard", 2).unwrap();
        assert_eq!(vc.n_models(), 3);
        assert_eq!(vc.voting_type(), "hard");
        assert_eq!(vc.n_classes(), 2);
        assert_eq!(vc.get_classes(), vec![0.0, 1.0]);
    }

    #[test]
    fn test_to_string() {
        let predictions = vec![0.0, 1.0, 0.0, 1.0];
        let vc = voting_classifier_impl(&predictions, 2, &[], "soft", 2).unwrap();
        let s = vc.to_string_js();
        assert!(s.contains("VotingClassifier"));
        assert!(s.contains("models=2"));
        assert!(s.contains("soft"));
    }
}
