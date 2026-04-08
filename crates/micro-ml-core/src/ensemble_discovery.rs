/// Ensemble Discovery and Quality-Weighted Prediction
///
/// Run multiple algorithms, rank by quality, and combine predictions:
/// - Ensemble discovery (find best model combination)
/// - Consensus scoring (agreement between models)
/// - Quality-weighted prediction (weight by model quality)
/// - Pruned ensemble (remove low-quality models)
///
/// Use cases: Model comparison, robustness through diversity, automated model selection
use wasm_bindgen::prelude::*;
use crate::error::MlError;

/// Ensemble Member with quality score
#[wasm_bindgen]
pub struct EnsembleMember {
    predictions: Vec<f64>,
    quality: f64,
    weight: f64,
}

#[wasm_bindgen]
impl EnsembleMember {
    #[wasm_bindgen(getter, js_name = "quality")]
    pub fn quality(&self) -> f64 { self.quality }

    #[wasm_bindgen(getter, js_name = "weight")]
    pub fn weight(&self) -> f64 { self.weight }

    #[wasm_bindgen(js_name = "getPredictions")]
    pub fn get_predictions(&self) -> Vec<f64> {
        self.predictions.clone()
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("EnsembleMember(quality={:.4}, weight={:.4}, n_predictions={})",
                self.quality, self.weight, self.predictions.len())
    }
}

/// Ensemble Discovery Result
///
/// Contains ranked ensemble members and consensus scores.
#[wasm_bindgen]
pub struct EnsembleDiscovery {
    members: Vec<EnsembleMember>,
    consensus_scores: Vec<f64>,
}

#[wasm_bindgen]
impl EnsembleDiscovery {
    #[wasm_bindgen(getter, js_name = "nMembers")]
    pub fn n_members(&self) -> usize { self.members.len() }

    #[wasm_bindgen(getter, js_name = "consensusScores")]
    pub fn consensus_scores(&self) -> Vec<f64> {
        self.consensus_scores.clone()
    }

    /// Get weighted prediction (quality-weighted average)
    #[wasm_bindgen(js_name = "weightedPrediction")]
    pub fn weighted_prediction(&self) -> Vec<f64> {
        if self.members.is_empty() {
            return Vec::new();
        }

        let n_predictions = self.members[0].predictions.len();
        let mut result = vec![0.0; n_predictions];

        for member in &self.members {
            let weight = member.weight;
            for (i, &pred) in member.predictions.iter().enumerate() {
                result[i] += weight * pred;
            }
        }

        result
    }

    /// Get majority vote prediction
    #[wasm_bindgen(js_name = "majorityVote")]
    pub fn majority_vote(&self) -> Vec<f64> {
        if self.members.is_empty() {
            return Vec::new();
        }

        let n_predictions = self.members[0].predictions.len();
        let mut result = Vec::new();

        for i in 0..n_predictions {
            // Count occurrences of each prediction value using a vector
            let mut unique_preds: Vec<f64> = Vec::new();
            let mut counts: Vec<usize> = Vec::new();

            for member in &self.members {
                let pred = member.predictions[i];
                // Find if this prediction already exists
                if let Some(pos) = unique_preds.iter().position(|&x| (x - pred).abs() < 1e-9) {
                    counts[pos] += 1;
                } else {
                    unique_preds.push(pred);
                    counts.push(1);
                }
            }

            // Find most common prediction
            let max_idx = counts.iter().enumerate()
                .max_by_key(|(_, &count)| count)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            result.push(unique_preds[max_idx]);
        }

        result
    }

    /// Prune ensemble to top-k members
    #[wasm_bindgen(js_name = "prune")]
    pub fn prune(&mut self, k: usize) {
        self.members.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap());
        self.members.truncate(k);

        // Renormalize weights
        let total_weight: f64 = self.members.iter().map(|m| m.weight).sum();
        if total_weight > 0.0 {
            for member in &mut self.members {
                member.weight /= total_weight;
            }
        }
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("EnsembleDiscovery(n_members={}, avg_quality={:.4})",
                self.members.len(),
                self.members.iter().map(|m| m.quality).sum::<f64>() / self.members.len() as f64)
    }
}

/// Discover ensemble from multiple model predictions
///
/// # Parameters
/// - `predictions`: Flat array [n_models * n_predictions]
/// - `n_models`: Number of models/ensemble members
/// - `n_predictions`: Number of predictions per model
/// - `qualities`: Quality scores for each model (e.g., R², accuracy)
///
/// # Returns
/// EnsembleDiscovery with ranked members
#[wasm_bindgen(js_name = "ensembleDiscovery")]
pub fn ensemble_discovery(
    predictions: &[f64],
    n_models: usize,
    n_predictions: usize,
    qualities: &[f64],
) -> Result<EnsembleDiscovery, JsError> {
    ensemble_discovery_impl(predictions, n_models, n_predictions, qualities)
        .map_err(|e| JsError::new(&e.message))
}

pub fn ensemble_discovery_impl(
    predictions: &[f64],
    n_models: usize,
    n_predictions: usize,
    qualities: &[f64],
) -> Result<EnsembleDiscovery, MlError> {
    if predictions.is_empty() {
        return Err(MlError::new("predictions cannot be empty"));
    }
    if n_models == 0 {
        return Err(MlError::new("n_models must be > 0"));
    }
    if qualities.len() != n_models {
        return Err(MlError::new("qualities length must equal n_models"));
    }

    // Create ensemble members
    let mut members = Vec::new();
    for i in 0..n_models {
        let start = i * n_predictions;
        let end = start + n_predictions;
        let member_predictions = predictions[start..end].to_vec();

        members.push(EnsembleMember {
            predictions: member_predictions,
            quality: qualities[i],
            weight: qualities[i], // Initial weight = quality
        });
    }

    // Sort by quality (descending)
    members.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap());

    // Normalize weights to sum to 1
    let total_quality: f64 = members.iter().map(|m| m.weight).sum();
    if total_quality > 0.0 {
        for member in &mut members {
            member.weight /= total_quality;
        }
    }

    // Compute consensus scores (agreement between top members)
    let mut consensus_scores = Vec::new();
    let n_consensus = (n_models + 1) / 2; // Majority

    for i in 0..n_predictions {
        let mut agreement = 0.0;
        for j in 0..n_consensus {
            for k in (j + 1)..n_models {
                let diff = (members[j].predictions[i] - members[k].predictions[i]).abs();
                if diff < 1e-6 {
                    agreement += 1.0;
                }
            }
        }
        let max_agreement = (n_consensus * (n_consensus - 1) / 2) as f64;
        let consensus = if max_agreement > 0.0 {
            agreement / max_agreement
        } else {
            0.0
        };
        consensus_scores.push(consensus);
    }

    Ok(EnsembleDiscovery {
        members,
        consensus_scores,
    })
}

/// Compute consensus scores between predictions
///
/// Measures agreement between multiple prediction vectors.
#[wasm_bindgen(js_name = "consensusScores")]
pub fn consensus_scores(
    predictions: &[f64],
    n_models: usize,
    n_predictions: usize,
) -> Result<Vec<f64>, JsError> {
    if n_models < 2 {
        return Err(JsError::new("need at least 2 models for consensus"));
    }

    let mut scores = Vec::new();

    for i in 0..n_predictions {
        let mut agreements = 0;
        let mut total_comparisons = 0;

        for j in 0..n_models {
            for k in (j + 1)..n_models {
                let pred_j = predictions[j * n_predictions + i];
                let pred_k = predictions[k * n_predictions + i];
                total_comparisons += 1;

                if (pred_j - pred_k).abs() < 1e-6 {
                    agreements += 1;
                }
            }
        }

        let score = if total_comparisons > 0 {
            agreements as f64 / total_comparisons as f64
        } else {
            0.0
        };
        scores.push(score);
    }

    Ok(scores)
}

/// Quality-weighted prediction (direct function)
///
/// Weight predictions by model quality scores.
#[wasm_bindgen(js_name = "qualityWeightedPrediction")]
pub fn quality_weighted_prediction(
    predictions: &[f64],
    n_models: usize,
    n_predictions: usize,
    qualities: &[f64],
) -> Result<Vec<f64>, JsError> {
    let discovery = ensemble_discovery_impl(predictions, n_models, n_predictions, qualities)
        .map_err(|e| JsError::new(&e.message))?;

    Ok(discovery.weighted_prediction())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_discovery_basic() {
        let predictions = vec![
            1.0, 2.0, 3.0,  // Model 1
            1.1, 2.1, 3.1,  // Model 2 (better quality)
            0.9, 2.0, 3.0,  // Model 3 (worse quality)
        ];
        let qualities = vec![0.8, 0.9, 0.7];

        let result = ensemble_discovery_impl(&predictions, 3, 3, &qualities).unwrap();

        assert_eq!(result.n_members(), 3);
        assert_eq!(result.members[0].quality, 0.9); // Best model first
    }

    #[test]
    fn test_weighted_prediction() {
        let predictions = vec![
            10.0, 20.0, 30.0,
            12.0, 22.0, 32.0,
        ];
        let qualities = vec![0.6, 0.4];
        let n_models = 2;
        let n_predictions = 3;

        let total_quality: f64 = qualities.iter().sum();
        let mut expected = Vec::new();
        for i in 0..n_predictions {
            let weighted = (predictions[i] * qualities[0] + predictions[i + n_predictions] * qualities[1]) / total_quality;
            expected.push(weighted);
        }

        let result = quality_weighted_prediction(&predictions, n_models, n_predictions, &qualities).unwrap();

        assert_eq!(result.len(), 3);
        for i in 0..3 {
            assert!((result[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_consensus_scores() {
        let predictions = vec![
            1.0, 1.0, 1.0,  // Model 1 (agrees with model 2)
            1.0, 2.0, 1.0,  // Model 2
            2.0, 2.0, 2.0,  // Model 3 (disagrees)
        ];

        let scores = consensus_scores(&predictions, 3, 3).unwrap();

        assert_eq!(scores.len(), 3);
        // Position 0: models 1&2 agree on 1.0, model 3 has 2.0 (1/3 = 0.333 agreements)
        // Position 1: model 1 has 1.0, models 2&3 agree on 2.0 (1/3 = 0.333 agreements)
        // Position 2: models 1&2 agree on 1.0, model 3 has 2.0 (1/3 = 0.333 agreements)
        assert!(scores[0] > 0.0); // Some agreement at position 0
        assert!(scores[1] > 0.0); // Some agreement at position 1
    }

    #[test]
    fn test_prune_ensemble() {
        let predictions = vec![
            1.0, 2.0, 3.0,
            1.1, 2.1, 3.1,
            0.9, 2.0, 3.0,
        ];
        let qualities = vec![0.8, 0.9, 0.7];

        let mut result = ensemble_discovery_impl(&predictions, 3, 3, &qualities).unwrap();
        assert_eq!(result.n_members(), 3);

        result.prune(2);
        assert_eq!(result.n_members(), 2);
        assert_eq!(result.members[0].quality, 0.9);
        assert_eq!(result.members[1].quality, 0.8);
    }

    #[test]
    fn test_majority_vote() {
        let predictions = vec![
            1.0, 2.0, 3.0,  // Model 1
            1.0, 3.0, 3.0,  // Model 2
            2.0, 3.0, 4.0,  // Model 3
        ];
        let qualities = vec![1.0, 1.0, 1.0];

        let result = ensemble_discovery_impl(&predictions, 3, 3, &qualities).unwrap();
        let vote = result.majority_vote();

        assert_eq!(vote[0], 1.0); // Majority: 1
        assert_eq!(vote[1], 3.0); // Majority: 3
        assert_eq!(vote[2], 3.0); // Majority: 3
    }

    #[test]
    fn test_empty_predictions() {
        let result = ensemble_discovery_impl(&[], 1, 1, &[]);
        assert!(result.is_err());
    }
}
