/// NGram Sequence Prediction
///
/// Markov chain-based sequence prediction using n-grams.
/// Predicts the next item in a sequence based on historical context.
///
/// Use cases: Process mining (next activity prediction), text prediction,
/// recommendation systems, anomaly detection in sequences
use wasm_bindgen::prelude::*;
use crate::error::MlError;
use std::collections::HashMap;

/// NGram Model
///
/// Markov chain of order n for sequence prediction.
#[wasm_bindgen]
pub struct NGramModel {
    n: usize,              // Order of the n-gram (1 = unigram, 2 = bigram, etc.)
    vocab_size: usize,     // Number of unique items in vocabulary
    transition_counts: HashMap<Vec<usize>, HashMap<usize, usize>>, // (context) -> (next_item) -> count
    vocab_indices: HashMap<usize, usize>, // Item -> index mapping (for consistency)
    total_counts: HashMap<Vec<usize>, usize>, // (context) -> total count
}

#[wasm_bindgen]
impl NGramModel {
    #[wasm_bindgen(getter, js_name = "n")]
    pub fn n(&self) -> usize { self.n }

    #[wasm_bindgen(getter, js_name = "vocabSize")]
    pub fn vocab_size(&self) -> usize { self.vocab_size }

    /// Get the vocabulary (unique items)
    #[wasm_bindgen(js_name = "getVocabulary")]
    pub fn get_vocabulary(&self) -> Vec<usize> {
        self.vocab_indices.keys().copied().collect()
    }

    /// Predict next item(s) given a context
    /// Returns top-k items with probabilities
    #[wasm_bindgen(js_name = "predict")]
    pub fn predict(&self, context: &[usize], k: usize) -> Result<Vec<f64>, JsError> {
        if context.is_empty() {
            return Err(JsError::new("context cannot be empty"));
        }

        // Use last n-1 items as context
        let start = if context.len() >= self.n { context.len() - self.n + 1 } else { 0 };
        let context_key: Vec<usize> = context[start..].to_vec();

        match self.transition_counts.get(&context_key) {
            Some(transitions) => {
                let mut results: Vec<(usize, f64)> = transitions.iter()
                    .map(|(&item, &count)| (item, count as f64))
                    .collect();

                // Sort by count (descending)
                results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Normalize to probabilities
                let total = *self.total_counts.get(&context_key).unwrap_or(&1) as f64;
                results.iter_mut().for_each(|(_, prob)| *prob /= total);

                // Take top-k and flatten
                let top_k = results.iter().take(k).flat_map(|&(item, prob)| vec![item as f64, prob]).collect();
                Ok(top_k)
            }
            None => {
                // No data for this context - return empty
                Ok(Vec::new())
            }
        }
    }

    /// Get probability of a specific next item given context
    #[wasm_bindgen(js_name = "probability")]
    pub fn probability(&self, context: &[usize], next_item: usize) -> f64 {
        if context.is_empty() {
            return 0.0;
        }

        let start = if context.len() >= self.n { context.len() - self.n + 1 } else { 0 };
        let context_key: Vec<usize> = context[start..].to_vec();

        match self.transition_counts.get(&context_key) {
            Some(transitions) => {
                let count = *transitions.get(&next_item).unwrap_or(&0) as f64;
                let total = *self.total_counts.get(&context_key).unwrap_or(&1) as f64;
                if total > 0.0 { count / total } else { 0.0 }
            }
            None => 0.0
        }
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("NGramModel(n={}, vocab_size={}, transitions={})",
                self.n, self.vocab_size, self.transition_counts.len())
    }
}

/// Fit an NGram model to sequence data
///
/// # Parameters
/// - `sequences`: Flat array where each sequence is concatenated
/// - `sequence_lengths`: Length of each sequence (for splitting)
/// - `n`: Order of the n-gram (1 = unigram, 2 = bigram, etc.)
///
/// # Returns
/// Trained NGramModel
#[wasm_bindgen(js_name = "ngramFit")]
pub fn ngram_fit(
    sequences: &[usize],
    sequence_lengths: &[usize],
    n: usize,
) -> Result<NGramModel, JsError> {
    ngram_fit_impl(sequences, sequence_lengths, n)
        .map_err(|e| JsError::new(&e.message))
}

pub fn ngram_fit_impl(
    sequences: &[usize],
    sequence_lengths: &[usize],
    n: usize,
) -> Result<NGramModel, MlError> {
    if n == 0 {
        return Err(MlError::new("n must be >= 1"));
    }
    if sequences.is_empty() {
        return Err(MlError::new("sequences cannot be empty"));
    }
    if sequence_lengths.is_empty() {
        return Err(MlError::new("sequence_lengths cannot be empty"));
    }
    if sequences.len() != sequence_lengths.iter().sum::<usize>() {
        return Err(MlError::new("sequences length must equal sum of sequence_lengths"));
    }

    let mut vocab_indices: HashMap<usize, usize> = HashMap::new();
    let mut transition_counts: HashMap<Vec<usize>, HashMap<usize, usize>> = HashMap::new();
    let mut total_counts: HashMap<Vec<usize>, usize> = HashMap::new();

    // Build vocabulary and count transitions
    let mut offset = 0;
    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // Build vocabulary
        for &item in seq {
            let next_idx = vocab_indices.len();
            vocab_indices.entry(item).or_insert(next_idx);
        }

        // Count transitions
        for i in 0..seq.len().saturating_sub(n) {
            let context_start = i;
            let context_end = i + n - 1;
            let next_item = seq[context_end];

            // Extract context (n-1 items)
            let context: Vec<usize> = seq[context_start..context_end].to_vec();

            // Update transition counts
            *transition_counts.entry(context.clone()).or_insert_with(HashMap::new)
                .entry(next_item).or_insert(0) += 1;

            // Update total count for this context
            *total_counts.entry(context).or_insert(0) += 1;
        }
    }

    Ok(NGramModel {
        n,
        vocab_size: vocab_indices.len(),
        transition_counts,
        vocab_indices,
        total_counts,
    })
}

/// Batch prediction for multiple contexts
///
/// # Parameters
/// - `sequences`: Flat array of input sequences
/// - `sequence_lengths`: Length of each sequence
/// - `n`: NGram order
/// - `k`: Number of top predictions to return
///
/// # Returns
/// Flat array of predictions for each sequence
#[wasm_bindgen(js_name = "ngramPredictBatch")]
pub fn ngram_predict_batch(
    sequences: &[usize],
    sequence_lengths: &[usize],
    n: usize,
    k: usize,
) -> Result<Vec<f64>, JsError> {
    let model = ngram_fit_impl(sequences, sequence_lengths, n)
        .map_err(|e| JsError::new(&e.message))?;

    let mut all_predictions = Vec::new();
    let mut offset = 0;

    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // Use all but the last element as context (predict the last element)
        let context_len = if seq.len() > 0 { seq.len() - 1 } else { 0 };
        let context = &seq[..context_len];

        if !context.is_empty() {
            let predictions = model.predict(context, k)?;
            all_predictions.extend(predictions);
        }
    }

    Ok(all_predictions)
}

/// Laplace-smoothed probability (add-one smoothing)
///
/// Prevents zero probability for unseen transitions
#[wasm_bindgen(js_name = "ngramProbabilitySmooth")]
pub fn ngram_probability_smooth(
    sequences: &[usize],
    sequence_lengths: &[usize],
    n: usize,
    context: &[usize],
    next_item: usize,
) -> Result<f64, JsError> {
    let model = ngram_fit_impl(sequences, sequence_lengths, n)
        .map_err(|e| JsError::new(&e.message))?;

    let vocab_size = model.vocab_size();
    let _base_prob = model.probability(context, next_item);

    // Laplace smoothing: P = (count + 1) / (total + V)
    // where V is vocabulary size
    if context.is_empty() {
        return Ok(0.0);
    }

    let start = if context.len() >= n { context.len() - n + 1 } else { 0 };
    let context_key: Vec<usize> = context[start..].to_vec();

    let count = *model.transition_counts.get(&context_key)
        .and_then(|t| t.get(&next_item))
        .unwrap_or(&0) as f64;

    let total = *model.total_counts.get(&context_key).unwrap_or(&1) as f64;

    // Laplace smoothed probability
    let smoothed = (count + 1.0) / (total + vocab_size as f64);
    Ok(smoothed.min(1.0))
}

/// Compute perplexity of sequences under the model
///
/// Lower perplexity = better model fit
#[wasm_bindgen(js_name = "ngramPerplexity")]
pub fn ngram_perplexity(
    sequences: &[usize],
    sequence_lengths: &[usize],
    n: usize,
) -> Result<f64, JsError> {
    let model = ngram_fit_impl(sequences, sequence_lengths, n)
        .map_err(|e| JsError::new(&e.message))?;

    if model.vocab_size() == 0 {
        return Ok(f64::INFINITY);
    }

    let mut log_prob_sum = 0.0;
    let mut n_predictions = 0;

    let mut offset = 0;
    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // Compute probability for each position (after the first n-1 items)
        for i in (n - 1)..seq.len() {
            let context = &seq[0..i];
            let next_item = seq[i];

            // Use smoothed probability to avoid log(0)
            let vocab_size = model.vocab_size();

            let start = if context.len() >= n { context.len() - n + 1 } else { 0 };
            let context_key: Vec<usize> = context[start..].to_vec();

            let count = *model.transition_counts.get(&context_key)
                .and_then(|t| t.get(&next_item))
                .unwrap_or(&0) as f64;

            let total = *model.total_counts.get(&context_key).unwrap_or(&1) as f64;
            let prob = (count + 1.0) / (total + vocab_size as f64);

            log_prob_sum += prob.ln();
            n_predictions += 1;
        }
    }

    if n_predictions == 0 {
        return Ok(f64::INFINITY);
    }

    // Perplexity = exp(-1/N * sum(log(P)))
    Ok((-log_prob_sum / n_predictions as f64).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_fit_basic() {
        let sequences = vec![1, 2, 3, 1, 2, 4];
        let sequence_lengths = vec![6];
        let model = ngram_fit_impl(&sequences, &sequence_lengths, 2).unwrap();

        assert_eq!(model.n(), 2);
        assert!(model.vocab_size() >= 3);
    }

    #[test]
    fn test_ngram_fit_invalid_n() {
        let sequences = vec![1, 2, 3];
        let sequence_lengths = vec![3];
        let result = ngram_fit_impl(&sequences, &sequence_lengths, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ngram_predict() {
        let sequences = vec![1, 2, 3, 1, 2, 3, 1, 2, 4];
        let sequence_lengths = vec![9];
        let model = ngram_fit_impl(&sequences, &sequence_lengths, 2).unwrap();

        // Context [1, 2] should predict 3 or 4
        let predictions = model.predict(&[1, 2], 2).unwrap();
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_ngram_probability() {
        let sequences = vec![1, 2, 3, 1, 2, 3, 1, 2, 4];
        let sequence_lengths = vec![9];
        let model = ngram_fit_impl(&sequences, &sequence_lengths, 2).unwrap();

        let prob = model.probability(&[1, 2], 3);
        assert!(prob > 0.0);
    }

    #[test]
    fn test_ngram_probability_unseen() {
        let sequences = vec![1, 2, 3];
        let sequence_lengths = vec![3];
        let model = ngram_fit_impl(&sequences, &sequence_lengths, 2).unwrap();

        // Unseen transition
        let prob = model.probability(&[9, 9], 9);
        assert_eq!(prob, 0.0);
    }

    #[test]
    fn test_ngram_batch_prediction() {
        // Use longer sequences where last context has transitions
        let sequences = vec![1, 2, 3, 1, 2, 4, 1, 2, 5];
        let sequence_lengths = vec![9];
        let predictions = ngram_predict_batch(&sequences, &sequence_lengths, 2, 2).unwrap();

        // Should return predictions (context [2] predicts [3, 4, 5])
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_ngram_smoothed_probability() {
        let sequences = vec![1, 2, 3];
        let sequence_lengths = vec![3];
        let prob = ngram_probability_smooth(&sequences, &sequence_lengths, 2, &[1, 2], 99).unwrap();

        // Smoothed probability should be > 0 even for unseen item
        assert!(prob > 0.0);
    }

    #[test]
    fn test_ngram_perplexity() {
        let sequences = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        let sequence_lengths = vec![9];
        let perplexity = ngram_perplexity(&sequences, &sequence_lengths, 2).unwrap();

        // Perplexity should be finite and positive
        assert!(perplexity.is_finite());
        assert!(perplexity > 0.0);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_ngram_empty_context() {
        let sequences = vec![1, 2, 3];
        let sequence_lengths = vec![3];
        let model = ngram_fit_impl(&sequences, &sequence_lengths, 2).unwrap();

        let result = model.predict(&[], 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_ngram_multiple_sequences() {
        let sequences = vec![1, 2, 3, 4, 5, 6];
        let sequence_lengths = vec![3, 3];
        let model = ngram_fit_impl(&sequences, &sequence_lengths, 2).unwrap();

        // Should handle multiple sequences correctly
        assert_eq!(model.n(), 2);
    }
}
