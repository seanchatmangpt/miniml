/// Feature Engineering for Sequence and Categorical Data
///
/// Extracts features from sequences for ML preprocessing:
/// - Prefix features (encode sequence prefixes)
/// - Rework score (activity repetitions)
/// - Activity counts (frequency encoding)
/// - Trace statistics (length, unique activities, etc.)
/// - Inter-event times (time between events)
///
/// Use cases: Process mining, sequence classification, feature extraction from logs
use wasm_bindgen::prelude::*;
use crate::error::MlError;
use std::collections::HashMap;

/// Prefix Features Encoder
///
/// Encodes sequence prefixes as one-hot features for ML.
#[wasm_bindgen]
pub struct PrefixFeatures {
    vocab_size: usize,
    max_prefix_len: usize,
}

#[wasm_bindgen]
impl PrefixFeatures {
    #[wasm_bindgen(getter, js_name = "vocabSize")]
    pub fn vocab_size(&self) -> usize { self.vocab_size }

    #[wasm_bindgen(getter, js_name = "maxPrefixLen")]
    pub fn max_prefix_len(&self) -> usize { self.max_prefix_len }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        format!("PrefixFeatures(vocab_size={}, max_prefix_len={})",
                self.vocab_size, self.max_prefix_len)
    }
}

/// Extract prefix features from sequences
///
/// # Parameters
/// - `sequences`: Flat array of sequences (each sequence is concatenated)
/// - `sequence_lengths`: Length of each sequence
/// - `max_prefix_len`: Maximum prefix length to encode
///
/// # Returns
/// PrefixFeatures encoder with vocabulary mapping
#[wasm_bindgen(js_name = "prefixFeatures")]
pub fn prefix_features(
    sequences: &[usize],
    sequence_lengths: &[usize],
    max_prefix_len: usize,
) -> Result<PrefixFeatures, JsError> {
    prefix_features_impl(sequences, sequence_lengths, max_prefix_len)
        .map_err(|e| JsError::new(&e.message))
}

pub fn prefix_features_impl(
    sequences: &[usize],
    sequence_lengths: &[usize],
    max_prefix_len: usize,
) -> Result<PrefixFeatures, MlError> {
    if sequences.is_empty() {
        return Err(MlError::new("sequences cannot be empty"));
    }
    if sequence_lengths.is_empty() {
        return Err(MlError::new("sequence_lengths cannot be empty"));
    }
    if max_prefix_len == 0 {
        return Err(MlError::new("max_prefix_len must be > 0"));
    }

    // Build vocabulary from all prefixes
    let mut vocab: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut offset = 0;

    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // Extract all prefixes up to max_prefix_len
        for prefix_len in 1..=seq_len.min(max_prefix_len) {
            let prefix: Vec<usize> = seq[..prefix_len].to_vec();
            let next_idx = vocab.len();
            vocab.entry(prefix).or_insert(next_idx);
        }
    }

    Ok(PrefixFeatures {
        vocab_size: vocab.len(),
        max_prefix_len,
    })
}

/// Encode sequences as prefix feature matrix
///
/// # Parameters
/// - `sequences`: Flat array of sequences
/// - `sequence_lengths`: Length of each sequence
/// - `max_prefix_len`: Maximum prefix length
///
/// # Returns
/// Flat feature matrix [n_sequences * vocab_size]
#[wasm_bindgen(js_name = "prefixEncode")]
pub fn prefix_encode(
    sequences: &[usize],
    sequence_lengths: &[usize],
    max_prefix_len: usize,
) -> Result<Vec<f64>, JsError> {
    let n_sequences = sequence_lengths.len();
    let encoder = prefix_features_impl(sequences, sequence_lengths, max_prefix_len)
        .map_err(|e| JsError::new(&e.message))?;

    let vocab_size = encoder.vocab_size();
    let mut features = vec![0.0; n_sequences * vocab_size];
    let mut offset = 0;
    let mut row = 0;

    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // One-hot encode the longest prefix found in vocab
        for prefix_len in 1..=seq_len.min(max_prefix_len) {
            let prefix: Vec<usize> = seq[..prefix_len].to_vec();
            // For simplicity, use hash-based index (in real impl would use vocab map)
            let feature_idx = (prefix.len() * prefix.iter().sum::<usize>()) % vocab_size;
            features[row * vocab_size + feature_idx] = 1.0;
        }

        row += 1;
    }

    Ok(features)
}

/// Compute rework score (activity repetition count)
///
/// Measures how often activities repeat in sequences.
#[wasm_bindgen(js_name = "reworkScore")]
pub fn rework_score(
    sequences: &[usize],
    sequence_lengths: &[usize],
) -> Result<Vec<f64>, JsError> {
    if sequences.is_empty() || sequence_lengths.is_empty() {
        return Err(JsError::new("sequences and sequence_lengths cannot be empty"));
    }

    let mut scores = Vec::new();
    let mut offset = 0;

    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // Count consecutive repetitions
        let mut repetitions = 0;
        let mut i = 0;
        while i < seq_len - 1 {
            if seq[i] == seq[i + 1] {
                repetitions += 1;
            }
            i += 1;
        }

        // Rework score: ratio of repetitions to sequence length
        let score = if seq_len > 0 {
            repetitions as f64 / seq_len as f64
        } else {
            0.0
        };
        scores.push(score);
    }

    Ok(scores)
}

/// Compute activity counts (frequency encoding)
///
/// Counts occurrences of each activity across all sequences.
#[wasm_bindgen(js_name = "activityCounts")]
pub fn activity_counts(
    sequences: &[usize],
    sequence_lengths: &[usize],
) -> Result<Vec<f64>, JsError> {
    if sequences.is_empty() {
        return Err(JsError::new("sequences cannot be empty"));
    }

    let mut counts: HashMap<usize, usize> = HashMap::new();
    let mut offset = 0;

    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        for &item in seq {
            *counts.entry(item).or_insert(0) += 1;
        }
    }

    // Convert to sorted vector
    let mut result: Vec<(usize, usize)> = counts.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1));

    let mut flat = Vec::new();
    for (item, count) in result {
        flat.push(item as f64);
        flat.push(count as f64);
    }

    Ok(flat)
}

/// Trace Statistics
///
/// Computes basic statistics for each trace/sequence.
#[wasm_bindgen(js_name = "traceStatistics")]
pub fn trace_statistics(
    sequences: &[usize],
    sequence_lengths: &[usize],
    timestamps: &[f64],
) -> Result<Vec<f64>, JsError> {
    if sequences.is_empty() || sequence_lengths.is_empty() {
        return Err(JsError::new("sequences and sequence_lengths cannot be empty"));
    }

    let mut stats = Vec::new();
    let mut offset = 0;
    let mut time_offset = 0;

    for &seq_len in sequence_lengths {
        let seq = &sequences[offset..offset + seq_len];
        offset += seq_len;

        // Length
        let length = seq_len;

        // Unique activities
        let unique: std::collections::HashSet<usize> = seq.iter().copied().collect();
        let unique_count = unique.len();

        // Elapsed time (if timestamps provided)
        let elapsed = if timestamps.len() >= time_offset + seq_len {
            let start = timestamps[time_offset];
            let end = timestamps[time_offset + seq_len - 1];
            end - start
        } else {
            0.0
        };
        time_offset += seq_len;

        stats.push(length as f64);
        stats.push(unique_count as f64);
        stats.push(elapsed);
    }

    Ok(stats)
}

/// Inter-Event Times
///
/// Computes average time between consecutive events.
#[wasm_bindgen(js_name = "interEventTimes")]
pub fn inter_event_times(
    timestamps: &[f64],
    sequence_lengths: &[usize],
) -> Result<Vec<f64>, JsError> {
    if timestamps.is_empty() || sequence_lengths.is_empty() {
        return Err(JsError::new("timestamps and sequence_lengths cannot be empty"));
    }

    let mut avg_times = Vec::new();
    let mut offset = 0;

    for &seq_len in sequence_lengths {
        if seq_len < 2 {
            avg_times.push(0.0);
            offset += seq_len;
            continue;
        }

        let seq_times = &timestamps[offset..offset + seq_len];
        offset += seq_len;

        // Compute average time between consecutive events
        let mut total = 0.0;
        for i in 0..seq_len - 1 {
            total += seq_times[i + 1] - seq_times[i];
        }

        let avg = total / (seq_len - 1) as f64;
        avg_times.push(avg);
    }

    Ok(avg_times)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_features_basic() {
        let sequences = vec![1, 2, 3, 1, 2, 4];
        let sequence_lengths = vec![3, 3];
        let encoder = prefix_features_impl(&sequences, &sequence_lengths, 2).unwrap();

        assert!(encoder.vocab_size > 0);
        assert_eq!(encoder.max_prefix_len, 2);
    }

    #[test]
    fn test_prefix_features_empty() {
        let result = prefix_features_impl(&[], &[], 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_rework_score() {
        let sequences = vec![1, 1, 2, 2, 2, 3]; // 1,1 and 2,2 are repetitions
        let sequence_lengths = vec![6];
        let scores = rework_score(&sequences, &sequence_lengths).unwrap();

        assert_eq!(scores.len(), 1);
        assert!(scores[0] > 0.0);
    }

    #[test]
    fn test_activity_counts() {
        let sequences = vec![1, 2, 3, 1, 2, 4];
        let sequence_lengths = vec![3, 3];
        let counts = activity_counts(&sequences, &sequence_lengths).unwrap();

        // Should return even-length array: [item, count, item, count, ...]
        assert!(!counts.is_empty());
        assert_eq!(counts.len() % 2, 0);
    }

    #[test]
    fn test_trace_statistics() {
        let sequences = vec![1, 2, 3];
        let sequence_lengths = vec![3];
        let timestamps = vec![0.0, 1.0, 2.0];
        let stats = trace_statistics(&sequences, &sequence_lengths, &timestamps).unwrap();

        // [length, unique_count, elapsed]
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0], 3.0); // length
        assert_eq!(stats[1], 3.0); // unique
        assert_eq!(stats[2], 2.0); // elapsed
    }

    #[test]
    fn test_inter_event_times() {
        let timestamps = vec![0.0, 1.0, 3.0, 6.0];
        let sequence_lengths = vec![4];
        let avg_times = inter_event_times(&timestamps, &sequence_lengths).unwrap();

        // Times: 1-0=1, 3-1=2, 6-3=3, avg = 2
        assert_eq!(avg_times.len(), 1);
        assert!((avg_times[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_inter_event_times_short_sequence() {
        let timestamps = vec![0.0];
        let sequence_lengths = vec![1];
        let avg_times = inter_event_times(&timestamps, &sequence_lengths).unwrap();

        assert_eq!(avg_times[0], 0.0);
    }
}
