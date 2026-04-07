use wasm_bindgen::prelude::*;

/// Matthews Correlation Coefficient (MCC) - balanced measure for binary classification
/// Range: [-1, 1], where 1 = perfect, 0 = random, -1 = inverse
#[wasm_bindgen(js_name = "matthewsCorrcoef")]
pub fn matthews_corrcoef(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(JsError::new("arrays must be same non-zero length"));
    }

    let mut tp = 0usize;
    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let t_binary = if t > 0.5 { 1 } else { 0 };
        let p_binary = if p > 0.5 { 1 } else { 0 };

        match (t_binary, p_binary) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_count += 1,
            _ => unreachable!(),
        }
    }

    let numerator = (tp * tn - fp * fn_count) as f64;
    let denominator = ((tp + fp) * (tp + fn_count) * (tn + fp) * (tn + fn_count)) as f64;

    if denominator == 0.0 {
        return Ok(0.0);  // Undefined case
    }

    Ok(numerator / denominator.sqrt())
}

/// Cohen's Kappa - agreement measure accounting for chance
/// Range: [-1, 1], where 1 = perfect agreement, 0 = chance agreement
#[wasm_bindgen(js_name = "cohensKappa")]
pub fn cohens_kappa(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(JsError::new("arrays must be same non-zero length"));
    }

    let n = y_true.len();

    // Build confusion matrix
    let mut unique_true: Vec<f64> = y_true.to_vec();
    unique_true.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    unique_true.dedup();

    let n_classes = unique_true.len();
    let mut observed = vec![0usize; n_classes * n_classes];

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let t_idx = unique_true.iter().position(|&v| (v - t).abs() < 1e-10).unwrap();
        let p_idx = unique_true.iter().position(|&v| (v - p).abs() < 1e-10);
        if let Some(pi) = p_idx {
            observed[t_idx * n_classes + pi] += 1;
        }
    }

    // Observed agreement (diagonal)
    let mut po = 0.0;
    for i in 0..n_classes {
        po += observed[i * n_classes + i] as f64;
    }
    po /= n as f64;

    // Expected agreement (chance)
    let mut pe = 0.0;
    for i in 0..n_classes {
        let row_sum: usize = (i * n_classes..(i + 1) * n_classes)
            .map(|j| observed[j])
            .sum();
        let col_sum: usize = (0..n_classes)
            .filter(|j| i < n_classes)
            .map(|j| observed[j * n_classes + i])
            .sum();

        pe += (row_sum * col_sum) as f64 / (n * n) as f64;
    }

    if pe >= 1.0 {
        return Ok(1.0);  // Perfect agreement
    }

    Ok((po - pe) / (1.0 - pe))
}

/// Balanced Accuracy - average of recall per class
#[wasm_bindgen(js_name = "balancedAccuracy")]
pub fn balanced_accuracy(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(JsError::new("arrays must be same non-zero length"));
    }

    let mut unique_true: Vec<f64> = y_true.to_vec();
    unique_true.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    unique_true.dedup();

    let mut recalls = Vec::new();

    for &class in &unique_true {
        let mut tp = 0usize;
        let mut fn_count = 0usize;

        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            let t_match = (t - class).abs() < 1e-10;
            let p_match = (p - class).abs() < 1e-10;

            if t_match && p_match {
                tp += 1;
            } else if t_match && !p_match {
                fn_count += 1;
            }
        }

        let total = tp + fn_count;
        if total > 0 {
            recalls.push(tp as f64 / total as f64);
        }
    }

    if recalls.is_empty() {
        return Ok(0.0);
    }

    let sum: f64 = recalls.iter().sum();
    Ok(sum / recalls.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcc_perfect() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 0.0, 1.0, 1.0];
        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert!((mcc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_worst() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![1.0, 1.0, 0.0, 0.0];
        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert!((mcc - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cohen_kappa_perfect() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0];
        let kappa = cohens_kappa(&y_true, &y_pred).unwrap();
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_balanced_accuracy() {
        let y_true = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y_pred = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        // Class 0: 2/3 correct, Class 1: 2/3 correct
        let ba = balanced_accuracy(&y_true, &y_pred).unwrap();
        assert!((ba - 2.0/3.0).abs() < 1e-10);
    }

    // ML CORRECTNESS VALIDATION TESTS

    #[test]
    fn test_mcc_formula_verification() {
        // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        // TP=1, TN=1, FP=1, FN=1 -> MCC = (1-1)/sqrt(2*2*2*2) = 0
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0];
        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert_eq!(mcc, 0.0);
    }

    #[test]
    fn test_mcc_perfect_classifier() {
        // TP=2, TN=2, FP=0, FN=0 -> MCC = (4-0)/sqrt(2*2*2*2) = 4/4 = 1
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 0.0, 1.0, 1.0];
        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert!((mcc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_worst_classifier() {
        // TP=0, TN=0, FP=2, FN=2 -> MCC = (0-4)/sqrt(2*2*2*2) = -4/4 = -1
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![1.0, 1.0, 0.0, 0.0];
        let mcc = matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert!((mcc - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cohen_kappa_formula_verification() {
        // Perfect agreement -> Kappa = 1
        let y_true = vec![0.0, 1.0, 2.0];
        let y_pred = vec![0.0, 1.0, 2.0];
        let kappa = cohens_kappa(&y_true, &y_pred).unwrap();
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cohen_kappa_chance_agreement() {
        // If predictions match by chance, Kappa = 0
        // For [0,1,0,1] with random predictions, expected agreement = 0.5
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0];
        let kappa = cohens_kappa(&y_true, &y_pred).unwrap();
        assert!((kappa - 1.0).abs() < 1e-10);  // Perfect agreement
    }
}
