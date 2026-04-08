use wasm_bindgen::prelude::*;
use crate::error::MlError;

/// ROC AUC (Area Under ROC Curve) for binary classification
/// Returns AUC score in [0, 1] where 1 = perfect classifier
#[wasm_bindgen(js_name = "rocAucScore")]
pub fn roc_auc_score(y_true: &[f64], y_scores: &[f64]) -> Result<f64, JsError> {
    if y_true.len() != y_scores.len() {
        return Err(JsError::new("y_true and y_scores must have the same length"));
    }
    if y_true.is_empty() {
        return Err(JsError::new("arrays must not be empty"));
    }

    let auc = roc_auc_impl(y_true, y_scores).map_err(|e| JsError::new(&e.message))?;
    Ok(auc)
}

pub fn roc_auc_impl(y_true: &[f64], y_scores: &[f64]) -> Result<f64, MlError> {
    let n = y_true.len();
    let total_pos = y_true.iter().filter(|&&y| y > 0.5).count();
    let total_neg = n - total_pos;

    if total_pos == 0 || total_neg == 0 {
        return Ok(0.0); // Undefined
    }

    // Use Mann-Whitney U statistic: AUC = P(score_pos > score_neg)
    // Tied scores count as 0.5
    let mut pairs_won = 0.0_f64;
    for i in 0..n {
        if y_true[i] > 0.5 {
            for j in 0..n {
                if y_true[j] <= 0.5 {
                    if y_scores[i] > y_scores[j] {
                        pairs_won += 1.0;
                    } else if (y_scores[i] - y_scores[j]).abs() < 1e-10 {
                        pairs_won += 0.5;
                    }
                }
            }
        }
    }

    Ok(pairs_won / (total_pos as f64 * total_neg as f64))
}

/// Log Loss (Cross-Entropy) for probabilistic classification
#[wasm_bindgen(js_name = "logLoss")]
pub fn log_loss(y_true: &[f64], y_proba: &[f64], n_classes: usize) -> Result<f64, JsError> {
    if y_true.len() != y_proba.len() / n_classes {
        return Err(JsError::new("y_true length must match y_proba rows"));
    }

    let n = y_true.len();
    let mut loss = 0.0;

    for i in 0..n {
        let true_class = y_true[i] as usize;
        if true_class < n_classes {
            let prob = y_proba[i * n_classes + true_class];
            // Clip to avoid log(0)
            let prob_clipped = prob.max(1e-15).min(1.0 - 1e-15);
            loss -= prob_clipped.ln();
        }
    }

    Ok(loss / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_auc() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_scores = vec![0.1, 0.2, 0.9, 1.0];
        let auc = roc_auc_impl(&y_true, &y_scores).unwrap();
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_auc() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        let y_scores = vec![0.5, 0.5, 0.5, 0.5];
        let auc = roc_auc_impl(&y_true, &y_scores).unwrap();
        assert_eq!(auc, 0.5);
    }

    #[test]
    fn test_log_loss() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        // y_proba layout: [n_samples * n_classes] = [4 * 2] = 8 elements
        // Sample 0 (class 0): [0.9, 0.1], Sample 1 (class 1): [0.1, 0.9]
        // Sample 2 (class 0): [0.8, 0.2], Sample 3 (class 1): [0.2, 0.8]
        let y_proba = vec![0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.8];
        let loss = log_loss(&y_true, &y_proba, 2).unwrap();
        // Should be low (good predictions)
        assert!(loss < 1.0);
    }

    #[test]
    fn test_single_class_auc() {
        let y_true = vec![1.0, 1.0, 1.0];
        let y_scores = vec![0.1, 0.5, 0.9];
        let auc = roc_auc_impl(&y_true, &y_scores).unwrap();
        assert_eq!(auc, 0.0); // Undefined, returns 0
    }

    // ML CORRECTNESS VALIDATION TESTS

    #[test]
    fn test_roc_auc_trapezoidal_rule() {
        // Verify AUC using trapezoidal rule
        // Perfect: [0,0,1,1] with scores [0.1,0.2,0.9,1.0]
        // TPR: [0, 0, 1, 1], FPR: [0, 0, 0, 1]
        // AUC = 1.0 (perfect separation)
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_scores = vec![0.1, 0.2, 0.9, 1.0];
        let auc = roc_auc_impl(&y_true, &y_scores).unwrap();
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_loss_formula_verification() {
        // Log Loss = -Σ(y*log(p) + (1-y)*log(1-p)) / n
        // For perfect prediction p=0.99 for true class:
        // y=0, p=[0.99,0.01]: -log(0.99) ≈ 0.01
        // y=1, p=[0.01,0.99]: -log(0.99) ≈ 0.01
        let y_true = vec![0.0, 1.0];
        let y_proba = vec![0.99, 0.01, 0.01, 0.99];
        let loss = log_loss(&y_true, &y_proba, 2).unwrap();

        // Should be small (good predictions)
        assert!(loss < 0.1);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_log_loss_worst_case() {
        // Worst case: predicting opposite
        // y=1, p=[0.99, 0.01] -> -log(0.01) = 4.6
        let y_true = vec![1.0];
        let y_proba = vec![0.99, 0.01];
        let loss = log_loss(&y_true, &y_proba, 2).unwrap();

        // Should be high (bad prediction)
        assert!(loss > 4.0);
    }
}
