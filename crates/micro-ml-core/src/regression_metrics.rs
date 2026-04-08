use wasm_bindgen::prelude::*;

/// R² (Coefficient of Determination) - proportion of variance explained
#[wasm_bindgen(js_name = "r2Score")]
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(JsError::new("arrays must be same non-zero length"));
    }

    let n = y_true.len();
    let mean_true: f64 = y_true.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y_true.iter().map(|y| (y - mean_true).powi(2)).sum();
    let ss_res: f64 = y_true.iter().zip(y_pred.iter()).map(|(t, p)| (t - p).powi(2)).sum();

    if ss_tot == 0.0 {
        return Ok(1.0); // Perfect fit already
    }

    Ok(1.0 - ss_res / ss_tot)
}

/// Mean Squared Error
#[wasm_bindgen(js_name = "meanSquaredError")]
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(JsError::new("arrays must be same non-zero length"));
    }
    let mse = y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>() / y_true.len() as f64;
    Ok(mse)
}

/// Root Mean Squared Error
#[wasm_bindgen(js_name = "rootMeanSquaredError")]
pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    let mse = mean_squared_error(y_true, y_pred)?;
    Ok(mse.sqrt())
}

/// Mean Absolute Error
#[wasm_bindgen(js_name = "meanAbsoluteError")]
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(JsError::new("arrays must be same non-zero length"));
    }
    let mae = y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>() / y_true.len() as f64;
    Ok(mae)
}

/// Median Absolute Error (robust to outliers)
#[wasm_bindgen(js_name = "medianAbsoluteError")]
pub fn median_absolute_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64, JsError> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(JsError::new("arrays must be same non-zero length"));
    }

    let mut errors: Vec<f64> = y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .collect();
    errors.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if errors.len().is_multiple_of(2) {
        (errors[errors.len() / 2 - 1] + errors[errors.len() / 2]) / 2.0
    } else {
        errors[errors.len() / 2]
    };

    Ok(median)
}

/// Mean Absolute Percentage Error (with epsilon for division by zero)
#[wasm_bindgen(js_name = "meanAbsolutePercentageError")]
pub fn mean_absolute_percentage_error(y_true: &[f64], y_pred: &[f64], epsilon: f64) -> Result<f64, JsError> {
    if y_true.len() != y_pred.len() || y_true.is_empty() {
        return Err(JsError::new("arrays must be same non-zero length"));
    }

    let mape: f64 = y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| {
            let denom = t.abs().max(epsilon);
            ((t - p).abs() / denom) * 100.0
        })
        .sum::<f64>() / y_true.len() as f64;

    Ok(mape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r2_perfect() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        assert!((r2_score(&y_true, &y_pred).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_half() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.5, 1.5, 1.5, 1.5]; // Mean = 2.5, predictions are constant
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!(r2 < 0.0); // Negative R² is possible
    }

    #[test]
    fn test_mse() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.1, 7.8];
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert!(mse > 0.0);
    }

    #[test]
    fn test_mae() {
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.1, 7.8];
        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!(mae > 0.0);
    }

    #[test]
    fn test_mape() {
        let y_true = vec![100.0, 200.0, 300.0];
        let y_pred = vec![110.0, 190.0, 330.0];
        let mape = mean_absolute_percentage_error(&y_true, &y_pred, 1e-10).unwrap();
        // Should be ~10% error
        assert!(mape > 5.0 && mape < 15.0);
    }

    #[test]
    fn test_median_ae_outlier_robust() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 100.0]; // Outlier at end
        let y_pred = vec![1.0, 2.0, 3.0, 4.0, 1.0]; // Wrong prediction for outlier
        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        let med_ae = median_absolute_error(&y_true, &y_pred).unwrap();
        // Median should be much lower than mean due to outlier
        assert!(med_ae < mae);
    }

    // ML CORRECTNESS VALIDATION TESTS

    #[test]
    fn test_r2_formula_verification() {
        // Verify R² formula: R² = 1 - SS_res/SS_tot
        // y = [1,2,3,4], y_pred = [1,2,3,4] -> SS_res = 0, R² = 1
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        let r2 = r2_score(&y_true, &y_pred).unwrap();
        assert!((r2 - 1.0).abs() < 1e-10);

        // y = [1,2,3], y_pred = [2,3,4] -> mean=2, SS_tot = 2, SS_res = 3, R² = 1 - 3/2 = -0.5
        let y_true2 = vec![1.0, 2.0, 3.0];
        let y_pred2 = vec![2.0, 3.0, 4.0];
        let r2_2 = r2_score(&y_true2, &y_pred2).unwrap();
        assert!((r2_2 - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_mse_formula_verification() {
        // MSE = (1/n) * Σ(y_true - y_pred)²
        // y_true = [0, 1, 2], y_pred = [1, 2, 3] -> errors = [-1, -1, -1] -> MSE = (1+1+1)/3 = 1
        let y_true = vec![0.0, 1.0, 2.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        assert!((mse - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rmse_sqrt_relationship() {
        // RMSE should equal sqrt(MSE)
        let y_true = vec![0.0, 1.0, 2.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        let mse = mean_squared_error(&y_true, &y_pred).unwrap();
        let rmse = root_mean_squared_error(&y_true, &y_pred).unwrap();
        assert!((rmse - mse.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_mae_formula_verification() {
        // MAE = (1/n) * Σ|y_true - y_pred|
        // y_true = [0, 1, 2], y_pred = [1, 2, 4] -> errors = [1, 1, 2] -> MAE = 4/3
        let y_true = vec![0.0, 1.0, 2.0];
        let y_pred = vec![1.0, 2.0, 4.0];
        let mae = mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!((mae - 4.0/3.0).abs() < 1e-10);
    }
}
