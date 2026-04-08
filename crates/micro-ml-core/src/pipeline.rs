use crate::error::MlError;
use crate::matrix::validate_matrix;
use wasm_bindgen::prelude::*;

/// A single step in a transformation pipeline.
///
/// Each step stores pre-computed parameters so the pipeline can be
/// serialized and applied without fitting. The JS side fits scalers,
/// extracts their params, and passes them here to prevent data leakage.
#[wasm_bindgen]
pub struct PipelineStep {
    name: String,
    step_type: String,
    params: Vec<f64>,
    n_features: usize,
}

#[wasm_bindgen]
impl PipelineStep {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter, js_name = "stepType")]
    pub fn step_type(&self) -> String {
        self.step_type.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn params(&self) -> Vec<f64> {
        self.params.clone()
    }

    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}

/// A sequential transformation pipeline for WASM ML.
///
/// Stores pre-computed transformation steps and applies them in order.
/// This design avoids passing generic model objects across the WASM
/// boundary: the JS side fits scalers/encoders, extracts parameters,
/// then passes those parameters to the Pipeline.
#[wasm_bindgen]
pub struct Pipeline {
    steps: Vec<PipelineStep>,
    n_features: usize,
}

#[wasm_bindgen]
impl Pipeline {
    /// Number of features this pipeline expects per sample.
    #[wasm_bindgen(getter, js_name = "nFeatures")]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of transformation steps in the pipeline.
    #[wasm_bindgen(getter, js_name = "nSteps")]
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    /// Add a transformation step to the pipeline.
    ///
    /// # Arguments
    /// * `name` - Human-readable step name (e.g., "standard_scaler")
    /// * `step_type` - One of: "scaler_standard", "scaler_minmax",
    ///   "scaler_robust", "scaler_normalizer_l2", "scaler_normalizer_l1",
    ///   "imputer_mean", "selector"
    /// * `params` - Pre-computed parameters for the step. Layout depends
    ///   on step_type (see module docs for each step's parameter format).
    #[wasm_bindgen]
    pub fn add_step(&mut self, name: &str, step_type: &str, params: &[f64]) {
        self.steps.push(PipelineStep {
            name: name.to_string(),
            step_type: step_type.to_string(),
            params: params.to_vec(),
            n_features: self.n_features,
        });
    }

    /// Apply all pipeline steps sequentially to the input data.
    ///
    /// The input is a flat row-major matrix of shape `(n_samples, n_features)`.
    /// Each step transforms the entire matrix in place before passing to the
    /// next step. Some steps (e.g., "selector") may change the effective
    /// feature count between steps.
    #[wasm_bindgen]
    pub fn transform(&self, data: &[f64]) -> Result<Vec<f64>, JsError> {
        pipeline_transform_impl(self, data).map_err(|e| JsError::new(&e.message))
    }

    /// Return step metadata as a flat array for JS inspection.
    ///
    /// Layout: `[step_count, step_0_n_features, step_0_param_count, ...params, step_1_...]`
    /// This allows the JS side to reconstruct step information for serialization.
    #[wasm_bindgen(js_name = "getSteps")]
    pub fn get_steps(&self) -> Vec<f64> {
        let mut result = Vec::new();
        result.push(self.steps.len() as f64);

        for step in &self.steps {
            result.push(step.n_features as f64);
            result.push(step.params.len() as f64);
            result.extend_from_slice(&step.params);
        }

        result
    }

    #[wasm_bindgen(js_name = "toString")]
    pub fn to_string_js(&self) -> String {
        let step_names: Vec<&str> = self.steps.iter().map(|s| s.name.as_str()).collect();
        format!(
            "Pipeline(features={}, steps=[{}])",
            self.n_features,
            step_names.join(", ")
        )
    }
}

/// Create a new Pipeline for the given number of features.
#[wasm_bindgen(js_name = "pipeline")]
pub fn pipeline(n_features: usize) -> Pipeline {
    Pipeline {
        steps: Vec::new(),
        n_features,
    }
}

fn pipeline_transform_impl(pipeline: &Pipeline, data: &[f64]) -> Result<Vec<f64>, MlError> {
    if data.is_empty() {
        return Err(MlError::new("data must not be empty"));
    }

    let mut current = data.to_vec();
    let mut current_n_features = pipeline.n_features;

    for step in &pipeline.steps {
        validate_matrix(&current, current_n_features)?;
        current = apply_step(&current, current_n_features, &step.step_type, &step.params)?;

        // Selector changes the feature count
        if step.step_type == "selector" {
            current_n_features = step.params.iter().filter(|&&p| p == 1.0).count();
            if current_n_features == 0 {
                return Err(MlError::new("selector step removed all features"));
            }
        }
    }

    Ok(current)
}

/// Apply a single transformation step to the data matrix.
fn apply_step(
    data: &[f64],
    n_features: usize,
    step_type: &str,
    params: &[f64],
) -> Result<Vec<f64>, MlError> {
    match step_type {
        "scaler_standard" => apply_standard_scaler(data, n_features, params),
        "scaler_minmax" => apply_minmax_scaler(data, n_features, params),
        "scaler_robust" => apply_robust_scaler(data, n_features, params),
        "scaler_normalizer_l2" => apply_normalizer(data, n_features, "l2"),
        "scaler_normalizer_l1" => apply_normalizer(data, n_features, "l1"),
        "imputer_mean" => apply_imputer_mean(data, n_features, params),
        "selector" => apply_selector(data, n_features, params),
        _ => Err(MlError::new(format!("unknown step type: {}", step_type))),
    }
}

/// Standard scaler: params = [mean_0, std_0, mean_1, std_1, ...]
/// Transforms: (x - mean) / std
fn apply_standard_scaler(
    data: &[f64],
    n_features: usize,
    params: &[f64],
) -> Result<Vec<f64>, MlError> {
    if params.len() != n_features * 2 {
        return Err(MlError::new(format!(
            "scaler_standard expects {} params (mean_0, std_0, ...), got {}",
            n_features * 2,
            params.len()
        )));
    }

    let n_samples = data.len() / n_features;
    let mut result = Vec::with_capacity(data.len());

    for i in 0..n_samples {
        for j in 0..n_features {
            let val = data[i * n_features + j];
            let mean = params[j * 2];
            let std = params[j * 2 + 1];
            let scale = std.max(1e-8);
            result.push((val - mean) / scale);
        }
    }

    Ok(result)
}

/// Min-max scaler: params = [min_0, max_0, min_1, max_1, ...]
/// Transforms: (x - min) / (max - min)
fn apply_minmax_scaler(
    data: &[f64],
    n_features: usize,
    params: &[f64],
) -> Result<Vec<f64>, MlError> {
    if params.len() != n_features * 2 {
        return Err(MlError::new(format!(
            "scaler_minmax expects {} params (min_0, max_0, ...), got {}",
            n_features * 2,
            params.len()
        )));
    }

    let n_samples = data.len() / n_features;
    let mut result = Vec::with_capacity(data.len());

    for i in 0..n_samples {
        for j in 0..n_features {
            let val = data[i * n_features + j];
            let min = params[j * 2];
            let max = params[j * 2 + 1];
            let range = (max - min).max(1e-8);
            result.push((val - min) / range);
        }
    }

    Ok(result)
}

/// Robust scaler: params = [median_0, iqr_0, median_1, iqr_1, ...]
/// Transforms: (x - median) / iqr
fn apply_robust_scaler(
    data: &[f64],
    n_features: usize,
    params: &[f64],
) -> Result<Vec<f64>, MlError> {
    if params.len() != n_features * 2 {
        return Err(MlError::new(format!(
            "scaler_robust expects {} params (median_0, iqr_0, ...), got {}",
            n_features * 2,
            params.len()
        )));
    }

    let n_samples = data.len() / n_features;
    let mut result = Vec::with_capacity(data.len());

    for i in 0..n_samples {
        for j in 0..n_features {
            let val = data[i * n_features + j];
            let median = params[j * 2];
            let iqr = params[j * 2 + 1];
            let scale = iqr.max(1e-8);
            result.push((val - median) / scale);
        }
    }

    Ok(result)
}

/// Per-row normalizer (L1 or L2). No params needed.
fn apply_normalizer(data: &[f64], n_features: usize, norm: &str) -> Result<Vec<f64>, MlError> {
    let n_samples = data.len() / n_features;
    let mut result = Vec::with_capacity(data.len());

    for i in 0..n_samples {
        let row = &data[i * n_features..(i + 1) * n_features];
        let norm_value = match norm {
            "l1" => row.iter().map(|&v| v.abs()).sum::<f64>(),
            "l2" => row.iter().map(|&v| v * v).sum::<f64>().sqrt(),
            _ => return Err(MlError::new(format!("unknown norm type: {}", norm))),
        };

        if norm_value > 1e-12 {
            for &val in row {
                result.push(val / norm_value);
            }
        } else {
            result.extend_from_slice(row);
        }
    }

    Ok(result)
}

/// Mean imputer: params = [mean_0, mean_1, ...]
/// Replaces NaN values with the feature mean.
fn apply_imputer_mean(
    data: &[f64],
    n_features: usize,
    params: &[f64],
) -> Result<Vec<f64>, MlError> {
    if params.len() != n_features {
        return Err(MlError::new(format!(
            "imputer_mean expects {} params (one mean per feature), got {}",
            n_features,
            params.len()
        )));
    }

    let mut result = Vec::with_capacity(data.len());

    for val in data {
        if val.is_nan() {
            // Determine which feature column this value belongs to
            let idx = result.len();
            let j = idx % n_features;
            result.push(params[j]);
        } else {
            result.push(*val);
        }
    }

    Ok(result)
}

/// Feature selector: params = [mask_0, mask_1, ...] where 1 = keep, 0 = drop
fn apply_selector(data: &[f64], n_features: usize, params: &[f64]) -> Result<Vec<f64>, MlError> {
    if params.len() != n_features {
        return Err(MlError::new(format!(
            "selector expects {} params (one mask per feature), got {}",
            n_features,
            params.len()
        )));
    }

    let n_samples = data.len() / n_features;
    let mut result = Vec::new();

    for i in 0..n_samples {
        for j in 0..n_features {
            if params[j] == 1.0 {
                result.push(data[i * n_features + j]);
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_scaler_pipeline() {
        // Training data: 5 samples, 2 features
        // Feature 0: [1, 2, 3, 4, 5] -> mean=3, std=sqrt(2) ~ 1.4142
        // Feature 1: [10, 20, 30, 40, 50] -> mean=30, std=sqrt(200) ~ 14.1421
        let train_data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0];

        // Compute params manually (matching standard_scaler.rs logic: population std)
        // Feature 0: mean=3, variance=((1-3)^2+(2-3)^2+(3-3)^2+(4-3)^2+(5-3)^2)/5 = 10/5 = 2, std=sqrt(2)
        // Feature 1: mean=30, variance=((10-30)^2+...)/5 = 2000/5 = 400, std=sqrt(400)=20
        let mean_0 = 3.0;
        let std_0 = (2.0_f64).sqrt();
        let mean_1 = 30.0;
        let std_1 = 20.0;

        let mut pipe = pipeline(2);
        pipe.add_step(
            "standard_scaler",
            "scaler_standard",
            &[mean_0, std_0, mean_1, std_1],
        );

        // Transform the same training data — means should be ~0
        let transformed = pipe.transform(&train_data).unwrap();

        // Feature 0 mean check
        let mut sum_0 = 0.0;
        for i in 0..5 {
            sum_0 += transformed[i * 2];
        }
        assert!((sum_0 / 5.0).abs() < 1e-10, "Feature 0 mean should be ~0");

        // Feature 1 mean check
        let mut sum_1 = 0.0;
        for i in 0..5 {
            sum_1 += transformed[i * 2 + 1];
        }
        assert!((sum_1 / 5.0).abs() < 1e-10, "Feature 1 mean should be ~0");

        // Transform new data point [3, 30] -> should map to [0, 0]
        let new_data = vec![3.0, 30.0];
        let result = pipe.transform(&new_data).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);

        // Transform [5, 50] -> should map to [(5-3)/sqrt(2), (50-30)/20] = [sqrt(2), 1]
        let new_data2 = vec![5.0, 50.0];
        let result2 = pipe.transform(&new_data2).unwrap();
        assert!((result2[0] - std_0).abs() < 1e-10);
        assert!((result2[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_step_pipeline() {
        // 3 features: [age, income, score]
        // Training data
        let train_data = vec![
            25.0, 50000.0, 0.8, 30.0, 60000.0, 0.9, 35.0, 70000.0, 0.7, 40.0, 80000.0, 0.85, 45.0,
            90000.0, 0.95,
        ];

        // Step 1: Standard scaler for all 3 features
        // Feature 0 (age): mean=35, var=50, std=sqrt(50)
        // Feature 1 (income): mean=70000, var=200_000_000, std=sqrt(200M)
        // Feature 2 (score): mean=0.84, var=0.0076, std=sqrt(0.0076)
        let mean_0 = 35.0;
        let std_0 = (50.0_f64).sqrt();
        let mean_1 = 70000.0;
        let std_1 = (200_000_000.0_f64).sqrt();
        let mean_2 = 0.84;
        let std_2 = (0.0076_f64).sqrt();

        // Step 2: Selector — keep feature 0 (age) and feature 2 (score), drop income
        let selector_mask = vec![1.0, 0.0, 1.0];

        let mut pipe = pipeline(3);
        pipe.add_step(
            "standard_scaler",
            "scaler_standard",
            &[mean_0, std_0, mean_1, std_1, mean_2, std_2],
        );
        pipe.add_step("feature_selection", "selector", &selector_mask);

        assert_eq!(pipe.n_steps(), 2);

        // Transform — should produce 5 samples with 2 features each (age_scaled, score_scaled)
        let result = pipe.transform(&train_data).unwrap();
        assert_eq!(result.len(), 5 * 2);

        // Verify the first sample [25, 50000, 0.8]:
        // After standard scaler: [(25-35)/sqrt(50), (50000-70000)/sqrt(200M), (0.8-0.84)/sqrt(0.0076)]
        // After selector: [age_scaled, score_scaled]
        let age_scaled = (25.0 - mean_0) / std_0;
        let score_scaled = (0.8 - mean_2) / std_2;

        assert!((result[0] - age_scaled).abs() < 1e-8);
        assert!((result[1] - score_scaled).abs() < 1e-8);

        // Verify last sample [45, 90000, 0.95]
        let age_scaled_4 = (45.0 - mean_0) / std_0;
        let score_scaled_4 = (0.95 - mean_2) / std_2;
        assert!((result[8] - age_scaled_4).abs() < 1e-8);
        assert!((result[9] - score_scaled_4).abs() < 1e-8);
    }

    #[test]
    fn test_minmax_scaler() {
        // Data: [[0, 10], [5, 20], [10, 30]]
        // Feature 0: min=0, max=10
        // Feature 1: min=10, max=30
        let mut pipe = pipeline(2);
        pipe.add_step("minmax", "scaler_minmax", &[0.0, 10.0, 10.0, 30.0]);

        let data = vec![0.0, 10.0, 5.0, 20.0, 10.0, 30.0];
        let result = pipe.transform(&data).unwrap();

        // [0, 10] -> [0, 0]
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);

        // [5, 20] -> [0.5, 0.5]
        assert!((result[2] - 0.5).abs() < 1e-10);
        assert!((result[3] - 0.5).abs() < 1e-10);

        // [10, 30] -> [1, 1]
        assert!((result[4] - 1.0).abs() < 1e-10);
        assert!((result[5] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_robust_scaler() {
        // Feature 0: median=10, iqr=5
        let mut pipe = pipeline(1);
        pipe.add_step("robust", "scaler_robust", &[10.0, 5.0]);

        let data = vec![5.0, 10.0, 15.0];
        let result = pipe.transform(&data).unwrap();

        // (5-10)/5 = -1, (10-10)/5 = 0, (15-10)/5 = 1
        assert!((result[0] - (-1.0)).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_l2() {
        let mut pipe = pipeline(2);
        pipe.add_step("norm_l2", "scaler_normalizer_l2", &[]);

        let data = vec![3.0, 4.0];
        let result = pipe.transform(&data).unwrap();

        // L2 norm of [3, 4] = 5
        assert!((result[0] - 0.6).abs() < 1e-10);
        assert!((result[1] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_l1() {
        let mut pipe = pipeline(3);
        pipe.add_step("norm_l1", "scaler_normalizer_l1", &[]);

        let data = vec![1.0, 2.0, 3.0];
        let result = pipe.transform(&data).unwrap();

        // L1 norm = 6
        assert!((result[0] - 1.0 / 6.0).abs() < 1e-10);
        assert!((result[1] - 2.0 / 6.0).abs() < 1e-10);
        assert!((result[2] - 3.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_imputer_mean() {
        let mut pipe = pipeline(2);
        pipe.add_step("imputer", "imputer_mean", &[5.0, 10.0]);

        let data = vec![1.0, f64::NAN, f64::NAN, 20.0, 3.0, f64::NAN];
        let result = pipe.transform(&data).unwrap();

        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 10.0).abs() < 1e-10); // NaN replaced with mean
        assert!(result[2].is_nan() == false);
        assert!((result[2] - 5.0).abs() < 1e-10); // NaN replaced with mean
        assert!((result[3] - 20.0).abs() < 1e-10);
        assert!((result[4] - 3.0).abs() < 1e-10);
        assert!((result[5] - 10.0).abs() < 1e-10); // NaN replaced with mean
    }

    #[test]
    fn test_selector() {
        let mut pipe = pipeline(4);
        pipe.add_step("select_first_and_last", "selector", &[1.0, 0.0, 0.0, 1.0]);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = pipe.transform(&data).unwrap();

        // Should select columns 0 and 3
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
        assert!((result[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_data_error() {
        let pipe = pipeline(2);
        assert!(pipeline_transform_impl(&pipe, &[]).is_err());
    }

    #[test]
    fn test_wrong_param_count() {
        let mut pipe = pipeline(2);
        pipe.add_step("bad", "scaler_standard", &[1.0, 2.0]); // needs 4 params

        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(pipeline_transform_impl(&pipe, &data).is_err());
    }

    #[test]
    fn test_unknown_step_type() {
        let mut pipe = pipeline(2);
        pipe.add_step("bad", "unknown_step", &[]);

        let data = vec![1.0, 2.0];
        assert!(pipeline_transform_impl(&pipe, &data).is_err());
    }

    #[test]
    fn test_get_steps() {
        let mut pipe = pipeline(3);
        pipe.add_step("scaler", "scaler_standard", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        pipe.add_step("selector", "selector", &[1.0, 0.0, 1.0]);

        let meta = pipe.get_steps();

        // [step_count=2, n_features_0=3, param_count_0=6, ...6 params, n_features_1=3, param_count_1=3, ...3 params]
        assert_eq!(meta.len(), 1 + 2 + 6 + 2 + 3);
        assert!((meta[0] - 2.0).abs() < 1e-10); // 2 steps
        assert!((meta[1] - 3.0).abs() < 1e-10); // first step n_features
        assert!((meta[2] - 6.0).abs() < 1e-10); // first step param count
    }

    #[test]
    fn test_getters() {
        let mut pipe = pipeline(5);
        assert_eq!(pipe.n_features(), 5);
        assert_eq!(pipe.n_steps(), 0);

        pipe.add_step("s1", "scaler_normalizer_l2", &[]);
        assert_eq!(pipe.n_steps(), 1);
    }

    #[test]
    fn test_to_string() {
        let mut pipe = pipeline(3);
        pipe.add_step("scale", "scaler_standard", &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        pipe.add_step("select", "selector", &[1.0, 0.0, 1.0]);

        let s = pipe.to_string_js();
        assert!(s.contains("features=3"));
        assert!(s.contains("scale"));
        assert!(s.contains("select"));
    }

    #[test]
    fn test_selector_removes_all_features_error() {
        let mut pipe = pipeline(2);
        pipe.add_step("drop_all", "selector", &[0.0, 0.0]);

        let data = vec![1.0, 2.0];
        assert!(pipeline_transform_impl(&pipe, &data).is_err());
    }
}
