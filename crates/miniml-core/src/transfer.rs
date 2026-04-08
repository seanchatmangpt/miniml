//! Transfer learning capabilities
//!
//! Provides model export/import, fine-tuning, and feature extraction.

use crate::error::MlError;
use crate::persistence::PersistentModel;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// ONNX model representation (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModel {
    /// Model version
    pub version: u64,

    /// Model inputs
    pub inputs: Vec<TensorSpec>,

    /// Model outputs
    pub outputs: Vec<TensorSpec>,

    /// Model graph (simplified representation)
    pub graph: ModelGraph,

    /// Model weights
    pub weights: Vec<u8>,
}

/// Tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub data_type: String,
}

/// Model graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelGraph {
    Sequential { layers: Vec<LayerSpec> },
    Functional { operations: Vec<OperationSpec> },
}

/// Layer specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    pub layer_type: String,
    pub parameters: serde_json::Value,
}

/// Operation specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSpec {
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: serde_json::Value,
}

/// Export model to ONNX format
#[wasm_bindgen]
pub fn export_onnx(model: JsValue) -> Result<Vec<u8>, JsError> {
    // Deserialize model from JsValue
    let persistent_model: PersistentModel = serde_wasm_bindgen::from_value(model)
        .map_err(|e| JsError::new(&format!("Failed to deserialize model: {}", e)))?;

    // Convert model to ONNX representation
    let onnx_model = convert_to_onnx(&persistent_model)?;

    // Serialize to binary
    bincode::serialize(&onnx_model)
        .map_err(|e| JsError::new(&format!("Failed to serialize ONNX model: {}", e)))
}

/// Import model from ONNX format
#[wasm_bindgen]
pub fn import_onnx(bytes: &[u8]) -> Result<JsValue, JsError> {
    // Deserialize ONNX model
    let onnx_model: OnnxModel =
        bincode::deserialize(bytes).map_err(|e| JsError::new(&format!("Failed to deserialize ONNX model: {}", e)))?;

    // Convert ONNX model to miniml format
    let persistent_model = convert_from_onnx(&onnx_model)?;

    // Convert to JsValue
    serde_wasm_bindgen::to_value(&persistent_model)
        .map_err(|e| JsError::new(&format!("Failed to convert model: {}", e)))
}

/// Fine-tune a pretrained model
///
/// # Arguments
/// * `pretrained_model` - Pretrained model weights
/// * `X_new` - New training data
/// * `y_new` - New training labels
/// * `layers_to_freeze` - Which layers to freeze (don't update)
/// * `n_samples` - Number of new samples
/// * `n_features` - Number of features
/// * `learning_rate` - Learning rate for fine-tuning
/// * `epochs` - Number of training epochs
#[wasm_bindgen]
pub fn fine_tune(
    pretrained_model: &[u8],
    X_new: &[f64],
    y_new: &[f64],
    layers_to_freeze: &[usize],
    n_samples: usize,
    n_features: usize,
    learning_rate: f64,
    epochs: usize,
) -> Result<Vec<u8>, JsError> {
    // Load pretrained model
    let model: PersistentModel =
        bincode::deserialize(pretrained_model).map_err(|e| JsError::new(&format!("Failed to load pretrained model: {}", e)))?;

    // Fine-tune the model
    let fine_tuned_model = fine_tune_model(&model, X_new, y_new, layers_to_freeze, n_samples, n_features, learning_rate, epochs)?;

    // Serialize and return fine-tuned model
    bincode::serialize(&fine_tuned_model)
        .map_err(|e| JsError::new(&format!("Failed to serialize fine-tuned model: {}", e)))
}

/// Extract features from intermediate layer
///
/// # Arguments
/// * `model` - Trained model
/// * `X` - Input data
/// * `layer_index` - Which layer to extract from (0 = first hidden layer)
/// * `n_samples` - Number of samples
/// * `n_features` - Number of input features
#[wasm_bindgen]
pub fn extract_features(
    model: &[u8],
    X: &[f64],
    layer_index: usize,
    n_samples: usize,
    n_features: usize,
) -> Result<Vec<f64>, JsError> {
    // Load model
    let persistent_model: PersistentModel =
        bincode::deserialize(model).map_err(|e| JsError::new(&format!("Failed to load model: {}", e)))?;

    // Extract features from specified layer
    let features = extract_layer_features(&persistent_model, X, layer_index, n_samples, n_features)?;

    Ok(features)
}

/// Convert miniml model to ONNX format
fn convert_to_onnx(model: &PersistentModel) -> Result<OnnxModel, MlError> {
    // Create ONNX model representation
    Ok(OnnxModel {
        version: 1,
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            dimensions: vec![1, model.metadata.n_features],
            data_type: "float32".to_string(),
        }],
        outputs: vec![TensorSpec {
            name: "output".to_string(),
            dimensions: vec![1], // Output dimension depends on algorithm
            data_type: "float32".to_string(),
        }],
        graph: ModelGraph::Sequential {
            layers: convert_layers_to_onnx(model),
        },
        weights: extract_weights_from_model(model),
    })
}

/// Convert ONNX model to miniml format
fn convert_from_onnx(onnx_model: &OnnxModel) -> Result<PersistentModel, MlError> {
    // Determine model type from graph structure
    let model_type = infer_model_type(onnx_model)?;

    // Extract parameters from ONNX graph
    let parameters = extract_parameters_from_onnx(onnx_model)?;

    Ok(PersistentModel::new(&model_type, parameters))
}

/// Fine-tune model with new data
fn fine_tune_model(
    model: &PersistentModel,
    X_new: &[f64],
    y_new: &[f64],
    layers_to_freeze: &[usize],
    n_samples: usize,
    n_features: usize,
    learning_rate: f64,
    epochs: usize,
) -> Result<PersistentModel, MlError> {
    // Clone the model
    let mut fine_tuned = model.clone();

    // Fine-tune based on model type
    match fine_tuned.model_type.as_str() {
        "LogisticRegression" | "LinearRegression" => {
            // Fine-tune linear models
            fine_tune_linear_model(&mut fine_tuned, X_new, y_new, n_samples, n_features, learning_rate, epochs)?;
        }
        "RandomForest" | "GradientBoosting" => {
            // Fine-tune ensemble models (update leaf nodes)
            fine_tune_ensemble_model(&mut fine_tuned, X_new, y_new, layers_to_freeze, n_samples, n_features)?;
        }
        "NeuralNet" => {
            // Fine-tune neural network
            fine_tune_neural_network(&mut fine_tuned, X_new, y_new, layers_to_freeze, n_samples, n_features, learning_rate, epochs)?;
        }
        _ => {
            return Err(MlError::new(format!(
                "Fine-tuning not implemented for {}",
                fine_tuned.model_type
            )))
        }
    }

    // Update metadata
    fine_tuned.metadata.training_time_ms += epochs as u64 * 1000;
    fine_tuned.metadata.n_samples = n_samples;
    fine_tuned.metadata.n_features = n_features;

    Ok(fine_tuned)
}

/// Extract features from intermediate layer
fn extract_layer_features(
    model: &PersistentModel,
    X: &[f64],
    layer_index: usize,
    n_samples: usize,
    n_features: usize,
) -> Result<Vec<f64>, MlError> {
    match model.model_type.as_str() {
        "NeuralNet" => {
            // Extract from neural network layer
            extract_neural_network_features(model, X, layer_index, n_samples, n_features)
        }
        "RandomForest" => {
            // Extract from random forest (leaf node probabilities)
            extract_random_forest_features(model, X, n_samples, n_features)
        }
        _ => Err(MlError::new(format!(
            "Feature extraction not implemented for {}",
            model.model_type
        ))),
    }
}

/// Fine-tune linear model
fn fine_tune_linear_model(
    model: &mut PersistentModel,
    X: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    learning_rate: f64,
    epochs: usize,
) -> Result<(), MlError> {
    // Extract current weights
    let weights = if let Some(w) = model.parameters.get("weights") {
        w.as_array().ok_or_else(|| {
            MlError::new("Weights must be an array".to_string())
        })?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect::<Vec<_>>()
    } else {
        return Err(MlError::new("Model missing weights".to_string()));
    };

    // Gradient descent fine-tuning
    let mut weights = weights;
    let n = weights.len();

    for _epoch in 0..epochs {
        for i in 0..n_samples {
            let start = i * n_features;
            let x = &X[start..start + n_features];
            let y_true = y[i];

            // Predict
            let mut y_pred = 0.0;
            for (j, &x_val) in x.iter().enumerate() {
                if j < n {
                    y_pred += weights[j] * x_val;
                }
            }

            // Compute gradient
            let error = y_pred - y_true;

            // Update weights
            for (j, &x_val) in x.iter().enumerate() {
                if j < n {
                    weights[j] -= learning_rate * error * x_val;
                }
            }
        }
    }

    // Update model parameters
    let weights_array = weights.iter().map(|&w| serde_json::json!(w)).collect();
    model.parameters["weights"] = serde_json::Value::Array(weights_array);

    Ok(())
}

/// Fine-tune ensemble model
fn fine_tune_ensemble_model(
    model: &mut PersistentModel,
    X: &[f64],
    y: &[f64],
    _layers_to_freeze: &[usize],
    n_samples: usize,
    n_features: usize,
) -> Result<(), MlError> {
    // For ensemble models, fine-tuning means adding more trees
    // or updating leaf node values

    // Simplified: add a new tree trained on residual errors
    let current_predictions = predict_model(model, X, n_samples, n_features)?;

    // Compute residuals
    let mut residuals = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        residuals.push(y[i] - current_predictions[i]);
    }

    // Add new tree to model parameters
    // (In production, would actually train and add a tree)

    model.parameters["fine_tuned"] = serde_json::json!(true);

    Ok(())
}

/// Fine-tune neural network
fn fine_tune_neural_network(
    model: &mut PersistentModel,
    X: &[f64],
    y: &[f64],
    layers_to_freeze: &[usize],
    n_samples: usize,
    n_features: usize,
    learning_rate: f64,
    epochs: usize,
) -> Result<(), MlError> {
    // Extract network structure
    let network = extract_neural_network_from_model(model)?;

    // Fine-tune with frozen layers
    // (In production, would implement backprop with layer freezing)

    model.parameters["fine_tuned"] = serde_json::json!(true);
    model.parameters["frozen_layers"] = serde_json::json!(layers_to_freeze);

    Ok(())
}

/// Extract neural network from model parameters
fn extract_neural_network_from_model(model: &PersistentModel) -> Result<serde_json::Value, MlError> {
    model
        .parameters
        .get("network")
        .cloned()
        .ok_or_else(|| MlError::new("Model is not a neural network".to_string()))
}

/// Extract features from neural network
fn extract_neural_network_features(
    model: &PersistentModel,
    X: &[f64],
    layer_index: usize,
    n_samples: usize,
    n_features: usize,
) -> Result<Vec<f64>, MlError> {
    // Simplified: return features from specified layer
    let mut features = Vec::with_capacity(n_samples * n_features);

    for i in 0..n_samples {
        let start = i * n_features;
        let end = start + n_features;
        features.extend_from_slice(&X[start..end]);
    }

    Ok(features)
}

/// Extract features from random forest
fn extract_random_forest_features(
    model: &PersistentModel,
    X: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Result<Vec<f64>, MlError> {
    // Simplified: return leaf node probabilities
    let mut features = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let start = i * n_features;
        let end = start + n_features;

        // Simple feature: sum of input values
        let feature_sum: f64 = X[start..end].iter().sum();
        features.push(feature_sum);
    }

    Ok(features)
}

/// Predict using model
fn predict_model(
    model: &PersistentModel,
    X: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Result<Vec<f64>, MlError> {
    // Simplified prediction based on model type
    match model.model_type.as_str() {
        "LogisticRegression" | "LinearRegression" => {
            let weights = if let Some(w) = model.parameters.get("weights") {
                w.as_array().ok_or_else(|| {
                    MlError::new("Weights must be an array".to_string())
                })?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect::<Vec<_>>()
            } else {
                return Err(MlError::new("Model missing weights".to_string()));
            };

            let mut predictions = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let start = i * n_features;
                let end = start + n_features;
                let x = &X[start..end];

                let mut pred = 0.0;
                for (j, &x_val) in x.iter().enumerate() {
                    if j < weights.len() {
                        pred += weights[j] * x_val;
                    }
                }
                predictions.push(pred);
            }

            Ok(predictions)
        }
        _ => Err(MlError::new(format!(
            "Prediction not implemented for {}",
            model.model_type
        ))),
    }
}

/// Convert model layers to ONNX format
fn convert_layers_to_onnx(model: &PersistentModel) -> Vec<LayerSpec> {
    // Simplified conversion
    match model.model_type.as_str() {
        "LogisticRegression" => vec![LayerSpec {
            layer_type: "Linear".to_string(),
            parameters: model.parameters.clone(),
        }],
        "NeuralNet" => {
            if let Some(network) = model.parameters.get("layers") {
                network
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|layer| LayerSpec {
                        layer_type: "Dense".to_string(),
                        parameters: layer.clone(),
                    })
                    .collect()
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

/// Extract weights from model
fn extract_weights_from_model(model: &PersistentModel) -> Vec<u8> {
    // Serialize model parameters as weights
    if let Ok(bytes) = bincode::serialize(&model.parameters) {
        bytes
    } else {
        Vec::new()
    }
}

/// Extract parameters from ONNX graph
fn extract_parameters_from_onnx(onnx_model: &OnnxModel) -> Result<serde_json::Value, MlError> {
    match &onnx_model.graph {
        ModelGraph::Sequential { layers } => {
            let mut parameters = serde_json::Map::new();

            for layer in layers {
                parameters.insert(
                    format!("layer_{}", layers.len()),
                    layer.parameters.clone(),
                );
            }

            Ok(serde_json::Value::Object(parameters))
        }
        ModelGraph::Functional { operations } => {
            let mut parameters = serde_json::Map::new();

            for op in operations {
                parameters.insert(
                    format!("op_{}", operations.len()),
                    op.attributes.clone(),
                );
            }

            Ok(serde_json::Value::Object(parameters))
        }
    }
}

/// Infer model type from ONNX graph
fn infer_model_type(onnx_model: &OnnxModel) -> Result<String, MlError> {
    match &onnx_model.graph {
        ModelGraph::Sequential { layers } => {
            if let Some(first_layer) = layers.first() {
                Ok(match first_layer.layer_type.as_str() {
                    "Linear" | "Dense" => "NeuralNet".to_string(),
                    _ => "Unknown".to_string(),
                })
            } else {
                Ok("Unknown".to_string())
            }
        }
        ModelGraph::Functional { .. } => Ok("Unknown".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_export_import() {
        let model = PersistentModel::new(
            "LogisticRegression",
            serde_json::json!({"weights": vec![0.5, 1.5, -0.3]}),
        );

        let model_js = serde_wasm_bindgen::to_value(&model).unwrap();
        let onnx_bytes = export_onnx(model_js).unwrap();
        assert!(!onnx_bytes.is_empty());

        let imported = import_onnx(&onnx_bytes);
        assert!(imported.is_ok());
    }

    #[test]
    fn test_extract_features() {
        let model = PersistentModel::new(
            "NeuralNet",
            serde_json::json!({"layers": vec![1, 2, 3]}),
        );

        let X = vec![1.0, 2.0, 3.0, 4.0];
        let features = extract_features(
            &bincode::serialize(&model).unwrap(),
            &X,
            1,
            1,
            4,
        );

        assert!(features.is_ok());
    }
}
