mod error;
mod matrix;
mod linear;
mod polynomial;
mod exponential;
mod timeseries;
mod kmeans;
mod knn;
mod logistic;
mod dbscan;
mod naive_bayes;
mod decision_tree;
mod pca;
mod perceptron;
mod random_forest;
mod silhouette;
mod confusion_matrix;
mod cross_validation;
mod data_split;
mod feature_importance;
mod standard_scaler;

use wasm_bindgen::prelude::*;
pub use error::MlError;

#[wasm_bindgen(start)]
pub fn init() {}

// Re-export all public types and functions
pub use linear::*;
pub use polynomial::*;
pub use exponential::*;
pub use timeseries::*;
pub use kmeans::*;
pub use knn::*;
pub use logistic::*;
pub use dbscan::*;
pub use naive_bayes::*;
pub use decision_tree::*;
pub use pca::*;
pub use perceptron::*;
pub use random_forest::*;
pub use silhouette::*;
pub use confusion_matrix::*;
pub use cross_validation::*;
pub use data_split::*;
pub use feature_importance::*;
pub use standard_scaler::*;
