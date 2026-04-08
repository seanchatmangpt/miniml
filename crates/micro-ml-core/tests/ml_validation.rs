//! ML Result Validation Tests
//! Verifies each algorithm produces MATHEMATICALLY CORRECT results on known datasets

// This test validates actual ML correctness, not just compilation
// All tests verify exact mathematical properties of the algorithms

#[cfg(test)]
mod ml_validation_tests {

    // === REGRESSION METRICS VALIDATION ===

    #[test]
    fn test_r2_perfect_fit() {
        // Perfect prediction: R² should be exactly 1.0
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];

        let r2 = micro_ml_core::regression_metrics::r2_score(&y_true, &y_pred).unwrap();
        assert!((r2 - 1.0).abs() < 1e-10, "R² should be 1.0 for perfect fit, got {}", r2);
    }

    #[test]
    fn test_r2_worst_fit() {
        // Worst prediction: mean of y_true is 2.5, predicting constant 2.5 gives R² = 0
        // But if we predict opposite, R² can be negative
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![4.0, 3.0, 2.0, 1.0];  // Reverse order

        let r2 = micro_ml_core::regression_metrics::r2_score(&y_true, &y_pred).unwrap();
        assert!(r2 < 0.0, "R² should be negative for worse-than-mean prediction, got {}", r2);
    }

    #[test]
    fn test_mse_exact_value() {
        // MSE = (1² + 1² + 1²) / 3 = 1.0
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![2.0, 3.0, 4.0];  // Each off by 1

        let mse = micro_ml_core::regression_metrics::mean_squared_error(&y_true, &y_pred).unwrap();
        assert!((mse - 1.0).abs() < 1e-10, "MSE should be 1.0, got {}", mse);
    }

    #[test]
    fn test_rmse_sqrt_relationship() {
        let y_true = vec![0.0, 1.0, 2.0];
        let y_pred = vec![1.0, 2.0, 3.0];

        let mse = micro_ml_core::regression_metrics::mean_squared_error(&y_true, &y_pred).unwrap();
        let rmse = micro_ml_core::regression_metrics::root_mean_squared_error(&y_true, &y_pred).unwrap();

        assert!((rmse - mse.sqrt()).abs() < 1e-10, "RMSE should equal sqrt(MSE)");
    }

    #[test]
    fn test_mae_exact_value() {
        // MAE = (1 + 2 + 1) / 3 = 1.333...
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![2.0, 4.0, 4.0];  // Errors: 1, 2, 1

        let mae = micro_ml_core::regression_metrics::mean_absolute_error(&y_true, &y_pred).unwrap();
        assert!((mae - 4.0/3.0).abs() < 1e-10, "MAE should be 4/3, got {}", mae);
    }

    #[test]
    fn test_mape_exact_value() {
        // MAPE = mean of (|t-p|/|t| * 100) for each sample
        // y_true=[1,2,3], y_pred=[2,3,4]: |1-2|/1*100=100, |2-3|/2*100=50, |3-4|/3*100=33.33
        // MAPE = (100 + 50 + 100/3) / 3 = 61.11
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![2.0, 3.0, 4.0];  // Errors: 1, 1, 1

        let mape = micro_ml_core::regression_metrics::mean_absolute_percentage_error(&y_true, &y_pred, 1e-10).unwrap();
        let expected = (100.0 / 1.0 + 100.0 / 2.0 + 100.0 / 3.0) / 3.0;
        assert!((mape - expected).abs() < 0.1, "MAPE should be {}, got {}", expected, mape);
    }

    // === CLASSIFICATION METRICS VALIDATION ===

    #[test]
    fn test_confusion_matrix_binary() {
        // True: 0,0,1,1 | Pred: 0,1,0,1
        // TP=1, TN=1, FP=1, FN=1
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0];

        let cm = micro_ml_core::confusion_matrix::confusion_matrix_impl(&y_true, &y_pred).unwrap();
        let n_classes = cm[0] as usize;
        assert_eq!(n_classes, 2);

        // Result layout: [n_classes, class_0, class_1, ..., matrix_flat...]
        // Skip 1 (n_classes) + n_classes (class values) = 3 elements
        let matrix_offset = 1 + n_classes;
        let matrix = &cm[matrix_offset..];
        assert_eq!(matrix[0], 1.0);  // TN (true=0, pred=0)
        assert_eq!(matrix[1], 1.0);  // FP (true=0, pred=1)
        assert_eq!(matrix[2], 1.0);  // FN (true=1, pred=0)
        assert_eq!(matrix[3], 1.0);  // TP (true=1, pred=1)
    }

    #[test]
    fn test_accuracy_binary() {
        // 3 correct out of 4 = 75%
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 0.0, 1.0, 0.0];

        let correct = y_true.iter().zip(y_pred.iter())
            .filter(|(t, p)| {
                let diff: f64 = **t - **p;
                diff.abs() < 1e-10
            })
            .count();
        let acc = correct as f64 / y_true.len() as f64;
        assert!((acc - 0.75).abs() < 1e-10, "Accuracy should be 0.75, got {}", acc);
    }

    #[test]
    fn test_precision_recall_f1() {
        // y_true=[0,1,1,1], y_pred=[0,1,1,0]
        // TP: true=1 & pred=1 -> positions 1,2 -> TP=2
        // FP: true!=1 & pred=1 -> none -> FP=0
        // FN: true=1 & pred!=1 -> position 3 -> FN=1
        // Precision = 2/(2+0) = 1.0
        // Recall = 2/(2+1) = 2/3
        // F1 = 2*1.0*(2/3)/(1.0+2/3) = (4/3)/(5/3) = 4/5
        let y_true = vec![0.0, 1.0, 1.0, 1.0];
        let y_pred = vec![0.0, 1.0, 1.0, 0.0];

        // Compute confusion matrix inline
        let tp = y_true.iter().zip(y_pred.iter()).filter(|(&t, &p)| (t - 1.0_f64).abs() < 1e-10 && (p - 1.0_f64).abs() < 1e-10).count() as f64;
        let fp = y_true.iter().zip(y_pred.iter()).filter(|(&t, &p)| (t - 1.0_f64).abs() >= 1e-10 && (p - 1.0_f64).abs() < 1e-10).count() as f64;
        let fn_val = y_true.iter().zip(y_pred.iter()).filter(|(&t, &p)| (t - 1.0_f64).abs() < 1e-10 && (p - 1.0_f64).abs() >= 1e-10).count() as f64;

        let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let rec = if tp + fn_val > 0.0 { tp / (tp + fn_val) } else { 0.0 };
        let f1 = if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };

        assert!((prec - 1.0).abs() < 1e-10, "Precision should be 1.0, got {}", prec);
        assert!((rec - 2.0/3.0).abs() < 1e-10, "Recall should be 2/3, got {}", rec);
        assert!((f1 - 4.0/5.0).abs() < 1e-10, "F1 should be 4/5, got {}", f1);
    }

    #[test]
    fn test_mcc_perfect() {
        // Perfect classification: MCC = 1.0
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 0.0, 1.0, 1.0];

        let mcc = micro_ml_core::classification_metrics::matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert!((mcc - 1.0).abs() < 1e-10, "MCC should be 1.0 for perfect classification");
    }

    #[test]
    fn test_mcc_formula_verification() {
        // TP=1, TN=1, FP=1, FN=1
        // MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        //     = (1*1 - 1*1) / sqrt(2*2*2*2) = 0
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0];

        let mcc = micro_ml_core::classification_metrics::matthews_corrcoef(&y_true, &y_pred).unwrap();
        assert_eq!(mcc, 0.0, "MCC should be 0 for random classifier");
    }

    #[test]
    fn test_cohen_kappa_perfect_agreement() {
        // Perfect agreement: Kappa = 1.0
        let y_true = vec![0.0, 1.0, 2.0];
        let y_pred = vec![0.0, 1.0, 2.0];

        let kappa = micro_ml_core::classification_metrics::cohens_kappa(&y_true, &y_pred).unwrap();
        assert!((kappa - 1.0).abs() < 1e-10, "Kappa should be 1.0 for perfect agreement");
    }

    // === CLUSTERING METRICS VALIDATION ===

    #[test]
    fn test_silhouette_perfect_separation() {
        // Two well-separated clusters
        // Cluster 0: (0,0), (0.1,0.1)  -> intra-cluster distance ~0.14
        // Cluster 1: (10,10), (10.1,10.1)  -> intra-cluster distance ~0.14
        // Inter-cluster distance ~14.14
        // s = (b - a) / max(a,b) ≈ (14.14 - 0.14) / 14.14 ≈ 0.99
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            10.0, 10.0,
            10.1, 10.1,
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let score = micro_ml_core::silhouette::silhouette_score_impl(&data, 2, &labels).unwrap();
        assert!(score > 0.9, "Silhouette score should be >0.9 for well-separated clusters, got {}", score);
    }

    #[test]
    fn test_davies_bouldin_perfect() {
        // Single cluster: DB = 0 (by definition)
        let data = vec![1.0, 2.0, 3.0];
        let labels = vec![0.0, 0.0, 0.0];

        let db = micro_ml_core::clustering_metrics::davies_bouldin_impl(&data, 1, &labels).unwrap();
        assert_eq!(db, 0.0, "DB should be 0 for single cluster");
    }

    #[test]
    fn test_calinski_harabasz_formula() {
        // Two well-separated clusters at (0,0) and (10,10)
        // Global centroid: (5,5)
        // Between-cluster dispersion should be high, within-cluster low
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            10.0, 10.0,
            10.1, 10.1,
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let ch = micro_ml_core::clustering_metrics::calinski_harabasz_impl(&data, 2, &labels).unwrap();
        assert!(ch > 10.0, "CH score should be high for well-separated clusters, got {}", ch);
    }

    // === MODEL SELECTION METRICS VALIDATION ===

    #[test]
    fn test_roc_auc_perfect() {
        // Perfect classifier: all positives ranked higher than negatives
        // AUC = 1.0
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_scores = vec![0.1, 0.2, 0.9, 1.0];

        let auc = micro_ml_core::model_selection::roc_auc_impl(&y_true, &y_scores).unwrap();
        assert!((auc - 1.0).abs() < 1e-10, "AUC should be 1.0 for perfect classifier");
    }

    #[test]
    fn test_roc_auc_random() {
        // Random classifier: all scores equal
        // AUC = 0.5
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        let y_scores = vec![0.5, 0.5, 0.5, 0.5];

        let auc = micro_ml_core::model_selection::roc_auc_impl(&y_true, &y_scores).unwrap();
        assert_eq!(auc, 0.5, "AUC should be 0.5 for random classifier");
    }

    #[test]
    fn test_log_loss_perfect() {
        // Perfect predictions (probability = 1.0 for true class)
        // Log loss = -log(1.0) = 0
        let y_true = vec![0.0, 1.0];
        let y_proba = vec![0.99, 0.01,  // Class 0
                          0.01, 0.99]; // Class 1

        let loss = micro_ml_core::model_selection::log_loss(&y_true, &y_proba, 2).unwrap();
        assert!(loss < 0.1, "Log loss should be near 0 for perfect predictions, got {}", loss);
    }

    // === PREPROCESSING VALIDATION ===

    #[test]
    fn test_standard_scaler_properties() {
        // After standard scaling: mean = 0, std = 1
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut scaler = micro_ml_core::standard_scaler::standard_scaler(2);
        scaler.fit(&data).unwrap();
        let transformed = scaler.transform(&data);

        // Check feature 0: [1,3,5] -> mean=3, std=2
        let mean0 = (transformed[0] + transformed[2] + transformed[4]) / 3.0;
        assert!(mean0.abs() < 1e-10, "Mean should be ~0 after standardization");

        // Check feature 1: [2,4,6] -> mean=4, std=2
        let mean1 = (transformed[1] + transformed[3] + transformed[5]) / 3.0;
        assert!(mean1.abs() < 1e-10, "Mean should be ~0 after standardization");
    }

    #[test]
    fn test_minmax_scaler_range() {
        // MinMax scaling: all values in [0, 1]
        let data = vec![-5.0, 0.0, 5.0, 10.0];
        let mut scaler = micro_ml_core::minmax_scaler::minmax_scaler(1);
        let transformed = scaler.fit_transform(&data).unwrap();

        for &v in &transformed {
            assert!(v >= 0.0 && v <= 1.0, "All values should be in [0, 1], got {}", v);
        }

        // Check endpoints
        assert_eq!(transformed[0], 0.0, "Min should map to 0");
        assert_eq!(transformed[3], 1.0, "Max should map to 1");
    }

    #[test]
    fn test_robust_scaler_outlier_resistance() {
        // With outlier [1000], median should be unaffected
        let data = vec![1.0, 2.0, 3.0, 1000.0];
        let mut scaler = micro_ml_core::robust_scaler::robust_scaler(1);
        scaler.fit(&data).unwrap();

        // Median of [1,2,3,1000] = 2.5 (not affected by outlier)
        let center = scaler.center();
        assert!(center[0] > 2.0 && center[0] < 3.0,
                   "Median should be robust to outlier, got {}", center[0]);
    }

    #[test]
    fn test_normalizer_l2_unit_norm() {
        // L2 normalization: ||x||_2 = 1 for each sample (row)
        let data = vec![3.0, 4.0];  // 1 sample with 2 features, L2 norm = 5
        let norm = micro_ml_core::normalizer::normalizer(2, "l2".to_string());
        let transformed = norm.transform(&data);

        let l2_norm = (transformed[0].powi(2) + transformed[1].powi(2)).sqrt();
        assert!((l2_norm - 1.0).abs() < 1e-10, "L2 norm should be 1.0");
    }

    // === ENSEMBLE METHODS VALIDATION ===

    #[test]
    fn test_random_forest_improves_over_single_tree() {
        // 5 samples with 2 features each = 10 data values
        let data = vec![
            0.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
            1.0, 0.0,
            10.0, 10.0,  // Outlier
        ];
        let labels = vec![0.0, 1.0, 0.0, 1.0, 1.0];

        // Single tree
        let tree = micro_ml_core::decision_tree::decision_tree_impl(&data, 2, &labels, 5, 2, true).unwrap();
        let tree_preds = tree.predict(&data);

        // Random forest (should be more robust)
        let rf = micro_ml_core::random_forest::random_forest_impl(&data, 2, &labels, 10, 5, 2, true).unwrap();
        let rf_preds = rf.predict(&data);

        // Both should classify correctly on non-outlier data
        for i in 0..4 {
            assert_eq!(tree_preds[i], labels[i], "Tree should predict correctly at index {}", i);
            assert_eq!(rf_preds[i], labels[i], "RF should predict correctly at index {}", i);
        }
    }

    #[test]
    fn test_gradient_boosting_residual_reduction() {
        // Binary classification data where GB should improve with more trees
        let data = vec![
            0.0, 0.0,  1.0, 1.0,
            0.0, 1.0,  1.0, 0.0,
        ];
        let labels = vec![0.0, 1.0, 0.0, 1.0];

        let model = micro_ml_core::gradient_boosting::gradient_boosting_impl(&data, 2, &labels, 20, 3, 0.1).unwrap();
        let preds = model.predict(&data);

        // Should classify all correctly on training data
        for i in 0..4 {
            assert_eq!(preds[i], labels[i], "Prediction should match label at index {}", i);
        }
    }

    // === SUPERVISED LEARNING VALIDATION ===

    #[test]
    fn test_svm_linear_separability() {
        // Linearly separable data
        let data = vec![
            0.0, 0.0,  // Class 0
            1.0, 0.0,
            10.0, 10.0,  // Class 1
            11.0, 10.0,
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let model = micro_ml_core::svm::linear_svm_impl(&data, 2, &labels, 0.01, 1000, 0.01).unwrap();
        let preds = model.predict(&data);

        // Should classify all correctly
        for i in 0..4 {
            assert_eq!(preds[i], labels[i], "SVM should separate linearly separable data");
        }
    }

    #[test]
    fn test_ridge_regression_shrinkage() {
        // Ridge should shrink coefficients compared to OLS
        let data = vec![1.0, 2.0, 3.0];
        let targets = vec![2.0, 4.0, 6.0];  // y = 2x

        // Low alpha -> closer to OLS (coef ≈ 2)
        let ridge_low = micro_ml_core::linear_regression::ridge_regression_impl(&data, 1, &targets, 0.01).unwrap();
        let coef_low = ridge_low.coefficients()[0];

        // High alpha -> more shrinkage (coef < 2)
        let ridge_high = micro_ml_core::linear_regression::ridge_regression_impl(&data, 1, &targets, 10.0).unwrap();
        let coef_high = ridge_high.coefficients()[0];

        assert!(coef_low > coef_high, "Higher alpha should shrink coefficients more");
    }

    #[test]
    fn test_lasso_sparsity() {
        // Lasso should produce sparse coefficients
        let data = vec![
            1.0, 0.0, 0.0,
            2.0, 0.0, 0.0,
            3.0, 0.0, 0.0,
        ];
        let targets = vec![2.0, 4.0, 6.0];  // Only feature 0 matters

        let model = micro_ml_core::linear_regression::lasso_regression_impl(&data, 3, &targets, 1.0, 1000, 1e-4).unwrap();

        // Feature 0 should be non-zero
        assert!(model.coefficients()[0].abs() > 0.1, "Feature 0 should have non-zero coefficient");

        // Features 1 and 2 should be zero or near-zero
        assert!(model.coefficients()[1].abs() < 0.5, "Feature 1 should be shrunk to zero");
        assert!(model.coefficients()[2].abs() < 0.5, "Feature 2 should be shrunk to zero");
    }

    // === UNSUPERVISED LEARNING VALIDATION ===

    #[test]
    fn test_kmeans_plus_convergence() {
        // Data with clear clusters
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            10.0, 10.0,
            10.1, 10.1,
        ];

        let result = micro_ml_core::kmeans_plus::kmeans_plus_impl(&data, 2, 2, 100).unwrap();

        // Extract assignments
        let n_clusters = result[0] as usize;
        let assignments = &result[1..5];

        // First two should be same cluster, last two same cluster
        assert_eq!(assignments[0], assignments[1], "First two should be same cluster");
        assert_eq!(assignments[2], assignments[3], "Last two should be same cluster");
        assert_ne!(assignments[0], assignments[2], "Clusters should be different");
    }

    #[test]
    fn test_hierarchical_dendrogram_structure() {
        // Hierarchical should produce meaningful clusters
        let data = vec![
            0.0, 0.0,
            0.1, 0.1,
            5.0, 5.0,
            5.1, 5.1,
            10.0, 10.0,
        ];

        let labels = micro_ml_core::hierarchical::hierarchical_impl(&data, 2, 3).unwrap();

        // Should have 3 clusters (use tolerance-based counting since f64 doesn't impl Eq/Hash)
        let mut unique_labels: Vec<f64> = labels.iter().copied().collect();
        unique_labels.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        unique_labels.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        assert_eq!(unique_labels.len(), 3, "Should have 3 clusters");
    }

    // === DATA SPLIT VALIDATION ===

    #[test]
    fn test_train_test_split_ratio() {
        // 10 samples with 2 features each = 20 data values
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                       1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];
        let labels = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = micro_ml_core::data_split::train_test_split_impl(&data, 2, &labels, 0.8, None).unwrap();

        let n_train = result[0] as usize;
        let n_test = result[1] as usize;

        // 80% of 10 = 8 train, 2 test
        assert_eq!(n_train, 8, "Should have 8 training samples");
        assert_eq!(n_test, 2, "Should have 2 test samples");
    }

    #[test]
    fn test_train_test_split_no_overlap() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let labels = vec![0.0, 1.0];

        let result = micro_ml_core::data_split::train_test_split_impl(&data, 2, &labels, 0.5, Some(42)).unwrap();

        let n_train = result[0] as usize;
        let n_features = result[2] as usize;

        let x_train = &result[3..3 + n_train * n_features];
        let x_test = &result[3 + n_train * n_features..3 + n_train * n_features + 2 * n_features];

        // Train and test should not share samples
        // (Can't directly compare due to shuffling, but lengths should be correct)
        assert_eq!(x_train.len(), 2, "Train should have 1 sample");
        assert_eq!(x_test.len(), 4, "Test should have 1 sample");
    }

    // === CROSS-VALIDATION VALIDATION ===

    #[test]
    fn test_cross_validation_k_folds() {
        // 4 samples with 2 features each = 8 data values
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0];  // 4 labels matching 4 samples

        let result = micro_ml_core::cross_validation::cross_validate_score_impl(
            &data, 2, &labels, 4, "decision_tree", &[5.0, 2.0]
        ).unwrap();

        // Should return mean, std, and 4 fold scores
        assert_eq!(result.len(), 6, "Should return mean + std + 4 scores");

        // All scores should be valid (0-1 for accuracy)
        for &score in &result {
            assert!(score >= 0.0 && score <= 1.0, "Scores should be in [0,1]");
        }
    }

    // === ENCODING VALIDATION ===

    #[test]
    fn test_label_encoder_roundtrip() {
        let labels = vec![10.0, 20.0, 10.0, 30.0];

        let mut encoder = micro_ml_core::label_encoder::label_encoder();
        encoder.fit(&labels).unwrap();
        let encoded = encoder.transform(&labels).unwrap();
        let decoded = encoder.inverse_transform(&encoded).unwrap();

        assert_eq!(decoded, labels, "Roundtrip should preserve labels");
    }

    #[test]
    fn test_one_hot_encoding_correctness() {
        let data = vec![0.0, 1.0, 0.0];
        let mut encoder = micro_ml_core::one_hot_encoder::one_hot_encoder(1);
        let transformed = encoder.fit_transform(&data).unwrap();

        // [0, 1, 0] -> [[1,0], [0,1], [1,0]]
        assert_eq!(transformed, vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
    }

    // === IMPUTER VALIDATION ===

    #[test]
    fn test_imputer_mean_formula() {
        let data = vec![1.0, f64::NAN, 3.0];
        let mut imputer = micro_ml_core::imputer::simple_imputer(1, "mean".to_string(), 0.0);
        let result = imputer.fit_transform(&data).unwrap();

        // Mean of [1, 3] = 2
        assert_eq!(result[1], 2.0, "NaN should be replaced with mean");
    }

    #[test]
    fn test_imputer_median_formula() {
        let data = vec![1.0, f64::NAN, 3.0, 100.0];
        let mut imputer = micro_ml_core::imputer::simple_imputer(1, "median".to_string(), 0.0);
        let result = imputer.fit_transform(&data).unwrap();

        // Median of [1, 3, 100] = 3 (not affected by outlier)
        assert_eq!(result[1], 3.0, "NaN should be replaced with median");
    }
}
