# -*- coding: utf-8 -*-
"""
Created on Mon May 26 20:09:31 2025

@author: Ashmitha
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import tempfile
import os
import gradio as gr
import scipy.linalg as la
RANDOM_STATE=42

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Add
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import scipy.linalg as la

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Add
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import scipy.linalg as la

def DeepMap(
    trainX, trainy,
    valX=None, valy=None,
    testX=None, testy=None,
    K_train=None, K_val=None, K_test=None,
    # --- DNN (Base Learner) Hyperparameters ---
    epochs=1000,
    batch_size=64,
    learning_rate=0.0001,
    l2_reg=0.01,
    dropout_rate=0.8,
    # --- GBLUP (Base Learner) Hyperparameters ---
    gblup_lambda=0.7,
    # --- SVR (Meta-Learner) Hyperparameters ---
    svr_kernel='rbf',
    svr_C=5.0,
    svr_gamma='scale',
    # --- Misc ---
    verbose=1
):
    """
    DeepMap: A hybrid DNN + GBLUP model with SVR as the meta-learner.
   
    Returns:
        predicted_train, predicted_val, predicted_test: Final predictions (in original scale)
        history: DNN training history
        svr_meta: Trained SVR meta-learner
        dnn_model: Trained DNN base learner
    """
   
    # Initialize results
    predicted_test = None
   
    # -------------------------------- Feature Scaling
    feature_scaler = StandardScaler()
    trainX_scaled = feature_scaler.fit_transform(trainX)
    valX_scaled = feature_scaler.transform(valX) if valX is not None else None
    testX_scaled = feature_scaler.transform(testX) if testX is not None else None
   
    # -------------------------------- Target Scaling
    target_scaler = StandardScaler()
    trainy_scaled = target_scaler.fit_transform(trainy.reshape(-1, 1)).flatten()
    valy_scaled = target_scaler.transform(valy.reshape(-1, 1)).flatten() if valy is not None else None
    testy_scaled = target_scaler.transform(testy.reshape(-1, 1)).flatten() if testy is not None else None
   
    # -------------------------------- Build DNN Model (Base Learner)
    def build_dnn_model(input_shape):
        inputs = tf.keras.Input(shape=(input_shape,))  
        x = Dense(512, kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
       
        # First residual block
        res = x
        x = Dense(256, kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)  
       
        if res.shape[-1] != x.shape[-1]:
            res = Dense(256, kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l2_reg))(res)
        x = Add()([x, res])
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
       
        # Second residual block
        res = x
        x = Dense(128, kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)  
       
        if res.shape[-1] != x.shape[-1]:
            res = Dense(128, kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l2_reg))(res)
        x = Add()([x, res])
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Final layers
        x = Dense(64, kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Dense(32, kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU(alpha=0.1)(x)
       
        outputs = Dense(1, activation="relu")(x)
        model = tf.keras.Model(inputs, outputs)
       
        model.compile(
            loss=tf.keras.losses.Huber(delta=0.1),
            optimizer=Adam(learning_rate=learning_rate, clipvalue=0.1),
            metrics=['mse']
        )
        return model
   
    dnn_model = build_dnn_model(trainX.shape[1])
   
    # -------------------------------- Train DNN Model
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            verbose=verbose,
            restore_best_weights=True,
            patience=15
        )
    ]
   
    if valX is not None and valy is not None:
        validation_data = (valX_scaled, valy_scaled)
        validation_split = 0.0
    else:
        validation_data = None
        validation_split = 0.1
   
    history = dnn_model.fit(
        trainX_scaled,
        trainy_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        validation_split=validation_split,
        verbose=verbose,
        callbacks=callbacks,
        
    )
   
    # -------------------------------- GBLUP (Base Learner)
    def run_gblup(y, K_train, K_val=None, K_test=None, lambda_=0.01):
        """Run GBLUP model"""
        n = len(y)
        K_reg = K_train + lambda_ * np.eye(n)
        L = la.cholesky(K_reg, lower=True)
        alpha = la.solve_triangular(L.T, la.solve_triangular(L, y, lower=True))
        u_hat = K_train @ alpha
       
        pred_train = u_hat
        pred_val = K_val @ alpha if K_val is not None else None
        pred_test = K_test @ alpha if K_test is not None else None
       
        return pred_train, pred_val, pred_test
   
    pred_train_gblup, pred_val_gblup, pred_test_gblup = run_gblup(
        trainy_scaled, K_train, K_val, K_test, lambda_=gblup_lambda
    )
   
    # -------------------------------- Base Model Predictions
    predicted_train_dnn_scaled = dnn_model.predict(trainX_scaled).flatten()
    predicted_val_dnn_scaled = dnn_model.predict(valX_scaled).flatten() if valX is not None else None
    predicted_test_dnn_scaled = dnn_model.predict(testX_scaled).flatten() if testX is not None else None
   
    # Stack predictions for meta-learner
    if predicted_train_dnn_scaled is None or pred_train_gblup is None:
        raise ValueError(
            "One of the base-learner prediction is None."  
            )
    if predicted_train_dnn_scaled.shape != pred_train_gblup.shape:
        raise ValueError(
            f"Shape mismatch between DNN and GBLUP prediction:"
            f"DNN has {predicted_train_dnn_scaled.shape},"
            f"GLUP has {pred_train_gblup.shape}"
            )
       
    meta_train_features = np.vstack((predicted_train_dnn_scaled, pred_train_gblup)).T
    meta_val_features = np.vstack((predicted_val_dnn_scaled, pred_val_gblup)).T if valX is not None else None
    meta_test_features = np.vstack((predicted_test_dnn_scaled, pred_test_gblup)).T if testX is not None else None
   
    meta_scaler = StandardScaler()
    meta_train_features_scaled = meta_scaler.fit_transform(meta_train_features)
    meta_val_features_scaled = meta_scaler.transform(meta_val_features) if meta_val_features is not None else None
    meta_test_features_scaled = meta_scaler.transform(meta_test_features) if meta_test_features is not None else None
   
    # -------------------------------- SVR Meta-Learner
    svr_meta = SVR(kernel=svr_kernel, C=svr_C, gamma=svr_gamma)
    svr_meta.fit(meta_train_features_scaled, trainy_scaled)
   
    # -------------------------------- Final Predictions
    predicted_train_scaled = svr_meta.predict(meta_train_features_scaled)
    predicted_val_scaled = svr_meta.predict(meta_val_features_scaled) if meta_val_features_scaled is not None else None
    predicted_test_scaled = svr_meta.predict(meta_test_features_scaled) if meta_test_features_scaled is not None else None
   
    # Inverse scaling
    predicted_train = target_scaler.inverse_transform(predicted_train_scaled.reshape(-1, 1)).flatten()
    predicted_val = target_scaler.inverse_transform(predicted_val_scaled.reshape(-1, 1)).flatten() if predicted_val_scaled is not None else None
    predicted_test = target_scaler.inverse_transform(predicted_test_scaled.reshape(-1, 1)).flatten() if predicted_test_scaled is not None else None
   
    # Base model predictions (original scale)
    predicted_train_dnn = target_scaler.inverse_transform(predicted_train_dnn_scaled.reshape(-1, 1)).flatten()
    predicted_val_dnn = target_scaler.inverse_transform(predicted_val_dnn_scaled.reshape(-1, 1)).flatten() if predicted_val_dnn_scaled is not None else None
    predicted_test_dnn = target_scaler.inverse_transform(predicted_test_dnn_scaled.reshape(-1, 1)).flatten() if predicted_test_dnn_scaled is not None else None
   
    predicted_train_gblup = target_scaler.inverse_transform(pred_train_gblup.reshape(-1, 1)).flatten()
    predicted_val_gblup = target_scaler.inverse_transform(pred_val_gblup.reshape(-1, 1)).flatten() if pred_val_gblup is not None else None
    predicted_test_gblup = target_scaler.inverse_transform(pred_test_gblup.reshape(-1, 1)).flatten() if pred_test_gblup is not None else None
   
    # -------------------------------- Results Summary
    if verbose > 0:
        print("\n=== Training Summary ===")
        print(f"Train samples: {len(trainX)}, Validation samples: {len(valX) if valX is not None else 'N/A'}")
       
        train_mse_dnn = mean_squared_error(trainy, predicted_train_dnn)
        train_mse_gblup = mean_squared_error(trainy, predicted_train_gblup)
        train_mse_stacked = mean_squared_error(trainy, predicted_train)
        print(f"\nTraining MSE - DNN: {train_mse_dnn:.4f}, GBLUP: {train_mse_gblup:.4f}, Stacked Ensemble: {train_mse_stacked:.4f}")
       
        if valX is not None:
            val_mse_dnn = mean_squared_error(valy, predicted_val_dnn)
            val_mse_gblup = mean_squared_error(valy, predicted_val_gblup)
            val_mse_stacked = mean_squared_error(valy, predicted_val)
            print(f"Validation MSE - DNN: {val_mse_dnn:.4f}, GBLUP: {val_mse_gblup:.4f}, Stacked Ensemble: {val_mse_stacked:.4f}")
   
    return predicted_train, predicted_val, predicted_test, history, svr_meta, dnn_model
def compute_genomic_features(X, ref_features=None, is_train=False):
    
    if is_train and ref_features is None:
        # For training data when no reference provided
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_markers = X_scaled.shape[1]
        
        # Compute relationship matrices for training data only
        G_train = np.dot(X_scaled, X_scaled.T) / n_markers
        I_train = G_train * G_train
        
        # Store reference statistics
        mean_diag = np.mean(np.diag(I_train))
        I_train_norm = I_train / mean_diag
        
        # Combine features
        X_final = np.concatenate([G_train, I_train_norm], axis=1)
        
        # Store reference info for validation/test
        ref_features = {
            'scaler': scaler,
            'mean_diag': mean_diag,
            'X_train_scaled': X_scaled
        }
        
        # Return both features and kinship matrix
        return X_final, ref_features, G_train
        
    elif not is_train and ref_features is not None:
        # For validation/test data with reference features
        X_scaled = ref_features['scaler'].transform(X)
        n_markers = X_scaled.shape[1]
        
        # Compute relationship with training samples only
        G_val = np.dot(X_scaled, ref_features['X_train_scaled'].T) / n_markers
        I_val = G_val * G_val
        
        # Normalize using training statistics
        I_val_norm = I_val / ref_features['mean_diag']
        
        # Construct features
        X_final = np.concatenate([G_val, I_val_norm], axis=1)
        
        # Return features and cross-kinship matrix
        return X_final, ref_features, G_val
    
    else:
        raise ValueError("Invalid combination of is_train and ref_features parameters")

def calculate_metrics(true_values, predicted_values):
    """Compute performance metrics between true and predicted values"""
    mask = ~np.isnan(predicted_values)
    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan, np.nan
    true_values = true_values[mask]
    predicted_values = predicted_values[mask]
    
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    corr = pearsonr(true_values, predicted_values)[0]
    return mse, rmse, corr, r2

def KFoldCrossValidation(training_data, training_additive, testing_data, testing_additive,
                        epochs=1000, learning_rate=0.0001, batch_size=64,
                        outer_n_splits=10, output_file='cross_validation_results.csv',
                        train_pred_file='train_predictions.csv', 
                        val_pred_file='validation_predictions.csv',
                        test_pred_file='test_predictions.csv',
                        feature_selection=True):
    
    # -------------------------------- Data Validation
    assert isinstance(training_data, pd.DataFrame), "Training data must be DataFrame"
    assert isinstance(testing_data, pd.DataFrame), "Testing data must be DataFrame"
    
    train_ids = set(training_data.iloc[:, 0].values)
    test_ids = set(testing_data.iloc[:, 0].values)
    assert len(train_ids & test_ids) == 0, "Training and testing sets must be distinct"
    
    # -------------------------------- Data Preparation
    training_additive_raw = training_additive.iloc[:, 1:].values
    testing_additive_raw = testing_additive.iloc[:, 1:].values
    phenotypic_info = training_data['phenotypes'].values
    
    has_test_phenotypes = 'phenotypes' in testing_data.columns
    phenotypic_test_info = testing_data['phenotypes'].values if has_test_phenotypes else None
    test_sample_ids = testing_data.iloc[:, 0].values
    
    # -------------------------------- Outer CV Loop
    outer_kf = KFold(n_splits=outer_n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []
    train_predictions = []
    val_predictions = []

    for outer_fold, (outer_train_index, outer_val_index) in enumerate(outer_kf.split(training_additive_raw), 1):
        print(f"\n=== Outer Fold {outer_fold}/{outer_n_splits} ===")
        
        # Split data
        outer_trainX = training_additive_raw[outer_train_index]
        outer_valX = training_additive_raw[outer_val_index]
        outer_trainy = phenotypic_info[outer_train_index]
        outer_valy = phenotypic_info[outer_val_index]
        
        # Process features and compute kinship matrices without leakage
        X_train_genomic, ref_features, K_train = compute_genomic_features(
            outer_trainX, 
            ref_features=None,
            is_train=True
        )
        
        X_val_genomic, _, K_val = compute_genomic_features(
            outer_valX, 
            ref_features=ref_features,
            is_train=False
        )
        
        # Feature selection without leakage
        if feature_selection:
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), 
                threshold="1.25*median"
            )
            selector.fit(X_train_genomic, outer_trainy)
            X_train_final = selector.transform(X_train_genomic)
            X_val_final = selector.transform(X_val_genomic)
        else:
            X_train_final = X_train_genomic
            X_val_final = X_val_genomic
            
        # Model training with kinship matrices
        pred_train, pred_val, _, history, _,_ = DeepMap(
            trainX=X_train_final, 
            trainy=outer_trainy,
            valX=X_val_final,
            valy=outer_valy,
            K_train=K_train,  # Pass kinship matrix for training
            K_val=K_val,      # Pass cross-kinship matrix for validation
            testX=None,
            testy=None,
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            verbose=1
        )
        
        # Store predictions
        train_predictions.append(pd.DataFrame({
            'Sample_ID': training_data.iloc[outer_train_index, 0].values,
            'True_Phenotype': outer_trainy,
            'Predicted_Phenotype': pred_train,
            'Fold': outer_fold
        }))
        
        val_predictions.append(pd.DataFrame({
            'Sample_ID': training_data.iloc[outer_val_index, 0].values,
            'True_Phenotype': outer_valy,
            'Predicted_Phenotype': pred_val,
            'Fold': outer_fold
        }))
        
        # Calculate metrics
        mse_train, rmse_train, corr_train, r2_train = calculate_metrics(outer_trainy, pred_train)
        mse_val, rmse_val, corr_val, r2_val = calculate_metrics(outer_valy, pred_val)
        
        results.append({
            'Fold': outer_fold,
            'Train_MSE': mse_train, 'Train_RMSE': rmse_train,
            'Train_R2': r2_train, 'Train_Corr': corr_train,
            'Val_MSE': mse_val, 'Val_RMSE': rmse_val,
            'Val_R2': r2_val, 'Val_Corr': corr_val
        })
    
    # -------------------------------- FINAL MODEL TRAINING
    print("\n==============================Training Final model on ALL training data")
    
    # Process ALL training data
    X_train_raw = training_additive_raw
    y_train_raw = phenotypic_info

    # Feature processing and kinship computation
    X_train_genomic, ref_features, K_train = compute_genomic_features(
        X_train_raw, 
        ref_features=None, 
        is_train=True
    )
    X_test_genomic, _, K_test = compute_genomic_features(
        testing_additive_raw, 
        ref_features=ref_features, 
        is_train=False
    )

    # Feature selection
    if feature_selection:
        selector = SelectFromModel(
            RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE), 
            threshold="1.25*median"
        )
        selector.fit(X_train_genomic, y_train_raw)
        X_train_final = selector.transform(X_train_genomic)
        X_test_final = selector.transform(X_test_genomic)
    else:
        X_train_final = X_train_genomic
        X_test_final = X_test_genomic

    # Train final model with kinship matrices
    _, _, pred_test_final, _, _,_ = DeepMap(
        trainX=X_train_final, 
        trainy=y_train_raw,
        valX=None,
        valy=None,
        testX=X_test_final,
        testy=phenotypic_test_info if has_test_phenotypes else None,
        K_train=K_train,  # Full training kinship
        K_test=K_test,     # Test-training cross-kinship
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=1
    )
    
    # Create final test predictions
    test_pred_final_df = pd.DataFrame({
        'Sample_ID': test_sample_ids,
        'Predicted_Phenotype': pred_test_final,
        'Model': 'Final'
    })
    if has_test_phenotypes:
        test_pred_final_df['True_Phenotype'] = phenotypic_test_info
    
    # Calculate final test metrics if available
    if has_test_phenotypes:
        mse_test_final, rmse_test_final, corr_test_final, r2_test_final = calculate_metrics(
            phenotypic_test_info, pred_test_final
        )
        results.append({
            'Fold': 'Final_Model',
            'Test_MSE': mse_test_final, 'Test_RMSE': rmse_test_final,
            'Test_R2': r2_test_final, 'Test_Corr': corr_test_final
        })
        
        print(f"\n=== Final Test Results ===")
        print(f"MSE: {mse_test_final:.4f}, RMSE: {rmse_test_final:.4f}")
        print(f"R²: {r2_test_final:.4f}, Correlation: {corr_test_final:.4f}")
    
    # -------------------------------- Output Preparation (same as before)
    train_pred_df = pd.concat(train_predictions, ignore_index=True)
    val_pred_df = pd.concat(val_predictions, ignore_index=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    train_pred_df.to_csv(train_pred_file, index=False)
    val_pred_df.to_csv(val_pred_file, index=False)
    test_pred_final_df.to_csv(test_pred_file, index=False)
    
    # -------------------------------- Generate Plots (same as before)
    def generate_plot(true_vals, pred_vals, title, is_test=False):
        plt.figure(figsize=(10, 6))
        if is_test and not has_test_phenotypes:
            pred_values = pred_vals.dropna()
            if len(pred_values) > 0:
                plt.hist(pred_values, bins=30)
                plt.xlabel('Predicted Phenotype')
                plt.ylabel('Frequency')
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        else:
            valid_mask = (~pd.isna(pred_vals)) & (~pd.isna(true_vals))
            if valid_mask.any():
                plt.scatter(true_vals[valid_mask], pred_vals[valid_mask], alpha=0.5)
                plt.xlabel('True Phenotype')
                plt.ylabel('Predicted Phenotype')
                coef = np.polyfit(true_vals[valid_mask], pred_vals[valid_mask], 1)
                poly1d_fn = np.poly1d(coef)
                plt.plot(true_vals[valid_mask], poly1d_fn(true_vals[valid_mask]), '--k')
            else:
                plt.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
        
        plt.title(title)
        plot_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        plt.savefig(plot_file)
        plt.close()
        return plot_file
    
    plot_files_train = [generate_plot(train_pred_df['True_Phenotype'], 
                                    train_pred_df['Predicted_Phenotype'], 
                                    'Training Set Predictions')]
    
    plot_files_val = [generate_plot(val_pred_df['True_Phenotype'], 
                                  val_pred_df['Predicted_Phenotype'], 
                                  'Validation Set Predictions')]
    
    plot_files_test = [generate_plot(test_pred_final_df.get('True_Phenotype', None),
                                   test_pred_final_df['Predicted_Phenotype'],
                                   'Test Set Predictions (Final Model)', 
                                   is_test=True)]
    
    return results_df, train_pred_df, val_pred_df, test_pred_final_df, plot_files_train, plot_files_val, plot_files_test

def run_cross_validation(training_file, training_additive_file, testing_file, testing_additive_file, 
                        feature_selection=True, learning_rate=0.0001, **kwargs):
    """
    Run cross-validation with the fixed model that prevents data leakage
    """
    
    # Load data
    training_data = pd.read_csv(training_file)
    training_additive = pd.read_csv(training_additive_file)
    testing_data = pd.read_csv(testing_file)
    testing_additive = pd.read_csv(testing_additive_file)
    
    # Check required columns
    if 'phenotypes' not in training_data.columns:
        raise ValueError("Training data must contain 'phenotypes' column")
    
    # Run cross-validation
    results, train_pred, val_pred, test_pred, train_plot, val_plot, test_plot = KFoldCrossValidation(
        training_data=training_data,
        training_additive=training_additive,
        testing_data=testing_data,
        testing_additive=testing_additive,
        epochs=1000,  # Reasonable number for demonstration
        batch_size=64,
        learning_rate=learning_rate,
        feature_selection=feature_selection,
        outer_n_splits=10
    )
    
    # Prepare files for download
    def save_to_temp(df, prefix):
        path = os.path.join(tempfile.gettempdir(), f"{prefix}_predictions.csv")
        df.to_csv(path, index=False)
        return path
    
    train_csv = save_to_temp(train_pred, "train")
    val_csv = save_to_temp(val_pred, "val")
    test_csv = save_to_temp(test_pred, "test")
    
    return (
        train_pred,  # train_output
        val_pred,    # val_output
        test_pred,   # test_output
        train_plot[0] if train_plot else None,
        val_plot[0] if val_plot else None,
        test_plot[0] if test_plot else None,
        train_csv,
        val_csv,
        test_csv
    )

# Gradio interface
with gr.Blocks(title="DeepMap-1.0 Genomic Prediction") as interface:
    gr.Markdown("# DeepMap-1.0 Genomic Prediction")
    
    with gr.Row():
        with gr.Column():
            training_file = gr.File(label="Training Data (CSV)", file_types=[".csv"])
            training_additive_file = gr.File(label="Training Additive Data (CSV)", file_types=[".csv"])
        with gr.Column():
            testing_file = gr.File(label="Testing Data (CSV)", file_types=[".csv"])
            testing_additive_file = gr.File(label="Testing Additive Data (CSV)", file_types=[".csv"])
    
    with gr.Accordion("Advanced Parameters", open=False):
        feature_selection = gr.Checkbox(label="Enable Feature Selection", value=True)
        learning_rate = gr.Slider(label="Learning Rate", minimum=0.0001, maximum=0.1, value=0.001, step=0.0001)
        
    with gr.Row():
        with gr.Column():
            train_output = gr.Dataframe(label="Training Predictions")
            train_plot = gr.Image(label="Training Plot")
            train_download = gr.File(label="Download Training Predictions", visible=True)
        with gr.Column():
            val_output = gr.Dataframe(label="Validation Predictions")
            val_plot = gr.Image(label="Validation Plot")
            val_download = gr.File(label="Download Validation Predictions", visible=True)
        with gr.Column():
            test_output = gr.Dataframe(label="Test Predictions")
            test_plot = gr.Image(label="Test Plot")
            test_download = gr.File(label="Download Test Predictions", visible=True)
    
    submit_btn = gr.Button("Run Prediction", variant="primary")
    
    submit_btn.click(
        fn=run_cross_validation,
        inputs=[
            training_file, training_additive_file, 
            testing_file, testing_additive_file,
            feature_selection, learning_rate
        ],
        outputs=[
            train_output, val_output, test_output,
            train_plot, val_plot, test_plot,
            train_download, val_download, test_download
        ]
    )

# Launch the interface
if __name__ == "__main__":
    interface.launch()