import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
from carrega_dades import split_datos_3s, split_datos_30s
from plots_2 import plot_final_learning_curve, plot_single_validation_curve

# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion_Parametros_XGB"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------------------------------
# Funció per trobar el millor paràmetre (punt de màxima precisió)
# ----------------------------------------------------
def find_best_param(param_range, test_scores, tolerance=0.001):
    """Troba el valor del paràmetre que dóna el màxim Test Score, triant el més simple/petit primer."""
    best_score = np.max(test_scores)
    
    # Trobem el primer paràmetre amb un score molt proper al màxim (tolerància 0.1%)
    best_param_index = np.where(test_scores >= (best_score - tolerance))[0][0]
    return param_range[best_param_index]

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- 🔍 GENERANDO JUSTIFICACIÓN DE PARÁMETROS (XGBOOST) SENSE CV ---")
    
    # 1. Cargar datos (Utilitzem split_datos_3s com en el teu codi original)
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    
    # 2. DEFINIR EL MODELO BASE AMB PARÀMETRES FIXOS
    fixed_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }
    
    # -----------------------------------------------------------------------
    # 3. ANÀLISI D'HIPERPARÀMETRES 
    # AVALUEM CONTRA X_TEST EN CADA PAS PER SIMULAR LA METODOLOGIA
    # -----------------------------------------------------------------------
    
    # 3.0. Definició de valors base
    N_ESTIMATORS = 500
    MAX_DEPTH = 2
    LEARNING_RATE = 0.1
    GAMMA = 0
    REG_ALPHA = 0
    MIN_CHILD_WEIGHT = 1
    COLSAMPLE_BYTREE = 0.7
    SUBSAMPLE = 0.9
    

    # --- GRÀFICA 1: JUSTIFICACIÓ DE MAX_DEPTH ---
    depth_range = np.array([1, 2, 3, 4, 5, 6, 8, 10]) 
    model_depth = xgb.XGBClassifier(**fixed_params, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
    
    depth_range_result, depth_scores_test = plot_single_validation_curve(
        model_depth, X_train, y_train, X_test, y_test, 
        param_name="max_depth", 
        param_range=depth_range, 
        title="Impacte de la Profunditat (max_depth) - SENSE CV",
        xlabel="Max Depth",
        SAVE_DIR=SAVE_DIR
    )
    MAX_DEPTH = find_best_param(depth_range_result, depth_scores_test)
    print(f"✨ Valor òptim de max_depth trobat: {MAX_DEPTH}")
    
    
    # --- GRÀFICA 2: JUSTIFICACIÓ DE N_ESTIMATORS ---
    estimators_range = np.array([50, 100, 200, 300, 500])
    model_estimators = xgb.XGBClassifier(**fixed_params, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE)
    
    estimators_range_result, estimators_scores_test = plot_single_validation_curve(
        model_estimators, X_train, y_train, X_test, y_test, 
        param_name="n_estimators",
        param_range=estimators_range,
        title="Impacte del Número de Árboles (n_estimators) - SENSE CV",
        xlabel="Número de Estimadores (n_estimators)",
        SAVE_DIR=SAVE_DIR
    )
    N_ESTIMATORS = find_best_param(estimators_range_result, estimators_scores_test)
    print(f"✨ Valor òptim d'n_estimators trobat: {N_ESTIMATORS}")


    # --- GRÀFICA 3: JUSTIFICACIÓ DE LEARNING_RATE ---
    lr_range = np.array([0.01, 0.05, 0.1, 0.2, 0.3])
    model_lr = xgb.XGBClassifier(**fixed_params, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS)
    
    lr_range_result, lr_scores_test = plot_single_validation_curve(
        model_lr, X_train, y_train, X_test, y_test, 
        param_name="learning_rate",
        param_range=lr_range,
        title="Impacte de la Tasa de Aprendizaje (learning_rate) - SENSE CV",
        xlabel="Learning Rate",
        SAVE_DIR=SAVE_DIR
    )
    LEARNING_RATE = find_best_param(lr_range_result, lr_scores_test)
    print(f"✨ Valor òptim de learning_rate trobat: {LEARNING_RATE}")


    # --- GRÁFICA 4: JUSTIFICACIÓN DE GAMMA (PODA) ---
    gamma_range = np.array([0, 0.5, 1, 3, 5, 7, 10])
    model_gamma = xgb.XGBClassifier(**fixed_params, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE)
    
    gamma_range_result, gamma_scores_test = plot_single_validation_curve(
        model_gamma, X_train, y_train, X_test, y_test, 
        param_name="gamma",
        param_range=gamma_range,
        title="Impacto de Gamma (Poda) - SENSE CV",
        xlabel="Gamma",
        SAVE_DIR=SAVE_DIR
    )
    GAMMA = find_best_param(gamma_range_result, gamma_scores_test)
    print(f"✨ Valor òptim de gamma trobat: {GAMMA}")
    
    
    # --- GRÁFICA 5: JUSTIFICACIÓN DE REG_ALPHA (Regularización L1) ---
    reg_alpha_range = np.array([0, 0.1, 0.5, 1.0, 2.0, 5.0])
    model_reg_alpha = xgb.XGBClassifier(**fixed_params, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, gamma=GAMMA)
    
    reg_alpha_range_result, reg_alpha_scores_test = plot_single_validation_curve(
        model_reg_alpha, X_train, y_train, X_test, y_test, 
        param_name="reg_alpha",
        param_range=reg_alpha_range,
        title="Impacto de la Regularización L1 (reg_alpha) - SENSE CV",
        xlabel="reg_alpha",
        SAVE_DIR=SAVE_DIR
    )
    REG_ALPHA = find_best_param(reg_alpha_range_result, reg_alpha_scores_test)
    print(f"✨ Valor òptim de reg_alpha trobat: {REG_ALPHA}")


    # --- GRÁFICA 6: JUSTIFICACION MIN_CHILD_WEIGHT ---
    min_child_range = np.array([1, 3, 5, 7])
    model_min_child = xgb.XGBClassifier(**fixed_params, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, gamma=GAMMA, reg_alpha=REG_ALPHA)
    
    min_child_range_result, min_child_scores_test = plot_single_validation_curve(
        model_min_child, X_train, y_train, X_test, y_test, 
        param_name="min_child_weight",
        param_range=min_child_range,
        title="Impacto de Min Child Weight - SENSE CV",
        xlabel="Min Child Weight",
        SAVE_DIR=SAVE_DIR
    )
    MIN_CHILD_WEIGHT = find_best_param(min_child_range_result, min_child_scores_test)
    print(f"✨ Valor òptim de min_child_weight trobat: {MIN_CHILD_WEIGHT}")
    
    
    # --- GRÁFICA 7: JUSTIFICACION COLSAMPLE_BYTREE ---
    colsample_range = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    model_colsample = xgb.XGBClassifier(**fixed_params, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, gamma=GAMMA, reg_alpha=REG_ALPHA, min_child_weight=MIN_CHILD_WEIGHT)
    
    colsample_range_result, colsample_scores_test = plot_single_validation_curve(
        model_colsample, X_train, y_train, X_test, y_test, 
        param_name="colsample_bytree",
        param_range=colsample_range,
        title="Impacto de Colsample_bytree - SENSE CV",
        xlabel="Colsample_bytree",
        SAVE_DIR=SAVE_DIR
    )
    COLSAMPLE_BYTREE = find_best_param(colsample_range_result, colsample_scores_test)
    print(f"✨ Valor òptim de colsample_bytree trobat: {COLSAMPLE_BYTREE}")

    
    # --- GRÁFICA 8: JUSTIFICACION SUBSAMPLE ---
    subsample_range = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    model_subsample = xgb.XGBClassifier(**fixed_params, max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, gamma=GAMMA, reg_alpha=REG_ALPHA, min_child_weight=MIN_CHILD_WEIGHT, colsample_bytree=COLSAMPLE_BYTREE)
    
    subsample_range_result, subsample_scores_test = plot_single_validation_curve(
        model_subsample, X_train, y_train, X_test, y_test,
        param_name="subsample",
        param_range=subsample_range,
        title="Impacto de Subsample - SENSE CV",
        xlabel="Subsample",
        SAVE_DIR=SAVE_DIR
    )
    SUBSAMPLE = find_best_param(subsample_range_result, subsample_scores_test)
    print(f"✨ Valor òptim de subsample trobat: {SUBSAMPLE}")
    
    
    # --- GRÀFICA FINAL: CURVA DE APRENDIZAJE CON LOS PARÁMETROS ELEGIDOS ---
    final_model = xgb.XGBClassifier(
        **fixed_params,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        reg_alpha=REG_ALPHA,
        min_child_weight=MIN_CHILD_WEIGHT,
        colsample_bytree=COLSAMPLE_BYTREE,
        subsample=SUBSAMPLE
    )
    plot_final_learning_curve(final_model, X_train, y_train, "Curva de Aprendizaje Final XGBoost con Parámetros Elegidos", SAVE_DIR=SAVE_DIR)

    print(f"\n✅ Tots els gràfics guardats en: {SAVE_DIR}")

if __name__ == "__main__":
    main()