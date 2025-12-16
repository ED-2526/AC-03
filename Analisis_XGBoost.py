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
SAVE_DIR = "Plots/Justificacion Parametros XGB (3s)"
os.makedirs(SAVE_DIR, exist_ok=True)

# Valors optims 3s: max_depth=2, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, gamma=7
# Valors optims 30s: max_depth=, n_estimators=, learning_rate=, subsample=, colsample_bytree=, gamma=

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- 🔍 GENERANDO JUSTIFICACIÓN DE PARÁMETROS (XGBOOST)---")
    
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
    # -----------------------------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # --- GRÀFICA 1: JUSTIFICACIÓ DE MAX_DEPTH ---
    depth_range = np.array([1, 2, 3, 4, 5, 6, 7, 8]) 
    model_depth = xgb.XGBClassifier(**fixed_params, n_estimators=300)
    
    plot_single_validation_curve(
        model_depth, X_train, y_train, X_test, y_test, 
        param_name="max_depth", 
        param_range=depth_range, 
        title="Impacte de la Profunditat (max_depth)",
        xlabel="Max Depth",
        ax=axes[0]
    )

    # --- GRÀFICA 2: JUSTIFICACIÓ DE N_ESTIMATORS ---
    estimators_range = np.array([50, 100, 200, 300, 400, 500])
    model_estimators = xgb.XGBClassifier(**fixed_params, max_depth=2)
    plot_single_validation_curve(
        model_estimators, X_train, y_train, X_test, y_test, 
        param_name="n_estimators", 
        param_range=estimators_range, 
        title="Impacte del Nombre d'Estimadors (n_estimators) - SENSE CV",
        xlabel="N Estimators",
        ax=axes[1]
    )

        # --- GRÀFICA 3: JUSTIFICACIÓ DE LEARNING_RATE ---
    learning_rate_range = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    model_lr = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300)
    plot_single_validation_curve(
        model_lr, X_train, y_train, X_test, y_test, 
        param_name="learning_rate",
        param_range=learning_rate_range,
        title="Impacte de la Taxa d'Aprenentatge (learning_rate)",
        xlabel="Learning Rate",
        ax=axes[2]
    )

    # --- GRÀFICA 4: JUSTIFICACIÓ DE SUBSAMPLE ---
    subsample_range = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    model_subsample = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300, learning_rate=0.05)
    plot_single_validation_curve(
        model_subsample, X_train, y_train, X_test, y_test,
        param_name="subsample",
        param_range=subsample_range,
        title="Impacte de Subsample",
        xlabel="Subsample",
        ax=axes[3]
    )

    # --- GRÀFICA 5: JUSTIFICACIÓ DE COLSAMPLE_BYTREE ---
    colsample_range = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    model_colsample = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300, learning_rate=0.05, subsample=0.8)
    plot_single_validation_curve(
        model_colsample, X_train, y_train, X_test, y_test,
        param_name="colsample_bytree",
        param_range=colsample_range,
        title="Impacte de Colsample_bytree",
        xlabel="Colsample_bytree",
        ax=axes[4]
    )

    # --- GRÀFICA 6: JUSTIFICACIÓ DE GAMMA ---
    gamma_range = np.array([0, 0.5, 1.0, 5.0, 7.0, 10.0])
    model_gamma = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    plot_single_validation_curve(
        model_gamma, X_train, y_train, X_test, y_test,
        param_name="gamma",
        param_range=gamma_range,
        title="Impacte de Gamma",
        xlabel="Gamma",
        ax=axes[5]
    )



    axes[0].legend(loc='best')
    plt.suptitle("Analisis Parámetros XGBoost", fontsize=16, fontweight="bold")
    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, "Justificacion_Parametros_XGB.png"))
    plt.close()

    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=2,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=7
    )

    plot_final_learning_curve(
        final_model, X_train, y_train, "Curva de Aprendizaje Final XGBoost amb Paràmetres Elegits",
        SAVE_DIR=SAVE_DIR,
    )
if __name__ == "__main__":
    main()