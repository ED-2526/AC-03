import matplotlib
matplotlib.use('Agg') # Evita errores de interfaz gráfica (Tkinter)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV
from carrega_dades import split_datos_3s, split_datos_30s
from plots_2 import plot_single_validation_curve, plot_final_learning_curve

# --- CONFIGURACIÓ GLOBAL ---

# --- CONFIGURACIÓN ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion_Parametros_XGB"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# BLOC 1: CERCA EXHAUSTIVA INICIAL (NO EJECUTADA POR DEFECTO)
# =============================================================================

def run_heavy_grid_search(X, y):
    """
    Aquesta funció conté la cerca inicial (45 minuts). 
    No s'executa per defecte, però es deixa aquí com a evidència del codi utilitzat.
    """
    print("\n⚠️  INICIANT CERCA EXHAUSTIVA (Pot trigar 45 minuts)...")
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid.fit(X, y)
    print(f"Millors paràmetres: {grid.best_params_}")
    pd.DataFrame(grid.cv_results_).to_csv("gridsearch_results_fase1.csv")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- 🔍 GENERANDO JUSTIFICACIÓN DE PARÁMETROS (XGBOOST) ---")
    
    # 1. Cargar datos
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    
    # 2. DEFINIR EL MODELO BASE CON TUS PARÁMETROS FIJOS
    fixed_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }

    # --- GRÁFICA 1: JUSTIFICACIÓN DE MAX_DEPTH ---
    # Probaremos de 1 a 10. Deberíamos ver que a partir de 3 el Test no sube.

    model_depth = xgb.XGBClassifier(**fixed_params, n_estimators=300, learning_rate=0.05)
    plot_single_validation_curve(
        model_depth, X_train, y_train, 
        param_name="max_depth", 
        param_range=[1, 2, 3, 4, 5, 6, 8, 10], 
        title="Impacto de la Profundidad",
        xlabel="Max Depth",
        SAVE_DIR=SAVE_DIR
    )

    #ESCOJEMOS 2 COMO VALOR ÓPTIMO ya que a partir de ahí la accuracy no mejora y el overfitting aumenta.

    # --- GRÁFICA 2: JUSTIFICACIÓN DE N_ESTIMATORS ---
    # Probaremos varios números de árboles. (Debido a grafica anterior, fijamos max_depth=2, ya que es óptimo según la gráfica (menos accuracy, però menos overfitting)).

    model_estimators = xgb.XGBClassifier(**fixed_params, max_depth=2, learning_rate=0.05)
    plot_single_validation_curve(
        model_estimators, X_train, y_train,
        param_name="n_estimators",
        param_range=[50, 100, 200, 300, 500],
        title="Impacto del Número de Árboles",
        xlabel="Número de Estimadores (n_estimators)",
        SAVE_DIR=SAVE_DIR
    )

    #ESCOJEMOS 300 COMO VALOR ÓPTIMO ya que a parte de que tarda menos, la accuracy no mejora mucho más allá de 300 y si reducimos el overfitting.

    # --- GRÁFICA 3: JUSTIFICACIÓN DE LEARNING_RATE ---
    # Probaremos varias tasas de aprendizaje. (Debido a gráficas anteriores, fijamos max_depth=2 y n_estimators=300).

    model_lr = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300)
    plot_single_validation_curve(
        model_lr, X_train, y_train,
        param_name="learning_rate",
        param_range=[0.01, 0.05, 0.1, 0.2, 0.3],
        title="Impacto de la Tasa de Aprendizaje",
        xlabel="Learning Rate",
        SAVE_DIR=SAVE_DIR
    )

    #ESCOJEMOS 0.05 COMO VALOR ÓPTIMO ya que es  reduciendo el riesgo de overfitting, manteniendo el accuracy.

    # ---- GRÁFICA 4: JUSTIFICACIÓN DE GAMMA (PODA) ---
    # Probaremos varios valores de gamma. Debido a gráficas anteriores, fijamos max_depth=2, n_estimators=300 y learning_rate=0.05.

    model_gamma = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300, learning_rate=0.05)
    plot_single_validation_curve(
        model_gamma, X_train, y_train,
        param_name="gamma",
        param_range=[0, 0.5, 1, 3, 5, 7, 10],
        title="Impacto de Gamma (Poda)",
        xlabel="Gamma",
        SAVE_DIR=SAVE_DIR
    )

    #ESCOJEMOS 7.0 COMO VALOR ÓPTIMO ya que reduce el overfitting sin sacrificar mucho el accuracy.

    # --- GRÁFICA 5: JUSTIFICACIÓN DE REG_ALPHA (Regularización L1) ---
    # Probaremos varios valores de reg_alpha. Debido a gráficas anteriores, fijamos max_depth=2, n_estimators=300 y learning_rate=0.05.

    model_reg_alpha = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300, learning_rate=0.05, gamma=7.0)
    plot_single_validation_curve(
        model_reg_alpha, X_train, y_train,
        param_name="reg_alpha",
        param_range=[0, 0.1, 0.5, 1.0, 2.0, 5.0],
        title="Impacto de la Regularización L1 (reg_alpha)",
        xlabel="reg_alpha",
        SAVE_DIR=SAVE_DIR
    )

    #ESCOJEMOS 2.0 COMO VALOR ÓPTIMO ya que reduce el overfitting sin sacrificar mucho el accuracy.

    # --- GRÁFICA 6: JUSTIFICACION MIN_CHILD_WEIGHT ---
    # Probaremos varios valores de min_child_weight. Debido a gráficas anteriores, fijamos max_depth=2, n_estimators=300, learning_rate=0.05 y reg_alpha=7.0. 
    
    model_min_child = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300, learning_rate=0.05, gamma=7.0, reg_alpha=2.0)
    plot_single_validation_curve(
        model_min_child, X_train, y_train,
        param_name="min_child_weight",
        param_range=[1, 3, 5, 7],
        title="Impacto de Min Child Weight",
        xlabel="Min Child Weight",
        SAVE_DIR=SAVE_DIR
    )

    #ESCOJEMOS 5 COMO VALOR ÓPTIMO aunque no varie mucho.

    # --- GRÁFICA 6: JUSTIFICACION COLSAMPLE_BYTREE ---
    # Probaremos varios valores de colsample_bytree. Debido a gráficas anteriores, fijamos max_depth=2, n_estimators=300, learning_rate=0.05, reg_alpha=2.0 y min_child_weight=5.
    
    model_colsample = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300, learning_rate=0.05, gamma=7.0, reg_alpha=2.0, min_child_weight=5)
    plot_single_validation_curve(
        model_colsample, X_train, y_train,
        param_name="colsample_bytree",
        param_range=[0.6, 0.7, 0.8, 0.9, 1.0],
        title="Impacto de Colsample_bytree",
        xlabel="Colsample_bytree",
        SAVE_DIR=SAVE_DIR
    )

    #ESCOJEMOS 0.8 COMO VALOR ÓPTIMO aunque no varie mucho.

    # --- GRÁFICA 7: JUSTIFICACION SUBSAMPLE ---
    # Probaremos varios valores de subsample. Debido a gráficas anteriores, fijamos max_depth=2, n_estimators=300, learning_rate=0.05, reg_alpha=2.0, min_child_weight=5 y colsample_bytree=0.8.
    
    model_subsample = xgb.XGBClassifier(**fixed_params, max_depth=2, n_estimators=300, learning_rate=0.05, gamma=7.0, reg_alpha=2.0, min_child_weight=5, colsample_bytree=0.8)
    plot_single_validation_curve(
        model_subsample, X_train, y_train,
        param_name="subsample",
        param_range=[0.6, 0.7, 0.8, 0.9, 1.0],
        title="Impacto de Subsample",
        xlabel="Subsample"
    )  

    #ESCOJEMOS 0.8 COMO VALOR ÓPTIMO aunque no varie mucho.
    
    # --- GRÁFICA FINAL: CURVA DE APRENDIZAJE CON LOS PARÁMETROS ELEGIDOS ---
    final_model = xgb.XGBClassifier(
        **fixed_params,
        n_estimators=300,
        max_depth=2,
        learning_rate=0.05,
        gamma=7.0,
        reg_alpha=2.0,
        min_child_weight=5,
        colsample_bytree=0.8,
        subsample=0.8
    )
    plot_final_learning_curve(final_model, X_train, y_train, "Curva de Aprendizaje Final con Parámetros Elegidos", SAVE_DIR=SAVE_DIR)

    print(f"\n✅ Todos los gráficos guardados en: {SAVE_DIR}")

    #NOTA: La cerca exhaustiva no s'executa per defecte per estalviar temps.
    # run_heavy_grid_search(X_train, y_train)
    #RESULTATS DE LA CERCA EXHAUSTIVA (FASE 1:
    #Millors paràmetres trobats:
    #{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.7}

if __name__ == "__main__":
    main()