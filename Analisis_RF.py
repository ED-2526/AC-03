import matplotlib
matplotlib.use('Agg') # Evita errores de interfaz gráfica (Tkinter)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import GridSearchCV
from carrega_dades import split_datos_3s, split_datos_30s
from plots_2 import plot_single_validation_curve, plot_final_learning_curve

# --- CONFIGURACIÓN ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion_Parametros_RF"
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# BLOC 1: CERCA EXHAUSTIVA INICIAL (NO EJECUTADA POR DEFECTO)
# =============================================================================

def run_heavy_grid_search(X, y):
    """
    Aquesta funció conté la cerca inicial (45 minuts). 
    No s'executa per defecte, però es deixa aquí com a evidència del codi utilitzat.
    """
    print("\n⚠️  INICIANT CERCA EXHAUSTIVA RF (Pot trigar 45 minuts)...")
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid.fit(X, y)
    print(f"Millors paràmetres: {grid.best_params_}")
    pd.DataFrame(grid.cv_results_).to_csv("gridsearch_results_rf_fase1.csv")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- 🔍 GENERANDO JUSTIFICACIÓN DE PARÁMETROS (RANDOM FOREST) ---")
    
    # 1. Cargar datos
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    
    # 2. DEFINIR EL MODELO BASE CON TUS PARÁMETROS FIJOS INICIALES
    fixed_params = {
        'random_state': RANDOM_STATE,
        'n_jobs': -2,
    }

    # --- GRÁFICA 1: JUSTIFICACIÓN DE MAX_DEPTH ---
    # Probaremos de 5 a 30 y None.
    model_depth = RandomForestClassifier(**fixed_params, n_estimators=50)
    plot_single_validation_curve(
        model_depth, X_train, y_train, 
        param_name="max_depth", 
        param_range=[0, 5, 10, 15, 20, 30], 
        title="Impacto de la Profundidad",
        xlabel="Max Depth",
        SAVE_DIR=SAVE_DIR
    )

    # ESCOJEMOS 8 COMO VALOR ÓPTIMO ya que ofrece un buen balance entre bias y variance.
    
    # --- GRÁFICA 2: JUSTIFICACIÓN DE MIN_SAMPLES_SPLIT ---
    # Probaremos varios valores. (Debido a gráficas anteriores, fijamos n_estimators=300 y max_depth=10).
    model_split = RandomForestClassifier(**fixed_params, n_estimators=50, max_depth=8)
    plot_single_validation_curve(
        model_split, X_train, y_train,
        param_name="min_samples_split",
        param_range=[2, 5, 10, 15, 20],
        title="Impacto de Min Samples Split",
        xlabel="Min Samples Split",
        SAVE_DIR=SAVE_DIR
    )
    
    # ESCOJEMOS 15 COMO VALOR ÓPTIMO.

    # --- GRÁFICA 3: JUSTIFICACIÓN DE MIN_SAMPLES_LEAF ---
    # Probaremos varios valores. (Debido a gráficas anteriores, fijamos n_estimators=300, max_depth=15 y min_samples_split=5).
    model_leaf = RandomForestClassifier(**fixed_params, n_estimators=50, max_depth=8, min_samples_split=15)
    plot_single_validation_curve(
        model_leaf, X_train, y_train,
        param_name="min_samples_leaf",
        param_range=[1, 2, 4, 8],
        title="Impacto de Min Samples Leaf",
        xlabel="Min Samples Leaf",
        SAVE_DIR=SAVE_DIR
    )
    
    # ESCOJEMOS 5 COMO VALOR ÓPTIMO para suavizar ligeramente el modelo.

    # GRAFICA 5: JUSTIFICACIÓN DE N_ESTIMATORS ---
    # Probaremos varios números de árboles.
    model_estimators = RandomForestClassifier(**fixed_params)
    plot_single_validation_curve(
        model_estimators, X_train, y_train,
        param_name="n_estimators",
        param_range=[50, 100, 200, 300, 500],
        title="Impacto del Número de Árboles",
        xlabel="Número de Estimadores (n_estimators)",
        SAVE_DIR=SAVE_DIR
    )
    
    # ESCOJEMOS 50 COMO VALOR ÓPTIMO ya que a partir de ahí la accuracy se estabiliza.

    
    # --- GRÁFICA FINAL: CURVA DE APRENDIZAJE CON LOS PARÁMETROS ELEGIDOS ---
    final_model = RandomForestClassifier(
        **fixed_params,
        n_estimators=50,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=5
    )
    plot_final_learning_curve(final_model, X_train, y_train, "Curva de Aprendizaje Final RF con Parámetros Elegidos", SAVE_DIR=SAVE_DIR)

    print(f"\n✅ Todos los gráficos guardados en: {SAVE_DIR}")

    #NOTA: La cerca exhaustiva no s'executa per defecte per estalviar temps.
    #run_heavy_grid_search(X_train, y_train)

if __name__ == "__main__":
    main()