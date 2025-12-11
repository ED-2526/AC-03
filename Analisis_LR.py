import matplotlib
matplotlib.use('Agg') # Evita errors de finestra gràfica (Tkinter)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
from sklearn.model_selection import GridSearchCV
from carrega_dades import split_datos_3s, split_datos_30s
from plots_2 import plot_single_validation_curve, plot_final_learning_curve

# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion_Parametros_LR"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# BLOC 1 - CERCA EXHAUSTIVA INICIAL (NO EJECUTADA POR DEFECTO)
# =============================================================================
def run_heavy_grid_search_lr(X_train, y_train):
    """
    Aquesta funció conté la cerca inicial per a Logistic Regression.
    No s'executa per defecte, però es deixa aquí com a evidència del codi utilitzat.
    """
    print("\n⚠️  INICIANT CERCA EXHAUSTIVA LR...")
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'max_iter': [100, 200, 500],
        'solver': ['lbfgs', 'newton-cg', 'sag']
    }
    model = LogisticRegression(multi_class='multinomial', random_state=RANDOM_STATE, n_jobs=-2)
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-2)
    grid.fit(X_train, y_train)
    print(f"Millors paràmetres: {grid.best_params_}")
    print(f"Millor Score de Validació Creuada: {grid.best_score_:.4f}")
    pd.DataFrame(grid.cv_results_).to_csv("gridsearch_results_lr_fase1.csv")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- 🔬 LABORATORI D'ANÀLISI REGRESSIÓ LOGÍSTICA ---")
    
    # 1. Carregar dades
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    
    # 2. Definir Model Base per a Exploració
    # 'lbfgs' és el solver per defecte i robust per a problemes multiclasse
    fixed_params = {
        'solver': 'lbfgs',            # Estándar robusto
        'multi_class': 'multinomial', # Para múltiples géneros
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # --- GRÁFICA 1: JUSTIFICACIÓN DE LA REGULARIZACIÓN (C) ---
    
    model_C = LogisticRegression(**fixed_params, max_iter=1000)
    plot_single_validation_curve(
        model_C, X_train, y_train, 
        param_name="C", 
        param_range=[0.01, 0.1, 0.3, 0.5, 1.0, 2.0], 
        title="Impacto de la Regularización (Parámetro C)", 
        xlabel="C (Escala Log)",
        SAVE_DIR=SAVE_DIR
    )

    # ESCOGEMOS 0.2 COMO VALOR OPTIMO ya que la curva empieza a estabilizarse a partir de este punto.

    # --- GRÁFICA 2: JUSTIFICACIÓN DE MAX_ITER ---
    #     
    model_iter = LogisticRegression(**fixed_params, C=0.2)
    plot_single_validation_curve(
        model_iter, X_train, y_train,
        param_name="max_iter",
        param_range=[50, 100, 200, 500, 1000],
        title="Convergencia del Modelo",
        xlabel="Máximo de Iteraciones",
        SAVE_DIR=SAVE_DIR
    )
    
    # ESCOGEMOS 50 COMO VALOR OPTIMO ya que la curva es plana a partir de este punto.

    # --- GRÁFICA FINAL: CURVA DE APRENDIZAJE ---

    final_model = LogisticRegression(
        **fixed_params,
        C=0.2,          # Ganador de gráfica 1
        max_iter=50    # Ganador de gráfica 2
    )
    
    plot_final_learning_curve(final_model, X_train, y_train, "Curva de Aprendizaje LR (C=1.0)", SAVE_DIR=SAVE_DIR)

    print(f"\n✅ Análisis LR completado. Gráficas en: {SAVE_DIR}")

    #run_grid_search(X_train, y_train)
    # RESULTADO DEL GRID SEARCH:
    # Millors Paràmetres: {'C': 10.0, 'max_iter': 200, 'solver': 'lbfgs'}

if __name__ == "__main__":
    main()