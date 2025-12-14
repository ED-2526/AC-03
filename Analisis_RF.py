# Importa necessària:
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve, learning_curve
from carrega_dades import (
    split_datos_3s, split_datos_30s, 
    cargar_y_preprocesar_datos_3s, cargar_y_preprocesar_datos_30s
)
import matplotlib
matplotlib.use('Agg') # Evita errors de finestra gràfica (Tkinter)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
# from sklearn.model_selection import GridSearchCV # Ja no es fa servir
# from carrega_dades import split_datos_3s, split_datos_30s
from plots_2 import (plot_final_learning_curve, plot_single_validation_curve )
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- (ASSUMIM que les funcions de simulació de dades i la funció modificada estan disponibles) ---

# --- CONFIGURACIÓN ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion_Parametros_RF"
os.makedirs(SAVE_DIR, exist_ok=True)
#Valors optims:  max depth=5, min_samples_leaf=16,min sample_split=5,n_estimators=500
# ----------------------------------------------------
# Funció per trobar el millor paràmetre (similar a la LR)
# ----------------------------------------------------
def find_best_param(param_range, test_scores):
    """Troba el valor del paràmetre que dóna el màxim Test Score."""
    best_index = np.argmax(test_scores)
    best_param = param_range[best_index]
    return best_param

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- 🔍 GENERANDO JUSTIFICACIÓN DE PARÁMETROS (RANDOM FOREST) SENSE CV ---")
    
    # 1. Cargar datos (Utilitzem split_datos_3s com en el teu codi original)
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    
    # 2. DEFINIR EL MODELO BASE CON PARÁMETROS FIJOS
    fixed_params = {
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }
    
    # -----------------------------------------------------------------------
    # 3. ANÀLISI D'HIPERPARÀMETRES SENSE CV
    # -----------------------------------------------------------------------
    
    
    # --- GRÀFICA 1: JUSTIFICACIÓ DE MAX_DEPTH ---
    # Convertim a np.array per compatibilitat de tipus
    depth_range = np.array([5, 10, 15, 20, 30]) 
    model_depth = RandomForestClassifier(**fixed_params, n_estimators=50)
    
    depth_range_result, depth_scores_test = plot_single_validation_curve(
        model_depth, X_train, y_train, X_test, y_test, # <--- PASSEM TEST SET
        param_name="max_depth", 
        param_range=depth_range, 
        title="Impacte de la Profunditat (max_depth) - SENSE CV",
        xlabel="Max Depth",
        SAVE_DIR=SAVE_DIR
    )
    best_depth = find_best_param(depth_range_result, depth_scores_test)
    print(f"✨ Valor òptim de max_depth trobat: {best_depth}")
    
    
    # --- GRÀFICA 2: JUSTIFICACIÓ DE MIN_SAMPLES_SPLIT ---
    split_range = np.array([2, 5, 10, 15, 20])
    model_split = RandomForestClassifier(**fixed_params, n_estimators=50, max_depth=best_depth)
    
    split_range_result, split_scores_test = plot_single_validation_curve(
        model_split, X_train, y_train, X_test, y_test, # <--- PASSEM TEST SET
        param_name="min_samples_split",
        param_range=split_range,
        title="Impacte de Min Samples Split - SENSE CV",
        xlabel="Min Samples Split",
        SAVE_DIR=SAVE_DIR
    )
    best_split = find_best_param(split_range_result, split_scores_test)
    print(f"✨ Valor òptim de min_samples_split trobat: {best_split}")

    
    # --- GRÀFICA 3: JUSTIFICACIÓ DE MIN_SAMPLES_LEAF ---
    leaf_range = np.array([1, 2, 4, 8, 16])
    model_leaf = RandomForestClassifier(**fixed_params, n_estimators=50, max_depth=best_depth, min_samples_split=best_split)
    
    leaf_range_result, leaf_scores_test = plot_single_validation_curve(
        model_leaf, X_train, y_train, X_test, y_test, # <--- PASSEM TEST SET
        param_name="min_samples_leaf",
        param_range=leaf_range,
        title="Impacte de Min Samples Leaf - SENSE CV",
        xlabel="Min Samples Leaf",
        SAVE_DIR=SAVE_DIR
    )
    best_leaf = find_best_param(leaf_range_result, leaf_scores_test)
    print(f"✨ Valor òptim de min_samples_leaf trobat: {best_leaf}")
    
    
    # --- GRÀFICA 4: JUSTIFICACIÓ DE N_ESTIMATORS ---
    estimators_range = np.array([50, 100, 200, 300, 500])
    model_estimators = RandomForestClassifier(
        **fixed_params, max_depth=best_depth, min_samples_split=best_split, min_samples_leaf=best_leaf
    )
    
    estimators_range_result, estimators_scores_test = plot_single_validation_curve(
        model_estimators, X_train, y_train, X_test, y_test, # <--- PASSEM TEST SET
        param_name="n_estimators",
        param_range=estimators_range,
        title="Impacte del Número de Árboles (n_estimators) - SENSE CV",
        xlabel="Número de Estimadores (n_estimators)",
        SAVE_DIR=SAVE_DIR
    )
    best_estimators = find_best_param(estimators_range_result, estimators_scores_test)
    # Cerquem el punt d'estabilització (el valor més petit que manté el màxim score)
    best_estimators_stable = estimators_range_result[estimators_scores_test >= (np.max(estimators_scores_test) - 0.005)][0]
    print(f"✨ Valor òptim d'n_estimators trobat (punt d'estabilització): {best_estimators_stable}")

    
    # --- GRÀFICA FINAL: CURVA DE APRENDIZAJE AMB ELS PARÀMETRES ESCOLLITS ---
    final_model = RandomForestClassifier(
        **fixed_params,
        n_estimators=best_estimators_stable,
        max_depth=best_depth,
        min_samples_split=best_split,
        min_samples_leaf=best_leaf
    )
    
    # plot_final_learning_curve es manté amb CV, que és l'estàndard per a aquesta corba.
    plot_final_learning_curve(final_model, X_train, y_train, "Curva de Aprendizaje Final RF amb Paràmetres Elegits", SAVE_DIR=SAVE_DIR)

    print(f"\n✅ Tots els gràfics guardats en: {SAVE_DIR}")

if __name__ == "__main__":
    main()