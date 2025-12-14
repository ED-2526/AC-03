import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split # Necessari per a la simulació de dades
from sklearn.metrics import accuracy_score
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
# Descomentar al teu entorn:
from carrega_dades import split_datos_3s, split_datos_30s
from plots_2 import plot_final_learning_curve, plot_single_validation_curve

# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion_Parametros_SVC_3s_FINAL"
os.makedirs(SAVE_DIR, exist_ok=True)

def find_best_param(param_range, test_scores, tolerance=0.001):
    best_score = np.max(test_scores)
    best_param_index = np.where(test_scores >= (best_score - tolerance))[0][0]
    return param_range[best_param_index]


#Valors optims: C= 0.1 ,Gamma= 0.001
# ======================================================
# MAIN
# ======================================================
def main():

    print("\n--- 🔬 LABORATORI D'ANÀLISI SVC (FINAL SENSE CV) ---")

    # 1. Carregar i preprocesar dades
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(
        random_state=RANDOM_STATE
    )

    # 2. Model base
    base_model = SVC(
        kernel='rbf',
        probability=True,
        random_state=RANDOM_STATE
    )

    # -------------------------
    # 3. VALIDATION CURVE — C
    # -------------------------
    C_range = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    
    C_range_result, C_scores_test = plot_single_validation_curve(
        base_model,
        X_train, y_train, X_test, y_test, 
        param_name="C",
        param_range=C_range,
        title="Impacte del paràmetre C en SVC",
        xlabel="C",
        SAVE_DIR=SAVE_DIR
    )
    best_C = find_best_param(C_range_result, C_scores_test)
    print(f"✨ Valor òptim de C trobat: {best_C}")


    # -------------------------
    # 4. VALIDATION CURVE — gamma
    # -------------------------
    gamma_range = np.array([0.0005, 0.001, 0.005, 0.01, 0.05])
    
    # Fixem C al valor òptim trobat
    model_gamma = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, C=best_C) 
    
    gamma_range_result, gamma_scores_test = plot_single_validation_curve(
        model_gamma,
        X_train, y_train, X_test, y_test, 
        param_name="gamma",
        param_range=gamma_range,
        title="Impacte del paràmetre gamma en SVC",
        xlabel="gamma",
        SAVE_DIR=SAVE_DIR
    )
    best_gamma = find_best_param(gamma_range_result, gamma_scores_test)
    print(f"✨ Valor òptim de gamma trobat: {best_gamma}")


    # -------------------------
    # 5. Model final amb paràmetres
    # -------------------------
    final_model = SVC(
        kernel='rbf',
        C=best_C,
        gamma=best_gamma,
        probability=True,
        random_state=RANDOM_STATE
    )

    # -------------------------
    # 6. Learning Curve final
    # -------------------------
    plot_final_learning_curve(
        final_model,
        X_train, y_train,
        title="Curva d'Aprenentatge SVC",
        SAVE_DIR=SAVE_DIR
    )

    print(f"\n✅ Anàlisi SVC completat. Gràfiques en: {SAVE_DIR}")


if __name__ == "__main__":
    main()