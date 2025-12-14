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
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
from plots_2 import (plot_final_learning_curve, plot_single_validation_curve )

# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion_Parametros_LR"
os.makedirs(SAVE_DIR, exist_ok=True)
# Amb 3s(C=0.001,iter=50),30s(c=,iter=)

def main():
    print("--- 🔬 LABORATORI D'ANÀLISI REGRESSIÓ LOGÍSTICA ---")
    
    # 1. Carregar dades (X_train i X_test són essencials)
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    
    # 2. Definir Model Base per a Exploració
    fixed_params = {
        'solver': 'lbfgs',
        'multi_class': 'multinomial', 
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # --- GRÀFICA 1: JUSTIFICACIÓ DE LA REGULARITZACIÓ (C) - SENSE CV ---
    
    # Rang C en escala logarítmica
    C_range = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0])
    
    model_C = LogisticRegression(**fixed_params, max_iter=1000)
    
    # CRIDA A LA NOVA FUNCIÓ: Ara passem X_test i y_test
    C_range_result, C_scores_test = plot_single_validation_curve(
        model_C, X_train, y_train, X_test, y_test, # <- PASSEM TEST SET
        param_name="C", 
        param_range=C_range, 
        title="Impacte de la Regularització (Paràmetre C)", 
        xlabel="C (Escala Log)",
        SAVE_DIR=SAVE_DIR
    )

    # 3. TROBAR EL MILLOR PARÀMETRE C BASAT EN EL RESULTAT DEL TEST SCORE
    best_C_index = np.argmax(C_scores_test)
    best_C = C_range_result[best_C_index]
    print(f"\n✨ Valor òptim de C trobat (basat en Test Score): {best_C}")
    
    # --- GRÀFICA 2: JUSTIFICACIÓ DE MAX_ITER ---
    
    iter_range = np.array([5, 10, 25, 50, 100, 200, 500])
    
    # Utilitzem el millor C trobat per avaluar max_iter
    model_iter = LogisticRegression(**fixed_params, C=best_C) 
    
    # CRIDA A LA NOVA FUNCIÓ
    iter_range_result, iter_scores_test = plot_single_validation_curve(
        model_iter, X_train, y_train, X_test, y_test, # <- PASSEM TEST SET
        param_name="max_iter",
        param_range=iter_range,
        title="Convergència del Model (max_iter)",
        xlabel="Màxim d'Iteracions",
        SAVE_DIR=SAVE_DIR
    )

    # 4. TROBAR EL MILLOR PARÀMETRE max_iter
    # Cerquem el punt d'estabilització (el valor més petit que manté el màxim score)
    best_iter_score = np.max(iter_scores_test)
    # Trobem el primer 'max_iter' on el score és proper al màxim (per eficiència)
    best_iter = iter_range_result[iter_scores_test >= (best_iter_score - 0.001)][0] 
    print(f"✨ Valor òptim de max_iter trobat (punt d'estabilització): {best_iter}")


    # --- GRÀFICA FINAL: CURVA DE APRENENTATGE ---
    final_model = LogisticRegression(
        **fixed_params,
        C=best_C, 
        max_iter=best_iter 
    )
    
    # plot_final_learning_curve es manté amb CV, que és l'estàndard per a aquesta corba.
    plot_final_learning_curve(final_model, X_train, y_train, "Curva de Aprendizaje LR Final", SAVE_DIR=SAVE_DIR)

    print(f"\n✅ Anàlisi LR completat. Gràfiques en: {SAVE_DIR}")

if __name__ == "__main__":
    main()