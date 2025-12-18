import matplotlib.pyplot as plt
import numpy as np
from carrega_dades import split_datos_3s, split_datos_30s
import os
from sklearn.model_selection import GridSearchCV
from plots_2 import plot_final_learning_curve, plot_single_validation_curve
from sklearn.linear_model import LogisticRegression

# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion Parametros LR"
os.makedirs(SAVE_DIR, exist_ok=True)

# Valors optims amb 3s: C = 5.0, iter = 200
# Valors optims amb 30s: C = 0.3, iter = 50

def main():
    print("--- 🔬 LABORATORI D'ANÀLISI REGRESSIÓ LOGÍSTICA ---")
    
    # 1. Carregar dades (X_train i X_test són essencials)

    #X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_30s(random_state=RANDOM_STATE)

    
    # 2. Definir Model Base per a Exploració
    fixed_params = {
        'solver': 'lbfgs',
        'multi_class': 'multinomial',
        'penalty': 'l2',
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    # --- GRÀFICA 1: JUSTIFICACIÓ DE LA REGULARITZACIÓ (C) - SENSE CV ---
    
    # Rang C en escala logarítmica
    C_range = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
    model_C = LogisticRegression(**fixed_params, max_iter=1000)
    
    # CRIDA A LA NOVA FUNCIÓ: Ara passem X_test i y_test
    plot_single_validation_curve(
        model_C, X_train, y_train, X_test, y_test, # <- PASSEM TEST SET
        param_name="C", 
        param_range=C_range, 
        title="Impacte de la Regularització (Paràmetre C)", 
        xlabel="C (Escala Log)",
        ax=axes[0]
    )

    # --- GRÀFICA 2: JUSTIFICACIÓ DE MAX_ITER ---
    
    iter_range = np.array([5, 10, 25, 50, 100, 200, 500, 1000])
    
    # Utilitzem el millor C trobat per avaluar max_iter
    model_iter = LogisticRegression(**fixed_params, C=0.3) # Amb 3s C=5,30s i amb 30 C=0.3)
    
    # CRIDA A LA NOVA FUNCIÓ
    plot_single_validation_curve(
        model_iter, X_train, y_train, X_test, y_test, # <- PASSEM TEST SET
        param_name="max_iter",
        param_range=iter_range,
        title="Convergència del Model (max_iter)",
        xlabel="Màxim d'Iteracions",
        ax=axes[1]
    )

    axes[0].legend(loc='best')
    plt.suptitle("Analisis Parámetros Regresion Logistica", fontsize=16, fontweight="bold")
    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, "Justificacion_Parametros_LR.png"))
    plt.close()


    # --- GRÀFICA FINAL: CURVA DE APRENENTATGE ---
    final_model = LogisticRegression(
        **fixed_params,
        C=0.3, # Amb 3s C=5,30s i amb 30 C=0.3
        max_iter=50 # Amb 3s max_iter=200 i amb 30s max_iter=50
    )
    
    # plot_final_learning_curve es manté amb CV, que és l'estàndard per a aquesta corba.
    plot_final_learning_curve(final_model, X_train, y_train, "Curva de Aprendizaje LR Final", SAVE_DIR=SAVE_DIR)

    print(f"\n✅ Anàlisi LR completat. Gràfiques en: {SAVE_DIR}")

if __name__ == "__main__":
    main()