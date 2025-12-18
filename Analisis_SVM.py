import matplotlib.pyplot as plt
import numpy as np
from carrega_dades import split_datos_3s, split_datos_30s
import os
from sklearn.model_selection import GridSearchCV
from plots_2 import plot_final_learning_curve, plot_single_validation_curve
from sklearn.svm import SVC

# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion Parametros SVC (30s)"
os.makedirs(SAVE_DIR, exist_ok=True)

# Valors optims 3s: C = 1.0 , Gamma = 0.005
# Valors optims 30s: C = 0.8, Gamma = 0.01

# ======================================================
# MAIN
# ======================================================
def main():

    print("\n--- 🔬 LABORATORI D'ANÀLISI SVC ---")

    # 1. Carregar i preprocesar dades
    #X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_30s(random_state=RANDOM_STATE)

    # 2. Model base
    fixed_params = {
        'kernel': 'rbf',
        'probability': True,
        'random_state': RANDOM_STATE,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    # -------------------------
    # 3. VALIDATION CURVE — C
    # -------------------------
    C_range = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0])
    model_c = SVC(**fixed_params)
    plot_single_validation_curve(
        model_c,
        X_train, y_train, X_test, y_test, 
        param_name="C",
        param_range=C_range,
        title="Impacte del paràmetre C en SVC",
        xlabel="C",
        ax=axes[0],
    )

    # -------------------------
    # 4. VALIDATION CURVE — gamma
    # -------------------------
    gamma_range = np.array([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3])
    model_gamma = SVC(**fixed_params, C=0.8) # Amb 3s C=1.0 i amb 30s C=0.8
    plot_single_validation_curve(
        model_gamma,
        X_train, y_train, X_test, y_test, 
        param_name="gamma",
        param_range=gamma_range,
        title="Impacte del paràmetre gamma en SVC",
        xlabel="gamma",
        ax=axes[1],
    )

    axes[0].legend(loc='best')
    plt.suptitle("Analisis Parámetros SVC", fontsize=16, fontweight="bold")
    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, "Justificacion_Parametros_SVC.png"))
    plt.close()

    # -------------------------
    # 5. Model final amb paràmetres
    # -------------------------
    final_model = SVC(
        kernel='rbf',
        C=0.8, # Amb 3s C=1 i amb 30s C=0.8
        gamma=0.01,  # Amb 3s C=0.005 i amb 30s C=0.01
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