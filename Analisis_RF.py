import matplotlib.pyplot as plt
import numpy as np
from carrega_dades import split_datos_3s, split_datos_30s
import os
from sklearn.model_selection import GridSearchCV
from plots_2 import plot_final_learning_curve, plot_single_validation_curve
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURACIÓN ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion Parametros RF (30s)"
os.makedirs(SAVE_DIR, exist_ok=True)

# Valors optims 3s:  n_estimators=300, max depth=8, min_samples_leaf=16, min_sample_split=2
# Valors optims 30s: n_estimators=50, max depth=7, min_samples_leaf=10, min_sample_split=2

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- 🔍 GENERANDO JUSTIFICACIÓN DE PARÁMETROS (RANDOM FOREST) SENSE CV ---")
    
    # 1. Cargar datos (Utilitzem split_datos_3s com en el teu codi original)
    #X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_30s(random_state=RANDOM_STATE)
    # 2. DEFINIR EL MODELO BASE CON PARÁMETROS FIJOS
    fixed_params = {
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }
    
    # -----------------------------------------------------------------------
    # 3. ANÀLISI D'HIPERPARÀMETRES SENSE CV
    # -----------------------------------------------------------------------
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # --- GRÀFICA 1: JUSTIFICACIÓ DE MAX_DEPTH ---
    # Convertim a np.array per compatibilitat de tipus
    depth_range = np.array([2, 5, 8, 10, 15, 20, 30]) 
    model_depth = RandomForestClassifier(**fixed_params, n_estimators=50)
    plot_single_validation_curve(
        model_depth, X_train, y_train, X_test, y_test, # <--- PASSEM TEST SET
        param_name="max_depth", 
        param_range=depth_range, 
        title="Impacte de la Profunditat (max_depth)",
        xlabel="Max Depth",
        ax=axes[0],
    )
    
    # --- GRÀFICA 2: JUSTIFICACIÓ DE MIN_SAMPLES_LEAF ---
    leaf_range = np.array([1, 2, 4, 8, 10, 12, 13, 14, 15, 16, 32])
    model_leaf = RandomForestClassifier(
        **fixed_params, n_estimators=50, max_depth=7
    )
    plot_single_validation_curve(
        model_leaf, X_train, y_train, X_test, y_test, # <--- PASSEM TEST SET
        param_name="min_samples_leaf",
        param_range=leaf_range,
        title="Impacte del Mínim de Mostres per Fulla (min_samples_leaf)",
        xlabel="Mínim de Mostres per Fulla (min_samples_leaf)",
        ax=axes[1],
    )

    # --- GRÀFICA 3: JUSTIFICACIÓ DE MIN_SAMPLES_SPLIT ---
    split_range = np.array([2, 5, 10, 15, 20, 50, 100])
    model_split = RandomForestClassifier(
        **fixed_params, n_estimators=50, max_depth=7, min_samples_leaf=10
    )
    plot_single_validation_curve(
        model_split, X_train, y_train, X_test, y_test, # <--- PASSEM TEST SET
        param_name="min_samples_split",
        param_range=split_range,
        title="Impacte del Mínim de Mostres per Divisió (min_samples_split)",
        xlabel="Mínim de Mostres per Divisió (min_samples_split)",
        ax=axes[2],
    )

    # --- GRÀFICA 4: JUSTIFICACIÓ DE N_ESTIMATORS ---
    estimators_range = np.array([50, 100, 200, 300, 500])
    model_estimators = RandomForestClassifier(
        **fixed_params, max_depth=7, min_samples_split=2, min_samples_leaf=10
    )
    plot_single_validation_curve(
        model_estimators, X_train, y_train, X_test, y_test, # <--- PASSEM TEST SET
        param_name="n_estimators",
        param_range=estimators_range,
        title="Impacte del Número de Árboles (n_estimators)",
        xlabel="Número de Estimadores (n_estimators)",
        ax=axes[3],
    )

    axes[0].legend(loc="best")
    plt.suptitle("Analisis Parámetros Random Forest", fontsize=16, fontweight="bold")
    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, "Justificacion_Parametros_RF.png"))
    plt.close()

    # --- GRÀFICA FINAL: CURVA DE APRENDIZAJE AMB ELS PARÀMETRES ESCOLLITS ---
    final_model = RandomForestClassifier(
        **fixed_params,
        n_estimators=50, # Amb 3s n_estimators=300 i amb 30s n_estimators=50
        max_depth=7, # Amb 3s max_depth=8 i amb 30s max_depth=7
        min_samples_split=2, # Amb 3s min_samples_split=2 i amb 30s min_samples_split=2
        min_samples_leaf=10 # Amb 3s min_samples_leaf=16 i amb 30s min_samples_leaf=10
    )
    
    # plot_final_learning_curve es manté amb CV, que és l'estàndard per a aquesta corba.
    plot_final_learning_curve(final_model, X_train, y_train, "Curva de Aprendizaje Final RF amb Paràmetres Elegits", SAVE_DIR=SAVE_DIR)

    print(f"\n✅ Tots els gràfics guardats en: {SAVE_DIR}")

if __name__ == "__main__":
    main()