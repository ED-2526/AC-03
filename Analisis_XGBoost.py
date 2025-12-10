import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import seaborn as sns
from sklearn.model_selection import validation_curve, GridSearchCV
from carrega_dades import cargar_y_preprocesar_datos_3s
# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42
SAVE_DIR = "Plots/Analisis_XGB"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# BLOC 1: GRID SEARCH EXHAUSTIU (FASE 1) - OPCIONAL (Targa 45 minuts aprox.)
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
# BLOC 2: GENERACIÓ DE CORBES DE VALIDACIÓ (1D)
# =============================================================================
def plot_validation_curve(estimator, X, y, param_name, param_range, title):
    print(f"   ⚙️  Generant corba 1D per a: {param_name}...")
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=3, scoring="accuracy", n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.plot(param_range, train_mean, label="Train", color="darkorange")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color="darkorange")
    plt.plot(param_range, test_mean, label="Test (CV)", color="navy")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color="navy")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--')
    plt.savefig(f"{SAVE_DIR}/VC_{param_name}.png")
    plt.close()

# =============================================================================
# MAIN (Execució Controlada)
# =============================================================================
def main():
    print("--- 🔬 LABORATORI D'ANÀLISI XGBOOST ---")
    
    # 1. Carregar dades
    X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_3s(random_state=RANDOM_STATE)
    
    # 2. Definir Model Base per a les Corbes de Validació (Exploració)
    xgb_base = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=RANDOM_STATE, n_jobs=-1)
    
    # 3. Generar Corbes individuals (1D) - Anàlisi de Sensibilitat

    plot_validation_curve(xgb_base, X_train, y_train, "max_depth", [1, 2, 3, 4, 5, 6, 8, 10], "Impacte Profunditat")
    plot_validation_curve(xgb_base, X_train, y_train, "n_estimators", [50, 100, 200, 300, 500], "Impacte Arbres")
    plot_validation_curve(xgb_base, X_train, y_train, "learning_rate", [0.01, 0.05, 0.1, 0.2, 0.3], "Impacte Taxa d'Aprenentatge")
    plot_validation_curve(xgb_base, X_train, y_train, "reg_alpha", [0, 0.1, 0.5, 1.0, 2.0, 5.0], "Impacte Regularitzacio L1")
    plot_validation_curve(xgb_base, X_train, y_train, "gamma", [0, 0.1, 0.2, 0.3, 0.5, 1.0], "Impacte Gamma (Min Split Loss)") 
    plot_validation_curve(xgb_base, X_train, y_train, "min_child_weight", [1, 3, 5, 7], "Impacte Min Child Weight")

    
    print("\n✅ Totes les gràfiques d'anàlisi han estat generades a 'Plots/Analisis_XGB'.")
    
    #NOTAR: La cerca exhaustiva no s'executa per defecte per estalviar temps.
    # run_heavy_grid_search(X_train, y_train)
    # RESULTATS DE LA CERCA EXHAUSTIVA (FASE 1:
    # Millors paràmetres trobats:
    # {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.7}

if __name__ == "__main__":
    main()
