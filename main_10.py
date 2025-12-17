import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Importacions del teu projecte original
from carrega_dades import split_datos_3s, split_datos_30s, cargar_y_preprocesar_datos_3s, cargar_y_preprocesar_datos_30s
from Models import (
    executar_knn, executar_random_forest, executar_svm, executar_xgboost, 
    executar_regressio_logistica, executar_decision_tree, executar_gmm_classifier, 
    executar_naive_bayes 
)
from plots_2 import plot_comparative_roc

# --- CONFIGURACIÓ ---
RANDOM_STATE = 42
TOP_N = 10

def obtenir_top_features(X_train_scaled, y_train, feature_names, n=10):
    """
    Entrena un Random Forest ràpid per identificar les n millors variables.
    """
    print(f"\n🔍 Calculant les {n} millors variables mitjançant Random Forest...")
    selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    selector.fit(X_train_scaled, y_train)
    
    importances = selector.feature_importances_
    indices = np.argsort(importances)[::-1][:n]
    
    top_features = [feature_names[i] for i in indices]
    print(f"✅ Top {n} detectat: {top_features}")
    return indices, top_features

def main():
    print("--- 🎼 EXPERIMENT: CLASSIFICACIÓ AMB TOP 10 FEATURES ---")
    
    # 1. SELECCIÓ DE DATASET (Simulem la teva lògica del main)
    print("Quin dataset vols utilitzar per l'experiment Top 10?")
    print(" [1] 3 segons | [2] 30 segons")
    choice_data = input("Tria 1 o 2: ").strip()
    
    if choice_data == '1':
        data_type, func_split, func_pre = '3s_Top10', split_datos_3s, cargar_y_preprocesar_datos_3s
    else:
        data_type, func_split, func_pre = '30s_Top10', split_datos_30s, cargar_y_preprocesar_datos_30s

    # 2. CARREGAR DADES I OBTENIR NOMS DE COLUMNES
    # Necessitem la càrrega pre-split per tenir els noms de les columnes (X és DataFrame)
    datos_raw = func_pre()
    X_raw = datos_raw[0] 
    feature_names = X_raw.columns.tolist()

    # 3. EXECUTAR SPLIT ORIGINAL (Dades escalades)
    X_train, X_test, y_train, y_test, label_encoder, scaler = func_split(random_state=RANDOM_STATE)

    # 4. SELECCIONAR ÍNDEXS DE LES TOP 10
    top_indices, top_names = obtenir_top_features(X_train, y_train, feature_names, n=TOP_N)

    # 5. FILTRAR MATRIUS
    X_train_top = X_train[:, top_indices]
    X_test_top = X_test[:, top_indices]

    # 6. EXECUTAR TOTS ELS MODELS AMB LES DADES REDUÏDES
    # Reutilitzem el teu mapa de models adaptant les noves X_train/X_test
    model_map = {
        'KNN': lambda: executar_knn(X_train_top, X_test_top, y_train, y_test, label_encoder, data_type),
        'RF': lambda: executar_random_forest(X_train_top, X_test_top, y_train, y_test, label_encoder, data_type),
        'SVM': lambda: executar_svm(X_train_top, X_test_top, y_train, y_test, label_encoder, data_type),
        'XGB': lambda: executar_xgboost(X_train_top, X_test_top, y_train, y_test, label_encoder, top_names, data_type),
        'LR': lambda: executar_regressio_logistica(X_train_top, X_test_top, y_train, y_test, label_encoder, data_type),
        'DT': lambda: executar_decision_tree(X_train_top, X_test_top, y_train, y_test, label_encoder, data_type),
        'GMM': lambda: executar_gmm_classifier(X_train_top, X_test_top, y_train, y_test, label_encoder, data_type),
        'NB': lambda: executar_naive_bayes(X_train_top, X_test_top, y_train, y_test, label_encoder, data_type),
    }

    results = []
    prob_dict = {}
    
    for name, func in model_map.items():
        res = func()
        results.append(res)
        if res.get('probabilities') is not None:
            prob_dict[res['model']] = res['probabilities']

    # 7. COMPARATIVA I RESUM
    if prob_dict:
        plot_comparative_roc(y_test, prob_dict, list(prob_dict.keys()), label_encoder.classes_, data_type)

    print("\n" + "="*60)
    print(f"🏆 RESUM DE RENDIMENT AMB NOMÉS {TOP_N} FEATURES ({data_type})")
    print("="*60)
    
    # Neteja de probabilitats per a la taula
    for r in results: r.pop('probabilities', None)
    
    df_final = pd.DataFrame(results).set_index('model')
    print(df_final.sort_values(by='f1_score', ascending=False).to_markdown(floatfmt=".4f"))

"""
============================================================
🏆 RESUM DE RENDIMENT AMB NOMÉS 10 FEATURES (3s_Top10)
============================================================
| model                               |   accuracy |   train_accuracy |   f1_score |
|:------------------------------------|-----------:|-----------------:|-----------:|
| SVM (3s_Top10)                      |     0.6073 |           0.6521 |     0.6034 |
| KNN (3s_Top10)                      |     0.5928 |           0.9987 |     0.5896 |
| Random Forest (3s_Top10)            |     0.5828 |           0.7002 |     0.5772 |
| GMM Classifier (3s_Top10)           |     0.5748 |           0.7257 |     0.5740 |
| XGBoost (3s_Top10)                  |     0.5663 |           0.6545 |     0.5593 |
| Regressió Logística (3s_Top10)      |     0.5313 |           0.5464 |     0.5243 |
| Decision Tree (3s_Top10)            |     0.5148 |           0.9987 |     0.5145 |
| Naive Bayes (GaussianNB - 3s_Top10) |     0.4532 |           0.4555 |     0.4334 |
"""
if __name__ == "__main__":
    main()