import os
import sys
import pandas as pd
import numpy as np

# Importacions des dels teus mòduls
from carrega_dades import split_datos_3s, split_datos_30s
from Models import (
    executar_knn, executar_random_forest, executar_svm, executar_xgboost, 
    executar_regressio_logistica, executar_decision_tree, executar_gmm_classifier, 
    executar_naive_bayes, executar_kmeans_clustering 
)
from plots_2 import plot_class_distribution, plot_comparative_roc

# Funcions EDA i PCA
from eda_pca import (
    eda_estadisticas_basicas, 
    eda_heatmap_correlacion, 
    eda_distribucion_clases, 
    ejecutar_pca, 
    graficar_pca_2d
)

# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42
PLOT_DIR = "plots"

def triar_dades():
    print("\n--- PAS 1: SELECCIÓ DE DADES ---")
    print("Quin dataset vols utilitzar?")
    print(" [1] Segments de 3 segons (amb GroupSplit)")
    print(" [2] Cançons de 30 segons (amb Stratified Split)")
    
    while True:
        choice = input("Introdueix 1 o 2: ").strip()
        if choice == '1':
            return '3s', split_datos_3s
        elif choice == '2':
            return '30s', split_datos_30s
        else:
            print("Selecció invàlida. Tria 1 o 2.")

def triar_model():
    print("\n--- PAS 3: SELECCIÓ D'ANÀLISI / MODEL ---")
    print("Quina opció vols executar?")
    print(" [0] EDA + PCA (exploració de dades)")
    print(" [1] K-Nearest Neighbors (KNN)")
    print(" [2] Random Forest (RF)")
    print(" [3] Support Vector Machine (SVM)")
    print(" [4] XGBoost")
    print(" [5] Regressió Logística")
    print(" [6] Decision Tree (DT)")
    print(" [7] GMM Classifier")
    print(" [8] Naive Bayes (NB)")
    print(" [9] K-Means Clustering") 
    print(" [10] Tots els models de CLASSIFICACIÓ (1-9)")
    
    while True:
        choice = input("Introdueix 0-10: ").strip()
        if choice.isdigit() and 0 <= int(choice) <= 10:
            return choice
        else:
            print("Selecció invàlida.")

def main():
    # 1. SELECCIÓ I CÀRREGA DE DADES
    data_type, funcio_carrega = triar_dades()
    
    try:
        X_train, X_test, y_train, y_test, label_encoder, scaler = funcio_carrega(random_state=RANDOM_STATE)
    except FileNotFoundError as e:
        print(f"Error crític: {e}")
        sys.exit(1)

    # 2. SELECCIÓ D'ANÀLISI / MODEL
    choice = triar_model()

    if choice == '0':
        # EDA + PCA
        print("\n--- PAS 2: EDA + PCA ---")
        eda_estadisticas_basicas(X_train, y_train)
        eda_heatmap_correlacion(pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])]), PLOT_DIR)
        eda_distribucion_clases(y_train, label_encoder, PLOT_DIR)

        X_train_pca, pca_model, pca_scaler = ejecutar_pca(X_train, n_components=2)
        graficar_pca_2d(X_train_pca, y_train, label_encoder, PLOT_DIR)
        print("✅ EDA + PCA completat.")
        return  # no executem models

    # --- Models de classificació ---
    model_map_classificacio = {
        '1': lambda: executar_knn(X_train, X_test, y_train, y_test, label_encoder, data_type, best_k=4, best_weights='distance', best_p=1),
        '2': lambda: executar_random_forest(X_train, X_test, y_train, y_test, label_encoder, data_type, n_estimators=50, max_depth=8, min_samples_split=15, min_samples_leaf=5),
        '3': lambda: executar_svm(X_train, X_test, y_train, y_test, label_encoder, data_type, C=10.0, gamma=0.001),
        '4': lambda: executar_xgboost(X_train, X_test, y_train, y_test, label_encoder, scaler, data_type),
        '5': lambda: executar_regressio_logistica(X_train, X_test, y_train, y_test, label_encoder, data_type),
        '6': lambda: executar_decision_tree(X_train, X_test, y_train, y_test, label_encoder, data_type),
        '7': lambda: executar_gmm_classifier(X_train, X_test, y_train, y_test, label_encoder, data_type),
        '8': lambda: executar_naive_bayes(X_train, X_test, y_train, y_test, label_encoder, data_type),
    }

    models_results = []
    if choice == '9':
        models_results.append(executar_kmeans_clustering(X_train, X_test, y_train, y_test, label_encoder, data_type))
    elif choice == '10':
        for key, func in model_map_classificacio.items():
            models_results.append(func())
    elif choice in model_map_classificacio:
        models_results.append(model_map_classificacio[choice]())
    else:
        print("Opció no reconeguda.")
        return

    # --- Resultats i ROC comparatiu ---
    if models_results:
        prob_dict = {}
        model_names_list = []
        for result in models_results:
            if result.get('probabilities') is not None:
                prob_dict[result['model']] = result['probabilities']
                model_names_list.append(result['model'])

        if choice == '10' and prob_dict:
            plot_comparative_roc(y_test, prob_dict, model_names_list, label_encoder.classes_, data_type)

        # TAULA FINAL
        print("\n" + "="*50)
        print(f"       ✨ RESUM DE RENDIMENT FINAL ({data_type.upper()}) ✨")
        print("="*50)
        for result in models_results:
            result.pop('probabilities', None)
        df_results = pd.DataFrame(models_results).set_index('model')
        if 'ARI Score' in df_results.columns:
            print(df_results.to_markdown(floatfmt=".4f"))
        else:
            print(df_results.sort_values(by='f1_score', ascending=False).to_markdown(floatfmt=".4f"))
        print("="*50)

if __name__ == "__main__":
    main()
