import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Importacions des dels teus mòduls
# ASEGÚRATE DE IMPORTAR TANTOS LOS SPLIT COMO LOS CARGAR
from carrega_dades import (
    split_datos_3s, split_datos_30s, 
    cargar_y_preprocesar_datos_3s, cargar_y_preprocesar_datos_30s
)

from Models import (
    executar_knn, executar_random_forest, executar_svm, executar_xgboost, 
    executar_regressio_logistica, executar_decision_tree, executar_gmm_classifier, 
    executar_naive_bayes 
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
    """
    Retorna l'identificador, la funció de càrrega completa i la funció de split.
    """
    print("\n--- PAS 1: SELECCIÓ DE DADES ---")
    print("Quin dataset vols utilitzar?")
    print(" [1] Segments de 3 segons (amb GroupSplit)")
    print(" [2] Cançons de 30 segons (amb Stratified Split)")
    
    while True:
        choice = input("Introdueix 1 o 2: ").strip()
        if choice == '1':
            # Retornem: ID, Funció Càrrega Completa, Funció Split
            return '3s', cargar_y_preprocesar_datos_3s, split_datos_3s
        elif choice == '2':
            return '30s', cargar_y_preprocesar_datos_30s, split_datos_30s
        else:
            print("Selecció invàlida. Tria 1 o 2.")

def triar_model():
    print("\n--- PAS 2: SELECCIÓ D'ANÀLISI / MODEL ---")
    print("Quina opció vols executar?")
    print(" [0] EDA + PCA (exploració de TOTES les dades)")
    print(" [1] K-Nearest Neighbors (KNN)")
    print(" [2] Random Forest (RF)")
    print(" [3] Support Vector Machine (SVM)")
    print(" [4] XGBoost")
    print(" [5] Regressió Logística")
    print(" [6] Decision Tree (DT)")
    print(" [7] GMM Classifier")
    print(" [8] Naive Bayes (NB)")
    print(" [9] Tots els models de CLASSIFICACIÓ (1-9)")
    
    while True:
        choice = input("Introdueix 0-9: ").strip()
        if choice.isdigit() and 0 <= int(choice) <= 10:
            return choice
        else:
            print("Selecció invàlida.")

def main():
    # 1. SELECCIÓ DE DATASET
    data_type, funcio_carrega_total, funcio_split = triar_dades()
    
    # 2. SELECCIÓ D'ANÀLISI / MODEL
    choice = triar_model()

    # --- OPCIÓ 0: EDA + PCA (UTILITZA TOT EL DATASET) ---
    if choice == '0':
        print(f"\n--- CARREGANT TOT EL DATASET ({data_type}) PER A EDA/PCA ---")
        try:
            # Càrrega dinàmica depenent de si retorna groups o no
            datos = funcio_carrega_total()
            
            if data_type == '3s':
                # Desempaquetem 4 valors
                X, y, groups, label_encoder = datos
            else:
                # Desempaquetem 3 valors
                X, y, label_encoder = datos
            
            print(f"✅ Dades carregades. Shape: {X.shape}")

            # 1. EDA BÀSIC (amb X que és un DataFrame, conserva noms de columnes)
            print("\n--- EXECUTANT EDA ---")
            eda_estadisticas_basicas(X, y)
            eda_heatmap_correlacion(X, PLOT_DIR) # X aquí té noms de columnes reals
            eda_distribucion_clases(y, label_encoder, PLOT_DIR)

            # 2. PCA (Necessita dades escalades)
            print("\n--- EXECUTANT PCA ---")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X) # Convertim a numpy array escalat

            X_pca, pca_model, pca_scaler = ejecutar_pca(X_scaled, n_components=2)
            graficar_pca_2d(X_pca, y, label_encoder, PLOT_DIR)
            
            print("✅ EDA + PCA completat sobre la totalitat de les dades.")
            return 

        except FileNotFoundError as e:
            print(f"Error crític carregant dades: {e}")
            sys.exit(1)

    # --- OPCIONS 1-10: MODELS (UTILITZA SPLIT TRAIN/TEST) ---
    else:
        print(f"\n--- GENERANT SPLIT TRAIN/TEST ({data_type}) PER A MODELS ---")
        try:
            # Aquí cridem a la funció de SPLIT, que ja escala i divideix
            X_train, X_test, y_train, y_test, label_encoder, scaler = funcio_split(random_state=RANDOM_STATE)
        except FileNotFoundError as e:
            print(f"Error crític: {e}")
            sys.exit(1)

        # Definició de models
        model_map_classificacio = {
            '1': lambda: executar_knn(X_train, X_test, y_train, y_test, label_encoder, data_type, best_k=4, best_weights='distance', best_p=1),
            '2': lambda: executar_random_forest(X_train, X_test, y_train, y_test, label_encoder, data_type, n_estimators=50, max_depth=8, min_samples_split=15, min_samples_leaf=5),
            '3': lambda: executar_svm(X_train, X_test, y_train, y_test, label_encoder, data_type, C=1, gamma=0.002), #3s(c=1,gamma=0.005), 30s (c=1,gamma=0.002)
            '4': lambda: executar_xgboost(X_train, X_test, y_train, y_test, label_encoder, scaler, data_type),
            '5': lambda: executar_regressio_logistica(X_train, X_test, y_train, y_test, label_encoder, data_type),
            '6': lambda: executar_decision_tree(X_train, X_test, y_train, y_test, label_encoder, data_type),
            '7': lambda: executar_gmm_classifier(X_train, X_test, y_train, y_test, label_encoder, data_type),
            '8': lambda: executar_naive_bayes(X_train, X_test, y_train, y_test, label_encoder, data_type),
        }

        models_results = []
        if choice == '9':
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

            if choice == '9' and prob_dict:
                plot_comparative_roc(y_test, prob_dict, model_names_list, label_encoder.classes_, data_type)

            # TAULA FINAL
            print("\n" + "="*50)
            print(f"       ✨ RESUM DE RENDIMENT FINAL ({data_type.upper()}) ✨")
            print("="*50)
            # Neteja probabilitats per a la visualització de la taula
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