import os
import sys
import pandas as pd
import numpy as np

# Importacions des dels teus mòduls
from carrega_dades import cargar_y_preprocesar_datos_3s, cargar_y_preprocesar_datos_30s
from Models import (
    executar_knn, executar_random_forest, executar_svm, executar_xgboost, 
    executar_regressio_logistica, executar_decision_tree, executar_gmm_classifier, 
    executar_naive_bayes, executar_kmeans_clustering 
    # Asegura't d'importar tots els models aquí
)
from plots_2 import plot_class_distribution, plot_comparative_roc

# --- CONFIGURACIÓ GLOBAL ---
RANDOM_STATE = 42

def triar_dades():
    """ Permet a l'usuari triar el tipus de dades a utilitzar. """
    print("\n--- PAS 1: SELECCIÓ DE DADES ---")
    print("Quin dataset vols utilitzar?")
    print(" [1] Segments de 3 segons (amb GroupSplit)")
    print(" [2] Cançons de 30 segons (amb Stratified Split)")
    
    while True:
        choice = input("Introdueix 1 o 2: ").strip()
        if choice == '1':
            return '3s', cargar_y_preprocesar_datos_3s
        elif choice == '2':
            return '30s', cargar_y_preprocesar_datos_30s
        else:
            print("Selecció invàlida. Tria 1 o 2.")

def triar_model():
    """ Permet a l'usuari triar el model a executar. """
    print("\n--- PAS 3: SELECCIÓ DE MODEL ---")
    print("Quin model vols executar?")
    print(" [1] K-Nearest Neighbors (KNN)")
    print(" [2] Random Forest (RF)")
    print(" [3] Support Vector Machine (SVM)")
    print(" [4] XGBoost")
    print(" [5] Regressió Logística")
    print(" [6] Decision Tree (DT)")
    print(" [7] GMM Classifier")
    print(" [8] Naive Bayes (NB)")
    print(" [9] K-Means Clustering") 
    print(" [10] Tots els models de CLASSIFICACIÓ (1-9)") # Tots els classificadors
    
    while True:
        choice = input("Introdueix 1-10: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= 11:
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
        
    # 2. GENERAR GRÀFIC DE DISTRIBUCIÓ DE CLASSES (Només una vegada)
    print("\n--- PAS 2: GENERACIÓ DE GRÀFICS DE DADES ---")
    plot_class_distribution(y_train, y_test, label_encoder)

    # 3. SELECCIÓ DE MODEL A EXECUTAR
    model_choice = triar_model()
    
    # --- 3.1. Mapa de Models (Classificació 1-9) ---
    model_map_classificacio = {
        '1': lambda: executar_knn(X_train, X_test, y_train, y_test, label_encoder, data_type, best_k=4, best_weights='distance', best_p=1),
        '2': lambda: executar_random_forest(X_train, X_test, y_train, y_test, label_encoder, data_type, n_estimators=250, max_depth=10, min_samples_split=2),
        '3': lambda: executar_svm(X_train, X_test, y_train, y_test, label_encoder, data_type, C=10.0, gamma=0.001),
        '4': lambda: executar_xgboost(X_train, X_test, y_train, y_test, label_encoder, scaler, data_type),
        '5': lambda: executar_regressio_logistica(X_train, X_test, y_train, y_test, label_encoder, data_type),
        '6': lambda: executar_decision_tree(X_train, X_test, y_train, y_test, label_encoder, data_type),
        '7': lambda: executar_gmm_classifier(X_train, X_test, y_train, y_test, label_encoder, data_type),
        '8': lambda: executar_naive_bayes(X_train, X_test, y_train, y_test, label_encoder, data_type), # NB
        
    }
    
    models_results = []
    
    # --- 3.2. Lògica d'Execució ---
    
    if model_choice == '9':
        # Executar K-Means Clustering (Clustering: No participa en ROC/F1)
        models_results.append(
            executar_kmeans_clustering(X_train, X_test, y_train, y_test, label_encoder, data_type)
        )
    
    elif model_choice == '10': 
        # Executar Tots els models de Classificació (1-9)
        for key, func in model_map_classificacio.items():
            models_results.append(func())
    
    elif model_choice in model_map_classificacio:
        # Executar un model de classificació individual
        models_results.append(model_map_classificacio[model_choice]())

    else:
        print("Opció no reconeguda.")
        
    # --- 4. ANÀLISI GLOBAL I GRÀFICS COMPARATIUS ---
    
    if models_results:
        
        prob_dict = {}
        model_names_list = []

        # Recollir les probabilitats de test i els noms dels models
        for result in models_results:
            # Només models de classificació tenen 'probabilities'
            if result.get('probabilities') is not None:
                prob_dict[result['model']] = result['probabilities']
                model_names_list.append(result['model'])
        
        # Generar el gràfic ROC comparatiu només si s'han executat Tots (Opció 11)
        if model_choice == '10' and prob_dict:
            print("\n--- PAS 5: GRÀFIC ROC COMPARATIU ---")
            plot_comparative_roc(
                y_test, 
                prob_dict, 
                model_names_list, 
                label_encoder.classes_, 
                data_type
            )
        elif model_choice == '10':
            print("\n⚠️ Advertència: No es pot generar el gràfic ROC comparatiu. Cap model retorna probabilitats (predict_proba).")

        # 5. TAULA RESUM FINAL
        print("\n" + "="*50)
        print(f"       ✨ RESUM DE RENDIMENT FINAL ({data_type.upper()}) ✨")
        print("="*50)
        
        # Preparar les dades per a la taula
        for result in models_results:
             # Eliminem 'probabilities' i altres claus no desitjades de la taula final
             result.pop('probabilities', None) 
        
        df_results = pd.DataFrame(models_results).set_index('model')
        
        # Si hem executat un model de clustering, utilitzem l'ARI com a clau de classificació
        if 'ARI Score' in df_results.columns:
             print(df_results.to_markdown(floatfmt=".4f"))
        else:
             # Ordenar per F1 Score per defecte (models de classificació)
             print(df_results.sort_values(by='f1_score', ascending=False).to_markdown(floatfmt=".4f"))
             
        print("="*50)


if __name__ == "__main__":
    main()