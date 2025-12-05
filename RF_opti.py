import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from carrega_dades import *

def optimitzar_random_forest_grid(X_train, y_train, random_state=42):
    # 1. Definició de la malla (Grid) de paràmetres a provar
    param_grid = {
        'n_estimators': [50, 100, 200],  # Nombre d'arbres al bosc
        'max_depth': [5, 10, 20, None],  # Profunditat màxima (None significa sense límit)
        'min_samples_split': [2, 5],     # Mínim de mostres necessàries per dividir un node
    }
    
    # Calcular el nombre total de models a provar
    num_combinations = (
        len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])
    )
    
    print("\n" + "="*70)
    print("Iniciant cerca de Grid Search per a Random Forest")
    print(f"Combinacions a provar: {num_combinations} (x 5 folds)")
    print("="*70)

    # 2. Inicialització de GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring='accuracy',  
        cv=5,               
        verbose=3,
        n_jobs=-1 
    )

    # 3. Execució de la cerca i entrenament
    grid_search.fit(X_train, y_train)

    return grid_search

def plot_rf_optimization_results(grid_search, save_path=None):
    
    results = grid_search.cv_results_
    df_results = pd.DataFrame(results)
    
    # Ens centrem en les variacions de max_depth, creant una línia per a cada n_estimators
    depths = df_results['param_max_depth'].unique()
    estimators = df_results['param_n_estimators'].unique()

    plt.figure(figsize=(14, 8))
    
    for n_est in estimators:
        # Filtrem per n_estimators i agrupem per max_depth
        subset = df_results[df_results['param_n_estimators'] == n_est]
        
        # Opció simplificada: Agrupem pel n_estimators i max_depth, agafant el millor score
        grouped = subset.groupby('param_max_depth')['mean_test_score'].max()

        # Reemplacem 'None' amb un string per a les etiquetes
        depth_labels = [str(d) for d in grouped.index]
        
        plt.plot(
            depth_labels,
            grouped.values,
            marker='o', 
            linestyle='-',
            label=f'{n_est} Estimadors'
        )

    plt.xlabel('Profunditat Màxima (max_depth)', fontsize=14)
    plt.ylabel(f"Puntuació Mitjana de Validació Creuada (Accuracy)", fontsize=14)
    plt.title('Rendiment del Random Forest segons Profunditat i Nombre d\'Arbres', fontsize=16)
    
    plt.legend(title='Nombre d\'Estimadors', fontsize=12)
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f" Gràfic de Random Forest Grid Search desat a: {save_path}")
            
    plt.show()

# 1. Carrega de dades 
try:
    X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_3s() # Per a 3 segons
    #X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_30s() # Per a 30 segons
    class_names = label_encoder.classes_
except Exception as e:
    print(e)
    exit()

# 2. Executar la cerca d'hiperparàmetres
resultats_grid_rf = optimitzar_random_forest_grid(X_train, y_train, random_state=42)

# 3. Imprimir i Analitzar els resultats
print("\n" + "="*50)
print("Millors Paràmetres Trobats (Random Forest):")
print(resultats_grid_rf.best_params_)
print("Millor Precisió (Accuracy) en Validació Creuada:")
print(f"{resultats_grid_rf.best_score_:.4f}")
print("="*50)

# 4. Generar la visualització
# Defineix la ruta on vols guardar la gràfica
GRAFIC_RF_PATH = './Plots/Random Forest/rf_grid_results.png' 
plot_rf_optimization_results(resultats_grid_rf, save_path=GRAFIC_RF_PATH)

