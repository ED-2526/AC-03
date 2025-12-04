import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns          
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os
from carrega_dades import *

def plot_grid_search_results(grid_search, save_path=None):
    results = grid_search.cv_results_
    df_results = pd.DataFrame(results)

    # Neteja i preparació de dades
    gammas = df_results['param_gamma'].astype(str)
    Cs = df_results['param_C'].astype(str)
    gamma_values = gammas.unique()
    C_values = Cs.unique().astype(float) 

    plt.figure(figsize=(14, 8))
    
    # Dibuixar una línia per a cada valor de gamma
    for gamma in gamma_values:
        subset = df_results[df_results['param_gamma'].astype(str) == gamma]
        subset = subset.sort_values(by='param_C') 

        plt.plot(
            subset['param_C'].astype(float),
            subset['mean_test_score'],       
            marker='o', 
            label=f'gamma={gamma}'
        )

    # Configuració i títols de la gràfica
    plt.xscale('log') 
    
    plt.xlabel('Valor del Paràmetre C (Escala Logarítmica)', fontsize=14)
    plt.ylabel(f"Puntuació Mitjana de Validació Creuada (Accuracy)", fontsize=14)
    plt.title('Impacte de C i gamma en el Rendiment del SVC (Grid Search CV)', fontsize=16)
    
    plt.xticks(C_values, labels=C_values, rotation=0) 
    plt.legend(title='Paràmetre gamma', fontsize=12)
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path)
            print(f"✅ Gràfic de Grid Search desat a: {save_path}")
        except Exception as e:
            print(f"❌ Error al guardar el gràfic: {e}")


### 1. Carrega i Preprocessament de les Dades
try:
    # Hem d'assegurar que passem el random_state si la funció el requereix
    X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_3s() 
    class_names = label_encoder.classes_
except Exception as e:
    print(e)
    exit()

# 2.2. Definició de la Malla (Grid) de Paràmetres C i gamma
param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': [0.001, 0.01, 0.1, 'scale'], 
    'kernel': ['rbf']
}

# 2.3. Inicialització i Execució de GridSearchCV
grid_search = GridSearchCV(
    estimator=SVC(random_state=42, probability=False),
    param_grid=param_grid,
    scoring='accuracy', 
    cv=5, 
    verbose=3,
    n_jobs=-1 
)

grid_search.fit(X_train, y_train)

# 2.4. Resultats de l'Optimització
print("\n" + "="*50)
print("🏆 Millors Paràmetres Trobats:")
print(grid_search.best_params_)
print("\nMillor Precisió (Accuracy) en Validació Creuada:")
print(f"{grid_search.best_score_:.4f}")
print("="*50)

# 1. Definició de les variables de ruta
NOM_GRÀFIC = 'resultats_grid_svm.png'
CARPETA_PLOT = 'Plots'
CARPETA_SVM = 'SVM'

# 2. Creació de la ruta completa
base_dir = os.path.dirname(os.path.abspath(__file__)) 
directori_desti = os.path.join(base_dir, CARPETA_PLOT, CARPETA_SVM)
path_complet_guardar = os.path.join(directori_desti, NOM_GRÀFIC)
os.makedirs(directori_desti, exist_ok=True) 

# 3. Cridar la funció per guardar
plot_grid_search_results(grid_search, save_path=path_complet_guardar)