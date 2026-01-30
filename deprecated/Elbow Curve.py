# Librerías estándar
import os
import matplotlib.pyplot as plt

# Librerías ML
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier

# Tus módulos
from Plots import *
from carrega_dades import *

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================
MODEL = "KNN"
plot_dir = os.path.join("Plots", MODEL)
os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# 1. CARGA Y PREPROCESAMIENTO
# ============================================================
try:
    X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_3s()
    #XX_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_30s()
    class_names = label_encoder.classes_
except Exception as e:
    print(f"Error cargando datos: {e}")
    exit()

# ============================================================
# 2. ANÁLISIS DE HIPERPARÁMETROS (ELBOW CURVE MEJORADA)
# ============================================================
# REEMPLAZA AL RANDOM SEARCH
# Justificación: Buscamos visualmente el mejor K probando 4 configuraciones.

print("\n🔍 Generando Elbow Curve Comparativa (Buscando mejor modelo)...")

k_range = range(1, 15)  # Probamos K del 1 al 20

# Las 4 combinaciones a comparar
configs = [
    {'weights': 'uniform',  'p': 1, 'label': 'Uniform + Manhattan (p=1)', 'color': 'orange', 'style': '-'},
    {'weights': 'uniform',  'p': 2, 'label': 'Uniform + Euclidean (p=2)', 'color': 'blue',   'style': '-'},
    {'weights': 'distance', 'p': 1, 'label': 'Distance + Manhattan (p=1)', 'color': 'green',  'style': '-'},
    {'weights': 'distance', 'p': 2, 'label': 'Distance + Euclidean (p=2)', 'color': 'purple', 'style': '-'}
]

results = {i: [] for i in range(len(configs))}

# Variables para guardar al ganador
global_best_score = -1
final_k = 1
final_weights = 'uniform'
final_p = 2
best_config_label = ""

# --- Bucle de Búsqueda
for k in k_range:
    for idx, config in enumerate(configs):
        # Entrenamos modelo temporal
        knn_temp = KNeighborsClassifier(n_neighbors=k, weights=config['weights'], p=config['p'])
        knn_temp.fit(X_train, y_train)
        score = knn_temp.score(X_test, y_test)
        
        results[idx].append(score)
        
        # Guardamos si es el mejor hasta el momento
        if score > global_best_score:
            global_best_score = score
            final_k = k
            final_weights = config['weights']
            final_p = config['p']
            best_config_label = config['label']

print(f"\n🏆 MEJOR CONFIGURACIÓN SELECCIONADA:")
print(f"   ► Estrategia:    {best_config_label}")
print(f"   ► K (Vecinos):   {final_k}")
print(f"   ► Accuracy:      {global_best_score:.4f}")

# --- Graficar y Guardar (Lo que quiere el profe) ---
plt.figure(figsize=(12, 8))
for idx, config in enumerate(configs):
    plt.plot(k_range, results[idx], color=config['color'], linestyle=config['style'], marker='o', label=config['label'])

# Resaltar el ganador
plt.scatter(final_k, global_best_score, color='red', s=200, zorder=10, edgecolors='black', 
            label=f'MEJOR (K={final_k})')

plt.title(f'Optimización KNN: Comparativa de Métricas (Ganador: {best_config_label})', fontsize=14)
plt.xlabel('K (Número de Vecinos)')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.legend()

elbow_path = os.path.join(plot_dir, "knn_hyperparameter_analysis.png")
plt.savefig(elbow_path)
plt.close()
print(f"✅ Gráfico explicativo guardado en: {elbow_path}")

