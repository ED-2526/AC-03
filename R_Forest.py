import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np
import pandas as pd
from carrega_dades import *
from Plots import *

MODEL = "Random Forest"
# 1. CARGA Y PREPROCESAMIENTO 
try:
    X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_3s() # Per a 3 segons
    #X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_30s() # Per a 30 segons
    class_names = label_encoder.classes_
except Exception as e:
    print(e)
    exit()


# --- 3. Generación del Gráfico de Distribució de Clases ---

print("\n--- 3. Generación del Gráfico de Distribución de Clases ---")

# CORRECCIÓN: Como ya no tenemos 'df', reconstruimos las etiquetas totales
# 1. Unimos las etiquetas de entrenamiento y test
y_total_encoded = np.concatenate([y_train, y_test])

# 2. Las convertimos de números (0, 1...) a nombres ('blues', 'classical'...)
y_total_names = label_encoder.inverse_transform(y_total_encoded)

# 3. Convertimos a Pandas Series para poder usar value_counts() fácilmente
y_labels = pd.Series(y_total_names)

# 4. Contar la frecuencia (Igual que antes)
class_counts = y_labels.value_counts().sort_index()

# 5. Crear el gráfico de barras
plt.figure(figsize=(12, 6))
class_counts.plot(kind='bar', color='darkgreen')

# Títulos y etiquetas
plt.title('Distribución de Géneros Musicales (Total Dataset)', fontsize=14)
plt.xlabel('Género Musical', fontsize=12)
plt.ylabel('Número de Muestras', fontsize=12)

# Ajustes visuales
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Poner valores sobre las barras
for index, value in enumerate(class_counts):
    plt.text(index, value, f'{value}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Guardar
output_file = 'rf_class_distribution_bar_chart.png'
plt.savefig(output_file)
print(f"✅ Gráfico de distribución guardado en '{output_file}'")
plt.close()

# --- 4. DEFINICIÓN, ENTRENAMIENTO Y EVALUACIÓN DEL MODELO ---

print("\n--- 4. ENTRENAMIENTO DE RANDOM FOREST ---")

# 4.1. Definición del Model
model = RandomForestClassifier(
    n_estimators=200,       # Abans: 100, Ara: 200
    max_depth=20,             # S'afegeix la profunditat màxima: 20
    min_samples_split=2, # S'afegeix el mínim de mostres per a la divisió: 2
    
    random_state=42, # Mantens la llavor per a la reproductibilitat
    n_jobs=-1        # Mantens l'ús de tots els nuclis
)

# 4.2. Entrenamiento
print("Entrenando el modelo...")
model.fit(X_train, y_train)
print("✅ Entrenamiento finalizado.")

# 4.3. Predicción
y_pred = model.predict(X_test) # ESTE ES EL RETORNO PRINCIPAL: numpy array de clases predichas
try:
    y_prob_test = model.predict_proba(X_test)
except AttributeError:
    y_prob_test = None
    print("El modelo no soporta predict_proba(). No se podrán generar curvas ROC/PR.")

# 4.4. Evaluación
print("\n--- 5. RESULTADOS DE LA EVALUACIÓN ---")

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy*100:.2f}%")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
# ============================================================
# 6. GRÁFICOS ADICIONALES PARA RANDOM FOREST
# ============================================================

print("\n--- Generando gráficos adicionales para Random Forest ---")

# --- 6.1. Per-Class Metrics (precision, recall, f1) ---

plot_per_class_metrics(y_test, y_pred, class_names, MODEL)

# --- 6.2. Confusion Matrix ---

plot_confusion_matrix(y_test, y_pred, class_names, MODEL)

# --- 6.3. ROC Curve ---

if y_prob_test is not None:
    plot_roc_curve(y_test, y_prob_test, MODEL, class_names)
    plot_general_roc_curve(y_test, y_prob_test, MODEL, class_names)

# --- 6.4. Precision-Recall Curve ---

if y_prob_test is not None:
    plot_precision_recall_curve(y_test, y_prob_test, MODEL, class_names)
    plot_general_pr_curve(y_test, y_prob_test, MODEL, class_names)