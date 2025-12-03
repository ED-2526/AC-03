import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np
import pandas as pd
# --- IMPORTAR TU NUEVO MÓDULO ---
from carrega_dades import cargar_y_preprocesar_datos

# 0. CONFIGURACIÓN
plot_dir = os.path.join(os.getcwd(), "Plots_RF")
os.makedirs(plot_dir, exist_ok=True)

# 1. CARGA Y PREPROCESAMIENTO (¡Solo una línea!)
try:
    X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos()
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
output_file = os.path.join(plot_dir, 'rf_class_distribution_bar_chart.png')
plt.savefig(output_file)
print(f"✅ Gráfico de distribución guardado en '{output_file}'")
plt.close()

# --- 4. DEFINICIÓN, ENTRENAMIENTO Y EVALUACIÓN DEL MODELO ---

print("\n--- 4. ENTRENAMIENTO DE RANDOM FOREST ---")

# 4.1. Definición del Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 4.2. Entrenamiento
print("Entrenando el modelo...")
model.fit(X_train, y_train)
print("✅ Entrenamiento finalizado.")

# 4.3. Predicción
y_pred = model.predict(X_test) # ESTE ES EL RETORNO PRINCIPAL: numpy array de clases predichas

# 4.4. Evaluación
print("\n--- 5. RESULTADOS DE LA EVALUACIÓN ---")

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy*100:.2f}%")

# ============================================================
# 6. GRÁFICOS ADICIONALES PARA RANDOM FOREST
# ============================================================

print("\n--- Generando gráficos adicionales para Random Forest ---")

# --- 6.1. Per-Class Metrics (precision, recall, f1) ---
p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
    y_test, y_pred, average=None # Importante: average=None para métricas por clase
)

plt.figure(figsize=(12, 6))
x = np.arange(len(class_names))
width = 0.25

plt.bar(x - width, p_per_class, width, label='Precision')
plt.bar(x, r_per_class, width, label='Recall')
plt.bar(x + width, f1_per_class, width, label='F1-score')

plt.xticks(x, class_names, rotation=45)
plt.ylabel("Score")
plt.title("Métricas por Clase (Random Forest)")
plt.legend()
plt.tight_layout()

# 👉 Guardar gráfico
plt.savefig(os.path.join(plot_dir, "rf_per_class_metrics.png"))
print(f"✅ Gráfico de métricas por clase guardado como: '{os.path.join(plot_dir, 'rf_per_class_metrics.png')}'")
plt.close()

# --- 6.2. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión (Random Forest)")
plt.tight_layout()

# 👉 Guardar gráfico
plt.savefig(os.path.join(plot_dir, "rf_confusion_matrix.png"))
print(f"✅ Gráfico de matriz de confusión guardado como: '{os.path.join(plot_dir, 'rf_confusion_matrix.png')}'")
plt.close()

print("\n--- Ejecución del script Random Forest finalizada ---")