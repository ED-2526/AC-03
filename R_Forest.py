import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt # Importar matplotlib
import seaborn as sns # Importar seaborn para la matriz de confusión

# --- Librerías de Machine Learning y Preprocesamiento (Scikit-learn) ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# ============================================================
# 0. CREAR CARPETA DE GUARDADO DE PLOTS
# ============================================================
plot_dir = os.path.join(os.getcwd(), "Plots_RF") # Carpeta específica para RF
os.makedirs(plot_dir, exist_ok=True)
print(f"✅ Carpeta de plots creada/verificada en: {plot_dir}")

# --- 1. CARGA DE DATOS (Igual) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
#features_path = os.path.join(current_dir, 'Data', 'features_30_sec.csv') 
features_path = os.path.join(current_dir, 'Data', 'features_3_sec.csv')


if not os.path.exists(features_path):
    print(f"❌ Error: No encuentro el archivo en: {features_path}")
    print("Asegúrate de que la carpeta 'Data' está en el mismo lugar que este script.")
    exit()

df = pd.read_csv(features_path)
print(f"✅ Datos cargados correctamente desde: {features_path}")
print("Tamaño:", df.shape)

# --- 2. PREPROCESAMIENTO (Igual) ---

# Eliminar columnas
X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']

# Codificar las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_ # Obtener nombres de las clases para los plots

# Dividir en Train y Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. Generación del Gráfico de Distribució de Clases ---

print("\n--- 3. Generación del Gráfico de Distribució de Clases ---")

# Obtenemos la variable objetivo antes del 'train_test_split'
y_labels = df['label']

# 1. Contar la frecuencia de cada género
class_counts = y_labels.value_counts().sort_index()

# 2. Crear el gráfico de barres
plt.figure(figsize=(12, 6))
class_counts.plot(kind='bar', color='darkgreen')

# 3. Títulos y etiquetas
plt.title('Distribución de Géneros Musicales (Classes) para el Análisis de Homogeneidad', fontsize=14)
plt.xlabel('Género Musical', fontsize=12)
plt.ylabel('Número de Muestras', fontsize=12)

# 4. Ajustes de visualización
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Afegir valors a sobre de les barres
for index, value in enumerate(class_counts):
    plt.text(index, value, f'{value}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# 5. Guardar la figura
output_file = os.path.join(plot_dir, 'rf_class_distribution_bar_chart.png') # Nombre específico
plt.savefig(output_file)
print(f"✅ Gráfico de distribución guardado como a '{output_file}'")
plt.close() # Cerrar la figura para liberar memoria

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