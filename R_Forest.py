import pandas as pd
import numpy as np
import os

# --- Lliberies de Machine Learning i Preprocesament (Scikit-learn) ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier # Nou model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. CARGA DE DADES (Igual) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(current_dir, 'Data', 'features_30_sec.csv')

if not os.path.exists(features_path):
    print(f"❌ Error: No encuentro el archivo en: {features_path}")
    print("Asegúrate de que la carpeta 'Data' está en el mismo lugar que este script.")
    exit()

df = pd.read_csv(features_path)
print(f"✅ Datos cargados correctamente desde: {features_path}")
print("Tamaño:", df.shape)

# --- 2. PREPROCESAMENT (Igual) ---

# Eliminar columnes
X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']

# Codificar les etiquetes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir en Train i Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Escalar els dades
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. ADAPTACIÓ PER A CLASSIFICADOR 2D (SIMPLIFICAT) ---

# Ja NO necessitem el reshape 3D ni el One-Hot Encoding (to_categorical)
# Les dades escalades X_train i X_test (que són 2D) ja són aptes per a Random Forest.
print(f"Forma de entrada para Random Forest: {X_train.shape}")
print(f"Número de clases: {len(label_encoder.classes_)}")
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------
# --- NOU BLOC DE CODI PER A LA GRÀFICA ---
# ----------------------------------------------------

print("\n--- 3. Generació del Gràfic de Distribució de Classes ---")

# Obtenim la variable objectiu abans del 'train_test_split'
y_labels = df['label']

# 1. Contar la freqüència de cada gènere
class_counts = y_labels.value_counts().sort_index()

# 2. Crear el gràfic de barres
plt.figure(figsize=(12, 6))
class_counts.plot(kind='bar', color='darkgreen')

# 3. Títols i etiquetes
plt.title('Distribució de Gèneres Musicals (Classes) per a l\'Anàlisi d\'Homogeneïtat', fontsize=14)
plt.xlabel('Gènere Musical', fontsize=12)
plt.ylabel('Nombre de Mostres', fontsize=12)

# 4. Ajustos de visualització
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Afegir valors a sobre de les barres
for index, value in enumerate(class_counts):
    plt.text(index, value, f'{value}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# 5. Guardar la figura
output_file = 'class_distribution_bar_chart.png'
plt.savefig(output_file)
print(f"✅ Gràfic de distribució guardat com a '{output_file}'")

# ----------------------------------------------------
# --- 4. DEFINICIÓ, ENTRENAMENT I AVALUACIÓ DEL MODEL ---

print("\n--- 4. ENTRENAMENT DE RANDOM FOREST ---")

# 4.1. Definició del Model
# Random Forest és un conjunt d'arbres de decisió (ensemble)
model = RandomForestClassifier(
    n_estimators=100,  # Número d'arbres
    random_state=42,
    n_jobs=-1          # Utilitzar tots els nuclis del processador
)

# 4.2. Entrenament
print("Entrenando el modelo...")
model.fit(X_train, y_train)
print("✅ Entrenamiento finalizado.")

# 4.3. Predicció
y_pred = model.predict(X_test)

# 4.4. Avaluació
print("\n--- 5. RESULTATS DE L'AVALUACIÓ ---")

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy*100:.2f}%")

# Report de classificació
print("\nReporte de Clasificación:")
# Utilitzem 'target_names' per mostrar els noms dels gèneres en lloc dels números codificats
report = classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_, 
    digits=3
)
print(report)

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))