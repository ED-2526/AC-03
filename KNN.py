import pandas as pd
import numpy as np
import os

# Librerías ML
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint

# Librerías visualización
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 0. CREAR CARPETA DE GUARDADO DE PLOTS
# ============================================================
plot_dir = os.path.join(os.getcwd(), "Plots_KNN")
os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(current_dir, 'Data', 'features_3_sec.csv')

if not os.path.exists(features_path):
    print(f"❌ Error: No encuentro el archivo en: {features_path}")
    exit()

df = pd.read_csv(features_path)
print(f"✅ Datos cargados correctamente desde: {features_path}")
print("Tamaño:", df.shape)

# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================
X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 3. KNN
# ============================================================
print("\n🔍 Buscando mejores hiperparámetros para KNN...")

param_grid = {
    'n_neighbors': randint(1, 15),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

knn = KNeighborsClassifier()

random_search_knn = RandomizedSearchCV(
    knn,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    random_state=42
)

random_search_knn.fit(X_train, y_train)
best_knn = random_search_knn.best_estimator_

print(f"✔️ Mejor modelo encontrado: {best_knn}")

# ============================================================
# 4. EVALUACIÓN
# ============================================================
y_pred_test = best_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

y_pred_train = best_knn.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)

print("\n📊 RESULTADOS KNN")
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================
class_names = label_encoder.classes_

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=class_names))

# ============================================================
# 6. METRICS GLOBALS
# ============================================================
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_test, average='weighted'
)

print("\n📐 Metricas Globales:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ============================================================
# 7. PER-CLASS METRICS (precision, recall, f1)
# ============================================================
p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
    y_test, y_pred_test
)

plt.figure(figsize=(12, 6))
x = np.arange(len(class_names))
width = 0.25

plt.bar(x - width, p_per_class, width, label='Precision')
plt.bar(x, r_per_class, width, label='Recall')
plt.bar(x + width, f1_per_class, width, label='F1-score')

plt.xticks(x, class_names, rotation=45)
plt.ylabel("Score")
plt.title("Per-Class Metrics (KNN)")
plt.legend()
plt.tight_layout()

# 👉 Guardar gráfico
plt.savefig(os.path.join(plot_dir, "per_class_metrics_knn.png"))
plt.close()

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")

plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión (KNN)")
plt.tight_layout()

# 👉 Guardar gráfico
plt.savefig(os.path.join(plot_dir, "confusion_matrix_knn.png"))
plt.close()

# ============================================================
# 9. PREDICCIONES EJEMPLO
# ============================================================
print("\n🧪 Ejemplos de predicciones:")
for i in range(10):
    true_genre = label_encoder.inverse_transform([y_test[i]])[0]
    pred_genre = label_encoder.inverse_transform([y_pred_test[i]])[0]
    print(f"Real: {true_genre} | Predicho: {pred_genre}")