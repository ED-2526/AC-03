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