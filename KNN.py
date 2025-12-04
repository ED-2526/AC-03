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

# Modulo de visualización
from Plots import *

#import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- IMPORTAR TU NUEVO MÓDULO ---
from carrega_dades import *

MODEL = "KNN"

# 1. CARGA Y PREPROCESAMIENTO
try:
    X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_3s() # Per a 3 segons
    #X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_30s() # Per a 30 segons
    class_names = label_encoder.classes_
except Exception as e:
    print(e)
    exit()


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

try:
    y_prob_test = best_knn.predict_proba(X_test)
except AttributeError:
    y_prob_test = None
    print("El modelo no soporta predict_proba(). No se podrán generar curvas ROC/PR.")

y_pred_train = best_knn.predict(X_train)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

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

plot_per_class_metrics(y_test, y_pred_test, class_names, MODEL)

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

plot_confusion_matrix(y_test, y_pred_test, class_names, MODEL)

# ============================================================
# 9. CURVAS ROC
# ============================================================

if y_prob_test is not None:
    plot_roc_curve(y_test, y_prob_test, MODEL, class_names)

# ============================================================
# 10. CURVAS PRECISION-RECALL
# ============================================================

if y_prob_test is not None:
    plot_precision_recall_curve(y_test, y_prob_test, MODEL, class_names)

