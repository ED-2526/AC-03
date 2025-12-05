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
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import uniform, loguniform

# Librerías visualización
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 0. CREAR CARPETA DE GUARDADO DE PLOTS
# ============================================================
plot_dir = os.path.join(os.getcwd(), "Plots")
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

# ============================================================
# 3. FEATURE ENGINEERING PARA CLUSTERS SOLAPADOS
# ============================================================
print("\n🔧 Aplicando Feature Engineering...")

# 3.1 Selección de características más relevantes
selector = SelectKBest(f_classif, k=min(30, X_train.shape[1]))
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 3.2 Añadir componentes principales
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 3.3 Combinar features originales seleccionadas + PCA
X_train_combined = np.hstack([X_train_selected, X_train_pca])
X_test_combined = np.hstack([X_test_selected, X_test_pca])

# 3.4 Escalar datos combinados
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_combined)
X_test_final = scaler.transform(X_test_combined)

print(f"✔️ Features originales: {X_train.shape[1]}")
print(f"✔️ Features después de selección: {X_train_selected.shape[1]}")
print(f"✔️ Features finales (con PCA): {X_train_final.shape[1]}")

# ============================================================
# 4. SVM MEJORADO PARA CLUSTERS SOLAPADOS
# ============================================================
print("\n🔍 Buscando mejores hiperparámetros para SVM...")

param_dist_svm = {
    "C": loguniform(1e-2, 1e3),          # Rango ampliado
    "gamma": loguniform(1e-5, 1e1),      # Más flexibilidad
    "kernel": ["rbf", "poly", "sigmoid"], # Múltiples kernels
    "degree": [2, 3, 4],                 # Para kernel poly
    "coef0": uniform(0, 10),             # Para poly y sigmoid
    "class_weight": ['balanced', None]   # Balance de clases
}

svm = SVC(probability=True)  # probability=True para ensemble posterior

random_search_svm = RandomizedSearchCV(
    svm,
    param_distributions=param_dist_svm,
    n_iter=50,           # Más iteraciones
    cv=10,               # Más folds
    random_state=42,
    verbose=2,
    n_jobs=-1,
    scoring='accuracy'
)

random_search_svm.fit(X_train_final, y_train)
best_svm = random_search_svm.best_estimator_

print(f"\n✔️ Mejor modelo SVM: {best_svm}")
print(f"✔️ Mejor score CV: {random_search_svm.best_score_:.4f}")

# ============================================================
# 5. ENSEMBLE DE SVMs (Bagging)
# ============================================================
print("\n🎯 Creando Ensemble de SVMs para mejor robustez...")

ensemble_svm = BaggingClassifier(
    estimator=best_svm,
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

ensemble_svm.fit(X_train_final, y_train)

# ============================================================
# 6. EVALUACIÓN - SVM Individual
# ============================================================
print("\n" + "="*60)
print("📊 RESULTADOS SVM INDIVIDUAL (Optimizado)")
print("="*60)

y_pred_test_svm = best_svm.predict(X_test_final)
y_pred_train_svm = best_svm.predict(X_train_final)

test_acc_svm = accuracy_score(y_test, y_pred_test_svm)
train_acc_svm = accuracy_score(y_train, y_pred_train_svm)

print(f"Train Accuracy: {train_acc_svm:.4f}")
print(f"Test Accuracy:  {test_acc_svm:.4f}")

# ============================================================
# 7. EVALUACIÓN - Ensemble
# ============================================================
print("\n" + "="*60)
print("📊 RESULTADOS ENSEMBLE SVM")
print("="*60)

y_pred_test_ens = ensemble_svm.predict(X_test_final)
y_pred_train_ens = ensemble_svm.predict(X_train_final)

test_acc_ens = accuracy_score(y_test, y_pred_test_ens)
train_acc_ens = accuracy_score(y_train, y_pred_train_ens)

print(f"Train Accuracy: {train_acc_ens:.4f}")
print(f"Test Accuracy:  {test_acc_ens:.4f}")

# Usar el mejor modelo (ensemble o individual)
if test_acc_ens > test_acc_svm:
    print(f"\n✅ Ensemble es mejor (+{(test_acc_ens-test_acc_svm)*100:.2f}%)")
    best_model = ensemble_svm
    y_pred_final = y_pred_test_ens
    final_acc = test_acc_ens
else:
    print(f"\n✅ SVM individual es mejor")
    best_model = best_svm
    y_pred_final = y_pred_test_svm
    final_acc = test_acc_svm

# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================
class_names = label_encoder.classes_

print("\n📄 Classification Report (Mejor Modelo):")
print(classification_report(y_test, y_pred_final, target_names=class_names))

# ============================================================
# 9. MÉTRICAS GLOBALES
# ============================================================
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_final, average='weighted'
)

print("\n📐 Métricas Globales:")
print(f"Accuracy:  {final_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# ============================================================
# 10. PER-CLASS METRICS
# ============================================================
p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
    y_test, y_pred_final
)

plt.figure(figsize=(14, 6))
x = np.arange(len(class_names))
width = 0.25

plt.bar(x - width, p_per_class, width, label='Precision', alpha=0.8)
plt.bar(x, r_per_class, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1_per_class, width, label='F1-score', alpha=0.8)

plt.xticks(x, class_names, rotation=45, ha='right')
plt.ylabel("Score")
plt.title("Per-Class Metrics (SVM Mejorado para Clusters Solapados)")
plt.legend()
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "per_class_metrics_svm_improved.png"), dpi=300)
plt.close()

# ============================================================
# 11. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Purples", cbar_kws={'label': 'Count'})

plt.xlabel("Predicho", fontsize=12)
plt.ylabel("Real", fontsize=12)
plt.title("Matriz de Confusión (SVM Mejorado)", fontsize=14)
plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "confusion_matrix_svm_improved.png"), dpi=300)
plt.close()

# ============================================================
# 12. COMPARACIÓN DE MODELOS
# ============================================================
models_comparison = pd.DataFrame({
    'Modelo': ['SVM Individual', 'Ensemble SVM'],
    'Train Acc': [train_acc_svm, train_acc_ens],
    'Test Acc': [test_acc_svm, test_acc_ens],
    'Diferencia': [
        abs(train_acc_svm - test_acc_svm),
        abs(train_acc_ens - test_acc_ens)
    ]
})

print("\n📊 Comparación de Modelos:")
print(models_comparison.to_string(index=False))

# ============================================================
# 13. ANÁLISIS DE FEATURES MÁS IMPORTANTES
# ============================================================
feature_scores = selector.scores_
feature_names = X.columns
top_features_idx = np.argsort(feature_scores)[-15:][::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features_idx)), feature_scores[top_features_idx])
plt.yticks(range(len(top_features_idx)), 
           [feature_names[i] for i in top_features_idx])
plt.xlabel("F-score")
plt.title("Top 15 Features Más Importantes")
plt.tight_layout()

plt.savefig(os.path.join(plot_dir, "top_features.png"), dpi=300)
plt.close()

# ============================================================
# 14. PREDICCIONES EJEMPLO
# ============================================================
print("\n🧪 Ejemplos de predicciones (primeras 15):")
print("-" * 50)
correct = 0
for i in range(min(15, len(y_test))):
    true_genre = label_encoder.inverse_transform([y_test[i]])[0]
    pred_genre = label_encoder.inverse_transform([y_pred_final[i]])[0]
    status = "✓" if true_genre == pred_genre else "✗"
    print(f"{status} Real: {true_genre:15} | Predicho: {pred_genre}")
    if true_genre == pred_genre:
        correct += 1

print("-" * 50)
print(f"Aciertos en muestra: {correct}/15 ({correct/15*100:.1f}%)")

print(f"\n✅ Todos los gráficos guardados en: {plot_dir}")