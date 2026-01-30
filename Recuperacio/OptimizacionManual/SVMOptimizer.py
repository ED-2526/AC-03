import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

from sklearn.inspection import permutation_importance

# 1. Cargar datos
print("Cargando dataset...")
df = pd.read_csv('../gtzan_pro_features2.csv')
df = df.dropna()

# --- [BLOQUE NUEVO 1] INGENIERÍA DE CARACTERÍSTICAS AL VUELO ---
print("Generando interacciones (Distortion & Punch)...")
if 'zcr_mean' in df.columns and 'contrast_mean' in df.columns:
    df['distortion_index'] = df['zcr_mean'] * df['contrast_mean']
else:
    df['distortion_index'] = 0

if 'onset_mean' in df.columns and 'rms_mean' in df.columns:
    df['punch_factor'] = df['onset_mean'] * df['rms_mean']
else:
    df['punch_factor'] = 0
# ---------------------------------------------------------------

# 2. Preprocesamiento
X = df.drop(['label', 'filename', 'song_id'], axis=1, errors='ignore')
y = df['label']
groups = df['song_id'] if 'song_id' in df.columns else np.arange(len(df))

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. División Rigurosa
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# --- [BLOQUE NUEVO 2] VERIFICACIÓN DE CLASES ---
print("\n--- Distribución del Test Set ---")
test_counts = pd.Series(le.inverse_transform(y_test)).value_counts().sort_index()
print(test_counts)
if 'classical' not in test_counts:
    print("¡ALERTA! 'classical' tiene 0 muestras en el Test Set. Cambia el random_state.")
# ------------------------------------------------

print(f"\nEntrenando con {len(X_train)} muestras. Testeando con {len(X_test)} muestras.")

# 4. Entrenar SVM (RBF)
# probability=True es necesario para ROC y soft voting; aquí lo usamos por ROC.
model = SVC(
    C=0.7,
    kernel='rbf',
    gamma=0.01,
    probability=True,
    class_weight='balanced',  # opcional, pero suele ayudar si hay desbalance
    random_state=42
)

print("Entrenando modelo (SVM)...")
model.fit(X_train, y_train)

# 5. Predicciones
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# --- [BLOQUE NUEVO 3] CÁLCULO DEL GAP (OVERFITTING) ---
print("\n--- ANÁLISIS DE GENERALIZACIÓN ---")
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
gap = train_acc - test_acc

print(f"Accuracy Train: {train_acc:.4f}")
print(f"Accuracy Test:  {test_acc:.4f}")
print(f"Gap:            {gap:.4f} ({gap:.2%})")

if gap > 0.15:
    print(">> DIAGNÓSTICO: Overfitting alto. Prueba bajar C (0.5 o 0.2) o ajustar gamma.")
elif gap < 0.05:
    print(">> DIAGNÓSTICO: Excelente generalización (Underfitting posible si el accuracy es bajo).")
else:
    print(">> DIAGNÓSTICO: Balance saludable.")
# ------------------------------------------------------

# --- VISUALIZACIÓN 1: MATRIZ DE CONFUSIÓN ---
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title(f'Matriz de Confusión SVM (Gap: {gap:.1%})')
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.tight_layout()
plt.show()

# --- VISUALIZACIÓN 2: CURVA ROC (One-vs-Rest) ---
y_test_bin = label_binarize(y_test, classes=range(len(classes)))
n_classes = len(classes)

fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = sns.color_palette("husl", n_classes)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC {classes[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Curva ROC por Género (SVM)')
plt.legend(loc="lower right")
plt.show()

# --- VISUALIZACIÓN 3: "IMPORTANCIA" DE CARACTERÍSTICAS (Permutation Importance) ---
print("\nGenerando importancia de características (Permutation Importance)...")
# Ojo: esto puede tardar un poco si hay muchas features; n_repeats controla estabilidad/tiempo.
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

importances = perm.importances_mean
indices = np.argsort(importances)[::-1][:20]
names = [X.columns[i] for i in indices]

plt.figure(figsize=(12, 8))
plt.title("Top 20 Características (Permutation Importance) - SVM")
plt.bar(range(len(indices)), importances[indices], align="center")
plt.xticks(range(len(indices)), names, rotation=45, ha='right')
plt.ylabel('Caída media en Accuracy al permutar')
plt.tight_layout()
plt.show()

print("\n--- TOP 5 CARACTERÍSTICAS (Permutation) ---")
for i in range(5):
    print(f"{i+1}. {names[i]} ({importances[indices[i]]:.6f})")

# Reporte
print("\n--- REPORTE FINAL ---")
print(classification_report(y_test, y_pred, target_names=classes))
