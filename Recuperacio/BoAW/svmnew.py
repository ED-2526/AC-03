import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.impute import SimpleImputer # <--- EL SALVAVIDAS PARA SVM

# 1. CARGAR DATOS
print("Cargando dataset con BoAW...")
# Ajusta la ruta si es necesario
try:
    df = pd.read_csv('../gtzan_complete_features.csv')
except FileNotFoundError:
    df = pd.read_csv('gtzan_complete_features.csv')

# --- INGENIERÍA DE CARACTERÍSTICAS (AYUDA AL SVM A SEPARAR CLASES) ---
print("Generando interacciones físicas (Distortion & Punch)...")
# Verificamos existencia para evitar errores
if 'zcr_mean' in df.columns and 'contrast_mean' in df.columns:
    df['distortion_index'] = df['zcr_mean'] * df['contrast_mean']
if 'onset_mean' in df.columns and 'rms_mean' in df.columns:
    df['punch_factor'] = df['onset_mean'] * df['rms_mean']
# ---------------------------------------------------------------------

# 2. PREPROCESAMIENTO ROBUSTO
# Definimos columnas a borrar (Metadatos)
cols_to_drop = ['label', 'filename', 'song_id']
# Solo borramos las que existen para evitar KeyError
X = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
y = df['label']

# Gestión de grupos para el split (evita data leakage)
if 'song_id' in df.columns:
    groups = df['song_id']
else:
    print("Aviso: 'song_id' no encontrado. Usando índices secuenciales.")
    groups = np.arange(len(df))

# Codificar etiquetas (Blues -> 0, Classical -> 1...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

# --- LIMPIEZA Y ESCALADO (CRÍTICO PARA SVM) ---
print(f"Limpiando {X.shape[1]} columnas (rellenando NaNs)...")

# A. Imputer: Rellena huecos vacíos con el promedio
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# B. Scaler: SVM necesita que todos los datos estén entre -1 y 1 aprox.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
# ----------------------------------------------

# 3. DIVISIÓN RIGUROSA (Train/Test Split)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# 4. ENTRENAMIENTO SVM
print(f"\nEntrenando SVM con {X_train.shape[1]} features...")
print(">> Ten paciencia: El SVM tarda más en calcular que el XGBoost...")

model = SVC(
    C=10,               # Penalización alta: fuerza al modelo a aprender bien los detalles
    kernel='rbf',       # Kernel Radial (Estándar para audio)
    gamma='scale',      # Ajuste automático de la curvatura
    probability=True,   # Para poder calcular probabilidades si fuera necesario
    random_state=42
)

model.fit(X_train, y_train)

# 5. EVALUACIÓN Y DIAGNÓSTICO
print("\nEvaluando modelo...")
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)
gap = train_acc - test_acc

print("\n--- ANÁLISIS DE GENERALIZACIÓN (SVM) ---")
print(f"Accuracy Train: {train_acc:.4f}")
print(f"Accuracy Test:  {test_acc:.4f}")
print(f"Gap:            {gap:.2%} (Diferencia Train vs Test)")

if test_acc > 0.70:
    print(">> VEREDICTO: ¡ÉXITO! El SVM maneja mejor el BoAW que los árboles.")
elif test_acc > 0.66:
    print(">> VEREDICTO: Mejor que XGBoost con BoAW, pero sin llegar al 70%.")
else:
    print(">> VEREDICTO: El BoAW ensucia demasiado, incluso para SVM.")

# Matriz de Confusión
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Purples',
            xticklabels=classes, yticklabels=classes)
plt.title(f'SVM Radial + BoAW + Engineering (Test Acc: {test_acc:.2%})')
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.tight_layout()
plt.show()

print("\n--- REPORTE FINAL ---")
print(classification_report(y_test, y_pred, target_names=classes))