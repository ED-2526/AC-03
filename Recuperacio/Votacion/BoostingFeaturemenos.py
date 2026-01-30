import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Los 4 Modelos
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Herramientas
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel # <--- LA HERRAMIENTA CLAVE

# 1. CARGA
print("--- CARGANDO PROYECTO (REDUCCIÓN DE DIMENSIONES) ---")

try:
    df = pd.read_csv('../gtzan_complete_features.csv')
except:
    print("❌ Error de carga.")

df = df.dropna()

# --- ELIMINAR ROCK (Mantenemos la ablación) ---
df = df[df['label'] != 'rock']

# 2. INGENIERÍA
if 'zcr_mean' in df.columns and 'contrast_mean' in df.columns:
    df['distortion_index'] = df['zcr_mean'] * df['contrast_mean']
else: df['distortion_index'] = 0

if 'onset_mean' in df.columns and 'rms_mean' in df.columns:
    df['punch_factor'] = df['onset_mean'] * df['rms_mean']
else: df['punch_factor'] = 0

cols_drop = [c for c in df.columns if c.startswith('boaw_')] + ['label', 'filename', 'song_id']
X = df.drop([c for c in cols_drop if c in df.columns], axis=1)
y = df['label']
groups = df['song_id'] if 'song_id' in df.columns else np.arange(len(df))

# Codificación y Escalado
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))
X_train_full, X_test_full = X_scaled[train_idx], X_scaled[test_idx] # Full features
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# --- PASO CRÍTICO: SELECCIÓN DE CARACTERÍSTICAS ---
print(f"\nDimensiones Originales: {X_train_full.shape[1]} columnas.")
print("Ejecutando Selección de Características (Eliminando ruido)...")

# Usamos un Random Forest rápido para juzgar qué columnas sirven
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold="1.25*mean" # <--- Solo las columnas mejores que la media (Agresivo)
)
selector.fit(X_train_full, y_train)

# Transformamos los datos (borramos columnas basura)
X_train = selector.transform(X_train_full)
X_test = selector.transform(X_test_full)

print(f"Dimensiones Reducidas:  {X_train.shape[1]} columnas. (Limpieza completada)")

# 4. ENTRENAMIENTO DEL SUPER-ENSEMBLE (Con datos limpios)
print("Entrenando Ensemble con Features VIP...")

clf1 = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, eval_metric='mlogloss', random_state=42)
clf2 = SVC(C=10, kernel='rbf', gamma='scale', probability=True, random_state=42)
# La Regresión Logística agradecerá MUCHO tener menos columnas
clf3 = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, C=5.0, random_state=42)
clf4 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)

ensemble = VotingClassifier(
    estimators=[('xgb', clf1), ('svm', clf2), ('lr', clf3), ('rf', clf4)],
    voting='soft', 
    weights=[2, 1, 1, 1] 
)

ensemble.fit(X_train, y_train)

# 5. RESULTADOS
y_train_pred = ensemble.predict(X_train)
y_test_pred = ensemble.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc

print("\n" + "="*40)
print(f"   TRAIN ACCURACY:      {train_acc:.4%}")
print(f"   TEST ACCURACY:       {test_acc:.4%} 🚀")
print(f"   GAP (Overfitting):   {gap:.2%}")
print("="*40)

# Matriz
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_test_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title(f"Modelo Reducido ({X_train.shape[1]} Features) - Acc: {test_acc:.2%}")
plt.show()

print(classification_report(y_test, y_test_pred, target_names=classes))