import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Los 4 Jinetes del Apocalipsis
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Herramientas
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. CARGA
print("--- CARGANDO PROYECTO (ABLACIÓN DE ROCK + 4 MODELOS) ---")
try:
    df = pd.read_csv('gtzan_pro_features_final.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features2.csv')
    except:
        print("❌ Error de carga.")

# 2. LIMPIEZA Y FILTRADO
df = df.dropna()

# --- ELIMINAR ROCK ---
print(f"   Tamaño original: {len(df)}")
df = df[df['label'] != 'rock']
print(f"   Tamaño sin Rock: {len(df)} (Clase eliminada)")

# 3. INGENIERÍA
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

# Codificación
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# 4. EL SUPER-ENSEMBLE (4 MODELOS)
print("\nEntrenando Super-Ensemble de 9 Clases...")

# A. XGBoost
clf1 = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4, 
    gamma=10, min_child_weight=3, subsample=0.6, colsample_bytree=0.8, 
    eval_metric='mlogloss', random_state=42
)

# B. SVM
clf2 = SVC(
    C=0.8, kernel='rbf', gamma='scale', probability=True,
    random_state=42
)

# C. Logistic Regression
clf3 = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000, # Aumentado para asegurar convergencia
    C=5.0, 
    random_state=42
)

# D. Random Forest (Vuelve al equipo)
clf4 = RandomForestClassifier(
    n_estimators=300, 
    max_depth=15, 
    min_samples_leaf=16,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[
        ('xgb', clf1), 
        ('svm', clf2), 
        ('lr', clf3),
        ('rf', clf4) # <--- ¡Bienvenido de nuevo!
    ],
    voting='soft', 
    # Ajuste de pesos: Le damos un poco más de voz a XGBoost y RF
    weights=[2, 1, 1, 1] 
)

ensemble.fit(X_train, y_train)

# 5. CÁLCULO DE RESULTADOS Y GAP
print("\nCalculando métricas...")
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

# Gráfica
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_test_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title(f"Modelo SIN Rock (4 Modelos) - Gap: {gap:.2%}")
plt.show()

print("\n--- REPORTE DETALLADO ---")
print(classification_report(y_test, y_test_pred, target_names=classes))