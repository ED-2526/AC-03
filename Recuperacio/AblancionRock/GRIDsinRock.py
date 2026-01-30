import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Los 4 Jinetes
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Herramientas de optimización
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV # <--- LA CLAVE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. CARGA
print("--- CARGANDO PROYECTO (GRID SEARCH OPTIMIZATION) ---")
try:
    df = pd.read_csv('gtzan_pro_features_final.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features2.csv')
    except:
        print("❌ Error de carga.")

# 2. LIMPIEZA
df = df.dropna()
# ABLACIÓN ROCK
df = df[df['label'] != 'rock']

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

# ==============================================================================
# 4. FASE DE OPTIMIZACIÓN (GRID SEARCH)
# ==============================================================================
print("\n🔥 INICIANDO BÚSQUEDA DE HIPERPARÁMETROS (Esto tomará unos minutos)...")

# --- A. OPTIMIZANDO XGBOOST ---
print("   1/4 Optimizando XGBoost...")
xgb_base = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb_params = {
    'n_estimators': [250, 300, 350],      # Alrededor de 300
    'max_depth': [3, 4, 5],               # Alrededor de 4
    'learning_rate': [0.04, 0.05, 0.06],  # Alrededor de 0.05
    'gamma': [8, 10, 12]                  # Alrededor de 10
}
grid_xgb = GridSearchCV(xgb_base, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print(f"      -> Mejor XGB: {grid_xgb.best_params_}")

# --- B. OPTIMIZANDO SVM ---
print("   2/4 Optimizando SVM...")
svm_base = SVC(probability=True, random_state=42)
svm_params = {
    'C': [0.5, 0.8, 1.0, 1.2],           # Alrededor de 0.8
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}
grid_svm = GridSearchCV(svm_base, svm_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train, y_train)
best_svm = grid_svm.best_estimator_
print(f"      -> Mejor SVM: {grid_svm.best_params_}")

# --- C. OPTIMIZANDO LOGISTIC REGRESSION ---
print("   3/4 Optimizando Logistic Regression...")
lr_base = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, random_state=42)
lr_params = {
    'C': [3.0, 5.0, 7.0, 10.0]           # Alrededor de 5.0
}
grid_lr = GridSearchCV(lr_base, lr_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
print(f"      -> Mejor LR: {grid_lr.best_params_}")

# --- D. OPTIMIZANDO RANDOM FOREST ---
print("   4/4 Optimizando Random Forest...")
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_params = {
    'n_estimators': [250, 300, 350],
    'max_depth': [12, 15, 18],           # Alrededor de 15
    'min_samples_leaf': [12, 16, 20]     # Alrededor de 16
}
grid_rf = GridSearchCV(rf_base, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
print(f"      -> Mejor RF: {grid_rf.best_params_}")

# ==============================================================================
# 5. ENSAMBLAJE FINAL (CON LOS MEJORES MODELOS ENCONTRADOS)
# ==============================================================================
print("\n🏆 Entrenando 'Dream Team' Ensemble...")

ensemble = VotingClassifier(
    estimators=[
        ('xgb', best_xgb), 
        ('svm', best_svm), 
        ('lr', best_lr),
        ('rf', best_rf)
    ],
    voting='soft', 
    weights=[2, 1, 1, 1] 
)

ensemble.fit(X_train, y_train)

# --- AUDITORÍA INDIVIDUAL ---
print("\n" + "="*60)
print(f"{'MODELO (OPTIMIZADO)':<20} | {'TRAIN ACC':<10} | {'TEST ACC':<10} | {'GAP':<10}")
print("-" * 60)

for name, model in ensemble.named_estimators_.items():
    train_p = model.predict(X_train)
    test_p = model.predict(X_test)
    tr_acc = accuracy_score(y_train, train_p)
    ts_acc = accuracy_score(y_test, test_p)
    g = tr_acc - ts_acc
    print(f"{name.upper():<20} | {tr_acc:.2%}    | {ts_acc:.2%}    | {g:.2%}")
print("="*60)

# RESULTADOS GLOBALES
y_train_pred = ensemble.predict(X_train)
y_test_pred = ensemble.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc

print("\n" + "="*40)
print(f"   TRAIN ACCURACY (GLOBAL): {train_acc:.4%}")
print(f"   TEST ACCURACY (GLOBAL):  {test_acc:.4%} 🚀")
print(f"   GAP (Overfitting):       {gap:.2%}")
print("="*40)

# Gráfica
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_test_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title(f"Ensemble Optimizado (Test: {test_acc:.2%})")
plt.show()

print("\n--- REPORTE DETALLADO ---")
print(classification_report(y_test, y_test_pred, target_names=classes))