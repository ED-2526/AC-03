import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Herramientas
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# 1. CARGA Y PREPARACIÓN
print("--- FASE 1: PREPARACIÓN DE LA JUNTA DE EXPERTOS ---")
df = pd.read_csv('../gtzan_pro_features1.csv') 

# INGENIERÍA (Damos las armas a los expertos)
if 'zcr_mean' in df.columns and 'contrast_mean' in df.columns:
    df['distortion_index'] = df['zcr_mean'] * df['contrast_mean'] # Para el Experto en Timbre
if 'onset_mean' in df.columns and 'rms_mean' in df.columns:
    df['punch_factor'] = df['onset_mean'] * df['rms_mean']     # Para el Experto en Ritmo

# Eliminamos BoAW (Ruido)
boaw_cols = [c for c in df.columns if c.startswith('boaw_')]
df = df.drop(boaw_cols, axis=1)

# Limpieza básica
X = df.drop(['label', 'filename', 'song_id'], axis=1, errors='ignore')
y = df['label']
groups = df['song_id'] if 'song_id' in df.columns else np.arange(len(df))

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

# DEFINICIÓN DE DOMINIOS (¿Qué columnas ve cada experto?)
# Esto es crítico: Cada experto solo debe ver SU especialidad
cols = X.columns
feat_rhythm = [c for c in cols if any(x in c for x in ['onset', 'tempo', 'rms', 'punch_factor', 'beat'])]
feat_timbre = [c for c in cols if any(x in c for x in ['mfcc', 'centroid', 'rolloff', 'contrast', 'zcr', 'distortion'])]
feat_harmony = [c for c in cols if any(x in c for x in ['chroma', 'tonnetz'])]

print(f"Experto Ritmo: {len(feat_rhythm)} variables")
print(f"Experto Timbre: {len(feat_timbre)} variables")
print(f"Experto Armonía: {len(feat_harmony)} variables")

# Escalado global (SVM lo necesita sí o sí)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split Riguroso
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))

# Separamos datasets completos
X_train_full, X_test_full = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# 2. ENTRENAMIENTO DE LOS EXPERTOS
print("\n--- FASE 2: ENTRENANDO ESPECIALISTAS ---")

# EXPERTO 1: RITMO (Random Forest es genial para datos ruidosos/temporales)
# Usamos tu 'punch_factor' aquí
clf_rhythm = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf_rhythm.fit(X_train_full[feat_rhythm], y_train)
print("✅ Experto en Ritmo entrenado.")

# EXPERTO 2: TIMBRE (XGBoost es el rey de los espectrogramas)
# Usamos tu 'distortion_index' y la configuración híbrida (Depth 3, Gamma 5)
clf_timbre = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, 
    gamma=5, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
)
clf_timbre.fit(X_train_full[feat_timbre], y_train)
print("✅ Experto en Timbre entrenado.")

# EXPERTO 3: ARMONÍA (SVM Radial para separar tonalidades complejas)
clf_harmony = SVC(C=10, kernel='rbf', probability=True, random_state=42)
clf_harmony.fit(X_train_full[feat_harmony], y_train)
print("✅ Experto en Armonía entrenado.")

# 3. GENERACIÓN DE META-FEATURES (EL STACKING)
print("\n--- FASE 3: REUNIÓN DE LA JUNTA (META-MODELO) ---")
# Para entrenar al "Jefe", necesitamos ver qué opinan los expertos sobre datos que NO han memorizado.
# Usamos cross_val_predict para generar predicciones "limpias" sobre el training set.

print("Generando opiniones internas...")
meta_train_rhythm = cross_val_predict(clf_rhythm, X_train_full[feat_rhythm], y_train, cv=3, method='predict_proba')
meta_train_timbre = cross_val_predict(clf_timbre, X_train_full[feat_timbre], y_train, cv=3, method='predict_proba')
meta_train_harmony = cross_val_predict(clf_harmony, X_train_full[feat_harmony], y_train, cv=3, method='predict_proba')

# Juntamos las opiniones (30 columnas: 10 generos * 3 expertos)
X_meta_train = np.hstack([meta_train_rhythm, meta_train_timbre, meta_train_harmony])

# Generamos las opiniones para el TEST set (aquí sí usamos los modelos entrenados)
meta_test_rhythm = clf_rhythm.predict_proba(X_test_full[feat_rhythm])
meta_test_timbre = clf_timbre.predict_proba(X_test_full[feat_timbre])
meta_test_harmony = clf_harmony.predict_proba(X_test_full[feat_harmony])

X_meta_test = np.hstack([meta_test_rhythm, meta_test_timbre, meta_test_harmony])

# 4. ENTRENAMIENTO DEL META-MODELO (EL JEFE)
# Usamos Regresión Logística para ver a quién le da más peso (transparencia)
meta_model = LogisticRegression(max_iter=1000, C=1.0)
meta_model.fit(X_meta_train, y_train)

# 5. EVALUACIÓN FINAL
y_pred = meta_model.predict(X_meta_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f"   ACCURACY FINAL (ARQUITECTURA EXPERTOS): {acc:.4%}   ")
print("="*40)

# Gráfica: ¿Quién manda? (Pesos del Meta-Modelo)
# Promediamos los pesos absolutos para ver qué experto influye más
weights = np.abs(meta_model.coef_)
imp_rhythm = np.mean(weights[:, 0:10])
imp_timbre = np.mean(weights[:, 10:20])
imp_harmony = np.mean(weights[:, 20:30])

plt.figure(figsize=(10, 6))
plt.bar(['Ritmo (RF)', 'Timbre (XGB)', 'Armonía (SVM)'], [imp_rhythm, imp_timbre, imp_harmony], color=['orange', 'green', 'purple'])
plt.title("¿A qué experto escucha más el Jefe?")
plt.ylabel("Peso promedio en la decisión final")
plt.show()

print(classification_report(y_test, y_pred, target_names=classes))