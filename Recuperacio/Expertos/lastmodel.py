import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Los 3 Tenores (Modelos)
from sklearn.ensemble import RandomForestClassifier # Experto en Ritmo (Datos sucios/temporales)
from sklearn.svm import SVC                         # Experto en Armonía (Espacios vectoriales)
from xgboost import XGBClassifier                   # Experto en Timbre (Espectrogramas)
from sklearn.linear_model import LogisticRegression # El Jefe (Meta-modelo)

# Utilería
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. CARGA DE DATOS
print("--- FASE 1: PREPARACIÓN DE LA JUNTA ---")
try:
    # Intenta cargar el archivo con las nuevas features (tempo, chroma, tonnetz)
    df = pd.read_csv('gtzan_pro_features_final.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('../gtzan_pro_features1.csv')
    except:
        print("¡ERROR! No encuentro el archivo csv.")

# 2. INGENIERÍA DE CARACTERÍSTICAS (Damos armas a los expertos)
# Distortion -> Para el Experto en Timbre
if 'zcr_mean' in df.columns and 'contrast_mean' in df.columns:
    df['distortion_index'] = df['zcr_mean'] * df['contrast_mean']
else:
    df['distortion_index'] = 0

# Punch -> Para el Experto en Ritmo (CRÍTICO: Ahora tienes onset y rms reales)
if 'onset_mean' in df.columns and 'rms_mean' in df.columns:
    df['punch_factor'] = df['onset_mean'] * df['rms_mean']
else:
    df['punch_factor'] = 0

# Limpieza
cols_to_drop = ['label', 'filename', 'song_id']
# Si hubiera boaw lo quitamos, ensucia la especialización
cols_to_drop += [c for c in df.columns if c.startswith('boaw_')]

X = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
y = df['label']
groups = df['song_id'] if 'song_id' in df.columns else np.arange(len(df))

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_
cols = X.columns.tolist()

# 3. ASIGNACIÓN DE CARTERAS (¿Qué ve cada experto?)

# A. EXPERTO EN RITMO (Velocidad y Fuerza)
# Le damos: Tempo, Onset, RMS, Punch y ZCR (el cruce por cero ayuda al ritmo percusivo)
feat_rhythm = [c for c in cols if any(x in c for x in ['tempo', 'onset', 'rms', 'punch', 'zcr'])]

# B. EXPERTO EN ARMONÍA (Color y Tonalidad)
# Le damos: Chroma, Tonnetz y Contrast (Contrast es clave para textura armónica)
feat_harmony = [c for c in cols if any(x in c for x in ['chroma', 'tonnetz', 'contrast'])]

# C. EXPERTO EN TIMBRE (La voz del instrumento)
# Le damos: MFCCs, Centroid, Rolloff, Distortion
# Excluimos lo que ya tienen los otros para obligar a especializarse
used = set(feat_rhythm + feat_harmony)
feat_timbre = [c for c in cols if c not in used]
# Añadimos 'distortion' explícitamente si se quedó fuera
if 'distortion_index' in cols and 'distortion_index' not in feat_timbre:
    feat_timbre.append('distortion_index')

print(f"\nREPARTO DE VARIABLES:")
print(f"- Experto Ritmo (RF):   {len(feat_rhythm)} vars (Tempo, Punch, ZCR...)")
print(f"- Experto Armonía (SVM):{len(feat_harmony)} vars (Chroma, Tonnetz...)")
print(f"- Experto Timbre (XGB): {len(feat_timbre)} vars (MFCC, Rolloff...)")

# Validación
if len(feat_rhythm) == 0 or len(feat_harmony) == 0:
    raise ValueError("Faltan variables críticas (Tempo o Chroma). Revisa el CSV.")

# Escalado
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split Riguroso
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))
X_train_full, X_test_full = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# 4. ENTRENAMIENTO DE LOS 3 EXPERTOS
print("\n--- FASE 2: ENTRENANDO ESPECIALISTAS ---")

# Ritmo -> Random Forest (Subimos n_estimators y bajamos profundidad para evitar memorizar)
print("Entrenando Experto Ritmo (RF)...")
clf_rhythm = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=4, random_state=42)
clf_rhythm.fit(X_train_full[feat_rhythm], y_train)

# Armonía -> SVM (C=10 funciona bien con Chromas limpios)
print("Entrenando Experto Armonía (SVM)...")
clf_harmony = SVC(C=10, kernel='rbf', probability=True, random_state=42)
clf_harmony.fit(X_train_full[feat_harmony], y_train)

# Timbre -> XGBoost (Tu config ganadora depth=3, gamma=5)
print("Entrenando Experto Timbre (XGBoost)...")
clf_timbre = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, 
    gamma=5, subsample=0.8, colsample_bytree=0.8, 
    random_state=42, eval_metric='mlogloss'
)
clf_timbre.fit(X_train_full[feat_timbre], y_train)

# 5. EL JEFE (META-MODELO)
print("\n--- FASE 3: LA DECISIÓN FINAL (STACKING) ---")
print("Consultando a los expertos (Cross-Validation)...")

# Generamos predicciones 'limpias' para entrenar al jefe
meta_train_rhythm = cross_val_predict(clf_rhythm, X_train_full[feat_rhythm], y_train, cv=3, method='predict_proba')
meta_train_harmony = cross_val_predict(clf_harmony, X_train_full[feat_harmony], y_train, cv=3, method='predict_proba')
meta_train_timbre = cross_val_predict(clf_timbre, X_train_full[feat_timbre], y_train, cv=3, method='predict_proba')

X_meta_train = np.hstack([meta_train_rhythm, meta_train_harmony, meta_train_timbre])

# Generamos predicciones sobre el test real
meta_test_rhythm = clf_rhythm.predict_proba(X_test_full[feat_rhythm])
meta_test_harmony = clf_harmony.predict_proba(X_test_full[feat_harmony])
meta_test_timbre = clf_timbre.predict_proba(X_test_full[feat_timbre])

X_meta_test = np.hstack([meta_test_rhythm, meta_test_harmony, meta_test_timbre])

# El Jefe: Logistic Regression (Pondera a quién creer)
meta_model = LogisticRegression(max_iter=1000, C=0.5) # C=0.5 para regularizar un poco al jefe
meta_model.fit(X_meta_train, y_train)

# 6. RESULTADOS
y_pred = meta_model.predict(X_meta_test)
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*45)
print(f"   ACCURACY FINAL (EXPERTOS): {acc:.4%}   ")
print("="*45)

# Visualización: ¿Quién tuvo la razón?
weights = np.abs(meta_model.coef_)
imp_rhythm = np.mean(weights[:, 0:10])
imp_harmony = np.mean(weights[:, 10:20])
imp_timbre = np.mean(weights[:, 20:30])

plt.figure(figsize=(10, 6))
bars = plt.bar(['Ritmo (RF)', 'Armonía (SVM)', 'Timbre (XGB)'], 
        [imp_rhythm, imp_harmony, imp_timbre], 
        color=['#e67e22', '#9b59b6', '#2ecc71'])
plt.title("Pesos de la Arquitectura de Expertos")
plt.ylabel("Influencia en la decisión final")
plt.grid(axis='y', alpha=0.3)
plt.show()

print("\n--- REPORTE DETALLADO ---")
print(classification_report(y_test, y_pred, target_names=classes))