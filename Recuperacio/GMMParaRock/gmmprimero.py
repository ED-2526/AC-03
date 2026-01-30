import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.mixture import GaussianMixture

# Herramientas
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. CARGA
print("--- CARGANDO PROYECTO 'TRINIDAD + GMM AUGMENTATION' ---")
try:
    df = pd.read_csv('gtzan_pro_features_final.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features2.csv')
    except:
        print("Error de carga.")

# 2. LIMPIEZA
df = df.dropna()

# 3. INGENIERÍA FÍSICA
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

# Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_encoded, groups))

# Separamos DataFrame para poder trabajar cómodamente
X_train_raw = X.iloc[train_idx].copy()
X_test_raw = X.iloc[test_idx].copy()
y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]

# --- PASO 4: LA MAGIA DEL GMM (FEATURE ENGINEERING) ---
print("\n1. Generando Features Probabilísticas (GMM)...")
# Vamos a añadir 10 columnas nuevas: "Probabilidad de ser Blues", "Probabilidad de ser Rock", etc.

# Matrices para guardar las nuevas features
gmm_train_feats = np.zeros((len(X_train_raw), len(classes)))
gmm_test_feats = np.zeros((len(X_test_raw), len(classes)))

# Entrenamos un GMM por cada género
for i, genre in enumerate(classes):
    # Cogemos solo las canciones de ese género en el train
    X_genre = X_train_raw[y_train == i]
    
    # Entrenamos GMM (diag es más robusto para muchas columnas)
    # Usamos n_components=1 para capturar el "centro" del género. 
    # Para Rock, al ser difuso, esto ayuda a encontrar su "centro de gravedad".
    gmm = GaussianMixture(n_components=1, covariance_type='diag', random_state=42)
    gmm.fit(X_genre)
    
    # Calculamos la "Log-Likelihood" (Puntuación de pertenencia)
    gmm_train_feats[:, i] = gmm.score_samples(X_train_raw)
    gmm_test_feats[:, i] = gmm.score_samples(X_test_raw)

print("   -> ¡10 Nuevas Variables Generadas!")

# Concatenamos las features físicas con las probabilísticas
scaler = StandardScaler()
# Escalamos las físicas
X_train_phys = scaler.fit_transform(X_train_raw)
X_test_phys = scaler.transform(X_test_raw)

# Escalamos las GMM (Importante para que el SVM no se vuelva loco)
scaler_gmm = StandardScaler()
X_train_gmm = scaler_gmm.fit_transform(gmm_train_feats)
X_test_gmm = scaler_gmm.transform(gmm_test_feats)

# Juntamos todo
X_train_final = np.hstack([X_train_phys, X_train_gmm])
X_test_final = np.hstack([X_test_phys, X_test_gmm])

print(f"   -> Dimensiones finales: {X_train_final.shape[1]} columnas (Física + GMM)")

# --- PASO 5: LA TRINIDAD PONDERADA ---
print("\n2. Entrenando la Trinidad con Super-Poderes...")

# A. XGBoost (Ahora ve probabilidades)
clf1 = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4, gamma=5, 
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8, 
    reg_alpha=0.5, reg_lambda=1, eval_metric='mlogloss', random_state=42
)

# B. SVM (Con ayuda para el Rock)
clf2 = SVC(
    C=10, kernel='rbf', gamma='scale', probability=True,
    class_weight='balanced', # <--- OBLIGATORIO para salvar el Rock
    random_state=42
)

# C. Random Forest (Robustez)
clf3 = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=2,
    class_weight='balanced', # <--- OBLIGATORIO
    random_state=42
)

# Ensemble
ensemble = VotingClassifier(
    estimators=[('xgb', clf1), ('svm', clf2), ('rf', clf3)],
    voting='soft', 
    weights=[1.5, 1, 1] # Bajamos un poco al XGBoost para que el SVM (que ama el Rock) tenga voz
)

ensemble.fit(X_train_final, y_train)

# --- RESULTADOS ---
y_pred = ensemble.predict(X_test_final)
test_acc = accuracy_score(y_test, y_pred)
gap = accuracy_score(y_train, ensemble.predict(X_train_final)) - test_acc

print("\n" + "="*40)
print(f" ACCURACY FINAL (GMM AUGMENTED): {test_acc:.4%} ")
print(f" GAP: {gap:.2%} ")
print("="*40)

# Matriz
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title(f"Trinidad + GMM Features (Acc: {test_acc:.2%})")
plt.show()

print("\n--- REPORTE DETALLADO ---")
print(classification_report(y_test, y_pred, target_names=classes))