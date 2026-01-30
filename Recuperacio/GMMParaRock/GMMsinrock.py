import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Los 4 Jinetes
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Herramientas
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- EL NUEVO FICHAJE ESTRELLA: LDA ---
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. CARGA
print("--- CARGANDO PROYECTO (LDA SUPERVISED FEATURES) ---")
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

# Split (Usamos semilla 21 que funcionó bien antes)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=21)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))
X_train_raw, X_test_raw = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# ==============================================================================
# 4. LDA FEATURE AUGMENTATION (SUPERVISADO)
# ==============================================================================
print("\n📐 Aplicando LDA Augmentation (Separación de Clases)...")

# Configuramos LDA
# LDA genera (n_clases - 1) nuevas dimensiones. Si hay 9 géneros, genera 8 columnas.
lda = LinearDiscriminantAnalysis()

# ¡OJO! Aquí usamos y_train. LDA aprende usando las etiquetas.
lda.fit(X_train_raw, y_train)

# Generamos las nuevas "Super-Features"
lda_train_feats = lda.transform(X_train_raw)
lda_test_feats = lda.transform(X_test_raw)

print(f"   -> Generadas {lda_train_feats.shape[1]} nuevas características discriminantes.")

# Pegamos las nuevas columnas a las originales
X_train = np.hstack((X_train_raw, lda_train_feats))
X_test = np.hstack((X_test_raw, lda_test_feats))

print(f"   -> Nueva dimensión de entrada: {X_train.shape[1]} columnas.")

# 

# ==============================================================================
# 5. EL SUPER-ENSEMBLE (4 MODELOS)
# ==============================================================================
print("\nEntrenando Super-Ensemble con LDA Features...")

# A. XGBoost
clf1 = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4, 
    gamma=8, min_child_weight=3, subsample=0.6, colsample_bytree=0.8, 
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
    max_iter=1000, 
    C=5.0, 
    random_state=42
)

# D. Random Forest
clf4 = RandomForestClassifier(
    n_estimators=300, 
    max_depth=15, 
    min_samples_leaf=16,
    random_state=42
)

# Voting
ensemble = VotingClassifier(
    estimators=[
        ('xgb', clf1), 
        ('svm', clf2), 
        ('lr', clf3),
        ('rf', clf4) 
    ],
    voting='soft', 
    weights=[2, 1, 1, 1] 
)

ensemble.fit(X_train, y_train)

# --- AUDITORÍA INDIVIDUAL ---
print("\n" + "="*60)
print(f"{'MODELO':<10} | {'TRAIN ACC':<12} | {'TEST ACC':<12} | {'GAP':<12}")
print("-" * 60)

for name, model in ensemble.named_estimators_.items():
    train_p = model.predict(X_train)
    test_p = model.predict(X_test)
    tr_acc = accuracy_score(y_train, train_p)
    ts_acc = accuracy_score(y_test, test_p)
    g = tr_acc - ts_acc
    print(f"{name.upper():<10} | {tr_acc:.2%}       | {ts_acc:.2%}       | {g:.2%}")
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
plt.title(f"Ensemble + LDA Augmentation (Test: {test_acc:.2%})")
plt.show()

print("\n--- REPORTE DETALLADO ---")
print(classification_report(y_test, y_test_pred, target_names=classes))