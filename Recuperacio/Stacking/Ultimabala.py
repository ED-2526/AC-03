import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos Base
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# EL ARMA SECRETA
from sklearn.ensemble import StackingClassifier

# Herramientas
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. CARGA
print("--- LA ÚLTIMA BALA: STACKING GENERAL ---")
try:
    df = pd.read_csv('gtzan_pro_features_final.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features2.csv')
    except:
        print("❌ Error de carga.")

# 2. LIMPIEZA
df = df.dropna()
df = df[df['label'] != 'rock'] # Sin Rock

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

# Split (Usamos la semilla mágica 21)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=21)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))
X_train_raw, X_test_raw = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# 4. LDA AUGMENTATION
print("\n📐 Aplicando LDA Augmentation...")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_raw, y_train)
lda_train_feats = lda.transform(X_train_raw)
lda_test_feats = lda.transform(X_test_raw)

X_train = np.hstack((X_train_raw, lda_train_feats))
X_test = np.hstack((X_test_raw, lda_test_feats))

# 5. DEFINICIÓN DE LOS SOLDADOS (Level 0)
# Configuraciones "apretadas" pero no locas
estimators = [
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, gamma=1, random_state=42)),
    ('svm', SVC(C=50, kernel='rbf', gamma='scale', probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42)),
    # Nota: Quitamos la LR de aquí porque será el Jefe Final
]

# 6. EL JEFE FINAL (Level 1)
# Una Regresión Logística que toma las decisiones basándose en los otros
final_estimator = LogisticRegression(max_iter=2000, C=10)

print("\n🏗️ Construyendo Arquitectura Stacking (Esto tarda más)...")
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5, # Validación cruzada interna para entrenar al jefe
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)

# 7. RESULTADOS
y_train_pred = stacking_model.predict(X_train)
y_test_pred = stacking_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc

print("\n" + "="*40)
print(f"   TRAIN ACCURACY: {train_acc:.4%}")
print(f"   TEST ACCURACY:  {test_acc:.4%} ")
if test_acc > 0.80:
    print("   🔥 ¡BARRERA ROTA! 🔥")
else:
    print("   (Límite matemático alcanzado)")
print("   GAP:            {:.2%}".format(gap))
print("="*40)

# Gráfica
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_test_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title(f"Stacking Final (Acc: {test_acc:.2%})")
plt.show()