import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.mixture import GaussianMixture  # <--- EL NUEVO FICHAJE

# Herramientas
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. CARGA
print("--- CARGANDO PROYECTO 'TRINIDAD + GMM SNIPERS' ---")
try:
    df = pd.read_csv('gtzan_pro_features_final.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features2.csv')
    except:
        print("Error de carga.")

# 2. LIMPIEZA
df = df.dropna()

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

# --- PASO 4: LA TRINIDAD (TU MODELO BASE) ---
print("\n1. Entrenando la Trinidad (Generalista)...")

clf1 = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4, gamma=10, 
    min_child_weight=3, subsample=0.6, colsample_bytree=0.8, 
    reg_alpha=0.5, reg_lambda=1, eval_metric='mlogloss', random_state=42
)
clf2 = SVC(C=10, kernel='rbf', gamma='scale', probability=True, random_state=42)
clf3 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42)

ensemble = VotingClassifier(
    estimators=[('xgb', clf1), ('svm', clf2), ('rf', clf3)],
    voting='soft', weights=[2, 1, 1]
)
ensemble.fit(X_train, y_train)

# Predicción inicial
y_pred_base = ensemble.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
print(f"   Accuracy Base: {acc_base:.4%}")

# --- PASO 5: DETECCIÓN Y ENTRENAMIENTO DE SNIPERS GMM ---
print("\n2. Desplegando Snipers Probabilísticos (GMM)...")


# Análisis de errores
cm = confusion_matrix(y_test, y_pred_base)
np.fill_diagonal(cm, 0)

conflicts = []
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        errores = cm[i, j] + cm[j, i]
        if errores > 4: 
            conflicts.append((classes[i], classes[j], errores))

conflicts.sort(key=lambda x: x[2], reverse=True)
top_conflicts = conflicts[:3] 

snipers = {}
print(f"   Conflictos detectados: {[f'{c[0]} vs {c[1]}' for c in top_conflicts]}")

for c1, c2, err in top_conflicts:
    print(f"   -> Entrenando GMM Dual para {c1} vs {c2}...")
    
    idx1 = np.where(classes == c1)[0][0]
    idx2 = np.where(classes == c2)[0][0]
    
    # Filtramos datos
    target_pair = [idx1, idx2]
    mask = np.isin(y_train, target_pair)
    X_pair = X_train[mask]
    y_pair = y_train[mask]
    
    # --- ESTRATEGIA GMM DUAL ---
    # Entrenamos un GMM para CADA clase por separado.
    # Aprendemos la "forma" de la clase 1 y la "forma" de la clase 2.
    
    # Datos solo de Clase 1
    X_c1 = X_pair[y_pair == idx1]
    gmm1 = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
    gmm1.fit(X_c1)
    
    # Datos solo de Clase 2
    X_c2 = X_pair[y_pair == idx2]
    gmm2 = GaussianMixture(n_components=2, covariance_type='diag', random_state=42)
    gmm2.fit(X_c2)
    
    # Guardamos los dos modelos generativos
    snipers[tuple(sorted((idx1, idx2)))] = {
        'gmm_0': gmm1, # Modelo para el primer índice
        'gmm_1': gmm2, # Modelo para el segundo índice
        'idx_0': idx1,
        'idx_1': idx2
    }

# --- PASO 6: INFERENCIA BAYESIANA ---
print("\n--- APLICANDO CORRECCIONES PROBABILÍSTICAS ---")
y_pred_final = y_pred_base.copy()
corrections = 0

for i in range(len(y_pred_final)):
    pred = y_pred_final[i]
    
    for (idx_a, idx_b), sniper_data in snipers.items():
        
        if pred == idx_a or pred == idx_b:
            sample = X_test[i].reshape(1, -1)
            
            # Calculamos la Log-Likelihood (¿Qué tanto encaja la canción en cada distribución?)
            score_a = sniper_data['gmm_0'].score_samples(sample)[0] # Score para idx_a
            score_b = sniper_data['gmm_1'].score_samples(sample)[0] # Score para idx_b
            
            # El que tenga mayor score gana (Máxima Verosimilitud)
            verdict_real = idx_a if score_a > score_b else idx_b
            
            if verdict_real != pred:
                y_pred_final[i] = verdict_real
                corrections += 1
                break 

# --- RESULTADOS ---
acc_final = accuracy_score(y_test, y_pred_final)

print("\n" + "="*40)
print(f" ACCURACY BASE:        {acc_base:.4%}")
print(f" ACCURACY FINAL (GMM): {acc_final:.4%}")
print(f" CORRECCIONES HECHAS:  {corrections}")
print("="*40)

# Matriz Visual
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred_final)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title(f"Trinidad + GMM Snipers (Acc: {acc_final:.2%})")
plt.show()

print("\n--- REPORTE FINAL ---")
print(classification_report(y_test, y_pred_final, target_names=classes))