import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. CARGA (DATASET COMPLETO, CON ROCK)
print("--- CARGANDO ARQUITECTURA JERÁRQUICA ---")
try:
    df = pd.read_csv('gtzan_complete_features:_;.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features2.csv')
    except:
        print("Error de carga.")

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

# --- NIVEL 1: EL PORTERO (XGBoost General) ---
print("\n1. Entrenando Nivel 1 (XGBoost Global)...")
# Este modelo separa bien Metal, Classical, Reggae, Disco, Pop, Hiphop.
# Pero falla en el "Cluster de Guitarras" (Rock/Jazz/Blues/Country).
model_l1 = XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.1, gamma=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
model_l1.fit(X_train, y_train)

# --- NIVEL 2: EL SOMMELIER (SVM Especialista) ---
print("2. Entrenando Nivel 2 (SVM para Guitarras)...")

# Definimos el "Cluster Confuso"
CONFUSING_GROUP = ['rock', 'jazz', 'blues', 'country']
confusing_indices = [np.where(classes == c)[0][0] for c in CONFUSING_GROUP]
print(f"   Especializándose en: {CONFUSING_GROUP} (Indices: {confusing_indices})")

# Filtramos SOLO datos de entrenamiento de estos géneros
mask_train = np.isin(y_train, confusing_indices)
X_train_sub = X_train[mask_train]
y_train_sub = y_train[mask_train]

# SVM necesita etiquetas 0, 1, 2, 3... así que re-codificamos localmente
le_sub = LabelEncoder()
y_train_sub_encoded = le_sub.fit_transform(y_train_sub)

# Entrenamos el SVM
# Usamos C=10 y rbf porque necesitamos fronteras curvas complejas entre Jazz y Blues
model_l2 = SVC(C=10, kernel='rbf', probability=True, random_state=42)
model_l2.fit(X_train_sub, y_train_sub_encoded)

# --- INFERENCIA JERÁRQUICA ---
print("\n--- EJECUTANDO CASCADA ---")
y_pred_final = model_l1.predict(X_test) # Predicción inicial
intervenciones = 0

for i in range(len(y_pred_final)):
    pred = y_pred_final[i]
    
    # SI el Portero dice que es una de las clases confusas...
    if pred in confusing_indices:
        # ... Pasamos la muestra al SVM
        sample = X_test[i].reshape(1, -1)
        
        # Predicción del SVM (devuelve 0..3)
        sub_pred = model_l2.predict(sample)[0]
        
        # Traducimos de vuelta al ID global
        # le_sub.inverse_transform devuelve el ID global original (ej. 9 para Rock)
        real_pred = le_sub.inverse_transform([sub_pred])[0]
        
        if real_pred != pred:
            intervenciones += 1
        
        # Sobrescribimos
        y_pred_final[i] = real_pred

# --- RESULTADOS ---
acc_base = accuracy_score(y_test, model_l1.predict(X_test))
acc_hier = accuracy_score(y_test, y_pred_final)

print("\n" + "="*40)
print(f"ACCURACY NIVEL 1 (Solo XGB): {acc_base:.4%}")
print(f"ACCURACY JERÁRQUICO (+SVM):  {acc_hier:.4%}")
print(f"INTERVENCIONES DEL SVM:      {intervenciones}")
print("="*40)

# Matriz Enfocada (Solo las clases difíciles)
# Para ver si el SVM arregló el lío
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred_final)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Blues')
plt.title(f"Jerárquico: XGBoost -> SVM (Acc: {acc_hier:.2%})")
plt.show()

print("\n--- REPORTE FINAL ---")
print(classification_report(y_test, y_pred_final, target_names=classes))