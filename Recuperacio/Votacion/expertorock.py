import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Herramientas
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. CARGA
print("--- CARGANDO PROYECTO 'TRINIDAD + EXPERTO ROCK' ---")
try:
    df = pd.read_csv('gtzan_pro_features_final.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features1.csv')
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

# Codificación Global
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

# --- PASO 4: LA TRINIDAD (TU MODELO FAVORITO) ---
print("\n1. Convocando a la Trinidad (Modelo General)...")

clf1 = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.1, gamma=10,
    subsample=0.6, colsample_bytree=0.8, reg_lambda=1.5,
    random_state=42, eval_metric='mlogloss'
)

clf2 = SVC(
    C=10, kernel='rbf', gamma='scale', probability=True,
    random_state=42
)

clf3 = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=2,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[('xgb', clf1), ('svm', clf2), ('rf', clf3)],
    voting='soft',
    weights=[2, 1, 1]
)

print("Entrenando Ensemble Principal...")
ensemble.fit(X_train, y_train)

# --- PASO 5: EL EXPERTO EN ROCK (Corrección de Etiquetas) ---
print("\n2. Entrenando al Especialista en Rock...")

# Definimos la zona conflictiva
CONFUSION_ZONE = ['rock', 'country', 'disco', 'blues']
confusion_indices = [np.where(classes == c)[0][0] for c in CONFUSION_ZONE if c in classes]
print(f"   Objetivo: Desempatar entre {CONFUSION_ZONE} (Indices globales: {confusion_indices})")

# Filtramos datos
mask_expert = np.isin(y_train, confusion_indices)
X_train_expert = X_train[mask_expert]
y_train_expert_global = y_train[mask_expert]

# --- AQUÍ ESTÁ EL ARREGLO ---
# Creamos un traductor local para que XGBoost vea 0, 1, 2, 3
le_expert = LabelEncoder()
y_train_expert_local = le_expert.fit_transform(y_train_expert_global)

print(f"   Clases locales para el experto: {le_expert.classes_} -> [0, 1, 2, 3]")

# Entrenamos el experto con etiquetas 0-3
rock_expert = XGBClassifier(
    n_estimators=200, 
    max_depth=5,           
    learning_rate=0.1, 
    gamma=2,               
    min_child_weight=1,
    random_state=42,
    eval_metric='mlogloss'
)
rock_expert.fit(X_train_expert, y_train_expert_local)

# --- PASO 6: LÓGICA DE CASCADA (INFERENCIA) ---
print("\n--- APLICANDO CORRECCIÓN DE EXPERTOS ---")


# 1. Predicción de la Trinidad
y_pred_final = ensemble.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_final)

# 2. Intervención del Experto
corrections = 0
for i in range(len(y_pred_final)):
    pred = y_pred_final[i]
    
    # Si la Trinidad predice algo en la zona de confusión...
    if pred in confusion_indices:
        sample = X_test[i].reshape(1, -1)
        
        # El experto nos da un número local (0, 1, 2 o 3)
        expert_pred_local = rock_expert.predict(sample)[0]
        
        # Traducimos de vuelta al ID global (ej. 9 para Rock)
        expert_pred_global = le_expert.inverse_transform([expert_pred_local])[0]
        
        # Si el experto cambia la opinión, lo anotamos
        if expert_pred_global != pred:
            y_pred_final[i] = expert_pred_global
            corrections += 1

# --- RESULTADOS ---
acc_final = accuracy_score(y_test, y_pred_final)
gap = accuracy_score(y_train, ensemble.predict(X_train)) - acc_final

print("\n" + "="*40)
print(f" ACCURACY BASE (Trinidad):   {acc_base:.4%}")
print(f" ACCURACY FINAL (+Experto):  {acc_final:.4%}")
print(f" CORRECCIONES REALIZADAS:    {corrections}")
print(f" GAP (Overfitting):          {gap:.2%}")
print("="*40)

# Matriz Visual
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred_final)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title(f"Trinidad + Experto Rock (Acc: {acc_final:.2%})")
plt.show()

print("\n--- REPORTE DETALLADO ---")
print(classification_report(y_test, y_pred_final, target_names=classes))