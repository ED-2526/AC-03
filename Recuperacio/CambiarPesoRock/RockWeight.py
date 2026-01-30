import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Los 3 Grandes
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# IMPORTANTE: Necesitamos esto para calcular los pesos
from sklearn.utils.class_weight import compute_sample_weight

# 1. CARGA
print("--- CARGANDO ESTRATEGIA PONDERADA ---")
try:
    df = pd.read_csv('gtzan_complete_featuresb.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features2.csv')
    except:
        print("Error de carga.")

df = df.dropna()

# 3. INGENIERÍA
if 'zcr_mean' in df.columns and 'contrast_mean' in df.columns:
    df['distortion_index'] = df['zcr_mean'] * df['contrast_mean']
else: df['distortion_index'] = 0

if 'onset_mean' in df.columns and 'rms_mean' in df.columns:
    df['punch_factor'] = df['onset_mean'] * df['rms_mean']
else: df['punch_factor'] = 0

# Limpieza
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

# --- 4. DEFINICIÓN DE LOS 3 MAESTROS ---
print("\nConvocando a los modelos...")

# A. XGBoost
clf1 = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,    
    max_depth=4,          
    gamma=10,              
    min_child_weight=3,   
    subsample=0.6,        
    colsample_bytree=0.8, 
    reg_alpha=0.5,        
    reg_lambda=1,      
    eval_metric='mlogloss',
    random_state=42
)

# B. SVM (Añadido class_weight='balanced' como refuerzo interno)
clf2 = SVC(
    C=1, kernel='rbf', gamma='scale', probability=True,
    class_weight='balanced', # <--- Refuerzo extra
    random_state=42
)

# C. Random Forest (Añadido class_weight='balanced')
clf3 = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=2,
    class_weight='balanced', # <--- Refuerzo extra
    random_state=42
)

# 5. EL ENSAMBLAJE
print("Creando el 'Voting Ensemble'...")
ensemble = VotingClassifier(
    estimators=[('xgb', clf1), ('svm', clf2), ('rf', clf3)],
    voting='soft', 
    weights=[2, 1, 1] 
)

# --- AQUI ESTÁ LA MAGIA DE LA PONDERACIÓN ---
print("Calculando 'Esteroides' para el Rock...")

# 1. Calculamos pesos base (para equilibrar todos los géneros)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 2. Identificamos el ID del Rock
try:
    rock_id = le.transform(['rock'])[0]
    print(f"   -> Rock detectado como clase {rock_id}")
    
    # 3. Multiplicador de castigo (x3.0)
    # Esto le dice al modelo: "Una canción de Rock vale por 3 de las normales"
    # Si fallas un Rock, el error se multiplica por 3.
    sample_weights[y_train == rock_id] *= 3.0 
    print("   -> Aplicado Multiplicador x3.0 al Rock")
except:
    print("⚠️ No se encontró la clase 'rock'. Usando pesos balanceados normales.")

# 6. ENTRENAMIENTO (Pasando los pesos)
print("Entrenando a la Trinidad Ponderada...")
# El parámetro sample_weight se pasa a los 3 modelos internos
ensemble.fit(X_train, y_train, sample_weight=sample_weights)

# 7. RESULTADOS
y_pred = ensemble.predict(X_test)
train_acc = accuracy_score(y_train, ensemble.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
gap = train_acc - test_acc

print("\n" + "="*40)
print(f"   ACCURACY (TRINIDAD PONDERADA): {test_acc:.4%}   ")
print(f"   GAP: {gap:.2%}                  ")
print("="*40)

# Visualización
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Greens')
plt.title(f"Ensemble con Rock Boosting (Acc: {test_acc:.2%})")
plt.show()

print("\n--- REPORTE DETALLADO ---")
print(classification_report(y_test, y_pred, target_names=classes))