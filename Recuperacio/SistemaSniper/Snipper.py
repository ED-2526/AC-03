import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. CARGA DE DATOS
print("--- CARGANDO ESTRATEGIA QUIRÚRGICA (CON GAP) ---")
try:
    df = pd.read_csv('gtzan_pro_features_final.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features1.csv')
    except:
        print("Error de carga. Revisa la ruta.")

# 2. SELECCIÓN DE CARACTERÍSTICAS (Tu Diagnóstico Maestro)
# Eliminamos el ruido (Skew/Kurt) para bajar el Gap y mejorar el SVM

# A. MFCC (Timbre) - Solo Mean y Var
mfcc_cols = [c for c in df.columns if 'mfcc' in c and ('mean' in c or 'var' in c)]
mfcc_cols = [c for c in mfcc_cols if 'skew' not in c and 'kurt' not in c]

# B. ESPECTRALES (Rock vs Metal) - Centroid, Rolloff, RMS (Solo Mean/Var)
spectral_cols = [c for c in df.columns if any(x in c for x in ['centroid', 'rolloff', 'rms']) and ('mean' in c or 'var' in c)]
spectral_cols = [c for c in spectral_cols if 'skew' not in c and 'kurt' not in c]

# C. ARMONÍA (Rock vs Country) - Mean
harmony_cols = [c for c in df.columns if any(x in c for x in ['chroma', 'tonnetz', 'zcr']) and 'mean' in c]

# D. DINÁMICA (Tu "Spectral Flux" - Usamos Onset)
dynamic_cols = [c for c in df.columns if 'onset' in c and ('mean' in c or 'var' in c)]
dynamic_cols = [c for c in dynamic_cols if 'skew' not in c and 'kurt' not in c]

# Lista Final Limpia
selected_features = mfcc_cols + spectral_cols + harmony_cols + dynamic_cols
print(f"Variables seleccionadas: {len(selected_features)} (Sin ruido de Skew/Kurt)")

X = df[selected_features]
y = df['label']
groups = df['song_id'] if 'song_id' in df.columns else np.arange(len(df))

# Codificación y Escalado
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Riguroso
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# 3. EL MODELO (SVM Radial)
print("\nEntrenando SVM...")
model = SVC(
    C=1,               
    kernel='rbf',       
    gamma='scale',      
    class_weight='balanced', # Ayuda al Rock
    random_state=42
)

model.fit(X_train, y_train)

# 4. CÁLCULO DEL GAP (LA PRUEBA DEL ALGODÓN)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
gap = train_acc - test_acc

print("\n" + "="*45)
print(f"   ACCURACY TRAIN: {train_acc:.4%}")
print(f"   ACCURACY TEST:  {test_acc:.4%}")
print(f"   GAP (Overfitting): {gap:.2%}")
print("="*45)

# Diagnóstico automático
if gap > 0.20:
    print(">> ALERTA: Overfitting alto. El modelo memoriza.")
    print(">> SUGERENCIA: Baja C a 1.0 o reduce más variables.")
elif gap < 0.05:
    print(">> ALERTA: Underfitting posible. ¿El Accuracy es bajo?")
else:
    print(">> ÉXITO: Gap saludable. El modelo generaliza bien.")

print("\n--- REPORTE DETALLADO POR GÉNERO ---")
print(classification_report(y_test, y_test_pred, target_names=classes))

# Matriz Visual
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_test_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='Purples')
plt.title(f"SVM Quirúrgico (Gap: {gap:.1%})")
plt.show()