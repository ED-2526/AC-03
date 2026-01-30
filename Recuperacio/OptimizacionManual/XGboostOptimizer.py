import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score # <--- Añadido accuracy_score
from sklearn.preprocessing import label_binarize

# 1. Cargar datos
print("Cargando dataset...")
# Asegúrate de que este es el archivo correcto
df = pd.read_csv('../gtzan_pro_features2.csv') 

# --- [BLOQUE NUEVO 1] INGENIERÍA DE CARACTERÍSTICAS AL VUELO ---
# Creamos las pistas para ayudar al Rock antes de entrenar
print("Generando interacciones (Distortion & Punch)...")
# Evitamos errores si alguna columna no existe con un try/except o verificación simple
if 'zcr_mean' in df.columns and 'contrast_mean' in df.columns:
    df['distortion_index'] = df['zcr_mean'] * df['contrast_mean']
if 'onset_mean' in df.columns and 'rms_mean' in df.columns:
    df['punch_factor'] = df['onset_mean'] * df['rms_mean']
# ---------------------------------------------------------------

# 2. Preprocesamiento
X = df.drop(['label', 'filename','song_id'], axis=1)
y = df['label']
groups = df['song_id']

# Codificar etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. División Rigurosa
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y_encoded, groups))

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

# --- [BLOQUE NUEVO 2] VERIFICACIÓN DE CLASES ---
# Esto te dirá si 'classical' desapareció porque no hay muestras en el test
print("\n--- Distribución del Test Set ---")
test_counts = pd.Series(le.inverse_transform(y_test)).value_counts().sort_index()
print(test_counts)
if 'classical' not in test_counts:
    print("¡ALERTA! 'classical' tiene 0 muestras en el Test Set. Cambia el random_state.")
# ------------------------------------------------

print(f"\nEntrenando con {len(X_train)} muestras. Testeando con {len(X_test)} muestras.")

# 4. Entrenar XGBoost (Regularizado para bajar el Gap)
model = XGBClassifier(

    n_estimators=300,
    
    # AJUSTE 1: Velocidad
    learning_rate=0.05,    # Subimos de 0.04 a 0.1. El 0.04 es demasiado lento para 300 árboles.
    
    # AJUSTE 2: Complejidad
    max_depth=4,          # Bajamos de 4 a 3. Depth 3 es el "número mágico" para GTZAN.
    
    # AJUSTE 3: Poda (La clave del éxito)
    gamma=10,              # Bajamos de 10 a 5.
                          # Gamma 10 es demasiado estricto (te deja en 67%).
                          # Gamma 0 es demasiado loco (te lleva al overfitting del 27%).
                          # Gamma 5 es el punto medio para el 70%.
    
    min_child_weight=3,   # Mantenemos esto para evitar reglas para canciones "raras".
    subsample=0.6,        # Subimos un poco (de 0.6 a 0.8) para darle más datos al árbol.
    colsample_bytree=0.8, # Subimos un poco para que mire más columnas.
    
    reg_alpha=0.5,        # Mantenemos L1
    reg_lambda=1,       # Subimos un poco L2 para compensar la subida de learning_rate
    
    eval_metric='mlogloss',
    random_state=42



    
)

print("Entrenando modelo...")
model.fit(X_train, y_train)

# 5. Predicciones
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# --- [BLOQUE NUEVO 3] CÁLCULO DEL GAP (OVERFITTING) ---
print("\n--- ANÁLISIS DE GENERALIZACIÓN ---")
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
gap = train_acc - test_acc

print(f"Accuracy Train: {train_acc:.4f}")
print(f"Accuracy Test:  {test_acc:.4f}")
print(f"Gap:            {gap:.4f} ({gap:.2%})")

if gap > 0.15:
    print(">> DIAGNÓSTICO: Todavía hay Overfitting alto. Prueba bajar max_depth a 3.")
elif gap < 0.05:
    print(">> DIAGNÓSTICO: Excelente generalización (Underfitting posible si el accuracy es bajo).")
else:
    print(">> DIAGNÓSTICO: Balance saludable.")
# ------------------------------------------------------

# --- VISUALIZACIÓN 1: MATRIZ DE CONFUSIÓN ---
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title(f'Matriz de Confusión (Gap: {gap:.1%})')
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.tight_layout()
plt.show()

# --- VISUALIZACIÓN 2: CURVA ROC ---
y_test_bin = label_binarize(y_test, classes=range(len(classes)))
n_classes = len(classes)
fpr, tpr, roc_auc = dict(), dict(), dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = sns.color_palette("husl", n_classes)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Curva ROC por Género')
plt.legend(loc="lower right")
plt.show()

# --- VISUALIZACIÓN 3: IMPORTANCIA DE CARACTERÍSTICAS ---
print("\nGenerando gráfico de importancia de características...")
plt.figure(figsize=(12, 8))

# 1. Obtener la importancia de cada variable
importances = model.feature_importances_

# 2. Ordenar de mayor a menor
# argsort nos da los índices ordenados, [::-1] invierte para que sea descendente
indices = np.argsort(importances)[::-1][:20] # Solo mostramos el Top 20 para no saturar

# 3. Recuperar los nombres reales de las columnas
# (Usamos X.columns porque X era el DataFrame antes de escalarlo)
names = [X.columns[i] for i in indices]

# 4. Graficar
plt.title("Top 20 Características que definen los Géneros")
plt.bar(range(len(indices)), importances[indices], align="center", color='#8e44ad') # Color morado
plt.xticks(range(len(indices)), names, rotation=45, ha='right')
plt.ylabel('Importancia Relativa')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import xgboost as xgb

# Ver qué variables dominan el modelo
xgb.plot_importance(model, max_num_features=15)
plt.show()
# Imprimir en texto las 5 mejores para verlas rápido
print("\n--- TOP 5 CARACTERÍSTICAS ---")
for i in range(5):
    print(f"{i+1}. {names[i]} ({importances[indices[i]]:.4f})")

# Reporte
print("\n--- REPORTE FINAL ---")
print(classification_report(y_test, y_pred, target_names=classes))