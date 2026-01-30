import pandas as pd
import numpy as np
import os

# Librerías ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# XGBoost
import xgboost as xgb

# ============================================================
# 0. CREAR CARPETA DE GUARDADO DE PLOTS
# ============================================================
plot_dir = os.path.join(os.getcwd(), "Plots")
os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(current_dir, 'Data', 'features_3_sec.csv')

if not os.path.exists(features_path):
    print(f"❌ Error: No encuentro el archivo en: {features_path}")
    exit()

df = pd.read_csv(features_path)
print(f"✅ Datos cargados correctamente desde: {features_path}")
print("Tamaño:", df.shape)

# ============================================================
# 2. PREPROCESAMIENTO
# ============================================================
X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 3. ENTRENAR XGBOOST
# ============================================================
model = xgb.XGBClassifier(
    objective='multi:softmax',   # para multiclass
    num_class=len(np.unique(y_encoded)),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ============================================================
# 4. EVALUACIÓN
# ============================================================
y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n✔️ Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ============================================================
# 5. IMPORTANCIA DE VARIABLES
# ============================================================
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=importances.head(20).values, y=importances.head(20).index, palette="viridis")
plt.title("Top 20 variables más importantes para XGBoost", fontsize=16)
plt.xlabel("Importancia")
plt.ylabel("Variables")
plt.tight_layout()

plot_path = os.path.join(plot_dir, "xgboost_feature_importances.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"📊 Gráfico de importancia guardado en: {plot_path}")
