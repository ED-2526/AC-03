import pandas as pd
import numpy as np
import os

# Librerías ML
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint

# Librerías visualización
import matplotlib.pyplot as plt
import seaborn as sns

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
# ============================================================
# 2.1 APLICAR PCA (después de preparar X e y)
# ============================================================
from sklearn.decomposition import PCA

# Escalado antes de PCA
scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X)

# Reducimos a 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Varianza explicada por PC1 y PC2:", pca.explained_variance_ratio_)

# ============================================================
# 2.2 GRAFICAR PCA
# ============================================================
plt.figure(figsize=(10, 7))

# Mapear colors por clase
classes = np.unique(y)
class_to_idx = {cls: i for i, cls in enumerate(classes)}

for cls in classes:
    idx = y == cls
    plt.scatter(
        X_pca[idx, 0], 
        X_pca[idx, 1], 
        label=cls,
        alpha=0.7,
        s=40
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Primeras 2 Componentes Principales")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "PCA_2D_scatter.png"))
plt.close()

print("📊 Gráfico PCA guardado en:", os.path.join(plot_dir, "PCA_2D_scatter.png"))
