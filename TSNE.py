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
# 2.1 APLICAR t-SNE (después de preparar X e y)
# ============================================================
from sklearn.manifold import TSNE

# Escalado antes de t-SNE (recomendable)
scaler_tsne = StandardScaler()
X_scaled = scaler_tsne.fit_transform(X)

# Reducimos a 2 componentes con t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42,
    init="pca"
)
X_tsne = tsne.fit_transform(X_scaled)

print("✅ t-SNE calculado. Shape de la proyección 2D:", X_tsne.shape)

# ============================================================
# 2.2 GRAFICAR t-SNE
# ============================================================
plt.figure(figsize=(10, 7))

# Mapear colores por clase
classes = np.unique(y)
class_to_idx = {cls: i for i, cls in enumerate(classes)}

for cls in classes:
    idx = y == cls
    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        label=cls,
        alpha=0.7,
        s=40
    )

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE - Proyección 2D")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "TSNE_2D_scatter.png"))
plt.close()

print("📊 Gráfico t-SNE guardado en:", os.path.join(plot_dir, "TSNE_2D_scatter.png"))
