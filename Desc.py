import pandas as pd
import numpy as np
import os
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
# Incluir todas las features excepto filename y label
X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']
classes = np.unique(y)

# ============================================================
# 3. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

# 3.1 Medidas centrales y dispersión por feature (todas las clases juntas)
desc_stats = X.describe().T
desc_stats['range'] = desc_stats['max'] - desc_stats['min']
print("📌 Estadísticas descriptivas generales por feature:")
print(desc_stats)

# 3.2 Medidas centrales y dispersión por clase
for cls in classes:
    X_cls = X[y == cls]
    desc_cls = X_cls.describe().T
    desc_cls['range'] = desc_cls['max'] - desc_cls['min']
    print(f"\n📌 Estadísticas descriptivas para la clase '{cls}':")
    print(desc_cls)

# 3.3 Detección de outliers simples (usando IQR) por feature y clase
def detect_outliers_iqr(df_feature):
    Q1 = df_feature.quantile(0.25)
    Q3 = df_feature.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_feature < Q1 - 1.5 * IQR) | (df_feature > Q3 + 1.5 * IQR))
    return outliers.sum()

outlier_summary = pd.DataFrame(index=X.columns, columns=classes)

for cls in classes:
    X_cls = X[y == cls]
    for feat in X.columns:
        outlier_summary.loc[feat, cls] = detect_outliers_iqr(X_cls[feat])

print("\n📌 Resumen de outliers por feature y clase (IQR method):")
print(outlier_summary)

# 3.4 Visualización: boxplots de primeras 5 features + tempo
features_to_plot = list(X.columns[:5]) + ['tempo']
plt.figure(figsize=(15, 8))
for i, feat in enumerate(features_to_plot):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=y, y=X[feat])
    plt.title(f"Boxplot - {feat}")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "boxplots_features.png"))
plt.close()
print("📊 Boxplots guardados en:", os.path.join(plot_dir, "boxplots_features.png"))
