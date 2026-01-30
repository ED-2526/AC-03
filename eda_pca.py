import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================================================================
# 1. ESTADÍSTICAS BÁSICAS
# =====================================================================
def eda_estadisticas_basicas(X, y, plot_dir="Plots"):
    """
    Muestra estadísticas descriptivas y boxplots.
    """
    os.makedirs(plot_dir, exist_ok=True)

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    
    classes = np.unique(y)

    print("===== ESTADÍSTICAS DE FEATURES =====")
    print(X.describe(include='all'))

    print("\n===== DISTRIBUCIÓN DE ETIQUETAS =====")
    print(pd.Series(y).value_counts())

    # --- Estadísticas descriptivas generales por feature ---
    desc_stats = X.describe().T
    desc_stats['range'] = desc_stats['max'] - desc_stats['min']
    print("\n📌 Estadísticas descriptivas generales por feature:")
    print(desc_stats)

    # --- Estadísticas por clase ---
    for cls in classes:
        X_cls = X[y == cls]
        desc_cls = X_cls.describe().T
        desc_cls['range'] = desc_cls['max'] - desc_cls['min']
        print(f"\n📌 Estadísticas descriptivas para la clase '{cls}':")
        print(desc_cls)

    # --- Detección de outliers simples (IQR) ---
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

    # --- Boxplots primeras 5 features + 'tempo' si existe ---
    features_to_plot = list(X.columns[:5])
    if 'tempo' in X.columns:
        features_to_plot.append('tempo')
    plt.figure(figsize=(15, 8))
    for i, feat in enumerate(features_to_plot):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=y, y=X[feat])
        plt.title(f"Boxplot - {feat}")
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "boxplots_features.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Boxplots guardados en: {save_path}")


# =====================================================================
# 2. HEATMAP DE CORRELACIÓN
# =====================================================================
def eda_heatmap_correlacion(X, plot_dir="Plots"):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(X.corr(), cmap='coolwarm', linewidths=0.5)
    plt.title("Heatmap de Correlación entre Features")
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "correlation_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Heatmap de correlación guardado en: {save_path}")


# =====================================================================
# 3. DISTRIBUCIÓN DE CLASES
# =====================================================================
def eda_distribucion_clases(y, le, plot_dir="Plots"):
    os.makedirs(plot_dir, exist_ok=True)
    labels_decoded = le.inverse_transform(y)
    counts = pd.Series(labels_decoded).value_counts()
    plt.figure(figsize=(12, 6))
    counts.plot(kind='bar', color='skyblue')
    plt.title("Distribución de clases")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "clases_distribucion.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Distribución de clases guardada en: {save_path}")


# =====================================================================
# 4. PCA: ENTRENAR PCA (ESCALADO AUTOMÁTICO)
# =====================================================================
def ejecutar_pca(X, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print("===== PCA =====")
    print("Varianza explicada:", pca.explained_variance_ratio_)
    return X_pca, pca, scaler


# =====================================================================
# 5. GRAFICAR PCA EN 2D
# =====================================================================
def graficar_pca_2d(X_pca, y, le, plot_dir="Plots"):
    os.makedirs(plot_dir, exist_ok=True)
    labels = le.inverse_transform(y)
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 7))
    for cls in unique_labels:
        idx = labels == cls
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=40, alpha=0.7, label=cls)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA – Primeras 2 Componentes")
    plt.legend(loc="best")
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "PCA_2D.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 PCA 2D guardado en: {save_path}")
