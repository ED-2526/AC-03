import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from itertools import combinations

# 1. CARGA
print("--- CARGANDO DATASET ---")
try:
    df = pd.read_csv('gtzan_pro_features_final.csv')
except:
    try:
        df = pd.read_csv('../gtzan_pro_features1.csv')
    except:
        print("❌ Error: No encuentro el archivo.")

# 2. PREPROCESAMIENTO
df = df.dropna()
cols_drop = [c for c in df.columns if c.startswith('boaw_')] + ['label', 'filename', 'song_id', 'length']
X = df.drop([c for c in cols_drop if c in df.columns], axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = np.unique(y) # Lista ordenada de clases

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. t-SNE
print("Calculando t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='random', learning_rate='auto')
X_embedded = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame(X_embedded, columns=['x', 'y'])
tsne_df['label'] = y.values

# --- 4. MAPA MAESTRO DE COLORES (LA CLAVE) ---
# Usamos el colormap 'tab10' que tiene 10 colores distintivos (perfecto para GTZAN)
# Asignamos un color fijo a cada género para siempre.
cmap = plt.get_cmap('tab10')
color_list = [cmap(i) for i in range(len(classes))]
color_map = dict(zip(classes, color_list))

print("✅ Mapa de colores fijado.")

# 5. SELECCIÓN DE CLUSTERS
print("Analizando distancias...")
centroids = tsne_df.groupby('label')[['x', 'y']].mean()

def get_separation_score(genres):
    points = centroids.loc[list(genres)].values
    dist = 0
    pairs = 0
    for p1 in range(len(points)):
        for p2 in range(p1+1, len(points)):
            dist += np.linalg.norm(points[p1] - points[p2])
            pairs += 1
    return dist / pairs

all_triplets = list(combinations(classes, 3))
scores = [(triplet, get_separation_score(triplet)) for triplet in all_triplets]
scores.sort(key=lambda x: x[1], reverse=True)

best_triplet = scores[0][0]
worst_triplet = scores[-1][0]

print(f"✅ Mejor separación: {best_triplet}")
print(f"⚠️ Peor solapamiento: {worst_triplet}")

# 6. FUNCIÓN DE PLOTEO COHERENTE
def plot_fixed_colors(df_subset, title, filename):
    plt.figure(figsize=(10, 7))
    
    # Obtenemos las clases presentes en este subset
    # Pero las ordenamos para que la leyenda salga bonita
    subset_classes = np.sort(df_subset['label'].unique())
    
    for cls in subset_classes:
        idx = df_subset['label'] == cls
        
        # AQUÍ ESTÁ EL TRUCO: Usamos 'c=[color_map[cls]]'
        # Forzamos a matplotlib a usar el color del diccionario maestro
        plt.scatter(
            df_subset.loc[idx, 'x'], 
            df_subset.loc[idx, 'y'], 
            label=cls,
            alpha=0.7,
            s=40,
            c=[color_map[cls]] # <--- Color forzado y consistente
        )
    
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"-> Gráfico guardado: {filename}")

# --- GENERAR GRÁFICOS ---

# 1. GLOBAL
plot_fixed_colors(tsne_df, "Mapa Global t-SNE", "tsne_global.png")



# 2. MEJORES
df_best = tsne_df[tsne_df['label'].isin(best_triplet)]
plot_fixed_colors(df_best, f"Clusters Diferenciados: {best_triplet}", "tsne_best.png")

# 3. PEORES
df_worst = tsne_df[tsne_df['label'].isin(worst_triplet)]
plot_fixed_colors(df_worst, f"Zona de Conflicto: {worst_triplet}", "tsne_worst.png")