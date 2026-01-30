import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
DATASET_PATH = 'AC-03/Data/genres_original' 
OUTPUT_CSV = 'gtzan_complete_features.csv'
SEGMENT_DURATION = 3 
SAMPLE_RATE = 22050
N_CLUSTERS = 50  # Este es el tamaño de tu "vocabulario" (50 palabras sonoras)

def get_stats(array, prefix):
    return {
        f'{prefix}_mean': np.mean(array),
        f'{prefix}_var': np.var(array),
        f'{prefix}_skew': skew(array, axis=None),
        f'{prefix}_kurt': kurtosis(array, axis=None)
    }

# --- FASE 1: CREACIÓN DEL VOCABULARIO (BOAW) ---
# Necesitamos aprender qué sonidos son comunes antes de poder contarlos
print("--- FASE 1: Aprendiendo el Vocabulario Sonoro ---")
all_mfcc_frames = []
# Muestreamos aleatoriamente el 10% del dataset para crear el diccionario rápido
for root, dirs, files in os.walk(DATASET_PATH):
    for filename in files:
        if filename.endswith('.wav') and np.random.rand() < 0.1:
            try:
                y, _ = librosa.load(os.path.join(root, filename), sr=SAMPLE_RATE, duration=5)
                mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
                all_mfcc_frames.append(mfcc.T) # Guardamos frames individuales
            except: pass

# Entrenamos los clusters (las "palabras")
X_vocab = np.vstack(all_mfcc_frames)
scaler_vocab = StandardScaler()
X_vocab_scaled = scaler_vocab.fit_transform(X_vocab)

kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=512)
kmeans.fit(X_vocab_scaled)
print(f"Vocabulario de {N_CLUSTERS} palabras creado.")

# --- FASE 2: EXTRACCIÓN PRINCIPAL ---
print("\n--- FASE 2: Extracción de Stats + BoAW ---")
features_list = []
genres = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    print(f"Procesando género: {genre}...")
    
    for filename in os.listdir(genre_path):
        if not filename.endswith('.wav'): continue
        file_path = os.path.join(genre_path, filename)
        
        try:
            y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            samples_per_segment = int(SEGMENT_DURATION * sr)
            n_segments = int(len(y_full) / samples_per_segment)
            
            for s in range(n_segments):
                start = s * samples_per_segment
                y = y_full[start:start + samples_per_segment]
                if len(y) < samples_per_segment: continue

                # A. CARACTERÍSTICAS DE TU SCRIPT ORIGINAL (Stats)
                S = np.abs(librosa.stft(y))
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                dict_features = {}
                
                # Timbre, Armonía y Ritmo (Tus funciones originales)
                dict_features.update(get_stats(librosa.feature.spectral_centroid(y=y, sr=sr), 'centroid'))
                dict_features.update(get_stats(librosa.feature.spectral_contrast(S=S, sr=sr), 'contrast'))
                dict_features.update(get_stats(librosa.feature.chroma_stft(y=y_harmonic, sr=sr), 'chroma'))
                
                # MFCC Stats
                mfcc_feat = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
                for i in range(13):
                    dict_features.update(get_stats(mfcc_feat[i], f'mfcc_{i+1}'))

                # B. EXTRACCIÓN BOAW (Histograma)
                # 1. Extraer MFCCs del segmento actual
                current_mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
                # 2. Normalizar con el escalador del vocabulario
                current_mfccs_scaled = scaler_vocab.transform(current_mfccs)
                # 3. Asignar cada frame a una "palabra" del K-Means
                words = kmeans.predict(current_mfccs_scaled)
                # 4. Contar frecuencias (crear el histograma)
                counts = np.bincount(words, minlength=N_CLUSTERS)
                hist = counts / np.sum(counts) # Normalizar para que sume 1
                
                for i in range(N_CLUSTERS):
                    dict_features[f'boaw_{i}'] = hist[i]

                # Metadatos
                dict_features['label'] = genre
                dict_features['song_id'] = filename  
                features_list.append(dict_features)
                
        except Exception as e:
            print(f"Error en {filename}: {e}")

df = pd.DataFrame(features_list)
df.to_csv(OUTPUT_CSV, index=False)
print(f"¡Hecho! Dataset con BoAW guardado en {OUTPUT_CSV}")