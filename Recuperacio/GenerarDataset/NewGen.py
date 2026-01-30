import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import warnings

# Ignorar advertencias de versiones futuras de librosa para mantener la consola limpia
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
# CAMBIA ESTO por la ruta donde tienes la carpeta 'genres' con las subcarpetas (blues, classical...)
DATASET_PATH = 'AC-03/Data/genres_original' 
OUTPUT_CSV = 'gtzan_pro_features.csv'
SEGMENT_DURATION = 3  # Segundos
SAMPLE_RATE = 22050

def get_stats(array, prefix):
    """
    Función auxiliar para no repetir código.
    Calcula: Media, Varianza, Skewness (Asimetría), Kurtosis
    """
    return {
        f'{prefix}_mean': np.mean(array),
        f'{prefix}_var': np.var(array),
        f'{prefix}_skew': skew(array, axis=None), # Axis None aplana el array
        f'{prefix}_kurt': kurtosis(array, axis=None)
    }

print("Iniciando extracción de características de alto nivel...")

features_list = []
genres = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    print(f"Procesando género: {genre}...")
    
    for filename in os.listdir(genre_path):
        if not filename.endswith('.wav'): continue
        
        file_path = os.path.join(genre_path, filename)
        
        try:
            # Cargar audio completo
            y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Calcular número de segmentos
            samples_per_segment = int(SEGMENT_DURATION * sr)
            n_segments = int(len(y_full) / samples_per_segment)
            
            for s in range(n_segments):
                # Cortar el segmento
                start = s * samples_per_segment
                end = start + samples_per_segment
                y = y_full[start:end]
                
                # Si el segmento es muy corto (final de canción), lo ignoramos
                if len(y) < samples_per_segment: continue

                # --- EXTRACCIÓN DE CARACTERÍSTICAS (MATRICES) ---
                
                # 1. ESPECTROGRAMA (STFT)
                S = np.abs(librosa.stft(y))
                
                # 2. SEPARACIÓN ARMÓNICO/PERCUSIVO
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                
                dict_features = {}
                
                # --- GRUPO TIMBRE (Basado en Armónicos y Espectro) ---
                
                # Spectral Centroid
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                dict_features.update(get_stats(cent, 'centroid'))
                
                # Spectral Rolloff
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                dict_features.update(get_stats(rolloff, 'rolloff'))
                
                # Spectral Contrast
                contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
                dict_features.update(get_stats(contrast, 'contrast'))

                # --- GRUPO ARMONÍA (NUEVO: AQUÍ ESTÁ LA MAGIA) ---
                # Usamos y_harmonic para limpiar el ruido de percusión
                
                # A. Chroma (Detecta si es Do, Re, Mi...) - Clave para escalas de Blues
                chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
                dict_features.update(get_stats(chroma, 'chroma'))
                
                # B. Tonnetz (Detecta quintas y terceras) - Clave para Jazz vs Country
                try:
                    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
                    dict_features.update(get_stats(tonnetz, 'tonnetz'))
                except Exception as e:
                    # Si falla (ej. silencio total), rellenamos con ceros
                    for k in ['mean', 'var', 'skew', 'kurt']:
                        dict_features[f'tonnetz_{k}'] = 0.0
                
                # --- FIN GRUPO ARMONÍA ---
                
                # MFCCs (Timbre general)
                mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=20)
                for i in range(1, 14): # MFCC 1 a 13
                    dict_features.update(get_stats(mfcc[i], f'mfcc_{i}'))
                
                # Zero Crossing Rate
                zcr = librosa.feature.zero_crossing_rate(y)
                dict_features.update(get_stats(zcr, 'zcr'))

                # --- GRUPO RITMO (Basado en Percusión y Tiempo) ---
                
                # RMS
                rms = librosa.feature.rms(y=y)
                dict_features.update(get_stats(rms, 'rms'))
                
                # Onset Strength
                onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)
                dict_features.update(get_stats(onset_env, 'onset'))
                
                # Tempo (BPM)
                # Nota: Usamos la función moderna si la antigua da warning, pero esta es la clásica
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
                dict_features['tempo'] = tempo[0] 
                
                # --- METADATOS ---
                dict_features['label'] = genre
                dict_features['filename'] = filename
                dict_features['song_id'] = filename  
                
                features_list.append(dict_features)
                
        except Exception as e:
            # El famoso jazz.00054.wav
            print(f"Error en {filename}: {e}")

# Crear DataFrame
df = pd.DataFrame(features_list)

# Guardar
df.to_csv(OUTPUT_CSV, index=False)
print(f"¡Hecho! Dataset guardado en {OUTPUT_CSV} con {df.shape[0]} filas.")