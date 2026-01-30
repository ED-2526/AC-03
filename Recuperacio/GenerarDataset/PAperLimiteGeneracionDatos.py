import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import warnings

# Ignorar advertencias de versiones futuras
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
DATASET_PATH = 'AC-03/Data/genres_original' # <--- VERIFICA TU RUTA
OUTPUT_CSV = 'gtzan_1sec_psychoacoustic.csv' # Nombre nuevo para no machacar el anterior
SEGMENT_DURATION = 1  # <--- EL CAMBIO CLAVE: 1 SEGUNDO
SAMPLE_RATE = 22050

def get_stats(array, prefix):
    """
    Calcula estadísticas robustas para cada segmento
    """
    return {
        f'{prefix}_mean': np.mean(array),
        f'{prefix}_var': np.var(array),
        f'{prefix}_skew': skew(array, axis=None),
        f'{prefix}_kurt': kurtosis(array, axis=None)
    }

print(f"--- INICIANDO EXTRACCIÓN FRAME-BASED ({SEGMENT_DURATION}s) ---")
print("Objetivo: Capturar micro-texturas y descriptores psicoacústicos.")

features_list = []
genres = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

for genre in genres:
    genre_path = os.path.join(DATASET_PATH, genre)
    print(f" >> Procesando género: {genre}...")
    
    # Listamos archivos
    files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
    
    for filename in files:
        file_path = os.path.join(genre_path, filename)
        
        try:
            # Cargar audio completo una vez
            y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Calcular segmentos
            samples_per_segment = int(SEGMENT_DURATION * sr)
            n_segments = int(len(y_full) / samples_per_segment)
            
            # Pre-cálculo de separación Armónico/Percusivo para toda la canción (Más rápido)
            # Esto ayuda a separar la melodía (Harmonic) del ritmo (Percussive)
            y_harmonic_full, y_percussive_full = librosa.effects.hpss(y_full)
            
            for s in range(n_segments):
                # Cortar índices
                start = s * samples_per_segment
                end = start + samples_per_segment
                
                # Extraer slices de los audios pre-procesados
                y = y_full[start:end]
                y_harm = y_harmonic_full[start:end]
                y_perc = y_percussive_full[start:end]
                
                # Descartar segmentos incompletos al final
                if len(y) < samples_per_segment: continue

                dict_features = {}

                # --- 1. CARACTERÍSTICAS PSICOACÚSTICAS & ESPECTRALES ---
                
                # Espectrograma base
                S = np.abs(librosa.stft(y))
                
                # A. Spectral Flatness (Wiener Entropy)
                # Vital para distinguir "ruido" (Rock distorsionado) de "tono" (Piano Clásico)
                flatness = librosa.feature.spectral_flatness(S=S)
                dict_features.update(get_stats(flatness, 'flatness'))

                # B. Spectral Bandwidth (Percepción de brillo/anchura)
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                dict_features.update(get_stats(bandwidth, 'bandwidth'))

                # C. Spectral Centroid (Brillo)
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                dict_features.update(get_stats(cent, 'centroid'))
                
                # D. Spectral Rolloff (Forma del espectro)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                dict_features.update(get_stats(rolloff, 'rolloff'))
                
                # E. Spectral Contrast (Picos vs Valles - Textura)
                contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
                dict_features.update(get_stats(contrast, 'contrast'))

                # --- 2. ARMONÍA (USANDO CAPA ARMÓNICA) ---
                
                # F. Chroma STFT (Notas musicales: Do, Re, Mi...)
                chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
                dict_features.update(get_stats(chroma, 'chroma'))
                
                # G. Tonnetz (Relaciones armónicas: Quintas, Terceras mayores/menores)
                # Esto es brutal para distinguir Jazz (acordes complejos) de Pop (acordes simples)
                try:
                    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
                    dict_features.update(get_stats(tonnetz, 'tonnetz'))
                except:
                    # Fallback si el segmento es silencio
                    pass

                # --- 3. TIMBRE (MFCCs) ---
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                for i in range(1, 14): # MFCC 1-13 (Los más importantes)
                    dict_features.update(get_stats(mfcc[i], f'mfcc_{i}'))
                
                # --- 4. RITMO & DINÁMICA ---
                
                # H. Zero Crossing Rate (Ruido/Metal)
                zcr = librosa.feature.zero_crossing_rate(y)
                dict_features.update(get_stats(zcr, 'zcr'))
                
                # I. RMS (Energía/Volumen)
                rms = librosa.feature.rms(y=y)
                dict_features.update(get_stats(rms, 'rms'))
                
                # J. Onset Strength (Fuerza de los golpes) - Usamos capa percusiva
                onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr)
                dict_features.update(get_stats(onset_env, 'onset'))
                
                # K. Tempo (BPM)
                # OJO: En 1 segundo el BPM es inestable, pero la media global ayudará
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
                dict_features['tempo'] = tempo[0] 
                
                # --- METADATOS ---
                dict_features['label'] = genre
                dict_features['filename'] = filename
                # IMPORTANTE: song_id permite luego agrupar los votos
                dict_features['song_id'] = filename 
                
                features_list.append(dict_features)
                
        except Exception as e:
            print(f"❌ Error en {filename}: {e}")

# Crear DataFrame y Guardar
df = pd.DataFrame(features_list)
print(f"\n--- EXTRACCIÓN COMPLETADA ---")
print(f"Total de segmentos (frames): {df.shape[0]}")
print(f"Guardando en: {OUTPUT_CSV}")
df.to_csv(OUTPUT_CSV, index=False)