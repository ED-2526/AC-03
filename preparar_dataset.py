import os
import shutil
import random
import re
from collections import defaultdict

# --- GESTIÓN DE RUTAS AUTOMÁTICA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ⚠️ CORRECCIÓN AQUÍ: Añadimos "AC-03" para que busque en la carpeta correcta
SOURCE_DIR = os.path.join(BASE_DIR, "Data", "images_cropped")
DEST_DIR = os.path.join(BASE_DIR, "Data", "dataset_split")

TEST_SPLIT = 0.2
RANDOM_SEED = 42

def extract_song_id(filename):
    match = re.search(r'\d+', filename)
    if match: return match.group()
    return None

def organizar_dataset():
    # Diagnóstico visual para que veas dónde busca
    print(f"🔍 Buscando imágenes en: {SOURCE_DIR}")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ ERROR: Sigue sin encontrar la carpeta.")
        print(f"   Ruta actual intentada: {SOURCE_DIR}")
        print("   Por favor, entra en tu explorador de archivos y verifica que 'Data' está dentro de 'AC-03'.")
        return

    # Limpiar destino previo si existe
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)

    random.seed(RANDOM_SEED)
    
    genres = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    print(f"📂 Géneros encontrados: {genres}")
    
    total_imgs = 0
    
    for genre in genres:
        genre_path = os.path.join(SOURCE_DIR, genre)
        files = [f for f in os.listdir(genre_path) if f.endswith('.png')]
        
        # Agrupar por canción (ID) para evitar Data Leakage
        song_dict = defaultdict(list)
        for f in files:
            sid = extract_song_id(f)
            if sid: song_dict[sid].append(f)

        unique_songs = list(song_dict.keys())
        random.shuffle(unique_songs)
        
        split_idx = int(len(unique_songs) * (1 - TEST_SPLIT))
        train_songs = unique_songs[:split_idx]
        test_songs = unique_songs[split_idx:]
        
        for mode, songs in [('train', train_songs), ('test', test_songs)]:
            save_path = os.path.join(DEST_DIR, mode, genre)
            os.makedirs(save_path, exist_ok=True)
            for s in songs:
                for f in song_dict[s]:
                    shutil.copy2(os.path.join(genre_path, f), os.path.join(save_path, f))
                    total_imgs += 1

    print("="*40)
    print(f"✅ DATASET ORGANIZADO: {total_imgs} imágenes.")
    print(f"📂 Ubicación: {DEST_DIR}")
    print("="*40)

if __name__ == "__main__":
    organizar_dataset()