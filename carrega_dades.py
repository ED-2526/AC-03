import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def cargar_y_preprocesar_datos_3s(filepath=None, test_size=0.2, random_state=42):
    """
    Carga los datos, soluciona el Data Leakage agrupando por el ID de la canción
    extraído del 'filename', escala los features y codifica los labels.
    
    Args:
        filepath (str): Ruta al csv. Si es None, busca en ./Data/features_3_sec.csv
        test_size (float): Proporción del test set (0.2 = 20%)
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        X_train, X_test, y_train, y_test, label_encoder, scaler
    """
    
    # --- 1. Gestión de la ruta del archivo ---
    if filepath is None:
        # Asume que esta función está en un script que puede deducir la ruta.
        # Ajusta esto si la estructura de carpetas es diferente.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Se asume la estructura original: ./Data/features_3_sec.csv
        filepath = os.path.join(current_dir, 'Data', 'features_3_sec.csv')

    if not os.path.exists(filepath):
        # Manejo de error si el archivo no se encuentra
        # Si no se puede determinar la ruta (__file__), se asume que está en el directorio actual
        if filepath.find('__file__') != -1: # Comprueba si falló la determinación de la ruta
             filepath = 'features_3_sec.csv'
             if not os.path.exists(filepath):
                 raise FileNotFoundError(f"❌ No se encontró el archivo en: {filepath} ni en la ruta absoluta.")
        else:
             raise FileNotFoundError(f"❌ No se encontró el archivo en: {filepath}")


    print(f"✅ Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)
    
    # --- 2. CREACIÓN DEL SONG_ID (SOLUCIÓN AL DATA LEAKAGE) ---
    # El nuevo método extrae el 'num_cancion' del 'filename' (e.g., 'blues.00000.0.wav')
    
    print("🛠️ Extrayendo ID de canción del filename...")
    # Ejemplo de filename: 'blues.00000.0.wav'
    # 1. Quitar la extensión: 'blues.00000.0'
    # 2. Dividir por el punto: ['blues', '00000', '0']
    # 3. Tomar el segundo elemento (índice 1), que es el num_cancion: '00000'
    df['song_id'] = df['filename'].apply(lambda x: x.split('.')[1])
    
    # Opcional: Combinar estilo + num_cancion para un ID único más robusto
    df['song_group'] = df['label'] + '_' + df['song_id']
    groups = df['song_group'] # Ahora agrupamos por esta columna
    
    # --- 3. Separar Features y Target ---
    # Eliminamos columnas que no sirven para predecir (y las columnas de ID de ayuda)
    X = df.drop(columns=['filename', 'length', 'label', 'song_id', 'song_group'])
    y = df['label']
    
    # --- 4. Codificar Etiquetas (String -> Número) ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # --- 5. SPLIT POR GRUPOS (GroupShuffleSplit) ---
    # Esto asegura que una canción completa (definida por 'song_group')
    # vaya a Train O a Test, nunca mezclada.
    print(f"⚙️ Realizando GroupShuffleSplit (test_size={test_size})...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Obtenemos los índices para separar
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    # Creamos los sets
    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]
    
    # --- 6. Escalado (StandardScaler) ---
    # Fit solo en Train para evitar data leakage del escalado
    print("📏 Escalando features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"✅ Datos procesados. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Verificación de Data Leakage
    train_songs = set(groups.iloc[train_idx])
    test_songs = set(groups.iloc[test_idx])
    shared_songs = len(train_songs.intersection(test_songs))
    
    print(f"✅ Data Leakage evitado: {shared_songs} canciones compartidas (debe ser 0).")
    
    return X_train, X_test, y_train, y_test, le, scaler

def cargar_y_preprocesar_datos_30s(filepath=None, test_size=0.2, random_state=42):
    # 1. Gestión de la ruta del archivo
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, 'Data', 'features_30_sec.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No se encontró el archivo en: {filepath}")

    print(f"✅ Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)

    # 2. Separar Features y Target
    # Eliminamos columnas que no sirven para predecir
    X = df.drop(columns=['filename', 'length', 'label'])
    y = df['label']

    # 3. Codificar Etiquetas (String -> Número)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. SPLIT (train_test_split)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # 5. Escalado (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"✅ Datos procesados. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, le, scaler

if __name__ == "__main__":
    # Bloque de prueba para cuando ejecutes este script solo
    try:
        X_tr, X_te, y_tr, y_te, le, sc = cargar_y_preprocesar_datos_3s()
        print("Prueba exitosa.")
    except Exception as e:
        print(f"Error en la prueba: {e}")