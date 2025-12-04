import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def cargar_y_preprocesar_datos_3s(filepath=None, test_size=0.2, random_state=42):
    """
    Carga los datos, soluciona el Data Leakage agrupando por canción,
    escala los features y codifica los labels.
    
    Args:
        filepath (str): Ruta al csv. Si es None, busca en ./Data/features_3_sec.csv
        test_size (float): Proporción del test set (0.2 = 20%)
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        X_train, X_test, y_train, y_test, label_encoder, scaler
    """
    
    # 1. Gestión de la ruta del archivo
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, 'Data', 'features_3_sec.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No se encontró el archivo en: {filepath}")

    print(f"✅ Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)

    # 2. CREACIÓN DEL SONG_ID (SOLUCIÓN AL DATA LEAKAGE)
    # Asumimos que las filas están ordenadas y hay 10 segmentos por canción
    df['song_id'] = df.index // 10

    # 3. Separar Features y Target
    # Eliminamos columnas que no sirven para predecir
    X = df.drop(columns=['filename', 'length', 'label', 'song_id'])
    y = df['label']
    groups = df['song_id'] # Guardamos los grupos para el split

    # 4. Codificar Etiquetas (String -> Número)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 5. SPLIT POR GRUPOS (GroupShuffleSplit)
    # Esto asegura que una canción entera vaya a Train O a Test, nunca mezclada
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Obtenemos los índices para separar
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    # Creamos los sets
    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    # 6. Escalado (StandardScaler)
    # Fit solo en Train para evitar data leakage del escalado
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"✅ Datos procesados. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"✅ Data Leakage evitado: {len(set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx])))} canciones compartidas (debe ser 0).")

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