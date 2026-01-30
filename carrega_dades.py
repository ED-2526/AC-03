import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def cargar_y_preprocesar_datos_3s(filepath=None):
    """
    Carga el CSV, genera song_id y song_group, separa X/y,
    codifica la etiqueta y devuelve X, y_encoded, groups y el LabelEncoder.
    """

    # 1. Gestionar ruta automáticamente si no se pasa filepath
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, 'Data', 'gtzan_pro_features2.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo en: {filepath}")

    print(f"✅ Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)

    # 2. Crear song_id y song_group
    print("🛠️ Extrayendo ID de canción del filename...")
    df['song_id'] = df['filename'].apply(lambda x: x.split('.')[1])
    df['song_group'] = df['label'] + '_' + df['song_id']
    df = df.dropna()
    groups = df['song_group']

    # 3. Separar features y target
    X = df.drop(columns=['filename', 'label','song_id', 'song_group'])
    y = df['label']

    # 4. Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("✅ Datos cargados y preprocesados correctamente.")

    return X, y_encoded, groups, le


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold # <--- NOVA EINA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# [La funció cargar_y_preprocesar_datos_3s() es manté igual, 
#  ja que fa una feina de preparació de dades i groups correcta.]

def split_datos_3s(filepath=None, test_size=0.2, random_state=42):
    """
    APLICACIÓ DE STRATIFIEDGROUPKFOLD per assegurar l'estratificació 
    mentre es manté la separació de grups (GroupShuffleSplit).
    """

    # --- 1. Cargar y preprocesar datos ---
    X, y, groups, label_encoder = cargar_y_preprocesar_datos_3s(filepath)

    print(f"⚙️ Realizando StratifiedGroupKFold (test_size={test_size})...")

    # --- 2. StratifiedGroupKFold ---
    
    # 2.1. Inicialitzar el validador
    # N_splits = 1/test_size per simular el percentatge de split
    # Amb test_size=0.2, necessitem n_splits=5 per obtenir una divisió 80/20.
    n_splits_for_test_size = int(1 / test_size)
    
    # L'eina StratifiedGroupKFold garanteix:
    # 1. Cap song_group compartit entre train i test.
    # 2. La proporció de 'y' és aproximadament igual en train i test.
    sgkf = StratifiedGroupKFold(n_splits=n_splits_for_test_size, shuffle=True, random_state=random_state)
    
    # Només agafem la primera (i única) divisió de les n_splits (que és la nostra partició 80/20)
    # y (la target) és necessari aquí per a l'estratificació!
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # --- 3. Escalar ---
    print("📏 Escalando features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # --- 4. Verificación de Data Leakage (La comprovació es manté) ---
    train_songs = set(groups.iloc[train_idx])
    test_songs = set(groups.iloc[test_idx])
    shared = train_songs.intersection(test_songs)

    print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"✅ Data Leakage evitado: {len(shared)} canciones compartidas (debe ser 0).")
    # Immediatament després de fer aquesta modificació i tornar a executar el main amb la funció d'anàlisi,
    # el nou gràfic de distribució de classes mostrarà barres molt més alineades.

    return X_train, X_test, y_train, y_test, label_encoder, scaler



def cargar_y_preprocesar_datos_30s(filepath=None):
    """
    Carga el CSV de 30s, limpia columnas y codifica etiquetas.
    Devuelve X, y_encoded y el LabelEncoder.
    Útil para EDA o PCA.
    """
    # 1. Gestión de la ruta
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, 'Data', 'features_30_sec.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No se encontró el archivo en: {filepath}")

    print(f"✅ Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath)

    # 2. Separar Features y Target
    X = df.drop(columns=['filename', 'length', 'label'])
    y = df['label']

    # 3. Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("✅ Datos cargados y preprocesados correctamente (raw).")
    return X, y_encoded, le

# ============================================================
# 2. FUNCIONES DE SPLIT Y ESCALADO
# ============================================================
def split_datos_30s(filepath=None, test_size=0.2, random_state=42):
    """
    Llama a cargar_y_preprocesar_datos_30s_raw(), luego hace train_test_split y escalado.
    No requiere pasar X, y desde el main.
    """
    # 1. Cargar datos raw
    X, y, label_encoder = cargar_y_preprocesar_datos_30s(filepath)

    # 2. Split estratificado
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Escalado
    print("📏 Escalando features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"✅ Datos procesados. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, label_encoder, scaler