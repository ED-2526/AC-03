import pandas as pd
import numpy as np
import os

# Librerías de Machine Learning y Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Librerías de Deep Learning (Keras/TensorFlow)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. CARGA DE DATOS ---
# Usaremos el dataset de 3 segundos para tener más datos (10,000 filas)
features_path = r'C:\Users\ikerb\.cache\kagglehub\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\features_3_sec.csv'
df = pd.read_csv(features_path)

print("Datos cargados. Tamaño:", df.shape)

# --- 2. PREPROCESAMIENTO ---

# Eliminar columnas que no son características de audio útiles para el modelo
# 'filename' es texto, 'length' es constante en este csv
X = df.drop(columns=['filename', 'length', 'label'])
y = df['label']

# Codificar las etiquetas (de texto "jazz" a números 0, 1, 2...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir en Train y Test (80% / 20%)
# Stratify asegura que haya la misma cantidad de cada género en ambos grupos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Escalar los datos (IMPORTANTE: fit solo en train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 3. ADAPTACIÓN PARA LSTM ---
# LSTM necesita entrada 3D: (Muestras, Pasos de tiempo, Características)
# Como son datos tabulares, usamos 1 paso de tiempo.
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Convertir etiquetas a One-Hot Encoding (matrices binarias) para la red neuronal
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print(f"Forma de entrada para LSTM: {X_train_reshaped.shape}")

# --- 4. ARQUITECTURA DEL MODELO ---
model = Sequential([
    # Capa de entrada
    BatchNormalization(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),

    # 1ª Capa LSTM Bidireccional
    Bidirectional(LSTM(256, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    # 2ª Capa LSTM Bidireccional
    Bidirectional(LSTM(128, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.3),

    # Capas Densas (Fully Connected)
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),

    # Capa de Salida (10 neuronas = 10 géneros)
    Dense(10, activation='softmax')
])

# --- 5. COMPILACIÓN ---
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', AUC(name='auc')]
)

# --- 6. ENTRENAMIENTO CON CALLBACKS ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print("Iniciando entrenamiento...")
history = model.fit(
    X_train_reshaped,
    y_train_cat,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test_cat),
    callbacks=[early_stopping, lr_reducer],
    verbose=1
)

print("Entrenamiento finalizado.")