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

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
# ... imports de keras ...

# --- IMPORTAR TU NUEVO MÓDULO ---
from carrega_dades import cargar_y_preprocesar_datos

# 1. CARGA BASE
X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos()

# 2. ADAPTACIÓN ESPECÍFICA PARA LSTM
# Reshape a 3D: (Samples, TimeSteps, Features)
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# One-Hot Encoding para el target
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# ... A partir de aquí defines tu modelo Sequential ...

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