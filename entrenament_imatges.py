import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# --- GESTIÓN DE RUTAS AUTOMÁTICA ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# ⚠️ CORRECCIÓN AQUÍ: Añadimos "AC-03" para que encuentre los datos
DATA_DIR = os.path.join(BASE_PATH, "Data", "dataset_split")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print(f"🔍 Buscando dataset en: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    print(f"❌ ERROR CRÍTICO: No encuentro los datos en: {DATA_DIR}")
    print("   Verifica que has ejecutado el script 'preparar_dataset.py' correctamente.")
    exit()

print("✅ Datos encontrados. Cargando...")

# --- CARGA DE DATASETS ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'train'),
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, 'test'),
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Detectar número de clases (géneros) automáticamente
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"🎵 Géneros detectados ({num_classes}): {class_names}")

# --- MODELO (EfficientNetV2 con Augmentation) ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
])

print("🏗️ Construyendo modelo...")

base_model = applications.EfficientNetV2B0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

base_model.trainable = True
# Congelamos todo excepto las últimas 20 capas
for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = applications.efficientnet.preprocess_input(x)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- ENTRENAMIENTO ---
print("🚀 Iniciando entrenamiento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25, 
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    ]
)

# --- RESULTADO FINAL ---
print("\n" + "="*40)
val_loss, val_acc = model.evaluate(val_ds)
print(f"🏆 Accuracy Final en Test: {val_acc:.4f}")
print("="*40)

# Guardar modelo (opcional)
save_path = os.path.join(BASE_PATH, "modelo_entrenado.keras")
model.save(save_path)
print(f"💾 Modelo guardado en: {save_path}")