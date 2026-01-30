import os
from PIL import Image, ImageChops

# --- GESTIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ⚠️ AJUSTE DE RUTAS: 
# Si este script YA está dentro de la carpeta AC-03, borra "AC-03" de las líneas de abajo.
# Si el script está fuera (en PROJECTE AC), déjalo como está.
INPUT_DIR = os.path.join(BASE_DIR, "Data", "images_original")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "images_cropped")

def trim_agresivo(im):
    """
    Recorta márgenes blancos usando un umbral de tolerancia.
    Si el píxel es 'casi' blanco, lo considera fondo y lo recorta.
    """
    # 1. Asegurar formato RGB (quita transparencias raras)
    im = im.convert("RGB")
    
    # 2. Crear una imagen completamente BLANCA del mismo tamaño
    bg = Image.new("RGB", im.size, (255, 255, 255))
    
    # 3. Calcular la diferencia entre tu imagen y el blanco puro
    diff = ImageChops.difference(im, bg)
    
    # 4. Convertir la diferencia a escala de grises
    diff = diff.convert("L")
    
    # 5. APLICAR UMBRAL (La parte mágica)
    # Todo lo que sea 'casi blanco' (diferencia < 20) se vuelve negro (se ignora).
    # Todo lo que sea contenido (colores fuertes) se vuelve blanco (se mantiene).
    # Puedes subir 'threshold' a 50 si todavía quedan bordes.
    threshold = 20
    mask = diff.point(lambda p: 255 if p > threshold else 0)
    
    # 6. Calcular la caja que contiene los píxeles útiles
    bbox = mask.getbbox()
    
    if bbox:
        # Añadimos un pequeño margen de seguridad (padding) opcional, o devolvemos directo
        return im.crop(bbox)
        
    return im # Si no encuentra nada (imagen blanca entera), devuelve la original

def procesar_imagenes():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ ERROR: No encuentro: {INPUT_DIR}")
        return

    print(f"✂️  Recortando (Modo Agresivo) desde: {INPUT_DIR}")
    print(f"💾 Guardando en: {OUTPUT_DIR}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    genres = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    count = 0
    for genre in genres:
        in_genre_dir = os.path.join(INPUT_DIR, genre)
        out_genre_dir = os.path.join(OUTPUT_DIR, genre)
        os.makedirs(out_genre_dir, exist_ok=True)
        
        files = [f for f in os.listdir(in_genre_dir) if f.endswith('.png')]
        
        for f in files:
            try:
                img = Image.open(os.path.join(in_genre_dir, f))
                
                # Usamos la nueva función agresiva
                cropped_img = trim_agresivo(img)
                
                # Forzar redimensión final (Opcional, pero recomendado para CNNs)
                # cropped_img = cropped_img.resize((224, 224)) 
                
                cropped_img.save(os.path.join(out_genre_dir, f))
                
                count += 1
                if count % 50 == 0: print(f"   Procesadas {count}...", end='\r')
            except Exception as e:
                print(f"Error en {f}: {e}")

    print(f"\n✅ ¡Listo! {count} imágenes recortadas.")
    print("👉 Ve a la carpeta y abre una imagen para verificar.")

if __name__ == "__main__":
    procesar_imagenes()