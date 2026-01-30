import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Datos de tu reporte final
genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
f1_scores = [0.72, 0.88, 0.62, 0.68, 0.70, 0.78, 0.79, 0.81, 0.67, 0.48]

# Colores: Destacamos los difíciles y los fáciles
colors = ['#3498db'] * 10
colors[1] = '#2ecc71' # Classical (Verde - Excelente)
colors[9] = '#e74c3c' # Rock (Rojo - El reto superado)

plt.figure(figsize=(12, 6))
bars = plt.bar(genres, f1_scores, color=colors)

# Línea del 70% (Tu aprobado)
plt.axhline(y=0.70, color='gray', linestyle='--', label='Objetivo del Proyecto (70%)')

plt.title("Rendimiento Final por Género (F1-Score)", fontsize=14, fontweight='bold')
plt.ylabel("F1 Score")
plt.ylim(0, 1.0)
plt.legend()

# Poner los numeritos encima
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()