import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from carrega_dades import *

try:
    X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_3s() # Per a 3 segons
    #X_train, X_test, y_train, y_test, label_encoder, scaler = cargar_y_preprocesar_datos_30s() # Per a 30 segons
    class_names = label_encoder.classes_
except Exception as e:
    print(e)
    exit()


### 2. Definició i Entrenament del Model SVC


print("\n" + "="*50)
print("     🚀 Iniciant l'Entrenament del Model SVC")
print("="*50)

# Inicialitzar el model SVC
# Valors inicials: C=1.0 i gamma='scale' (basat en l'escalat Standard)
svc_model = SVC(
    C=1.0, 
    kernel='rbf', 
    gamma='scale', 
    random_state=42, 
    verbose=True # Per veure el progrés de l'entrenament
)

# Entrenar el model amb les dades escalades
svc_model.fit(X_train, y_train)

print("\n✅ Entrenament del SVC finalitzat.")


### 3. Avaluació del Model

# Predicció sobre el conjunt de test (dades no vistes)
y_pred = svc_model.predict(X_test)

# Obtenció dels noms de les classes originals per a l'informe
class_names = label_encoder.classes_

print("\n" + "="*50)
print("             📊 Avaluació del Model SVC")
print("="*50)

# Informe de Classificació (mètriques clau)
print("\nInforme de Classificació (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred, target_names=class_names))

# Matriu de Confusió
print("\nMatriu de Confusió:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)