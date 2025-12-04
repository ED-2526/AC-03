import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from carrega_dades import *
from Plots import *

MODEL = "SVM"

### 1. Carrega i Preprocessament de les Dades
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
    C=10.0, 
    kernel='rbf', 
    gamma=0.01, 
    probability=True, # Necessari per obtenir probabilitats en la predicció
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

try:
    y_prob_test = svc_model.predict_proba(X_test)
except AttributeError:
    y_prob_test = None
    print("El modelo no soporta predict_proba(). No se podrán generar curvas ROC/PR.")


print("\n" + "="*50)
print("             📊 Avaluació del Model SVC")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("="*50)

# Informe de Classificació (mètriques clau)
print("\nInforme de Classificació (Precision, Recall, F1-Score):")
print(classification_report(y_test, y_pred, target_names=class_names))

# Matriu de Confusió
print("\nMatriu de Confusió:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Generació de Gràfics per a l'Avaluació del Model

# Gràfic de Mètriques per Classe
plot_per_class_metrics(y_test, y_pred, class_names, MODEL)

# Matriu de Confusió
plot_confusion_matrix(y_test, y_pred, class_names, MODEL)

# Corba ROC
if y_prob_test is not None:
    plot_roc_curve(y_test, y_prob_test, MODEL, class_names)

# Corba PR
if y_prob_test is not None:
    plot_precision_recall_curve(y_test, y_prob_test, MODEL, class_names)