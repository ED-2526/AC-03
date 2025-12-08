import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle

def plot_per_class_metrics(y_test, y_pred, class_names, model_name):
    
    output_dir = os.path.join("Plots", model_name)
    os.makedirs(output_dir, exist_ok=True)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, labels=range(len(class_names)), zero_division=0, average=None)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25

    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1_score, width, label='F1-score')

    plt.xticks(x, class_names, rotation=45)
    plt.ylabel("Score")
    plt.title(f"Per-Class Metrics ({model_name})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"per_class_metrics_{model_name.lower()}.png"))
    plt.close()
    print(f"Per-class metrics plot saved for model {model_name} in {output_dir}")

def plot_confusion_matrix(y_test, y_pred, class_names, model_name):
    output_dir = os.path.join("Plots", model_name)
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name.lower()}.png"))
    plt.close()
    print(f"Confusion matrix saved for model {model_name} in {output_dir}")

def plot_roc_curve(y_test, y_pred, model_name, class_names):
    # NOTA: y_pred aquí deben ser PROBABILIDADES (predict_proba), no etiquetas
    output_dir = os.path.join("Plots", model_name)
    os.makedirs(output_dir, exist_ok=True)

    y_test_binarized = label_binarize(y_test, classes=range(len(class_names)))
    n_classes = y_test_binarized.shape[1]
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    # Itera sobre cada clase calculando su curva individual contra el resto
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc:0.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Linia diagonal == Azar
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"roc_curve_{model_name.lower()}.png"))
    plt.close()
    print(f"ROC curve saved for model {model_name} in {output_dir}") 

def plot_precision_recall_curve(y_test, y_pred, model_name, class_names):
    # NOTA: y_pred aquí deben ser PROBABILIDADES (predict_proba)
    output_dir = os.path.join("Plots", model_name)
    os.makedirs(output_dir, exist_ok=True)

    y_test_binarized = label_binarize(y_test, classes=range(len(class_names)))
    n_classes = y_test_binarized.shape[1]

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])

    # Calcula Precisión vs Recall para cada clase.
    for i, color in zip(range(n_classes), colors):
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_pred[:, i])
        avg_precision = average_precision_score(y_test_binarized[:, i], y_pred[:, i])            
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'Precision-Recall curve of class {class_names[i]} (AP = {avg_precision:0.2f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"precision_recall_curve_{model_name.lower()}.png"))
    plt.close()
    print(f"Precision-Recall curve saved for model {model_name} in {output_dir}")


def plot_general_roc_curve(y_test, y_pred, model_name, class_names):
    """
    Genera una única curva ROC (Micro-average) que representa el rendimiento 
    global del modelo. Ideal para comparar entre diferentes modelos.
    """
    output_dir = os.path.join("Plots", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Binarizar para convertir etiquetas en formato one-hot
    y_test_binarized = label_binarize(y_test, classes=range(len(class_names)))
    
    # --- CÁLCULO MICRO-AVERAGE (GLOBAL) ---
    # .ravel() aplana los arrays para tratarlos como un único conjunto binario gigante
    fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_pred.ravel())
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'General ROC (Micro-avg) (AUC = {roc_auc:0.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Línea de azar
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'General ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"general_roc_{model_name.lower()}.png")
    plt.savefig(filename)
    plt.close()
    print(f"General ROC saved: {filename}")

def plot_general_pr_curve(y_test, y_pred, model_name, class_names):
    """
    Genera una única curva PR (Micro-average) para evaluación global.
    """
    output_dir = os.path.join("Plots", model_name)
    os.makedirs(output_dir, exist_ok=True)

    y_test_binarized = label_binarize(y_test, classes=range(len(class_names)))
    
    precision, recall, _ = precision_recall_curve(y_test_binarized.ravel(), y_pred.ravel())
    avg_precision = average_precision_score(y_test_binarized, y_pred, average="micro")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'General PR (Micro-avg) (AP = {avg_precision:0.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'General Precision-Recall Curve - {model_name}')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    filename = os.path.join(output_dir, f"general_pr_{model_name.lower()}.png")
    plt.savefig(filename)
    plt.close()
    print(f"General PR saved: {filename}")

def plot_class_distribution(y_train, y_test, label_encoder, filename='class_distribution_bar_chart.png'):
    """
    Genera i desa un gràfic de barres mostrant la distribució de classes (gèneres) 
    en tot el dataset (Train + Test).
    """
    
    # 1. Unim les etiquetes (codificades)
    y_total_encoded = np.concatenate([y_train, y_test])

    # 2. Descodifiquem a noms de classe
    y_total_names = label_encoder.inverse_transform(y_total_encoded)

    # 3. Contem la freqüència
    y_labels = pd.Series(y_total_names)
    class_counts = y_labels.value_counts().sort_index()
    
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Crear el gràfic de barres
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='bar', color='darkgreen')

    plt.title('Distribució de Gèneres Musicals (Total Dataset)', fontsize=14)
    plt.xlabel('Gènere Musical', fontsize=12)
    plt.ylabel('Número de Mostres', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Posa els valors sobre les barres
    for index, value in enumerate(class_counts):
        plt.text(index, value, f'{value}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Guardar
    output_file_path = os.path.join(output_dir, filename)
    plt.savefig(output_file_path)
    plt.close()
    print(f"✅ Gràfic de distribució de classes desat a '{output_file_path}'")


def plot_feature_importances(model, feature_names, model_name, top_n=20):
    """
    Dibuixa les top N variables més importants d'un model basat en arbres (RF, XGBoost).
    
    Args:
        model: Model entrenat (ha de tenir l'atribut .feature_importances_).
        feature_names (list): Noms de les columnes (features) originals.
        model_name (str): Nom del model per al títol i fitxer.
        top_n (int): Nombre de variables a mostrar.
    """
    
    output_dir = os.path.join("Plots", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="rocket")
    
    plt.title(f"Top {top_n} Variables Més Importants ({model_name})", fontsize=16)
    plt.xlabel("Importància (Score)")
    plt.ylabel("Variables")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"feature_importances_{model_name.lower()}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Gràfic d'importància de variables desat a: {plot_path}")

def plot_comparative_roc(y_test, probabilities_dict, model_names, class_names, data_type):
    """
    Genera una única corba ROC (Micro-average) per a tots els models comparats.

    Args:
        y_test (np.array): Etiquetes reals codificades (numèriques).
        probabilities_dict (dict): Diccionari {nom_model: y_prob_test}.
        model_names (list): Llista de noms dels models.
        class_names (np.array): Noms de les classes.
        data_type (str): Tipus de dades ('3s' o '30s').
    """
    
    output_dir = os.path.join("Plots", "Comparativa_Global")
    os.makedirs(output_dir, exist_ok=True)

    # Binaritzar per convertir etiquetes en format one-hot (necessari per a micro-average)
    y_test_binarized = label_binarize(y_test, classes=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    
    # 1. Iterar sobre cada model i dibuixar la seva corba ROC
    for name in model_names:
        y_prob = probabilities_dict[name]
        
        # Càlcul Micro-average ROC (Global)
        fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, 
                 label=f'{name} (AUC = {roc_auc:0.3f})')
        
    # 2. Afegir la línia de l'atzar
    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Línia diagonal == Azar

    # 3. Configuració del gràfic
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Comparativa ROC (Micro-Avg) - Dataset {data_type.upper()}')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"comparativa_roc_{data_type.lower()}.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Gràfic comparatiu ROC desat a: {filename}")