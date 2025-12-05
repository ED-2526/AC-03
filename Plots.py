import matplotlib.pyplot as plt
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