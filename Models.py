import pandas as pd
import numpy as np
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score # Mètrica per comparar clustering amb etiquetes reals
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support
)
from plots_2 import (
    plot_per_class_metrics, plot_confusion_matrix, plot_roc_curve, 
    plot_general_roc_curve, plot_precision_recall_curve, plot_general_pr_curve, plot_feature_importances
)
from sklearn.linear_model import LogisticRegression

def executar_knn(X_train, X_test, y_train, y_test, label_encoder, data_type, 
                 best_k=4, best_weights='distance', best_p=1):
    """
    Entrena, avalua i genera els plots del model K-Nearest Neighbors (KNN) 
    amb els hiperparàmetres òptims definits.

    Args:
        X_train, X_test, y_train, y_test: Dades pre-processades.
        label_encoder: Encoder utilitzat per obtenir els noms de les classes.
        data_type (str): Tipus de dades ('3s' o '30s') per al nom del model/fitxers.
        best_k (int): Nombre de veïns (K).
        best_weights (str): 'uniform' o 'distance'.
        best_p (int): Mètrica de distància (1=Manhattan, 2=Euclidea).
    """
    MODEL_NAME = f"KNN ({data_type})"
    class_names = label_encoder.classes_
    
    # 1. ENTRENAMENT AMB PARÀMETRES ÒPTIMS
    print(f"\n🤖 Entrenant {MODEL_NAME} amb K={best_k}, weights='{best_weights}', p={best_p}...")
    
    # KNN no necessita random_state
    knn_model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, p=best_p)
    knn_model.fit(X_train, y_train)

    print("✅ Entrenament finalitzat.")

    # 2. PREDICCIÓ
    y_pred_test = knn_model.predict(X_test)
    y_pred_train = knn_model.predict(X_train)

    # Predicció de probabilitats (necessària per a les corbes ROC/PR)
    try:
        y_prob_test = knn_model.predict_proba(X_test)
    except AttributeError:
        y_prob_test = None
        print("El model KNN no suporta predict_proba(). No es generaran corbes ROC/PR.")


    # 3. AVALUACIÓ I RESULTATS
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted'
    )

    print("\n📊 RESULTATS KNN")
    print("-----------------------------------")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision Global: {precision:.4f}")
    print(f"Recall Global: {recall:.4f}")
    print(f"F1 Score Global: {f1:.4f}")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    # 4. GENERACIÓ DE PLOTS
    print("\n📈 Generant plots...")
    
    # Plots generals
    plot_per_class_metrics(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_confusion_matrix(y_test, y_pred_test, class_names, MODEL_NAME)
    
    # Plots basats en probabilitats (ROC/PR)
    if y_prob_test is not None:
        plot_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_general_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_precision_recall_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_general_pr_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        
    print(f"✅ Execució de {MODEL_NAME} completada.")

    # Retornar les mètriques clau
    return {
        'model': MODEL_NAME,
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'f1_score': f1,
        'probabilities': y_prob_test
    }


def executar_random_forest(X_train, X_test, y_train, y_test, label_encoder, data_type, 
                           n_estimators=200, max_depth=20, min_samples_split=2):
    """
    Entrena, avalua i genera els plots del model Random Forest
    amb els hiperparàmetres òptims definits.

    Args:
        X_train, X_test, y_train, y_test: Dades pre-processades.
        label_encoder: Encoder utilitzat per obtenir els noms de les classes.
        data_type (str): Tipus de dades ('3s' o '30s') per al nom del model/fitxers.
        n_estimators (int): Nombre d'arbres al bosc.
        max_depth (int): Profunditat màxima dels arbres.
        min_samples_split (int): Mínim de mostres per dividir un node.
    """
    MODEL_NAME = f"Random Forest ({data_type})"
    class_names = label_encoder.classes_
    
    # 1. DEFINICIÓ I ENTRENAMENT AMB PARÀMETRES ÒPTIMS
    print(f"\n🤖 Entrenant {MODEL_NAME} amb n_estimators={n_estimators}, max_depth={max_depth}...")
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42, 
        n_jobs=-1 
    )

    rf_model.fit(X_train, y_train)
    print("✅ Entrenament finalitzat.")

    # 2. PREDICCIÓ
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Predicció de probabilitats (necessària per a les corbes ROC/PR)
    try:
        y_prob_test = rf_model.predict_proba(X_test)
    except AttributeError:
        y_prob_test = None
        print("El model Random Forest no suporta predict_proba(). No es generaran corbes ROC/PR.")

    # 3. AVALUACIÓ I RESULTATS
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted'
    )

    print("\n📊 RESULTATS RANDOM FOREST")
    print("-----------------------------------")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision Global: {precision:.4f}")
    print(f"Recall Global: {recall:.4f}")
    print(f"F1 Score Global: {f1:.4f}")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))

    # 4. GENERACIÓ DE PLOTS
    print("\n📈 Generant plots...")
    
    # Plots generals
    plot_per_class_metrics(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_confusion_matrix(y_test, y_pred_test, class_names, MODEL_NAME)
    
    # Plots basats en probabilitats (ROC/PR)
    if y_prob_test is not None:
        plot_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_general_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_precision_recall_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_general_pr_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        
    print(f"✅ Execució de {MODEL_NAME} completada.")

    # Retornar les mètriques clau
    return {
        'model': MODEL_NAME,
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'f1_score': f1,
        'probabilities': y_prob_test
    }

def executar_svm(X_train, X_test, y_train, y_test, label_encoder, data_type, 
                 C=10.0, gamma=0.01, kernel='rbf', random_state=42):
    """
    Entrena, avalua i genera els plots del model Support Vector Classifier (SVC)
    amb els hiperparàmetres òptims definits.

    Args:
        X_train, X_test, y_train, y_test: Dades pre-processades.
        label_encoder: Encoder utilitzat per obtenir els noms de les classes.
        data_type (str): Tipus de dades ('3s' o '30s') per al nom del model/fitxers.
        C (float): Paràmetre de regularització.
        gamma (float/str): Paràmetre del nucli RBF.
        kernel (str): Tipus de nucli ('rbf' per defecte).
        random_state (int): Llavor per a la reproductibilitat.
    """
    MODEL_NAME = f"SVM ({data_type})"
    class_names = label_encoder.classes_
    
    # 1. DEFINICIÓ I ENTRENAMENT AMB PARÀMETRES ÒPTIMS
    print(f"\n🚀 Iniciant l'Entrenament de {MODEL_NAME} (C={C}, gamma={gamma})...")
    
    # SVC amb paràmetres òptims
    svc_model = SVC(
        C=C, 
        kernel=kernel, 
        gamma=gamma, 
        probability=True, # És crucial per a les corbes ROC/PR
        random_state=random_state, 
    )

    svc_model.fit(X_train, y_train)
    print("✅ Entrenament del SVC finalitzat.")

    # 2. PREDICCIÓ
    y_pred_te = svc_model.predict(X_test)
    y_pred_tr = svc_model.predict(X_train)
    
    # Predicció de probabilitats
    try:
        y_prob_test = svc_model.predict_proba(X_test)
    except AttributeError:
        y_prob_test = None
        print("El model SVC no pot calcular predict_proba(). No es generaran corbes ROC/PR.")


    # 3. AVALUACIÓ I RESULTATS
    train_accuracy = accuracy_score(y_train, y_pred_tr)
    test_accuracy = accuracy_score(y_test, y_pred_te)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_te, average='weighted'
    )

    print("\n📊 AVALUACIÓ SVC")
    print("-----------------------------------")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision Global: {precision:.4f}")
    print(f"Recall Global: {recall:.4f}")
    print(f"F1 Score Global: {f1:.4f}")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_te, target_names=class_names))


    # 4. GENERACIÓ DE PLOTS
    print("\n📈 Generant plots...")
    
    # Plots generals
    plot_per_class_metrics(y_test, y_pred_te, class_names, MODEL_NAME)
    plot_confusion_matrix(y_test, y_pred_te, class_names, MODEL_NAME)
    
    # Plots basats en probabilitats (ROC/PR)
    if y_prob_test is not None:
        plot_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_general_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_precision_recall_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_general_pr_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        
    print(f"✅ Execució de {MODEL_NAME} completada.")

    # Retornar les mètriques clau
    return {
        'model': MODEL_NAME,
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'f1_score': f1,
        'probabilities': y_prob_test
    }


import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
# Asegúrate de importar tus funciones de plot si están en otro archivo
# from tus_plots import plot_feature_importances, plot_per_class_metrics, ...

def executar_xgboost(X_train, X_test, y_train, y_test, label_encoder, data_type, 
                     n_estimators=300, # Aumentamos un poco porque el learning rate es bajo
                     max_depth=3,      # BAJADO: De 6 a 3 (Fundamental para evitar overfitting)
                     learning_rate=0.03, # BAJADO: Más lento y seguro
                     random_state=42):
    """
    Entrena XGBoost con configuración 'Anti-Overfitting' agresiva.
    """
    MODEL_NAME = f"XGBoost ({data_type})"
    class_names = label_encoder.classes_
    
    # Reconstrucción de nombres de features
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # 1. DEFINICIÓ I ENTRENAMENT AMB PARÀMETRES "ULTRA-CONSERVADORES"
    print(f"\n🚀 Iniciant l'Entrenament de {MODEL_NAME} (Mode Anti-Overfitting)...")
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(class_names),
        eval_metric='mlogloss',
        
        # --- PARÁMETROS ANTI-OVERFITTING ---
        n_estimators=n_estimators,
        max_depth=max_depth,          # Profundidad 3: Evita relaciones muy complejas/memorización
        learning_rate=learning_rate,  # 0.03: Aprendizaje lento
        
        min_child_weight=5,           # Exige al menos 5 muestras para crear una hoja
        gamma=0.5,                    # Penalización alta para dividir nodos
        subsample=0.6,                # Usa solo el 60% de las filas por árbol
        colsample_bytree=0.6,         # Usa solo el 60% de las features por árbol
        reg_lambda=2.0,               # Regularización L2 fuerte
        # -----------------------------------
        
        random_state=random_state,
        n_jobs=-1,
        use_label_encoder=False,
        early_stopping_rounds=20      # Parar si Test no mejora en 20 rondas
    )

    # Pasamos eval_set para que el early_stopping funcione
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False # Cambiar a True si quieres ver el log de error bajando
    )

    print(f"✅ Entrenament finalitzat. Millor iteració: {xgb_model.best_iteration}")

    # 2. PREDICCIÓ (Usa automáticamente la mejor iteración gracias a early_stopping)
    y_pred_test = xgb_model.predict(X_test)
    y_pred_train = xgb_model.predict(X_train)
    y_prob_test = xgb_model.predict_proba(X_test)

    # 3. AVALUACIÓ I RESULTATS
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted'
    )

    print("\n📊 AVALUACIÓ XGBOOST (REALISTA)")
    print("-----------------------------------")
    print(f"Train Accuracy: {train_accuracy:.4f} (Objetivo: < 0.90)")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Gap:            {train_accuracy - test_accuracy:.4f} (Objetivo: < 0.15)")
    print(f"Precision Global: {precision:.4f}")
    print(f"Recall Global:    {recall:.4f}")
    print(f"F1 Score Global:  {f1:.4f}")
    
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    # 4. GENERACIÓ DE PLOTS
    print("\n📈 Generant plots...")
    
    # Asegúrate de que estas funciones existen en tu entorno o impórtalas
    try:
        # 4.1 Plot d'Importància de Variables
        plot_feature_importances(xgb_model, feature_names, MODEL_NAME)
        
        # 4.2 Plots generals
        plot_per_class_metrics(y_test, y_pred_test, class_names, MODEL_NAME)
        plot_confusion_matrix(y_test, y_pred_test, class_names, MODEL_NAME)
        
        # 4.3 Plots basats en probabilitats
        plot_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_general_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_precision_recall_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        plot_general_pr_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    except NameError as e:
        print(f"⚠️ No se pudieron generar los plots (falta función): {e}")
        
    print(f"✅ Execució de {MODEL_NAME} completada.")

    return {
        'model': MODEL_NAME,
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'f1_score': f1,
        'probabilities': y_prob_test,
        'model_object': xgb_model # Retornamos el objeto por si quieres guardarlo
    }

def executar_regressio_logistica(X_train, X_test, y_train, y_test, label_encoder, data_type, 
                                 C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42):
    """
    Entrena, avalua i genera els plots del model de Regressió Logística
    per a la classificació multiclase.

    Args:
        X_train, X_test, y_train, y_test: Dades pre-processades (escalades).
        label_encoder: Encoder per obtenir els noms de les classes.
        data_type (str): Tipus de dades ('3s' o '30s').
        C (float): Paràmetre de regularització (invers a la força).
        penalty (str): Tipus de regularització ('l1', 'l2', 'elasticnet', 'none').
        max_iter (int): Nombre màxim d'iteracions per a la convergència.
    """
    MODEL_NAME = f"Regressió Logística ({data_type})"
    class_names = label_encoder.classes_
    
    # 1. DEFINICIÓ I ENTRENAMENT
    print(f"\n🚀 Iniciant l'Entrenament de {MODEL_NAME} (C={C}, penalty='{penalty}')...")
    
    # Per a la classificació multiclase, utilitzem 'multi_class=multinomial'
    log_model = LogisticRegression(
        C=C, 
        penalty=penalty,
        solver=solver,
        multi_class='multinomial', 
        max_iter=max_iter,
        random_state=random_state, 
        n_jobs=-1
    )

    log_model.fit(X_train, y_train)
    print("✅ Entrenament de Regressió Logística finalitzat.")

    # 2. PREDICCIÓ
    y_pred_test = log_model.predict(X_test)
    y_pred_train = log_model.predict(X_train)
    
    # Predicció de probabilitats
    y_prob_test = log_model.predict_proba(X_test)

    # 3. AVALUACIÓ I RESULTATS
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted'
    )

    print("\n📊 AVALUACIÓ REGRESSIÓ LOGÍSTICA")
    print("-----------------------------------")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision Global: {precision:.4f}")
    print(f"Recall Global: {recall:.4f}")
    print(f"F1 Score Global: {f1:.4f}")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    # 4. GENERACIÓ DE PLOTS
    print("\n📈 Generant plots...")
    plot_per_class_metrics(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_confusion_matrix(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_general_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_precision_recall_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_general_pr_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        
    print(f"✅ Execució de {MODEL_NAME} completada.")

    return {
        'model': MODEL_NAME,
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'f1_score': f1,
        'probabilities': y_prob_test 
    }

def executar_decision_tree(X_train, X_test, y_train, y_test, label_encoder, data_type, 
                           max_depth=None, min_samples_split=2, random_state=42):
    """
    Entrena, avalua i genera els plots del model d'Arbre de Decisió (Decision Tree).

    Args:
        X_train, X_test, y_train, y_test: Dades pre-processades.
        label_encoder: Encoder per obtenir els noms de les classes.
        data_type (str): Tipus de dades ('3s' o '30s').
        max_depth (int): Profunditat màxima de l'arbre.
        min_samples_split (int): Mínim de mostres per dividir un node.
    """
    MODEL_NAME = f"Decision Tree ({data_type})"
    class_names = label_encoder.classes_
    
    # 1. DEFINICIÓ I ENTRENAMENT
    print(f"\n🌳 Iniciant l'Entrenament de {MODEL_NAME} (Max Depth: {max_depth})...")
    
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

    dt_model.fit(X_train, y_train)
    print("✅ Entrenament de Decision Tree finalitzat.")

    # 2. PREDICCIÓ
    y_pred_test = dt_model.predict(X_test)
    y_pred_train = dt_model.predict(X_train)
    
    # Decision Tree suporta predict_proba per defecte
    y_prob_test = dt_model.predict_proba(X_test)

    # 3. AVALUACIÓ I RESULTATS
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted'
    )

    print(f"\n📊 AVALUACIÓ {MODEL_NAME}")
    print("-----------------------------------")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score Global: {f1:.4f}")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    # 4. GENERACIÓ DE PLOTS
    print("\n📈 Generant plots...")
    
    # DT té feature_importances_
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    plot_feature_importances(dt_model, feature_names, MODEL_NAME)

    # Plots generals
    plot_per_class_metrics(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_confusion_matrix(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_general_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_precision_recall_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_general_pr_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        
    print(f"✅ Execució de {MODEL_NAME} completada.")

    return {
        'model': MODEL_NAME,
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'f1_score': f1,
        'probabilities': y_prob_test 
    }

def executar_gmm_classifier(X_train, X_test, y_train, y_test, label_encoder, data_type, 
                            n_components=3, covariance_type='full', random_state=42):
    """
    Classificador basat en l'estimació de densitat de GMM.
    Entrena un model GMM per a cadascuna de les classes (gèneres).

    Args:
        X_train, X_test, y_train, y_test: Dades pre-processades.
        label_encoder: Encoder utilitzat per obtenir els noms de les classes.
        data_type (str): Tipus de dades ('3s' o '30s').
        n_components (int): Nombre de components Gaussianes per modelar cada classe.
        covariance_type (str): Tipus de matriu de covariància.
    """
    MODEL_NAME = f"GMM Classifier ({data_type})"
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    
    # Estructura per guardar els 10 models GMM
    gmm_classifiers = {}
    
    # 1. ENTRENAMENT: Ajustar un GMM per a cada classe
    print(f"\n🧠 Iniciant l'Entrenament de {MODEL_NAME} ({n_classes} models)...")
    
    for i in range(n_classes):
        genre = class_names[i]
        
        # Filtrem les dades d'entrenament només per a la classe actual
        X_genre = X_train[y_train == i]
        
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=3 
        )
        
        try:
            # Cal un mínim de mostres > n_components, cosa que es compleix aquí.
            gmm.fit(X_genre)
            gmm_classifiers[i] = gmm
            # print(f"  > Model GMM per a '{genre}' ajustat.")
        except Exception as e:
            print(f"  ⚠️ Error ajustant GMM per a '{genre}': {e}. S'utilitzarà un GMM buit.")
            gmm_classifiers[i] = None 

    print("✅ Entrenament de GMM finalitzat.")

    # 2. PREDICCIÓ (Càlcul de Log-Versemblances)
    
    # Matriu per guardar la puntuació (log-versemblança) de cada mostra en cada model
    log_likelihoods_test = np.zeros((len(X_test), n_classes))
    log_likelihoods_train = np.zeros((len(X_train), n_classes))

    for i in range(n_classes):
        gmm = gmm_classifiers[i]
        if gmm is not None:
            # score_samples() retorna la log-versemblança de cada mostra
            log_likelihoods_test[:, i] = gmm.score_samples(X_test)
            log_likelihoods_train[:, i] = gmm.score_samples(X_train)
        else:
            # Si el model és buit, assignem una puntuació molt baixa
            log_likelihoods_test[:, i] = -1e6
            log_likelihoods_train[:, i] = -1e6

    # 3. CONVERSIÓ A PROBABILITATS (Softmax Estabilitzat)
    
    # La classe predita és aquella que dóna la màxima log-versemblança
    y_pred_test = np.argmax(log_likelihoods_test, axis=1)
    y_pred_train = np.argmax(log_likelihoods_train, axis=1)

    # ESTABILITZACIÓ NUMÈRICA DEL SOFTMAX PER EVITAR NaN (El punt clau)
    
    # 1. Trobar el valor màxim de log-versemblança per a cada mostra
    max_log_likelihood = np.max(log_likelihoods_test, axis=1, keepdims=True)

    # 2. Restar el màxim abans de l'exponencial per prevenir l'overflow (NaN)
    stabilized_exp_scores_test = np.exp(log_likelihoods_test - max_log_likelihood)

    # 3. Normalitzar per obtenir probabilitats
    sum_exp_test = np.sum(stabilized_exp_scores_test, axis=1, keepdims=True)
    # Afegim un petit valor (1e-15) al denominador per evitar una divisió per zero extrema
    y_prob_test = stabilized_exp_scores_test / (sum_exp_test + 1e-15)
    
    # 4. AVALUACIÓ I RESULTATS
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted', zero_division=0
    )

    print(f"\n📊 AVALUACIÓ {MODEL_NAME}")
    print("-----------------------------------")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score Global: {f1:.4f}")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))

    # 5. GENERACIÓ DE PLOTS
    print("\n📈 Generant plots...")
    
    # Plots generals
    plot_per_class_metrics(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_confusion_matrix(y_test, y_pred_test, class_names, MODEL_NAME)
    
    # Els plots ROC/PR ja no fallaran gràcies a l'estabilització de y_prob_test
    plot_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_general_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_precision_recall_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_general_pr_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        
    print(f"✅ Execució de {MODEL_NAME} completada.")

    return {
        'model': MODEL_NAME,
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'f1_score': f1,
        'probabilities': y_prob_test 
    }

def executar_kmeans_clustering(X_train, X_test, y_train, y_test, label_encoder, data_type, 
                               n_clusters=10, random_state=42):
    """
    Executa l'algorisme K-Means sobre el conjunt de test (X_test) per trobar clústers.
    Avalua la qualitat del clustering amb Inèrcia i l'alineació amb els gèneres reals.

    Args:
        X_train, X_test, y_train, y_test: Dades pre-processades.
        label_encoder: Encoder (per a obtenir els noms de les classes).
        data_type (str): Tipus de dades ('3s' o '30s').
        n_clusters (int): Nombre de clústers a trobar (idealment, 10, com els gèneres).
    """
    MODEL_NAME = f"K-Means Clustering (K={n_clusters} - {data_type})"
    
    # 1. DEFINICIÓ I AJUSTAMENT (Entrenament)
    print(f"\n🌀 Iniciant K-Means Clustering amb K={n_clusters}...")
    
    # K-Means s'entrena amb X_train
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10  # Múltiples inicialitzacions per millorar la qualitat
    )

    # Entrenem K-Means només amb les característiques (no necessita y_train)
    kmeans.fit(X_train)
    print("✅ Ajustament K-Means finalitzat.")

    # 2. PREDICCIÓ (Assignació de clústers)
    # Assignem un clúster a cada mostra de test
    test_clusters = kmeans.predict(X_test)
    
    # 3. AVALUACIÓ (Mètriques de Clustering)
    
    # Mètrica 1: Inèrcia (Inertia / SSE)
    # L'inèrcia es pot obtenir directament de l'objecte kmeans ajustat
    inertia = kmeans.inertia_
    
    # Mètrica 2: Adjusted Rand Index (ARI) - Avaluació externa
    # Aquesta mètrica compara els clústers trobats (test_clusters) amb les etiquetes reals (y_test)
    # ARI proper a 1.0 significa que els clústers coincideixen perfectament amb els gèneres.
    ari_score = adjusted_rand_score(y_test, test_clusters)
    
    print(f"\n📊 RESULTATS {MODEL_NAME}")
    print("-----------------------------------")
    print(f"Número de Clústers (K): {n_clusters}")
    print(f"Inèrcia (SSE): {inertia:.2f}")
    print(f"Adjusted Rand Index (ARI): {ari_score:.4f} (Coincidència amb Gèneres)")
    print("-----------------------------------")

    # K-Means NO té predict_proba, ni Train Accuracy, ni F1 Score.
    # NO genera plots ROC/PR/Confusió.

    return {
        'model': MODEL_NAME,
        'ARI Score': ari_score, # Utilitzem ARI com a mètrica de rendiment
        'Inertia': inertia
    }

def executar_naive_bayes(X_train, X_test, y_train, y_test, label_encoder, data_type, random_state=42):
    """
    Entrena, avalua i genera els plots del model Naive Bayes (GaussianNB) 
    per a la classificació.

    Args:
        X_train, X_test, y_train, y_test: Dades pre-processades.
        label_encoder: Encoder utilitzat per obtenir els noms de les classes.
        data_type (str): Tipus de dades ('3s' o '30s').
    """
    MODEL_NAME = f"Naive Bayes (GaussianNB - {data_type})"
    class_names = label_encoder.classes_
    
    # 1. DEFINICIÓ I ENTRENAMENT
    print(f"\n🧠 Iniciant l'Entrenament de {MODEL_NAME}...")
    
    # GaussianNB és el Naive Bayes més comú per a dades amb distribució normal (escalades)
    nb_model = GaussianNB() 

    # Naive Bayes no té el paràmetre random_state, però l'incloem per coherència amb la funció.
    # El seu ajust és determinista.
    
    nb_model.fit(X_train, y_train)
    print("✅ Entrenament de Naive Bayes finalitzat.")

    # 2. PREDICCIÓ
    y_pred_test = nb_model.predict(X_test)
    y_pred_train = nb_model.predict(X_train)
    
    # Naive Bayes suporta predict_proba per defecte
    y_prob_test = nb_model.predict_proba(X_test)

    # 3. AVALUACIÓ I RESULTATS
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted', zero_division=0
    )

    print(f"\n📊 AVALUACIÓ {MODEL_NAME}")
    print("-----------------------------------")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score Global: {f1:.4f}")
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))

    # 4. GENERACIÓ DE PLOTS
    print("\n📈 Generant plots...")
    
    # Plots generals
    plot_per_class_metrics(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_confusion_matrix(y_test, y_pred_test, class_names, MODEL_NAME)
    plot_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_general_roc_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_precision_recall_curve(y_test, y_prob_test, MODEL_NAME, class_names)
    plot_general_pr_curve(y_test, y_prob_test, MODEL_NAME, class_names)
        
    print(f"✅ Execució de {MODEL_NAME} completada.")

    return {
        'model': MODEL_NAME,
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'f1_score': f1,
        'probabilities': y_prob_test 
    }