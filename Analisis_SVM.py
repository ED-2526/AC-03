import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve, learning_curve
import os

from carrega_dades import split_datos_3s,split_datos_30s   

# ======================================================
# CONFIGURACIÓ
# ======================================================
RANDOM_STATE = 42
SAVE_DIR = "Plots/Justificacion_Parametros_SVC_30s"
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================================================
# FUNCIONS DE PLOT
# ======================================================
def plot_single_validation_curve(model, X, y, param_name, param_range, title, xlabel, SAVE_PATH):
    print(f"   ⚙️  Generando curva de validación para: {param_name}...")

    train_scores, test_scores = validation_curve(
        model,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        error_score=np.nan          # 🔥 CLAVE: no detener ejecución
    )

    train_mean = np.nanmean(train_scores, axis=1)
    test_mean = np.nanmean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_mean, label='Train', marker='o')
    plt.plot(param_range, test_mean, label='Validation', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    out = os.path.join(SAVE_PATH, f"VC_{param_name}.png")
    plt.savefig(out)
    plt.close()
    print(f"   ✅ Guardado: {out}")


def plot_final_learning_curve(model, X, y, title, SAVE_PATH):
    print("   📈 Generant Curva de Aprenentatge (Learning Curve)...")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
        error_score=np.nan    # 🔥 EVITA L’ERROR DE “1 sola classe”
    )

    train_mean = np.nanmean(train_scores, axis=1)
    test_mean = np.nanmean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label="Train", marker="o")
    plt.plot(train_sizes, test_mean, label="Validation", marker="o")
    plt.title(title)
    plt.xlabel("Número de mostres d'entrenament")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    out = os.path.join(SAVE_PATH, "Learning_Curve_Final.png")
    plt.savefig(out)
    plt.close()
    print(f"   ✅ Guardado: {out}")

# ======================================================
# MAIN
# ======================================================
def main():

    print("\n--- 🔬 LABORATORI D'ANÀLISI SVC ---")

    # -------------------------
    # 1. Carregar i preprocesar dades utilitzant *el teu* split_datos_3s
    # -------------------------
    X_train, X_test, y_train, y_test, label_encoder, scaler = split_datos_3s(
        random_state=RANDOM_STATE
    )

    # -------------------------
    # 2. Model base
    # -------------------------
    base_model = SVC(
        #C=1
        #gamma = 0.005
        kernel='rbf',
        probability=True,
        random_state=RANDOM_STATE
    )

    # -------------------------
    # 3. VALIDATION CURVE — C
    # -------------------------
    plot_single_validation_curve(
        base_model,
        X_train,
        y_train,
        param_name="C",
        param_range=[0.1, 0.5, 1.0, 2.0, 5.0,10.0],
        title="Impacte del paràmetre C en SVC",
        xlabel="C",
        SAVE_PATH=SAVE_DIR
    )

    # -------------------------
    # 4. VALIDATION CURVE — gamma 
    # -------------------------
    plot_single_validation_curve(
        base_model,
        X_train,
        y_train,
        param_name="gamma",
        param_range=[0.0005, 0.001, 0.005, 0.01, 0.05],
        title="Impacte del paràmetre gamma en SVC",
        xlabel="gamma",
        SAVE_PATH=SAVE_DIR
    )

    # -------------------------
    # 5. Model final triat manualment (ajusta si vols)
    # -------------------------
    final_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma=0.005,
        probability=True,
        random_state=RANDOM_STATE
    )

    # -------------------------
    # 6. Learning Curve final
    # -------------------------
    plot_final_learning_curve(
        final_model,
        X_train,
        y_train,
        title="Curva d'Aprenentatge SVC (model final)",
        SAVE_PATH=SAVE_DIR
    )

    print(f"\n✅ Anàlisi SVC completat. Gràfiques en: {SAVE_DIR}")


if __name__ == "__main__":
    main()
