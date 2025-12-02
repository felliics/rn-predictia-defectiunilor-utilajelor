import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Căi către seturi
TRAIN_PATH = os.path.join("data", "train", "train.csv")
VAL_PATH = os.path.join("data", "validation", "validation.csv")
TEST_PATH = os.path.join("data", "test", "test.csv")
MODEL_PATH = os.path.join("models", "nn_model.joblib")

FEATURE_COLS = ["temperature", "vibration", "runtime_hours", "noise_level"]
TARGET_COL = "status"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_and_train_model(X_train, y_train, X_val, y_val):
    # Rețea neuronală simplă: 2 straturi ascunse (16 și 8 neuroni)
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )

    print("[INFO] Încep antrenarea rețelei neuronale...")
    model.fit(X_train, y_train)
    print("[INFO] Antrenare terminată.")

    # Evaluare pe validation
    y_val_pred = model.predict(X_val)
    print("\n=== Rezultate pe setul de VALIDARE ===")
    print(classification_report(y_val, y_val_pred, digits=4))

    return model


def evaluate_on_test(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    print("\n=== Rezultate pe setul de TEST ===")
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Matrice de confuzie:")
    print(confusion_matrix(y_test, y_test_pred))
    plot_confusion_matrix(y_test, y_test_pred, "docs/plots/confusion_matrix_test.png")

def plot_confusion_matrix(y_true, y_pred, filename):
    os.makedirs("docs/plots", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["OK", "Defect"],
                yticklabels=["OK", "Defect"])
    plt.xlabel("Predicție")
    plt.ylabel("Adevăr")
    plt.title("Matrice de confuzie – set de test")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_loss_curve(model, filename):
    if not hasattr(model, "loss_curve_"):
        return
    os.makedirs("docs/plots", exist_ok=True)
    plt.figure()
    plt.plot(model.loss_curve_)
    plt.xlabel("Epocă")
    plt.ylabel("Loss")
    plt.title("Evoluția funcției de eroare (loss)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[INFO] Model salvat în: {MODEL_PATH}")


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    model = build_and_train_model(X_train, y_train, X_val, y_val)
    evaluate_on_test(model, X_test, y_test)
    save_model(model)
    plot_loss_curve(model, "docs/plots/loss_curve.png")

