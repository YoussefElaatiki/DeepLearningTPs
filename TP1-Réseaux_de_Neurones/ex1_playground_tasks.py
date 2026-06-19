"""
TP1 — Exercice 1 : Tâches équivalentes TensorFlow Playground
=============================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Reproduit quatre tâches classiques du TensorFlow Playground :
  Tâche 1 : Classification linéaire (données séparables)
  Tâche 2 : Problème XOR (non-linéaire, 1 couche cachée suffit)
  Tâche 3 : Classification en spirale (réseau profond)
  Tâche 4 : Ingénierie des features (features dérivées)

Usage :
    python ex1_playground_tasks.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend (no display required)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Reproductibilité ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Utilitaires ─────────────────────────────────────────────────────────────

def plot_decision_boundary(model, X, y, title, filename, feature_fn=None):
    """Trace la frontière de décision d'un modèle binaire 2D."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if feature_fn is not None:
        grid = feature_fn(grid)
    Z = model.predict(grid, verbose=0).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.4,
                colors=["#AAAAFF", "#FFAAAA"])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr",
                         edgecolors="k", s=30)
    ax.set_title(title)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


def plot_history(history, title, filename):
    """Trace les courbes perte/précision d'un historique Keras."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Époque")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train acc")
    axes[1].plot(history.history["val_accuracy"], label="Val acc")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Époque")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


def build_model(hidden_layers, activation="relu"):
    """Construit un MLP avec une architecture donnée.

    Args:
        hidden_layers: liste d'entiers, nombre de neurones par couche cachée.
                       Ex : [8] = 1 couche de 8 neurones.
        activation: fonction d'activation pour les couches cachées.

    Returns:
        model compilé (Adam, binary_crossentropy).
    """
    model = tf.keras.Sequential(name="MLP")
    model.add(tf.keras.layers.Input(shape=(2,)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ─── Génération des jeux de données ──────────────────────────────────────────

def make_linear_data(n=400):
    """Données linéairement séparables (deux gaussiennes)."""
    rng = np.random.default_rng(SEED)
    X0 = rng.multivariate_normal([-1.5, -1.5], [[1, 0], [0, 1]], n // 2)
    X1 = rng.multivariate_normal([1.5,  1.5], [[1, 0], [0, 1]], n // 2)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)])
    return X, y


def make_xor_data(n=400):
    """Données XOR (4 quadrants alternés)."""
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-1, 1, (n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(float)
    return X, y


def make_spiral_data(n_per_class=200, noise=0.3):
    """Spirale à 2 classes (souvent utilisée dans TF Playground)."""
    def spiral_arm(n, delta):
        t = np.linspace(0, 4 * np.pi, n)
        r = t / (4 * np.pi)
        x = r * np.cos(t + delta) + np.random.randn(n) * noise * 0.2
        y = r * np.sin(t + delta) + np.random.randn(n) * noise * 0.2
        return np.c_[x, y]

    rng_state = np.random.RandomState(SEED)
    np.random.seed(SEED)
    X0 = spiral_arm(n_per_class, 0)
    X1 = spiral_arm(n_per_class, np.pi)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    return X, y


def make_circles_data(n=400, noise=0.1):
    """Deux cercles concentriques (make_circles de sklearn)."""
    X, y = make_circles(n_samples=n, noise=noise, factor=0.5,
                        random_state=SEED)
    return X, y.astype(float)


def engineer_features(X):
    """Crée des features dérivées : [x1, x2, x1², x2², x1·x2, sin(x1), sin(x2)]."""
    x1, x2 = X[:, 0:1], X[:, 1:2]
    return np.hstack([x1, x2,
                      x1 ** 2, x2 ** 2,
                      x1 * x2,
                      np.sin(x1), np.sin(x2)])


# ─── Tâche 1 : Classification linéaire ───────────────────────────────────────

def task1_linear():
    print("\n=== Tâche 1 : Classification linéaire ===")
    X, y = make_linear_data()
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    # Pas de couche cachée → frontière linéaire
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50, batch_size=32, verbose=0)

    val_acc = history.history["val_accuracy"][-1]
    print(f"  Précision validation : {val_acc:.4f}")

    plot_decision_boundary(model, X, y,
                           "Tâche 1 — Linéaire (0 couche cachée)",
                           "t1_linear_boundary.png")
    plot_history(history, "Tâche 1 — Linéaire", "t1_linear_history.png")
    # Observation : une frontière linéaire sépare parfaitement les deux classes.


# ─── Tâche 2 : XOR ───────────────────────────────────────────────────────────

def task2_xor():
    print("\n=== Tâche 2 : Problème XOR ===")
    X, y = make_xor_data()
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    results = {}
    for config_name, hidden in [("Linéaire (0 CC)", []),
                                  ("1 CC (8 neurones)", [8]),
                                  ("1 CC (16 neurones)", [16])]:
        model = build_model(hidden, activation="tanh")
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=200, batch_size=32, verbose=0)
        val_acc = history.history["val_accuracy"][-1]
        results[config_name] = val_acc
        print(f"  {config_name} → val_accuracy = {val_acc:.4f}")
        safe = config_name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_decision_boundary(model, X, y,
                               f"Tâche 2 — XOR — {config_name}",
                               f"t2_xor_{safe}.png")

    # Observation : modèle linéaire ~50 % (chance), 1 CC ≥ 90 %.
    print("  Observation : Un perceptron linéaire échoue sur XOR. "
          "Une couche cachée non-linéaire le résout.")


# ─── Tâche 3 : Spirale ───────────────────────────────────────────────────────

def task3_spiral():
    print("\n=== Tâche 3 : Classification en spirale ===")
    X, y = make_spiral_data(n_per_class=200)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    configs = {
        "1 CC (8)": [8],
        "2 CC (16-16)": [16, 16],
        "3 CC (32-16-8)": [32, 16, 8],
    }
    for name, hidden in configs.items():
        model = build_model(hidden, activation="relu")
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=300, batch_size=32, verbose=0)
        val_acc = history.history["val_accuracy"][-1]
        print(f"  {name} → val_accuracy = {val_acc:.4f}")
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plot_decision_boundary(model, X, y,
                               f"Tâche 3 — Spirale — {name}",
                               f"t3_spiral_{safe}.png")
        plot_history(history, f"Tâche 3 — {name}", f"t3_spiral_hist_{safe}.png")

    print("  Observation : Plus le réseau est profond, meilleure est "
          "la frontière de décision sur les spirales.")


# ─── Tâche 4 : Ingénierie des features ───────────────────────────────────────

def task4_feature_engineering():
    print("\n=== Tâche 4 : Ingénierie des features ===")
    # Utilise les données circulaires (non-linéairement séparables)
    X, y = make_circles_data(n=600, noise=0.1)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    # Modèle sans features dérivées
    model_raw = build_model([8], activation="relu")
    hist_raw = model_raw.fit(X_train, y_train,
                             validation_data=(X_val, y_val),
                             epochs=150, batch_size=32, verbose=0)
    acc_raw = hist_raw.history["val_accuracy"][-1]
    print(f"  Features brutes (x1, x2)    → val_accuracy = {acc_raw:.4f}")

    # Features dérivées
    X_fe = engineer_features(X)
    X_fe_train, X_fe_val = X_fe[: len(X_train)], X_fe[len(X_train):]

    # Reshape train/val after engineer (need to re-split properly)
    X_fe_all = engineer_features(X)
    X_fe_tr, X_fe_v, y_tr, y_v = train_test_split(
        X_fe_all, y, test_size=0.2, random_state=SEED)

    model_fe = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),   # 7 features dérivées
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model_fe.compile(optimizer="adam", loss="binary_crossentropy",
                     metrics=["accuracy"])
    hist_fe = model_fe.fit(X_fe_tr, y_tr,
                           validation_data=(X_fe_v, y_v),
                           epochs=150, batch_size=32, verbose=0)
    acc_fe = hist_fe.history["val_accuracy"][-1]
    print(f"  Features dérivées (7 dims)  → val_accuracy = {acc_fe:.4f}")
    print("  Observation : L'ingénierie des features améliore significativement "
          "la précision, même avec un réseau simple.")

    # Comparaison visuelle (frontière de décision sur X brut)
    plot_decision_boundary(model_raw, X, y,
                           "Tâche 4 — Features brutes (x1, x2)",
                           "t4_raw_features.png")
    plot_decision_boundary(model_fe, X, y,
                           "Tâche 4 — Features dérivées (7 dims)",
                           "t4_engineered_features.png",
                           feature_fn=engineer_features)

    # Comparaison courbes d'apprentissage
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist_raw.history["val_accuracy"],
            label="Features brutes", linestyle="--")
    ax.plot(hist_fe.history["val_accuracy"],
            label="Features dérivées", linestyle="-")
    ax.set_title("Tâche 4 — Comparaison val_accuracy")
    ax.set_xlabel("Époque")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "t4_feature_comparison.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/t4_feature_comparison.png")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP1 — Exercice 1 : TensorFlow Playground Tasks")
    print("=" * 60)
    task1_linear()
    task2_xor()
    task3_spiral()
    task4_feature_engineering()
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
