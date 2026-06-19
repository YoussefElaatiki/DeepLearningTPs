"""
TP1 — Exercice 2 : Comparaison des fonctions d'activation
==========================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Compare quatre fonctions d'activation (ReLU, Tanh, Sigmoid, ELU) sur un
problème de classification binaire synthétique.

Usage :
    python ex2_mlp_activation_functions.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Reproductibilité ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Visualisation des fonctions d'activation ────────────────────────────────

def plot_activation_functions():
    """Trace les courbes des fonctions d'activation courantes."""
    x = np.linspace(-4, 4, 300)
    activations = {
        "ReLU":    np.maximum(0, x),
        "Tanh":    np.tanh(x),
        "Sigmoid": 1 / (1 + np.exp(-x)),
        "ELU":     np.where(x >= 0, x, np.exp(x) - 1),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, (name, vals) in zip(axes, activations.items()):
        ax.plot(x, vals, linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Fonctions d'activation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "activation_functions.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/activation_functions.png")


# ─── Construction & entraînement ─────────────────────────────────────────────

def build_mlp(activation, input_dim=2):
    """MLP à deux couches cachées avec l'activation spécifiée."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation=activation),
        tf.keras.layers.Dense(16, activation=activation),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name=f"MLP_{activation}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def run_comparison():
    """Entraîne un MLP avec chaque activation et compare les résultats."""
    # Jeu de données : deux lunes (non-linéairement séparables)
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=SEED)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    activations = ["relu", "tanh", "sigmoid", "elu"]
    histories = {}
    final_results = {}

    for act in activations:
        print(f"  Entraînement avec activation = {act} ...")
        model = build_mlp(act)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=32, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True)]
        )
        histories[act] = history
        val_acc = max(history.history["val_accuracy"])
        final_results[act] = val_acc
        print(f"    → Meilleure val_accuracy : {val_acc:.4f}")

    return histories, final_results, X, y


def plot_comparison(histories, final_results):
    """Trace les courbes de perte et de précision pour toutes les activations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for act, history in histories.items():
        axes[0].plot(history.history["val_loss"], label=act)
        axes[1].plot(history.history["val_accuracy"], label=act)

    axes[0].set_title("Perte de validation (val_loss)")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Précision de validation (val_accuracy)")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Comparaison des fonctions d'activation", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "activation_comparison_curves.png"),
                dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/activation_comparison_curves.png")

    # Bar chart des précisions finales
    fig, ax = plt.subplots(figsize=(7, 4))
    acts = list(final_results.keys())
    accs = [final_results[a] for a in acts]
    bars = ax.bar(acts, accs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Meilleure val_accuracy")
    ax.set_title("Comparaison finale des activations")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "activation_comparison_bar.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/activation_comparison_bar.png")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP1 — Exercice 2 : Fonctions d'activation")
    print("=" * 60)

    # 1. Visualiser les fonctions d'activation
    print("\n[1] Visualisation des fonctions d'activation")
    plot_activation_functions()

    # 2. Comparaison sur données synthétiques
    print("\n[2] Comparaison sur jeu de données 'deux lunes'")
    histories, final_results, X, y = run_comparison()

    print("\n[3] Génération des graphiques de comparaison")
    plot_comparison(histories, final_results)

    print("\n─── Observations ─────────────────────────────────────")
    print("• ReLU et ELU convergent plus vite grâce à leur gradient constant > 0.")
    print("• Sigmoid souffre du vanishing gradient : courbe de perte plus lente.")
    print("• Tanh est centré en 0 (meilleure symétrie), souvent proche de ReLU.")
    print("• ELU évite les 'neurones morts' en permettant des valeurs négatives.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
