"""
TP1 — Exercice 3 : Impact de la profondeur du réseau
=====================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Étudie l'effet du nombre de couches cachées (1 à 4) sur la précision
de classification et le risque de sur-apprentissage.

Usage :
    python ex3_hidden_layers_depth.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Reproductibilité ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Construction du modèle ───────────────────────────────────────────────────

def build_deep_model(n_hidden_layers, units_per_layer=16):
    """Construit un MLP avec n_hidden_layers couches cachées de taille fixe.

    Args:
        n_hidden_layers: nombre de couches cachées (1 à N).
        units_per_layer: nombre de neurones par couche cachée.

    Returns:
        Modèle Keras compilé.
    """
    layers = [tf.keras.layers.Input(shape=(2,))]
    for i in range(n_hidden_layers):
        layers.append(tf.keras.layers.Dense(units_per_layer, activation="relu",
                                            name=f"hidden_{i+1}"))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid", name="output"))

    model = tf.keras.Sequential(layers,
                                name=f"MLP_{n_hidden_layers}HL")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def count_parameters(model):
    """Retourne le nombre total de paramètres entraînables."""
    return model.count_params()


# ─── Expérience principale ───────────────────────────────────────────────────

def run_depth_experiment():
    """Entraîne des MLP de 1 à 4 couches cachées et compare les résultats."""

    # Petit jeu de données (200 points) pour mettre en évidence le sur-apprentissage
    X, y = make_circles(n_samples=300, noise=0.15, factor=0.4,
                        random_state=SEED)
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y.astype(float), test_size=0.3, random_state=SEED)

    configs = [1, 2, 3, 4]
    histories = {}
    param_counts = {}
    final_train_acc = {}
    final_val_acc = {}

    for n in configs:
        print(f"\n  Réseau avec {n} couche(s) cachée(s) :")
        model = build_deep_model(n, units_per_layer=16)
        model.summary()
        param_counts[n] = count_parameters(model)
        print(f"    → Nombre de paramètres : {param_counts[n]}")

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=300, batch_size=16, verbose=0
        )
        histories[n] = history
        final_train_acc[n] = history.history["accuracy"][-1]
        final_val_acc[n] = history.history["val_accuracy"][-1]
        gap = final_train_acc[n] - final_val_acc[n]
        print(f"    → Train acc : {final_train_acc[n]:.4f} | "
              f"Val acc : {final_val_acc[n]:.4f} | "
              f"Gap (sur-apprentissage) : {gap:.4f}")

    return histories, param_counts, final_train_acc, final_val_acc, X, y


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_depth_comparison(histories, param_counts, final_train, final_val):
    """Génère les graphiques de comparaison de profondeur."""

    configs = list(histories.keys())

    # Courbes de précision de validation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for n in configs:
        axes[0].plot(histories[n].history["val_loss"],
                     label=f"{n} CC")
        axes[1].plot(histories[n].history["val_accuracy"],
                     label=f"{n} CC")

    axes[0].set_title("Val Loss vs Profondeur")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend(title="Couches cachées")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Val Accuracy vs Profondeur")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(title="Couches cachées")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Impact de la profondeur du réseau", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "depth_comparison_curves.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/depth_comparison_curves.png")

    # Bar chart : train vs val accuracy (sur-apprentissage)
    x_pos = np.arange(len(configs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x_pos - width / 2,
                   [final_train[n] for n in configs],
                   width, label="Train accuracy", color="#4C72B0")
    bars2 = ax.bar(x_pos + width / 2,
                   [final_val[n] for n in configs],
                   width, label="Val accuracy", color="#DD8452")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{n} CC\n({param_counts[n]} params)" for n in configs])
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Val Accuracy — Impact de la profondeur")
    ax.legend()

    # Annoter le gap
    for i, n in enumerate(configs):
        gap = final_train[n] - final_val[n]
        ax.text(x_pos[i], 1.01, f"Δ={gap:.2f}", ha="center",
                fontsize=9, color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "depth_overfit_bar.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/depth_overfit_bar.png")

    # Tableau récapitulatif
    print("\n  ┌─────────────┬────────────┬────────────┬────────────┬──────────┐")
    print("  │  Profondeur │  Params    │ Train Acc  │  Val Acc   │   Gap    │")
    print("  ├─────────────┼────────────┼────────────┼────────────┼──────────┤")
    for n in configs:
        gap = final_train[n] - final_val[n]
        print(f"  │  {n} CC      │  {param_counts[n]:>8d}  │  {final_train[n]:.4f}    │  {final_val[n]:.4f}    │  {gap:.4f}  │")
    print("  └─────────────┴────────────┴────────────┴────────────┴──────────┘")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP1 — Exercice 3 : Impact de la profondeur du réseau")
    print("=" * 60)

    histories, param_counts, final_train, final_val, X, y = run_depth_experiment()
    plot_depth_comparison(histories, param_counts, final_train, final_val)

    print("\n─── Observations ─────────────────────────────────────")
    print("• Augmenter la profondeur augmente la capacité d'apprentissage.")
    print("• Sur un petit jeu de données, un réseau trop profond sur-apprend")
    print("  (grand écart train/val accuracy).")
    print("• Le nombre de paramètres croît avec la profondeur.")
    print("• Il faut trouver un compromis entre capacité et généralisation")
    print("  (utiliser Dropout, Early Stopping, régularisation L2).")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
