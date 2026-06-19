"""
TP3 — Exercice 1 : Implémentation LeNet-5 sur MNIST
====================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Implémente l'architecture LeNet-5 originale (LeCun et al., 1998) et
l'entraîne sur MNIST. Analyse :
  - Le nombre de paramètres par couche
  - La précision atteinte
  - Le risque de sur-apprentissage
  - L'effet de Dropout et de la régularisation

Usage :
    python ex1_lenet5_mnist.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

# ─── Reproductibilité ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Chargement des données ───────────────────────────────────────────────────

def load_mnist_lenet():
    """Charge MNIST et redimensionne de 28×28 à 32×32 (taille originale LeNet-5)."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normaliser
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0

    # Ajouter dimension canal
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]

    # Padding 28→32 (LeNet-5 original prend 32×32)
    X_train = tf.image.resize_with_pad(X_train, 32, 32).numpy()
    X_test  = tf.image.resize_with_pad(X_test,  32, 32).numpy()

    print(f"  Train shape : {X_train.shape}, Test shape : {X_test.shape}")
    return (X_train, y_train), (X_test, y_test)


# ─── Architecture LeNet-5 ─────────────────────────────────────────────────────

def build_lenet5(use_dropout=False, dropout_rate=0.3):
    """Construit LeNet-5 avec ou sans Dropout.

    Architecture originale (Tanh + AvgPooling) :
      Input (32×32×1)
      → Conv2D(6, 5×5, tanh) → AvgPool(2×2)
      → Conv2D(16, 5×5, tanh) → AvgPool(2×2)
      → Flatten
      → Dense(120, tanh)
      → Dense(84, tanh)
      → Dense(10, softmax)

    Args:
        use_dropout  : si True, ajoute des couches Dropout après les Dense.
        dropout_rate : taux de dropout.

    Returns:
        modèle Keras compilé.
    """
    layers = [
        tf.keras.layers.Input(shape=(32, 32, 1), name="input"),

        # C1 : 1ère couche de convolution — 6 filtres 5×5
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation="tanh",
                               padding="valid", name="C1"),
        # S2 : Sous-échantillonnage (Average Pooling 2×2)
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="S2"),

        # C3 : 2ème couche de convolution — 16 filtres 5×5
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation="tanh",
                               padding="valid", name="C3"),
        # S4 : Sous-échantillonnage (Average Pooling 2×2)
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="S4"),

        # Aplatir
        tf.keras.layers.Flatten(name="flatten"),

        # C5/F6 : Couches entièrement connectées
        tf.keras.layers.Dense(120, activation="tanh", name="C5"),
    ]
    if use_dropout:
        layers.append(tf.keras.layers.Dropout(dropout_rate, name="drop1"))

    layers.append(tf.keras.layers.Dense(84, activation="tanh", name="F6"))

    if use_dropout:
        layers.append(tf.keras.layers.Dropout(dropout_rate, name="drop2"))

    layers.append(tf.keras.layers.Dense(10, activation="softmax", name="output"))

    model = tf.keras.Sequential(layers,
                                name="LeNet5" + ("_Dropout" if use_dropout else ""))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Analyse des paramètres ───────────────────────────────────────────────────

def analyze_parameters(model):
    """Affiche un tableau détaillé du nombre de paramètres par couche."""
    print("\n  ┌─────────────────────┬────────────┬────────────┬─────────────────┐")
    print("  │ Couche              │ Shape sortie│  Paramètres│ Cumul            │")
    print("  ├─────────────────────┼────────────┼────────────┼─────────────────┤")
    cumul = 0
    for layer in model.layers:
        out_shape = str(layer.output_shape[1:])
        params = layer.count_params()
        cumul += params
        print(f"  │ {layer.name:<19s} │ {out_shape:<10s} │ {params:>10,d} │ {cumul:>15,d}  │")
    print("  └─────────────────────┴────────────┴────────────┴─────────────────┘")
    print(f"\n  Total paramètres entraînables : {model.count_params():,}")


# ─── Entraînement ────────────────────────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=128):
    """Entraîne le modèle et retourne l'historique."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_history(history, title, filename):
    """Trace les courbes perte/précision."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(history.history["loss"], label="Train loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train acc", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val acc", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


def compare_with_without_dropout(hist_no_drop, hist_drop):
    """Compare les deux courbes d'entraînement (avec / sans Dropout)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(hist_no_drop.history["train_loss"] if "train_loss" in hist_no_drop.history
                 else hist_no_drop.history["loss"],
                 label="Sans Dropout (train)", linestyle="--", color="#4C72B0")
    axes[0].plot(hist_no_drop.history["val_loss"],
                 label="Sans Dropout (val)", color="#4C72B0")
    axes[0].plot(hist_drop.history["loss"],
                 label="Avec Dropout (train)", linestyle="--", color="#DD8452")
    axes[0].plot(hist_drop.history["val_loss"],
                 label="Avec Dropout (val)", color="#DD8452")
    axes[0].set_title("Loss : Avec vs Sans Dropout")
    axes[0].set_xlabel("Époque")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist_no_drop.history["accuracy"],
                 label="Sans Dropout (train)", linestyle="--", color="#4C72B0")
    axes[1].plot(hist_no_drop.history["val_accuracy"],
                 label="Sans Dropout (val)", color="#4C72B0")
    axes[1].plot(hist_drop.history["accuracy"],
                 label="Avec Dropout (train)", linestyle="--", color="#DD8452")
    axes[1].plot(hist_drop.history["val_accuracy"],
                 label="Avec Dropout (val)", color="#DD8452")
    axes[1].set_title("Accuracy : Avec vs Sans Dropout")
    axes[1].set_xlabel("Époque")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("LeNet-5 sur MNIST — Effet du Dropout", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lenet5_dropout_comparison.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/lenet5_dropout_comparison.png")


def plot_confusion_matrix(model, X_test, y_test):
    """Affiche la matrice de confusion de LeNet-5 sur MNIST."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité terrain")
    ax.set_title("Matrice de confusion — LeNet-5 MNIST")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lenet5_confusion_matrix.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/lenet5_confusion_matrix.png")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP3 — Exercice 1 : LeNet-5 sur MNIST")
    print("=" * 60)

    # 1. Charger données
    print("\n[1] Chargement de MNIST (redimensionné 32×32)")
    (X_train, y_train), (X_test, y_test) = load_mnist_lenet()
    # Séparation train/val
    split = int(0.9 * len(X_train))
    X_val, y_val = X_train[split:], y_train[split:]
    X_tr,  y_tr  = X_train[:split], y_train[:split]

    # ── LeNet-5 sans Dropout ─────────────────────────────────────────────────
    print("\n[2] LeNet-5 sans Dropout")
    model_nodrop = build_lenet5(use_dropout=False)
    model_nodrop.summary()
    analyze_parameters(model_nodrop)
    hist_nodrop = train(model_nodrop, X_tr, y_tr, X_val, y_val, epochs=20)
    plot_history(hist_nodrop, "LeNet-5 MNIST (sans Dropout)",
                 "lenet5_nodrop_history.png")

    # ── LeNet-5 avec Dropout ─────────────────────────────────────────────────
    print("\n[3] LeNet-5 avec Dropout (0.3)")
    model_drop = build_lenet5(use_dropout=True, dropout_rate=0.3)
    hist_drop = train(model_drop, X_tr, y_tr, X_val, y_val, epochs=20)
    plot_history(hist_drop, "LeNet-5 MNIST (avec Dropout)",
                 "lenet5_drop_history.png")

    # ── Comparaison ──────────────────────────────────────────────────────────
    print("\n[4] Comparaison avec/sans Dropout")
    compare_with_without_dropout(hist_nodrop, hist_drop)

    # ── Évaluation finale ────────────────────────────────────────────────────
    print("\n[5] Évaluation sur le jeu de test")
    for name, model in [("Sans Dropout", model_nodrop), ("Avec Dropout", model_drop)]:
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"  {name:<20s} → Test loss : {loss:.4f} | Test accuracy : {acc:.4f}")

    # ── Matrice de confusion ─────────────────────────────────────────────────
    print("\n[6] Matrice de confusion (modèle avec Dropout)")
    try:
        import seaborn
        plot_confusion_matrix(model_drop, X_test, y_test)
    except ImportError:
        print("  (seaborn non disponible — matrice de confusion ignorée)")

    print("\n─── Observations ─────────────────────────────────────")
    print("• LeNet-5 atteint ~99% sur MNIST après quelques époques.")
    print("• Sans Dropout, la validation_loss peut remonter (sur-apprentissage).")
    print("• Avec Dropout (0.3), les courbes train/val sont plus proches.")
    print("• La plupart des paramètres (~97%) sont dans les couches Dense.")
    print("• Les couches Conv2D sont très efficaces avec peu de paramètres.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
