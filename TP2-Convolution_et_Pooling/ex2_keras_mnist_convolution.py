"""
TP2 — Exercice 2 : Convolution avec Keras sur MNIST + Feature Maps
===================================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Construit un CNN simple sur MNIST, l'entraîne, et visualise :
  - Les courbes perte/précision.
  - Les feature maps produites par la 1ère couche Conv2D.
  - Les poids (filtres) appris par chaque couche de convolution.

Usage :
    python ex2_keras_mnist_convolution.py
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

# ─── Chargement et préparation des données ───────────────────────────────────

def load_mnist():
    """Charge MNIST, normalise [0,1], reshape pour CNN (H,W,C), one-hot labels."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Ajouter dimension canal (grayscale → 1 canal)
    X_train = X_train.astype(np.float32)[..., np.newaxis] / 255.0
    X_test  = X_test.astype(np.float32)[..., np.newaxis]  / 255.0
    # Labels entiers (sparse_categorical_crossentropy)
    return (X_train, y_train), (X_test, y_test)


# ─── Architecture CNN ────────────────────────────────────────────────────────

def build_cnn():
    """CNN simple pour MNIST :
       Conv2D(32) → ReLU → MaxPool → Conv2D(64) → ReLU → MaxPool → Dense(128) → Dense(10)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1), name="input"),

        # Bloc 1
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
                               padding="same", name="conv1"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        # Bloc 2
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu",
                               padding="same", name="conv2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        # Classifier
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(128, activation="relu", name="dense1"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(10, activation="softmax", name="output"),
    ], name="SimpleCNN_MNIST")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Entraînement ────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_test, y_test):
    """Entraîne le modèle avec EarlyStopping et retourne l'historique."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")
    return history


# ─── Visualisation des courbes ───────────────────────────────────────────────

def plot_training_curves(history):
    """Trace et sauvegarde les courbes perte/précision d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Époque")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Époque")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("CNN MNIST — Courbes d'entraînement", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cnn_training_curves.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/cnn_training_curves.png")


# ─── Visualisation des feature maps ─────────────────────────────────────────

def visualize_feature_maps(model, sample_image, layer_name, filename):
    """Visualise les feature maps d'une couche donnée pour une image exemple.

    Args:
        model       : modèle Keras entraîné.
        sample_image: image de forme (1, H, W, C).
        layer_name  : nom de la couche Conv2D à visualiser.
        filename    : nom du fichier de sortie.
    """
    # Sous-modèle qui retourne la sortie de la couche demandée
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    feature_maps = feature_model.predict(sample_image, verbose=0)
    # feature_maps shape : (1, H_out, W_out, n_filters)

    n_filters = feature_maps.shape[-1]
    n_cols = 8
    n_rows = int(np.ceil(n_filters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2, n_rows * 2))
    axes = axes.ravel()

    for i in range(n_filters):
        axes[i].imshow(feature_maps[0, :, :, i], cmap="viridis")
        axes[i].set_title(f"F{i}", fontsize=7)
        axes[i].axis("off")

    # Masquer les axes vides
    for i in range(n_filters, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Feature Maps — couche «{layer_name}»",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


# ─── Visualisation des filtres appris ────────────────────────────────────────

def visualize_learned_filters(model, layer_name, filename):
    """Visualise les poids (filtres) de la 1ère couche Conv2D.

    Args:
        model     : modèle Keras entraîné.
        layer_name: nom de la couche Conv2D.
        filename  : nom du fichier de sortie.
    """
    weights = model.get_layer(layer_name).get_weights()[0]
    # weights shape : (kH, kW, in_channels, n_filters)
    n_filters = weights.shape[-1]
    n_cols = 8
    n_rows = int(np.ceil(n_filters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.8, n_rows * 1.8))
    axes = axes.ravel()

    for i in range(n_filters):
        f = weights[:, :, 0, i]      # canal 0 (grayscale)
        vmax = np.abs(f).max() or 1
        axes[i].imshow(f, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"F{i}", fontsize=7)
        axes[i].axis("off")

    for i in range(n_filters, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Filtres appris — couche «{layer_name}»",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP2 — Exercice 2 : CNN sur MNIST + Feature Maps (Keras)")
    print("=" * 60)

    # 1. Charger MNIST
    print("\n[1] Chargement de MNIST")
    (X_train, y_train), (X_test, y_test) = load_mnist()
    print(f"    Train : {X_train.shape}, Test : {X_test.shape}")

    # 2. Construire le CNN
    print("\n[2] Architecture du CNN")
    model = build_cnn()
    model.summary()

    # 3. Entraîner
    print("\n[3] Entraînement")
    history = train_model(model, X_train, y_train, X_test, y_test)

    # 4. Courbes d'entraînement
    print("\n[4] Courbes d'entraînement")
    plot_training_curves(history)

    # 5. Visualiser les feature maps après conv1 et conv2
    print("\n[5] Feature maps")
    sample = X_test[0:1]   # 1ère image de test, shape (1, 28, 28, 1)
    visualize_feature_maps(model, sample, "conv1",
                           "feature_maps_conv1.png")
    visualize_feature_maps(model, sample, "conv2",
                           "feature_maps_conv2.png")

    # 6. Visualiser les filtres appris par conv1
    print("\n[6] Filtres appris (conv1)")
    visualize_learned_filters(model, "conv1", "learned_filters_conv1.png")

    # 7. Afficher l'image test utilisée
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(X_test[0, :, :, 0], cmap="gray")
    ax.set_title(f"Image test — Label : {y_test[0]}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_test_image.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/sample_test_image.png")

    print("\n─── Observations ─────────────────────────────────────")
    print("• La 1ère couche Conv2D apprend des filtres de bas niveau (contours).")
    print("• La 2ème couche détecte des patterns plus abstraits (combinaisons).")
    print("• Les feature maps montrent les régions de l'image 'activées' par chaque filtre.")
    print("• Le CNN atteint ~99% de précision sur MNIST avec peu de paramètres.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
