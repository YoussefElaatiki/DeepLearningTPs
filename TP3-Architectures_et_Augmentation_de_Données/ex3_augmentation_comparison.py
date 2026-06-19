"""
TP3 — Exercice 3 : Comparaison avec/sans Augmentation de Données sur CIFAR-10
==============================================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Entraîne le même modèle CNN sur CIFAR-10 :
  1. Sans augmentation  → sur-apprentissage visible
  2. Avec augmentation  → meilleure généralisation

Compare les courbes d'entraînement et les précisions finales.

Usage :
    python ex3_augmentation_comparison.py
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

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─── Chargement des données ───────────────────────────────────────────────────

def load_cifar10():
    """Charge et normalise CIFAR-10."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0
    y_train = y_train.flatten()
    y_test  = y_test.flatten()
    return (X_train, y_train), (X_test, y_test)


# ─── Architecture CNN pour CIFAR-10 ──────────────────────────────────────────

def build_cifar_cnn():
    """CNN modéré pour CIFAR-10 (même architecture pour les deux expériences).

    Architecture :
        Conv2D(32, 3×3, relu) + BN + MaxPool
        Conv2D(64, 3×3, relu) + BN + MaxPool
        Conv2D(128, 3×3, relu) + BN + MaxPool
        Flatten
        Dense(256, relu) + Dropout(0.4)
        Dense(10, softmax)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),

        # Bloc 1
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Bloc 2
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Bloc 3
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="CIFAR10_CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Entraînement sans augmentation ──────────────────────────────────────────

def train_without_augmentation(model, X_train, y_train, X_val, y_val,
                                epochs=40, batch_size=64):
    """Entraîne directement sur les données brutes."""
    print("  ▶ Entraînement SANS augmentation...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
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


# ─── Entraînement avec augmentation ──────────────────────────────────────────

def train_with_augmentation(model, X_train, y_train, X_val, y_val,
                             epochs=40, batch_size=64):
    """Entraîne avec ImageDataGenerator pour l'augmentation à la volée."""
    print("  ▶ Entraînement AVEC augmentation (ImageDataGenerator)...")

    # Générateur d'augmentation (seulement pour le train)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest"
    )
    datagen.fit(X_train)   # calcule les stats si besoin

    # Générateur de validation (pas d'augmentation)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_gen = datagen.flow(X_train, y_train, batch_size=batch_size,
                             seed=SEED)
    val_gen   = val_datagen.flow(X_val, y_val, batch_size=batch_size,
                                  seed=SEED, shuffle=False)

    steps_per_epoch = len(X_train) // batch_size

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─── Visualisation comparative ───────────────────────────────────────────────

def plot_comparison(hist_no_aug, hist_aug, test_results):
    """Génère les figures de comparaison avec/sans augmentation."""

    # ── Courbes de perte ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(hist_no_aug.history["loss"],
                 label="Sans aug (train)", linestyle="--", color="#4C72B0", linewidth=2)
    axes[0].plot(hist_no_aug.history["val_loss"],
                 label="Sans aug (val)", color="#4C72B0", linewidth=2)
    axes[0].plot(hist_aug.history["loss"],
                 label="Avec aug (train)", linestyle="--", color="#DD8452", linewidth=2)
    axes[0].plot(hist_aug.history["val_loss"],
                 label="Avec aug (val)", color="#DD8452", linewidth=2)
    axes[0].set_title("Loss : Avec vs Sans Augmentation")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist_no_aug.history["accuracy"],
                 label="Sans aug (train)", linestyle="--", color="#4C72B0", linewidth=2)
    axes[1].plot(hist_no_aug.history["val_accuracy"],
                 label="Sans aug (val)", color="#4C72B0", linewidth=2)
    axes[1].plot(hist_aug.history["accuracy"],
                 label="Avec aug (train)", linestyle="--", color="#DD8452", linewidth=2)
    axes[1].plot(hist_aug.history["val_accuracy"],
                 label="Avec aug (val)", color="#DD8452", linewidth=2)
    axes[1].set_title("Accuracy : Avec vs Sans Augmentation")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("CIFAR-10 — Comparaison avec/sans Augmentation de Données",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "augmentation_comparison_curves.png"),
                dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/augmentation_comparison_curves.png")

    # ── Résultats finaux en barres ────────────────────────────────────────────
    labels = ["Sans augmentation", "Avec augmentation"]
    train_accs = [
        max(hist_no_aug.history["accuracy"]),
        max(hist_aug.history["accuracy"]),
    ]
    val_accs = [
        max(hist_no_aug.history["val_accuracy"]),
        max(hist_aug.history["val_accuracy"]),
    ]
    test_accs = [test_results["sans_aug"], test_results["avec_aug"]]
    gaps = [tr - va for tr, va in zip(train_accs, val_accs)]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, train_accs, width, label="Train acc", color="#4C72B0")
    ax.bar(x,         val_accs,   width, label="Val acc",   color="#DD8452")
    ax.bar(x + width, test_accs,  width, label="Test acc",  color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparaison finale : Train / Val / Test Accuracy")
    ax.legend()

    # Annoter le gap (sur-apprentissage)
    for i, gap in enumerate(gaps):
        ax.text(x[i], 1.02, f"Δ={gap:.2f}", ha="center",
                fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "augmentation_comparison_bar.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/augmentation_comparison_bar.png")

    # Rapport console
    print("\n  ─── Résumé ──────────────────────────────────────────────")
    print(f"  {'Méthode':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Gap':<8}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for i, label in enumerate(labels):
        print(f"  {label:<25} {train_accs[i]:.4f}    {val_accs[i]:.4f}    "
              f"{test_accs[i]:.4f}    {gaps[i]:.4f}")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP3 — Exercice 3 : Comparaison Augmentation sur CIFAR-10")
    print("=" * 60)

    # 1. Charger données
    print("\n[1] Chargement de CIFAR-10")
    (X_train, y_train), (X_test, y_test) = load_cifar10()
    split = int(0.9 * len(X_train))
    X_val, y_val = X_train[split:], y_train[split:]
    X_tr,  y_tr  = X_train[:split], y_train[:split]
    print(f"    Train : {X_tr.shape} | Val : {X_val.shape} | Test : {X_test.shape}")

    # 2. Entraîner SANS augmentation
    print("\n[2] Modèle 1 — Sans Augmentation")
    model_no_aug = build_cifar_cnn()
    model_no_aug.summary()
    hist_no_aug = train_without_augmentation(
        model_no_aug, X_tr, y_tr, X_val, y_val, epochs=40)

    # 3. Entraîner AVEC augmentation (même architecture, reinit)
    print("\n[3] Modèle 2 — Avec Augmentation (ImageDataGenerator)")
    model_aug = build_cifar_cnn()
    hist_aug = train_with_augmentation(
        model_aug, X_tr, y_tr, X_val, y_val, epochs=40)

    # 4. Évaluation finale
    print("\n[4] Évaluation sur le jeu de test")
    _, acc_no_aug = model_no_aug.evaluate(X_test, y_test, verbose=0)
    _, acc_aug    = model_aug.evaluate(X_test,    y_test, verbose=0)
    print(f"  Sans augmentation → Test accuracy : {acc_no_aug:.4f}")
    print(f"  Avec augmentation → Test accuracy : {acc_aug:.4f}")
    test_results = {"sans_aug": acc_no_aug, "avec_aug": acc_aug}

    # 5. Comparaison visuelle
    print("\n[5] Génération des graphiques de comparaison")
    plot_comparison(hist_no_aug, hist_aug, test_results)

    print("\n─── Observations ─────────────────────────────────────")
    print("• Sans augmentation : grand écart train/val = sur-apprentissage.")
    print("• Avec augmentation : l'écart train/val est réduit (meilleure généralisation).")
    print("• L'augmentation augmente le temps d'entraînement par époque.")
    print("• La précision de test est généralement meilleure avec augmentation.")
    print("• L'augmentation est une forme de régularisation implicite.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
