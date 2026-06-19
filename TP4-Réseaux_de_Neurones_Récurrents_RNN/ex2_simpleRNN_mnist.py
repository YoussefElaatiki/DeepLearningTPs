"""
TP4 — Exercice 2 : SimpleRNN pour la Classification MNIST
==========================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Traite les images MNIST comme des séquences temporelles :
  - Chaque image 28×28 est vue comme 28 pas de temps de 28 features.
  - Compare SimpleRNN vs MLP Dense vs CNN sur la même tâche.

Usage :
    python ex2_simpleRNN_mnist.py
"""

import os
import time
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

def load_mnist_sequential():
    """Charge MNIST dans le format séquentiel (28 pas × 28 features).

    Retourne :
        X : (N, 28, 28)  — 28 pas de temps, chaque pas = vecteur de 28 pixels
        y : labels entiers (0–9)
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32) / 255.0   # (60000, 28, 28)
    X_test  = X_test.astype(np.float32)  / 255.0   # (10000, 28, 28)
    return (X_train, y_train), (X_test, y_test)


def load_mnist_flat():
    """Charge MNIST aplati pour le MLP."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test  = X_test.reshape(-1, 784).astype(np.float32)  / 255.0
    return (X_train, y_train), (X_test, y_test)


def load_mnist_cnn():
    """Charge MNIST pour le CNN (ajout dimension canal)."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32)[..., np.newaxis] / 255.0
    X_test  = X_test.astype(np.float32)[..., np.newaxis]  / 255.0
    return (X_train, y_train), (X_test, y_test)


# ─── Architectures ───────────────────────────────────────────────────────────

def build_simple_rnn_model():
    """SimpleRNN pour MNIST séquentiel.
       Input  : (28, 28)  — 28 pas, 28 features par pas
       Output : (10,) — probabilités des 10 classes
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.SimpleRNN(128, activation="tanh",
                                  return_sequences=False, name="rnn"),
        tf.keras.layers.Dense(64, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(10, activation="softmax", name="output"),
    ], name="SimpleRNN_MNIST")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_mlp_model():
    """MLP Dense pour MNIST (entrée aplatie 784-dim)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="MLP_MNIST")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_cnn_model():
    """CNN simple pour MNIST."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="CNN_MNIST")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Entraînement ────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=128):
    """Entraîne le modèle et mesure le temps d'entraînement."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True),
    ]
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1
    )
    elapsed = time.time() - t0
    return history, elapsed


# ─── Visualisation ───────────────────────────────────────────────────────────

def visualize_sequential_processing(X_test, y_test, n=4):
    """Illustre comment une image MNIST est vue comme une séquence de lignes."""
    fig, axes = plt.subplots(n, 2, figsize=(9, n * 3))

    for i in range(n):
        img = X_test[i]  # (28, 28)
        label = y_test[i]

        # Image originale
        axes[i, 0].imshow(img, cmap="gray", aspect="auto")
        axes[i, 0].set_title(f"Image originale (label={label})", fontsize=9)
        axes[i, 0].set_xlabel("Pixel (feature)")
        axes[i, 0].set_ylabel("Ligne (pas de temps)")

        # Séquence de lignes (heatmap)
        axes[i, 1].imshow(img.T, cmap="hot", aspect="auto")
        axes[i, 1].set_title("Vu comme séquence (transposé)", fontsize=9)
        axes[i, 1].set_xlabel("Ligne (pas de temps)")
        axes[i, 1].set_ylabel("Pixel (feature)")

    plt.suptitle("MNIST comme séquence temporelle (28 pas × 28 features)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mnist_as_sequence.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/mnist_as_sequence.png")


def plot_comparison_histories(histories_dict):
    """Trace les courbes de validation accuracy de tous les modèles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"SimpleRNN": "#4C72B0", "MLP": "#DD8452", "CNN": "#55A868"}
    for name, hist in histories_dict.items():
        color = colors.get(name, "gray")
        axes[0].plot(hist.history["val_loss"],
                     label=name, color=color, linewidth=2)
        axes[1].plot(hist.history["val_accuracy"],
                     label=name, color=color, linewidth=2)

    axes[0].set_title("Val Loss — Comparaison")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Val Accuracy — Comparaison")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Comparaison SimpleRNN / MLP / CNN sur MNIST",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_mnist.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/model_comparison_mnist.png")


def print_results_table(results):
    """Affiche un tableau récapitulatif des résultats."""
    print("\n  ┌────────────┬────────────┬────────────┬────────────┬──────────────┐")
    print("  │ Modèle     │  Paramètres│ Test Acc   │ Test Loss  │ Temps (s)    │")
    print("  ├────────────┼────────────┼────────────┼────────────┼──────────────┤")
    for name, res in results.items():
        print(f"  │ {name:<10s} │ {res['params']:>10,d} │ {res['test_acc']:.4f}    │ "
              f"{res['test_loss']:.4f}    │ {res['time']:>12.1f} │")
    print("  └────────────┴────────────┴────────────┴────────────┴──────────────┘")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP4 — Exercice 2 : SimpleRNN pour MNIST")
    print("=" * 60)

    # 1. Charger données
    print("\n[1] Chargement de MNIST")
    (X_seq_train, y_train), (X_seq_test, y_test) = load_mnist_sequential()
    (X_flat_train, _), (X_flat_test, _) = load_mnist_flat()
    (X_cnn_train, _),  (X_cnn_test, _)  = load_mnist_cnn()

    # Séparation train/val (10%)
    split = int(0.9 * len(X_seq_train))
    X_seq_val,  y_val  = X_seq_train[split:],  y_train[split:]
    X_flat_val         = X_flat_train[split:]
    X_cnn_val          = X_cnn_train[split:]
    X_seq_tr,   y_tr   = X_seq_train[:split],  y_train[:split]
    X_flat_tr          = X_flat_train[:split]
    X_cnn_tr           = X_cnn_train[:split]

    # 2. Visualiser le concept séquentiel
    print("\n[2] Visualisation MNIST comme séquence")
    visualize_sequential_processing(X_seq_test, y_test)

    # 3. Entraîner SimpleRNN
    print("\n[3] SimpleRNN")
    model_rnn = build_simple_rnn_model()
    model_rnn.summary()
    hist_rnn, t_rnn = train_model(model_rnn, X_seq_tr, y_tr, X_seq_val, y_val)
    loss_rnn, acc_rnn = model_rnn.evaluate(X_seq_test, y_test, verbose=0)
    print(f"  SimpleRNN → Test acc : {acc_rnn:.4f} | Time : {t_rnn:.1f}s")

    # 4. Entraîner MLP
    print("\n[4] MLP Dense")
    model_mlp = build_mlp_model()
    model_mlp.summary()
    hist_mlp, t_mlp = train_model(model_mlp, X_flat_tr, y_tr, X_flat_val, y_val)
    loss_mlp, acc_mlp = model_mlp.evaluate(X_flat_test, y_test, verbose=0)
    print(f"  MLP → Test acc : {acc_mlp:.4f} | Time : {t_mlp:.1f}s")

    # 5. Entraîner CNN
    print("\n[5] CNN")
    model_cnn = build_cnn_model()
    model_cnn.summary()
    hist_cnn, t_cnn = train_model(model_cnn, X_cnn_tr, y_tr, X_cnn_val, y_val)
    loss_cnn, acc_cnn = model_cnn.evaluate(X_cnn_test, y_test, verbose=0)
    print(f"  CNN → Test acc : {acc_cnn:.4f} | Time : {t_cnn:.1f}s")

    # 6. Comparaison
    print("\n[6] Comparaison des modèles")
    histories = {"SimpleRNN": hist_rnn, "MLP": hist_mlp, "CNN": hist_cnn}
    plot_comparison_histories(histories)

    results = {
        "SimpleRNN": {"params": model_rnn.count_params(),
                      "test_acc": acc_rnn, "test_loss": loss_rnn, "time": t_rnn},
        "MLP":       {"params": model_mlp.count_params(),
                      "test_acc": acc_mlp, "test_loss": loss_mlp, "time": t_mlp},
        "CNN":       {"params": model_cnn.count_params(),
                      "test_acc": acc_cnn, "test_loss": loss_cnn, "time": t_cnn},
    }
    print_results_table(results)

    print("\n─── Observations ─────────────────────────────────────")
    print("• SimpleRNN peut classer MNIST de manière séquentielle (ligne par ligne).")
    print("• CNN est généralement le plus précis grâce à son invariance locale.")
    print("• MLP est rapide mais ne capture pas la structure spatiale/temporelle.")
    print("• SimpleRNN est plus lent que MLP car il traite séquentiellement.")
    print("• Un LSTM ou GRU améliorerait les performances du modèle récurrent.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
