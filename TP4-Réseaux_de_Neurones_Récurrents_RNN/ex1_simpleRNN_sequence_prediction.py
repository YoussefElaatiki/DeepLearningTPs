"""
TP4 — Exercice 1 : SimpleRNN pour la Prédiction de Séquences
=============================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Génère une séquence sinusoïdale et entraîne un SimpleRNN pour prédire
la prochaine valeur. Compare :
  - Différentes longueurs de séquence (seq_length = 10, 20, 50)
  - Activations tanh vs relu

Usage :
    python ex1_simpleRNN_sequence_prediction.py
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

# ─── Génération des données ───────────────────────────────────────────────────

def generate_sine_sequence(n_points=2000, noise_std=0.05, freq=1.0):
    """Génère une séquence sinusoïdale avec bruit gaussien.

    Args:
        n_points  : nombre de points dans la séquence.
        noise_std : écart-type du bruit gaussien ajouté.
        freq      : fréquence de la sinusoïde (en cycles par 2π).

    Returns:
        Tableau 1D numpy (n_points,) de valeurs en [-1, 1] environ.
    """
    t = np.linspace(0, 20 * np.pi, n_points)
    signal = np.sin(freq * t)
    noise  = np.random.normal(0, noise_std, n_points)
    return signal + noise


def create_sequences(data, seq_length):
    """Transforme un signal 1D en paires (entrée, cible) pour le RNN.

    Pour chaque position i, l'entrée est data[i : i+seq_length]
    et la cible est data[i+seq_length].

    Args:
        data      : tableau 1D.
        seq_length: longueur de la fenêtre glissante.

    Returns:
        X : (N, seq_length, 1), y : (N,)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(data[i + seq_length])
    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, T, 1)
    y = np.array(y, dtype=np.float32)                    # (N,)
    return X, y


def split_train_val_test(X, y, train_frac=0.7, val_frac=0.15):
    """Divise les données en train / val / test (pas de mélange pour séquences)."""
    n = len(X)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train: n_train + n_val], y[n_train: n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:], y[n_train + n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ─── Modèle SimpleRNN ────────────────────────────────────────────────────────

def build_rnn(seq_length, n_units=64, activation="tanh"):
    """SimpleRNN à 2 couches pour la régression (prédiction de valeur scalaire).

    Args:
        seq_length : longueur de la séquence d'entrée.
        n_units    : nombre de neurones cachés.
        activation : activation du SimpleRNN ('tanh' ou 'relu').

    Returns:
        Modèle Keras compilé (MSE, Adam).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, 1)),
        # 1ère couche RNN — retourne la séquence entière pour la 2ème couche
        tf.keras.layers.SimpleRNN(n_units, activation=activation,
                                  return_sequences=True,
                                  name="rnn1"),
        # 2ème couche RNN — retourne seulement l'état final
        tf.keras.layers.SimpleRNN(n_units // 2, activation=activation,
                                  return_sequences=False,
                                  name="rnn2"),
        tf.keras.layers.Dense(16, activation="relu", name="dense"),
        tf.keras.layers.Dense(1, name="output"),
    ], name=f"SimpleRNN_T{seq_length}_{activation}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"]
    )
    return model


# ─── Entraînement & évaluation ────────────────────────────────────────────────

def train_and_evaluate(model, train_data, val_data, test_data, epochs=30):
    """Entraîne et évalue un modèle RNN.

    Returns:
        (history, test_mse, test_mae)
    """
    X_train, y_train = train_data
    X_val,   y_val   = val_data
    X_test,  y_test  = test_data

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=64,
        callbacks=callbacks, verbose=0
    )
    test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
    return history, test_mse, test_mae


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_signal_and_predictions(data, model, seq_length, title, filename):
    """Trace le signal réel vs les prédictions du modèle."""
    X_all, y_all = create_sequences(data, seq_length)
    y_pred = model.predict(X_all, verbose=0).flatten()

    t_all = np.arange(seq_length, len(data))
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Signal complet avec prédictions
    axes[0].plot(data, label="Signal réel", alpha=0.7, linewidth=1.5)
    axes[0].plot(t_all, y_pred, label="Prédiction RNN",
                 alpha=0.8, linestyle="--", linewidth=1.5)
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("Valeur")

    # Résidus
    residuals = y_all - y_pred
    axes[1].plot(t_all, residuals, color="red", alpha=0.7, linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Résidus (Réel − Prédiction)")
    axes[1].set_xlabel("Pas de temps")
    axes[1].set_ylabel("Résidu")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


def plot_seq_length_comparison(results):
    """Compare les MSE / MAE pour différentes longueurs de séquence."""
    seq_lengths = list(results.keys())
    mses = [results[sl]["mse"] for sl in seq_lengths]
    maes = [results[sl]["mae"] for sl in seq_lengths]

    x = np.arange(len(seq_lengths))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(x, mses, color="#4C72B0")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"T={sl}" for sl in seq_lengths])
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("MSE vs Longueur de Séquence")
    for i, v in enumerate(mses):
        axes[0].text(i, v + 0.0002, f"{v:.4f}", ha="center", fontsize=9)

    axes[1].bar(x, maes, color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"T={sl}" for sl in seq_lengths])
    axes[1].set_ylabel("Test MAE")
    axes[1].set_title("MAE vs Longueur de Séquence")
    for i, v in enumerate(maes):
        axes[1].text(i, v + 0.0001, f"{v:.4f}", ha="center", fontsize=9)

    plt.suptitle("Impact de la longueur de séquence", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "seq_length_comparison.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/seq_length_comparison.png")


def plot_activation_comparison(results_act):
    """Compare tanh vs relu pour la prédiction."""
    activations = list(results_act.keys())
    mses = [results_act[a]["mse"] for a in activations]
    maes = [results_act[a]["mae"] for a in activations]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(activations, mses, color=["#4C72B0", "#DD8452"])
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("MSE : tanh vs relu")
    for i, v in enumerate(mses):
        axes[0].text(i, v + 0.0001, f"{v:.5f}", ha="center", fontsize=10)

    axes[1].bar(activations, maes, color=["#4C72B0", "#DD8452"])
    axes[1].set_ylabel("Test MAE")
    axes[1].set_title("MAE : tanh vs relu")
    for i, v in enumerate(maes):
        axes[1].text(i, v + 0.0001, f"{v:.5f}", ha="center", fontsize=10)

    plt.suptitle("Impact de la fonction d'activation (SimpleRNN, T=20)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "activation_comparison_rnn.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/activation_comparison_rnn.png")


def plot_training_curves_all(histories, labels, filename, metric="loss"):
    """Trace les courbes de perte pour plusieurs modèles sur le même graphe."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for hist, label in zip(histories, labels):
        ax.plot(hist.history[metric], label=f"{label} (train)", linewidth=1.5)
        ax.plot(hist.history[f"val_{metric}"], label=f"{label} (val)",
                linestyle="--", linewidth=1.5)
    ax.set_title(f"Comparaison {metric.upper()} — SimpleRNN")
    ax.set_xlabel("Époque")
    ax.set_ylabel(metric.upper())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP4 — Exercice 1 : SimpleRNN — Prédiction de Séquences")
    print("=" * 60)

    # 1. Générer le signal
    print("\n[1] Génération du signal sinusoïdal")
    data = generate_sine_sequence(n_points=3000, noise_std=0.05)
    print(f"  Signal shape : {data.shape}, min={data.min():.3f}, max={data.max():.3f}")

    # Visualiser le signal
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(data[:500], linewidth=1.5)
    ax.set_title("Signal sinusoïdal avec bruit (premiers 500 points)")
    ax.set_xlabel("Pas de temps")
    ax.set_ylabel("Valeur")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sine_signal.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/sine_signal.png")

    # ── Exp A : Impact de la longueur de séquence ─────────────────────────────
    print("\n[2] Expérience A — Impact de la longueur de séquence")
    seq_lengths = [10, 20, 50]
    results_seq = {}
    histories_seq = []
    labels_seq = []

    for sl in seq_lengths:
        print(f"\n  seq_length = {sl}")
        X, y = create_sequences(data, sl)
        train_data, val_data, test_data = split_train_val_test(X, y)
        model = build_rnn(sl, n_units=64, activation="tanh")
        model.summary()
        history, mse, mae = train_and_evaluate(model, train_data,
                                               val_data, test_data, epochs=30)
        results_seq[sl] = {"mse": mse, "mae": mae}
        histories_seq.append(history)
        labels_seq.append(f"T={sl}")
        print(f"    → Test MSE : {mse:.5f} | Test MAE : {mae:.5f}")
        plot_signal_and_predictions(data, model, sl,
                                    f"Prédictions SimpleRNN (T={sl})",
                                    f"rnn_pred_T{sl}.png")

    plot_seq_length_comparison(results_seq)
    plot_training_curves_all(histories_seq, labels_seq,
                             "rnn_seq_length_loss.png", "loss")

    # ── Exp B : tanh vs relu ──────────────────────────────────────────────────
    print("\n[3] Expérience B — Activation tanh vs relu (T=20)")
    seq_length = 20
    X, y = create_sequences(data, seq_length)
    train_data, val_data, test_data = split_train_val_test(X, y)
    results_act = {}
    histories_act = []

    for act in ["tanh", "relu"]:
        print(f"\n  Activation : {act}")
        model = build_rnn(seq_length, n_units=64, activation=act)
        history, mse, mae = train_and_evaluate(model, train_data,
                                               val_data, test_data, epochs=30)
        results_act[act] = {"mse": mse, "mae": mae}
        histories_act.append(history)
        print(f"    → Test MSE : {mse:.5f} | Test MAE : {mae:.5f}")

    plot_activation_comparison(results_act)
    plot_training_curves_all(histories_act, ["tanh", "relu"],
                             "rnn_activation_loss.png", "loss")

    print("\n─── Observations ─────────────────────────────────────")
    print("• Plus la séquence est longue, plus le RNN a de contexte → meilleure prédiction.")
    print("• Cependant, les séquences trop longues peuvent causer un vanishing gradient.")
    print("• Tanh est l'activation par défaut et généralement plus stable pour les RNN.")
    print("• ReLU dans un RNN peut exploser sur de longues séquences (gradient explosion).")
    print("• Pour des prédictions très précises sur longues séquences, LSTM/GRU sont préférables.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
