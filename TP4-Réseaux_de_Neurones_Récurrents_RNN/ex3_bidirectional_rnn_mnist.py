"""
TP4 — Exercice 3 : RNN Bidirectionnel sur MNIST
================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Compare un SimpleRNN unidirectionnel vs un RNN Bidirectionnel sur MNIST :
  - Nombre de paramètres
  - Précision de classification
  - Vitesse d'entraînement
  - Visualisation des activations cachées (t-SNE)

Usage :
    python ex3_bidirectional_rnn_mnist.py
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

CIFAR10_CLASSES = None   # non utilisé ici
MNIST_CLASSES = [str(i) for i in range(10)]

# ─── Chargement des données ───────────────────────────────────────────────────

def load_mnist():
    """Charge MNIST au format séquentiel (28, 28) et retourne train/val/test."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0

    split = int(0.9 * len(X_train))
    X_val, y_val = X_train[split:], y_train[split:]
    X_tr,  y_tr  = X_train[:split], y_train[:split]

    print(f"  Train : {X_tr.shape} | Val : {X_val.shape} | Test : {X_test.shape}")
    return (X_tr, y_tr), (X_val, y_val), (X_test, y_test)


# ─── Architectures ───────────────────────────────────────────────────────────

def build_unidirectional_rnn(n_units=128):
    """SimpleRNN unidirectionnel pour MNIST séquentiel.

    Input  : (28, 28) — 28 pas × 28 features
    Output : (10,) — softmax
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28), name="input"),
        tf.keras.layers.SimpleRNN(n_units, activation="tanh",
                                  return_sequences=False,
                                  name="simple_rnn"),
        tf.keras.layers.Dense(64, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(10, activation="softmax", name="output"),
    ], name="UniRNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_bidirectional_rnn(n_units=128):
    """SimpleRNN Bidirectionnel pour MNIST séquentiel.

    La couche Bidirectional enveloppe un SimpleRNN et l'exécute dans les
    deux sens (forward et backward). Les sorties sont concaténées.

    Nombre de paramètres = 2 × (unidirectionnel avec n_units) + Dense
    """
    # Note : avec Bidirectional, le SimpleRNN interne a n_units neurones,
    # mais la sortie est de taille 2*n_units (concaténation forward+backward).
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28), name="input"),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.SimpleRNN(n_units, activation="tanh",
                                      return_sequences=False),
            name="bidir_rnn"
        ),
        tf.keras.layers.Dense(64, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(10, activation="softmax", name="output"),
    ], name="BiRNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_bidirectional_lstm(n_units=64):
    """LSTM Bidirectionnel (comparaison supplémentaire pour l'étudiant)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28), name="input"),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(n_units, return_sequences=False),
            name="bidir_lstm"
        ),
        tf.keras.layers.Dense(64, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(10, activation="softmax", name="output"),
    ], name="BiLSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Entraînement ────────────────────────────────────────────────────────────

def train_model(model, train_data, val_data, epochs=15, batch_size=128):
    """Entraîne le modèle, mesure le temps d'entraînement."""
    X_tr, y_tr = train_data
    X_val, y_val = val_data

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1
    )
    elapsed = time.time() - t0
    return history, elapsed


# ─── Analyse des paramètres ───────────────────────────────────────────────────

def analyze_rnn_parameters(uni_model, bi_model):
    """Compare le nombre de paramètres des deux architectures."""
    print("\n  ─── Analyse des paramètres ──────────────────────────────")
    for name, model in [("UniRNN", uni_model), ("BiRNN", bi_model)]:
        print(f"\n  [{name}]")
        for layer in model.layers:
            if layer.count_params() > 0:
                print(f"    {layer.name:<20s} : {layer.count_params():>8,d} paramètres")
        print(f"    {'TOTAL':<20s} : {model.count_params():>8,d} paramètres")

    ratio = bi_model.count_params() / uni_model.count_params()
    print(f"\n  Ratio BiRNN/UniRNN : {ratio:.2f}x  "
          f"(attendu ≈ 2x pour la couche RNN)")


# ─── Visualisation ───────────────────────────────────────────────────────────

def plot_comparison_curves(hist_uni, hist_bi, hist_bilstm=None):
    """Trace les courbes de perte et de précision de validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(hist_uni.history["val_loss"],
                 label="UniRNN", color="#4C72B0", linewidth=2)
    axes[0].plot(hist_bi.history["val_loss"],
                 label="BiRNN", color="#DD8452", linewidth=2)
    if hist_bilstm:
        axes[0].plot(hist_bilstm.history["val_loss"],
                     label="BiLSTM", color="#55A868", linewidth=2)
    axes[0].set_title("Val Loss")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist_uni.history["val_accuracy"],
                 label="UniRNN", color="#4C72B0", linewidth=2)
    axes[1].plot(hist_bi.history["val_accuracy"],
                 label="BiRNN", color="#DD8452", linewidth=2)
    if hist_bilstm:
        axes[1].plot(hist_bilstm.history["val_accuracy"],
                     label="BiLSTM", color="#55A868", linewidth=2)
    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Comparaison UniRNN / BiRNN / BiLSTM sur MNIST",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bidir_comparison_curves.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/bidir_comparison_curves.png")


def plot_results_table_bar(results):
    """Barres de comparaison test accuracy pour tous les modèles."""
    names = list(results.keys())
    accs  = [results[n]["test_acc"] for n in names]
    times = [results[n]["time"] for n in names]
    params = [results[n]["params"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"][:len(names)]

    # Précision
    bars = axes[0].bar(names, accs, color=colors)
    axes[0].set_ylim(0.9, 1.0)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Précision de test")
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001, f"{acc:.4f}",
                     ha="center", fontsize=9)

    # Temps
    bars2 = axes[1].bar(names, times, color=colors)
    axes[1].set_ylabel("Temps (secondes)")
    axes[1].set_title("Temps d'entraînement")
    for bar, t in zip(bars2, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1, f"{t:.0f}s",
                     ha="center", fontsize=9)

    # Paramètres
    bars3 = axes[2].bar(names, params, color=colors)
    axes[2].set_ylabel("Nombre de paramètres")
    axes[2].set_title("Paramètres entraînables")
    for bar, p in zip(bars3, params):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 200, f"{p:,}",
                     ha="center", fontsize=8)

    plt.suptitle("Comparaison finale — MNIST Séquentiel",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bidir_results_bar.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/bidir_results_bar.png")


def visualize_hidden_states_tsne(model, X_test, y_test, model_name, filename,
                                 n_samples=1000):
    """Visualise les activations cachées de la couche RNN en 2D (t-SNE).

    Args:
        model     : modèle Keras entraîné.
        X_test    : données de test (N, 28, 28).
        y_test    : labels de test.
        model_name: nom pour le titre.
        filename  : fichier de sortie.
        n_samples : nombre d'échantillons à visualiser.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  (scikit-learn non disponible — t-SNE ignoré)")
        return

    # Construire un modèle intermédiaire qui retourne les activations RNN
    # Chercher la première couche Dense (juste après le RNN)
    rnn_output_layer = None
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense,)):
            rnn_output_layer = layer
            break

    if rnn_output_layer is None:
        print("  Impossible de trouver la couche Dense après le RNN.")
        return

    intermediate = tf.keras.Model(
        inputs=model.input,
        outputs=rnn_output_layer.output
    )

    # Obtenir les activations
    idx = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[idx]
    y_sample = y_test[idx]
    activations = intermediate.predict(X_sample, verbose=0)

    # t-SNE 2D
    print(f"  Calcul t-SNE sur {n_samples} échantillons...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=300)
    embeddings = tsne.fit_transform(activations)

    # Visualisation
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                         c=y_sample, cmap="tab10", alpha=0.7, s=8)
    plt.colorbar(scatter, ax=ax, ticks=range(10),
                 label="Classe MNIST")
    ax.set_title(f"t-SNE des activations cachées — {model_name}")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


def print_summary_table(results):
    """Affiche un tableau récapitulatif."""
    print("\n  ┌───────────┬────────────┬────────────┬────────────┬──────────┐")
    print("  │ Modèle    │  Paramètres│ Test Acc   │ Test Loss  │ Temps(s) │")
    print("  ├───────────┼────────────┼────────────┼────────────┼──────────┤")
    for name, res in results.items():
        print(f"  │ {name:<9s} │ {res['params']:>10,d} │ {res['test_acc']:.4f}    │"
              f" {res['test_loss']:.4f}    │ {res['time']:>8.1f} │")
    print("  └───────────┴────────────┴────────────┴────────────┴──────────┘")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP4 — Exercice 3 : RNN Bidirectionnel sur MNIST")
    print("=" * 60)

    # 1. Charger données
    print("\n[1] Chargement de MNIST")
    train_data, val_data, (X_test, y_test) = load_mnist()

    N_UNITS = 128   # neurones par couche RNN

    # 2. Modèle UniRNN
    print("\n[2] Modèle A — SimpleRNN Unidirectionnel")
    model_uni = build_unidirectional_rnn(N_UNITS)
    model_uni.summary()
    hist_uni, t_uni = train_model(model_uni, train_data, val_data, epochs=15)
    loss_uni, acc_uni = model_uni.evaluate(X_test, y_test, verbose=0)
    print(f"  UniRNN → Test acc : {acc_uni:.4f} | Time : {t_uni:.1f}s")

    # 3. Modèle BiRNN
    print("\n[3] Modèle B — SimpleRNN Bidirectionnel")
    model_bi = build_bidirectional_rnn(N_UNITS)
    model_bi.summary()
    hist_bi, t_bi = train_model(model_bi, train_data, val_data, epochs=15)
    loss_bi, acc_bi = model_bi.evaluate(X_test, y_test, verbose=0)
    print(f"  BiRNN → Test acc : {acc_bi:.4f} | Time : {t_bi:.1f}s")

    # 4. Modèle BiLSTM (comparaison bonus)
    print("\n[4] Modèle C — LSTM Bidirectionnel (comparaison bonus)")
    model_bilstm = build_bidirectional_lstm(N_UNITS // 2)
    model_bilstm.summary()
    hist_bilstm, t_bilstm = train_model(model_bilstm, train_data, val_data, epochs=15)
    loss_bilstm, acc_bilstm = model_bilstm.evaluate(X_test, y_test, verbose=0)
    print(f"  BiLSTM → Test acc : {acc_bilstm:.4f} | Time : {t_bilstm:.1f}s")

    # 5. Analyse paramètres
    analyze_rnn_parameters(model_uni, model_bi)

    # 6. Courbes de comparaison
    print("\n[5] Génération des graphiques de comparaison")
    plot_comparison_curves(hist_uni, hist_bi, hist_bilstm)

    results = {
        "UniRNN":  {"params": model_uni.count_params(),
                    "test_acc": acc_uni, "test_loss": loss_uni, "time": t_uni},
        "BiRNN":   {"params": model_bi.count_params(),
                    "test_acc": acc_bi, "test_loss": loss_bi, "time": t_bi},
        "BiLSTM":  {"params": model_bilstm.count_params(),
                    "test_acc": acc_bilstm, "test_loss": loss_bilstm, "time": t_bilstm},
    }
    plot_results_table_bar(results)
    print_summary_table(results)

    # 7. t-SNE des activations (optionnel, peut être lent)
    print("\n[6] Visualisation t-SNE des activations cachées")
    visualize_hidden_states_tsne(model_uni,    X_test, y_test,
                                 "UniRNN",  "tsne_uni_rnn.png")
    visualize_hidden_states_tsne(model_bi,     X_test, y_test,
                                 "BiRNN",   "tsne_bi_rnn.png")
    visualize_hidden_states_tsne(model_bilstm, X_test, y_test,
                                 "BiLSTM",  "tsne_bi_lstm.png")

    print("\n─── Observations ─────────────────────────────────────")
    print("• BiRNN a ~2× plus de paramètres que UniRNN dans la couche RNN.")
    print("• BiRNN obtient généralement une meilleure précision que UniRNN.")
    print("• BiLSTM est souvent le plus précis mais le plus lent.")
    print("• Le t-SNE montre des clusters plus séparés pour BiRNN/BiLSTM.")
    print("• La lecture dans les deux sens donne plus de contexte pour chaque pas.")
    print("• Pour MNIST, le RNN bidirectionnel peut 'voir' l'image du bas vers le haut.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
