"""
TP2 — Exercice 3 : Comparaison Max Pooling vs Average Pooling
=============================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Compare visuellement et quantitativement Max Pooling et Average Pooling
appliqués aux feature maps d'une image MNIST après une convolution.

Usage :
    python ex3_pooling_comparison.py
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

# ─── Utilitaires ─────────────────────────────────────────────────────────────

def apply_pooling_keras(feature_maps, pool_size=(2, 2), pool_type="max"):
    """Applique un pooling Keras (max ou average) sur un batch de feature maps.

    Args:
        feature_maps: numpy array (1, H, W, C).
        pool_size   : fenêtre de pooling.
        pool_type   : "max" ou "avg".

    Returns:
        numpy array (1, H_out, W_out, C).
    """
    inp = tf.keras.layers.Input(shape=feature_maps.shape[1:])
    if pool_type == "max":
        out = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_size)(inp)
    else:
        out = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_size)(inp)

    pool_model = tf.keras.Model(inp, out)
    return pool_model.predict(feature_maps, verbose=0)


def get_conv_feature_maps(image_batch, n_filters=16, kernel_size=3):
    """Applique une couche Conv2D sur une image et retourne les feature maps.

    Args:
        image_batch: numpy array (1, H, W, 1).
        n_filters  : nombre de filtres Conv2D.
        kernel_size: taille des filtres.

    Returns:
        numpy array (1, H, W, n_filters).
    """
    inp = tf.keras.layers.Input(shape=image_batch.shape[1:])
    out = tf.keras.layers.Conv2D(n_filters, kernel_size=kernel_size,
                                 padding="same", activation="relu",
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(SEED))(inp)
    conv_model = tf.keras.Model(inp, out)
    return conv_model.predict(image_batch, verbose=0)


# ─── Visualisation côte à côte ───────────────────────────────────────────────

def plot_pooling_comparison(image, conv_out, max_pool_out, avg_pool_out,
                            filter_idx=0, filename="pooling_comparison.png"):
    """Visualise : Original → Après Conv → Après MaxPool → Après AvgPool."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Image originale
    axes[0].imshow(image[0, :, :, 0], cmap="gray")
    axes[0].set_title(f"Original\n{image.shape[1]}×{image.shape[2]}", fontsize=10)
    axes[0].axis("off")

    # Feature map après convolution (filtre filter_idx)
    axes[1].imshow(conv_out[0, :, :, filter_idx], cmap="viridis")
    axes[1].set_title(f"Après Conv (filtre {filter_idx})\n"
                      f"{conv_out.shape[1]}×{conv_out.shape[2]}", fontsize=10)
    axes[1].axis("off")

    # Après Max Pooling
    axes[2].imshow(max_pool_out[0, :, :, filter_idx], cmap="viridis")
    axes[2].set_title(f"Après Max Pooling (2×2)\n"
                      f"{max_pool_out.shape[1]}×{max_pool_out.shape[2]}", fontsize=10)
    axes[2].axis("off")

    # Après Avg Pooling
    axes[3].imshow(avg_pool_out[0, :, :, filter_idx], cmap="viridis")
    axes[3].set_title(f"Après Avg Pooling (2×2)\n"
                      f"{avg_pool_out.shape[1]}×{avg_pool_out.shape[2]}", fontsize=10)
    axes[3].axis("off")

    plt.suptitle(f"Comparaison Max Pooling vs Avg Pooling (filtre {filter_idx})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


def plot_multiple_filters(conv_out, max_pool_out, avg_pool_out,
                          n_display=8, filename="pooling_all_filters.png"):
    """Affiche plusieurs filtres côte à côte : Conv | Max Pool | Avg Pool."""
    n_display = min(n_display, conv_out.shape[-1])
    fig, axes = plt.subplots(3, n_display, figsize=(n_display * 2, 6))

    for i in range(n_display):
        axes[0, i].imshow(conv_out[0, :, :, i], cmap="viridis")
        axes[0, i].set_title(f"F{i}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(max_pool_out[0, :, :, i], cmap="viridis")
        axes[1, i].axis("off")

        axes[2, i].imshow(avg_pool_out[0, :, :, i], cmap="viridis")
        axes[2, i].axis("off")

    # Étiquettes des lignes
    for i, label in enumerate(["Conv2D", "Max Pool", "Avg Pool"]):
        axes[i, 0].set_ylabel(label, fontsize=10, rotation=0,
                              labelpad=50, va="center")

    plt.suptitle("Comparaison des 8 premiers filtres : Conv → MaxPool | AvgPool",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


def plot_pooling_stats(conv_out, max_pool_out, avg_pool_out):
    """Compare statistiquement Max Pool et Avg Pool (valeurs moyennes, max, variance)."""
    stats = {
        "Conv (avant pool)": conv_out[0],
        "Max Pool":          max_pool_out[0],
        "Avg Pool":          avg_pool_out[0],
    }
    labels = list(stats.keys())
    means  = [np.mean(v)   for v in stats.values()]
    maxs   = [np.max(v)    for v in stats.values()]
    stds   = [np.std(v)    for v in stats.values()]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, means, width, label="Moyenne",  color="#4C72B0")
    ax.bar(x,         stds,  width, label="Écart-type", color="#DD8452")
    ax.bar(x + width, maxs,  width, label="Maximum",  color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valeur des activations")
    ax.set_title("Statistiques des activations après pooling")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pooling_statistics.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/pooling_statistics.png")

    print("\n  ┌────────────────────────┬──────────┬──────────┬──────────┐")
    print("  │        Méthode         │ Moyenne  │ Éc-type  │ Maximum  │")
    print("  ├────────────────────────┼──────────┼──────────┼──────────┤")
    for label, m, s, mx in zip(labels, means, stds, maxs):
        print(f"  │ {label:<22s} │ {m:8.4f} │ {s:8.4f} │ {mx:8.4f} │")
    print("  └────────────────────────┴──────────┴──────────┴──────────┘")


def compare_pooling_on_multiple_images(X_test, n_images=6):
    """Applique et compare les deux poolings sur plusieurs images MNIST."""
    samples = X_test[:n_images]
    conv_outputs    = get_conv_feature_maps(samples, n_filters=16)
    max_pool_out    = apply_pooling_keras(conv_outputs, (2, 2), "max")
    avg_pool_out    = apply_pooling_keras(conv_outputs, (2, 2), "avg")

    # Afficher image originale + pooling résultats (filtre 0)
    fig, axes = plt.subplots(3, n_images, figsize=(n_images * 2.5, 7))
    for i in range(n_images):
        axes[0, i].imshow(samples[i, :, :, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Img {i}", fontsize=8)

        axes[1, i].imshow(max_pool_out[i, :, :, 0], cmap="plasma")
        axes[1, i].axis("off")

        axes[2, i].imshow(avg_pool_out[i, :, :, 0], cmap="plasma")
        axes[2, i].axis("off")

    for i, label in enumerate(["Original", "Max Pool", "Avg Pool"]):
        axes[i, 0].set_ylabel(label, fontsize=10, rotation=0,
                              labelpad=55, va="center")

    plt.suptitle(f"Pooling sur {n_images} images MNIST (filtre 0)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pooling_multiple_images.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/pooling_multiple_images.png")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP2 — Exercice 3 : Comparaison Max Pooling vs Avg Pooling")
    print("=" * 60)

    # 1. Charger MNIST
    print("\n[1] Chargement de MNIST")
    (_, _), (X_test_raw, y_test) = tf.keras.datasets.mnist.load_data()
    X_test = X_test_raw.astype(np.float32)[..., np.newaxis] / 255.0
    sample = X_test[0:1]     # shape (1, 28, 28, 1)
    print(f"    Image shape : {sample.shape}")

    # 2. Obtenir les feature maps après convolution
    print("\n[2] Convolution 2D sur l'image test")
    conv_out = get_conv_feature_maps(sample, n_filters=16, kernel_size=3)
    print(f"    Feature maps après conv : {conv_out.shape}")

    # 3. Appliquer les deux poolings
    print("\n[3] Application du Max Pooling et Avg Pooling (2×2)")
    max_pool_out = apply_pooling_keras(conv_out, pool_size=(2, 2), pool_type="max")
    avg_pool_out = apply_pooling_keras(conv_out, pool_size=(2, 2), pool_type="avg")
    print(f"    Max Pool output : {max_pool_out.shape}")
    print(f"    Avg Pool output : {avg_pool_out.shape}")

    # 4. Visualisation côte à côte (plusieurs filtres)
    print("\n[4] Visualisations")
    for filt_idx in [0, 1, 2]:
        plot_pooling_comparison(sample, conv_out, max_pool_out, avg_pool_out,
                                filter_idx=filt_idx,
                                filename=f"pooling_comparison_f{filt_idx}.png")

    plot_multiple_filters(conv_out, max_pool_out, avg_pool_out,
                          n_display=8, filename="pooling_all_filters.png")

    # 5. Statistiques
    print("\n[5] Statistiques des activations")
    plot_pooling_stats(conv_out, max_pool_out, avg_pool_out)

    # 6. Plusieurs images
    print("\n[6] Comparaison sur plusieurs images MNIST")
    compare_pooling_on_multiple_images(X_test, n_images=6)

    print("\n─── Observations ─────────────────────────────────────")
    print("• Max Pooling conserve la valeur maximale → préserve les activations fortes.")
    print("• Avg Pooling calcule la moyenne → représentation plus lisse et globale.")
    print("• Max Pooling est plus courant en classification (retient les features saillantes).")
    print("• Avg Pooling est souvent utilisé en Global Average Pooling dans des architectures modernes.")
    print("• Les deux méthodes réduisent la dimension spatiale de ½ (stride 2).")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
