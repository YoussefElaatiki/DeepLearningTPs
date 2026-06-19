"""
TP2 — Exercice 1 : Convolution 2D Manuelle avec NumPy
======================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Implémente la convolution 2D sans bibliothèque de deep learning.
Applique différents filtres (Sobel, flou, Laplacien) à une image
en niveaux de gris et visualise les résultats.

Usage :
    python ex1_manual_convolution.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Reproductibilité ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Implémentation de la convolution 2D ────────────────────────────────────

def conv2d_manual(image, kernel, padding=0, stride=1):
    """Convolution 2D manuelle (corrélation croisée, convention deep learning).

    Args:
        image  : tableau NumPy 2D (H, W), valeurs float.
        kernel : tableau NumPy 2D (kH, kW), filtre de convolution.
        padding: nombre de pixels de zéro-padding ajoutés autour de l'image.
        stride : pas de déplacement du filtre.

    Returns:
        Feature map 2D (H_out, W_out).
    """
    kH, kW = kernel.shape
    H, W   = image.shape

    # Appliquer le zero-padding
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode="constant",
                       constant_values=0)

    H_pad, W_pad = image.shape
    H_out = (H_pad - kH) // stride + 1
    W_out = (W_pad - kW) // stride + 1

    feature_map = np.zeros((H_out, W_out), dtype=np.float64)

    for i in range(H_out):
        for j in range(W_out):
            patch = image[i * stride: i * stride + kH,
                          j * stride: j * stride + kW]
            # Produit élément par élément + somme = convolution
            feature_map[i, j] = np.sum(patch * kernel)

    return feature_map


def normalize_output(feature_map):
    """Normalise la feature map dans [0, 255] pour la visualisation."""
    f_min, f_max = feature_map.min(), feature_map.max()
    if f_max - f_min < 1e-8:
        return np.zeros_like(feature_map, dtype=np.uint8)
    return ((feature_map - f_min) / (f_max - f_min) * 255).astype(np.uint8)


# ─── Filtres classiques ───────────────────────────────────────────────────────

# Détection de contours — Sobel
SOBEL_H = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)  # contours horizontaux

SOBEL_V = np.array([[-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]], dtype=np.float64)  # contours verticaux

# Flou (moyenne) 5×5
BLUR_5 = np.ones((5, 5), dtype=np.float64) / 25.0

# Filtre de netteté (Laplacien discret)
LAPLACIAN = np.array([[ 0, -1,  0],
                      [-1,  4, -1],
                      [ 0, -1,  0]], dtype=np.float64)

# Renforcement des détails (Unsharp mask)
SHARPEN = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]], dtype=np.float64)

# Filtre de détection de coins (différence diagonale)
EMBOSS = np.array([[-2, -1,  0],
                   [-1,  1,  1],
                   [ 0,  1,  2]], dtype=np.float64)


# ─── Création d'une image synthétique ────────────────────────────────────────

def create_synthetic_image(size=64):
    """Génère une image synthétique en niveaux de gris avec des formes géométriques."""
    img = np.zeros((size, size), dtype=np.float64)

    # Rectangle blanc au centre
    img[15:50, 15:50] = 200.0

    # Cercle (approximation)
    cy, cx, r = size // 2, size // 2, 15
    Y, X = np.ogrid[:size, :size]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = 255.0

    # Lignes diagonales
    for k in range(0, size, 8):
        if k < size:
            img[k, :] = 150.0
    for k in range(0, size, 8):
        if k < size:
            img[:, k] = 150.0

    return img


# ─── Visualisation ───────────────────────────────────────────────────────────

def visualize_filters():
    """Visualise les noyaux de convolution utilisés."""
    filters = {
        "Sobel H": SOBEL_H,
        "Sobel V": SOBEL_V,
        "Blur 5×5": BLUR_5,
        "Laplacien": LAPLACIAN,
        "Sharpen": SHARPEN,
        "Emboss": EMBOSS,
    }
    fig, axes = plt.subplots(1, len(filters), figsize=(18, 3))
    for ax, (name, f) in zip(axes, filters.items()):
        im = ax.imshow(f, cmap="RdBu_r", vmin=-f.max(), vmax=f.max())
        ax.set_title(name, fontsize=9)
        ax.axis("off")
        for (r, c), val in np.ndenumerate(f):
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("Noyaux de convolution (filtres)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "filters_visualization.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/filters_visualization.png")


def apply_and_plot(image, kernels_dict, filename, padding=1):
    """Applique plusieurs filtres à une image et génère un plot comparatif."""
    n = len(kernels_dict) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    # Image originale
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, (name, kernel) in zip(axes[1:], kernels_dict.items()):
        fm = conv2d_manual(image, kernel, padding=padding)
        fm_norm = normalize_output(fm)
        ax.imshow(fm_norm, cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")

    plt.suptitle("Convolution 2D Manuelle — Comparaison des filtres",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")


def demo_stride_padding(image):
    """Démontre l'effet du stride et du padding sur la taille de sortie."""
    print("\n─── Effet du stride et du padding ────────────────────────────")
    kernel = SOBEL_H
    configs = [
        ("pad=0, stride=1", 0, 1),
        ("pad=1, stride=1", 1, 1),
        ("pad=0, stride=2", 0, 2),
        ("pad=1, stride=2", 1, 2),
    ]
    H, W = image.shape
    kH, kW = kernel.shape

    fig, axes = plt.subplots(1, len(configs), figsize=(16, 4))
    for ax, (name, pad, stride) in zip(axes, configs):
        fm = conv2d_manual(image, kernel, padding=pad, stride=stride)
        fm_norm = normalize_output(fm)
        ax.imshow(fm_norm, cmap="gray")
        ax.set_title(f"{name}\nSortie : {fm.shape[0]}×{fm.shape[1]}", fontsize=8)
        ax.axis("off")
        H_out_exp = (H + 2 * pad - kH) // stride + 1
        print(f"  {name:20s} → shape : {fm.shape} (théorique : {H_out_exp}×{H_out_exp})")

    plt.suptitle("Effet du Stride et du Padding (filtre Sobel H)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stride_padding_effect.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/stride_padding_effect.png")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP2 — Exercice 1 : Convolution 2D Manuelle avec NumPy")
    print("=" * 60)

    # 1. Visualiser les filtres
    print("\n[1] Visualisation des filtres")
    visualize_filters()

    # 2. Créer une image synthétique 64×64
    print("\n[2] Création d'une image synthétique 64×64")
    image = create_synthetic_image(64)
    print(f"    Image shape : {image.shape}, min={image.min()}, max={image.max()}")

    # 3. Appliquer tous les filtres
    print("\n[3] Application des filtres")
    kernels = {
        "Sobel H": SOBEL_H,
        "Sobel V": SOBEL_V,
        "Blur 5×5": BLUR_5,
        "Laplacien": LAPLACIAN,
        "Sharpen": SHARPEN,
        "Emboss": EMBOSS,
    }
    apply_and_plot(image, kernels, "convolution_results_synthetic.png", padding=1)

    # 4. Charger MNIST et appliquer sur un vrai chiffre
    print("\n[4] Application sur un chiffre MNIST")
    import tensorflow as tf
    (X_train, _), _ = tf.keras.datasets.mnist.load_data()
    mnist_digit = X_train[0].astype(np.float64)   # 28×28
    print(f"    MNIST digit shape : {mnist_digit.shape}")
    apply_and_plot(mnist_digit, kernels, "convolution_results_mnist.png", padding=1)

    # 5. Démonstration stride/padding
    demo_stride_padding(mnist_digit)

    print("\n─── Observations ─────────────────────────────────────")
    print("• Sobel H détecte les transitions verticales de niveau de gris.")
    print("• Sobel V détecte les transitions horizontales.")
    print("• Le flou (Blur) lisse l'image en moyennant les voisins.")
    print("• Le Laplacien détecte les contours dans toutes les directions.")
    print("• Un stride > 1 réduit la taille de sortie (sous-échantillonnage).")
    print("• Le padding 'same' maintient les dimensions spatiales.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
