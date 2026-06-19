"""
TP3 — Exercice 2 : Visualisation de l'Augmentation de Données sur CIFAR-10
==========================================================================
Université Ibn Tofail — Masters IICIA & IDA&SI 2025/2026

Démontre et visualise les différentes transformations d'augmentation de données
disponibles dans Keras ImageDataGenerator appliquées sur CIFAR-10.

Usage :
    python ex2_data_augmentation_visualization.py
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

# Noms des classes CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─── Chargement de CIFAR-10 ───────────────────────────────────────────────────

def load_cifar10():
    """Charge CIFAR-10 et normalise les valeurs de pixels dans [0, 1]."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32)  / 255.0
    y_train = y_train.flatten()
    y_test  = y_test.flatten()
    print(f"  Train : {X_train.shape}, Test : {X_test.shape}")
    print(f"  Classes : {CIFAR10_CLASSES}")
    return (X_train, y_train), (X_test, y_test)


# ─── Définitions des augmentations ───────────────────────────────────────────

def get_augmentation_configs():
    """Retourne un dictionnaire name → ImageDataGenerator config."""
    return {
        "Rotation (±20°)": dict(rotation_range=20),
        "Translation H (±15%)": dict(width_shift_range=0.15),
        "Translation V (±15%)": dict(height_shift_range=0.15),
        "Zoom (±20%)": dict(zoom_range=0.20),
        "Flip horizontal": dict(horizontal_flip=True),
        "Cisaillement": dict(shear_range=20),
        "Luminosité": dict(brightness_range=(0.5, 1.5)),
        "Toutes combinées": dict(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            shear_range=10,
            fill_mode="nearest",
        ),
    }


# ─── Génération des images augmentées ────────────────────────────────────────

def generate_augmented_samples(image, datagen_kwargs, n_samples=8):
    """Génère n_samples images augmentées à partir d'une image source.

    Args:
        image         : array (H, W, C) en float32 [0, 1].
        datagen_kwargs: dict des paramètres pour ImageDataGenerator.
        n_samples     : nombre d'images augmentées à générer.

    Returns:
        Liste de n_samples images augmentées (H, W, C) clippées dans [0, 1].
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    img_batch = image[np.newaxis]   # (1, H, W, C)
    augmented = []
    for batch in datagen.flow(img_batch, batch_size=1, seed=SEED):
        augmented.append(np.clip(batch[0], 0, 1))
        if len(augmented) >= n_samples:
            break
    return augmented


# ─── Visualisation par type d'augmentation ───────────────────────────────────

def visualize_single_augmentation(image, class_name, aug_name,
                                  aug_params, n_samples=8, filename=None):
    """Affiche l'image originale + n_samples versions augmentées."""
    augmented = generate_augmented_samples(image, aug_params, n_samples)

    fig, axes = plt.subplots(1, n_samples + 1, figsize=(2.2 * (n_samples + 1), 2.5))

    # Image originale
    axes[0].imshow(image)
    axes[0].set_title("Original\n" + class_name, fontsize=8)
    axes[0].axis("off")

    # Images augmentées
    for i, aug_img in enumerate(augmented):
        axes[i + 1].imshow(aug_img)
        axes[i + 1].set_title(f"Aug {i+1}", fontsize=8)
        axes[i + 1].axis("off")

    plt.suptitle(f"Augmentation : {aug_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if filename:
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
        plt.close()
        print(f"  ✔  Saved: {OUTPUT_DIR}/{filename}")
    else:
        plt.show()


def visualize_all_augmentations(image, class_name):
    """Affiche un aperçu de toutes les augmentations sur une seule figure."""
    aug_configs = get_augmentation_configs()
    n_augs = len(aug_configs)

    fig, axes = plt.subplots(n_augs, 5, figsize=(12, n_augs * 2))
    axes[0, 0].set_title("Original", fontsize=9, fontweight="bold")
    for j in range(1, 5):
        axes[0, j].set_title(f"Aug {j}", fontsize=9)

    for row, (aug_name, params) in enumerate(aug_configs.items()):
        augmented = generate_augmented_samples(image, params, n_samples=4)

        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(aug_name, fontsize=7, rotation=0,
                                labelpad=80, va="center")
        axes[row, 0].axis("off")

        for col, aug_img in enumerate(augmented):
            axes[row, col + 1].imshow(aug_img)
            axes[row, col + 1].axis("off")

    plt.suptitle(f"Aperçu des transformations d'augmentation — «{class_name}»",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "all_augmentations_overview.png"),
                dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/all_augmentations_overview.png")


def visualize_multiple_classes(X_train, y_train, n_per_class=1):
    """Montre l'augmentation 'toutes combinées' sur un exemple par classe."""
    aug_params = get_augmentation_configs()["Toutes combinées"]

    fig, axes = plt.subplots(10, 5, figsize=(12, 22))

    for cls_idx in range(10):
        # Trouver un exemple de cette classe
        idx = np.where(y_train == cls_idx)[0][0]
        image = X_train[idx]
        augmented = generate_augmented_samples(image, aug_params, n_samples=4)

        axes[cls_idx, 0].imshow(image)
        axes[cls_idx, 0].set_ylabel(CIFAR10_CLASSES[cls_idx],
                                    fontsize=9, rotation=0,
                                    labelpad=65, va="center")
        axes[cls_idx, 0].axis("off")
        for col, aug_img in enumerate(augmented):
            axes[cls_idx, col + 1].imshow(aug_img)
            axes[cls_idx, col + 1].axis("off")

    # En-têtes colonnes
    for j, title in enumerate(["Original", "Aug 1", "Aug 2", "Aug 3", "Aug 4"]):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold")

    plt.suptitle("Augmentation combinée sur les 10 classes CIFAR-10",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "augmentation_all_classes.png"), dpi=120)
    plt.close()
    print(f"  ✔  Saved: {OUTPUT_DIR}/augmentation_all_classes.png")


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TP3 — Exercice 2 : Visualisation Augmentation CIFAR-10")
    print("=" * 60)

    # 1. Charger CIFAR-10
    print("\n[1] Chargement de CIFAR-10")
    (X_train, y_train), _ = load_cifar10()

    # 2. Choisir une image représentative (ex: un chien)
    class_target = 5   # "dog"
    idx = np.where(y_train == class_target)[0][0]
    sample_image = X_train[idx]
    sample_name = CIFAR10_CLASSES[class_target]
    print(f"\n  Image sélectionnée : classe '{sample_name}' (index {idx})")

    # 3. Visualisation individuelle par type d'augmentation
    print("\n[2] Visualisation par type d'augmentation")
    aug_configs = get_augmentation_configs()
    for aug_name, params in aug_configs.items():
        safe_name = aug_name.replace(" ", "_").replace("(", "").replace(")", "").replace("±", "").replace("%", "").replace("°", "")
        filename = f"aug_{safe_name[:30]}.png"
        visualize_single_augmentation(
            sample_image, sample_name, aug_name, params,
            n_samples=8, filename=filename
        )

    # 4. Vue d'ensemble toutes augmentations
    print("\n[3] Vue d'ensemble de toutes les augmentations")
    visualize_all_augmentations(sample_image, sample_name)

    # 5. Toutes les classes
    print("\n[4] Augmentation sur toutes les classes CIFAR-10")
    visualize_multiple_classes(X_train, y_train)

    print("\n─── Observations ─────────────────────────────────────")
    print("• L'augmentation de données génère à la volée des variations d'images.")
    print("• Rotation et translation : utiles pour l'invariance aux transformations.")
    print("• Zoom : simule différentes distances de prise de vue.")
    print("• Flip horizontal : naturel pour certaines classes (voitures, animaux).")
    print("• La combinaison de plusieurs transformations enrichit davantage le set.")
    print("• L'augmentation ne crée pas de nouvelles informations mais régularise.")
    print("\n✅ Tous les graphiques ont été sauvegardés dans", OUTPUT_DIR)
