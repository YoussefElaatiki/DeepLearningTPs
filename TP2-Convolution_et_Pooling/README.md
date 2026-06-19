# TP2 — Convolution et Pooling

**Cours : Deep Learning — Masters IICIA & IDA&SI 2025/2026**
**Université Ibn Tofail**

---

## 🎯 Objectifs

- Comprendre le principe de la **convolution 2D** et l'implémenter manuellement avec NumPy.
- Utiliser des **couches Conv2D de Keras** pour extraire des features sur MNIST.
- Visualiser les **feature maps** produites après convolution.
- Comparer le **Max Pooling** et l'**Average Pooling** visuellement.

---

## 📋 Contenu

| Fichier | Description |
|---------|-------------|
| `ex1_manual_convolution.py` | Convolution 2D manuelle avec NumPy (différents filtres) |
| `ex2_keras_mnist_convolution.py` | CNN sur MNIST avec Keras, visualisation des feature maps |
| `ex3_pooling_comparison.py` | Comparaison Max Pooling vs Avg Pooling sur feature maps |
| `requirements.txt` | Dépendances Python |

---

## 📝 Exercices & Résumé

### Exercice 1 — Convolution 2D Manuelle (ex1_manual_convolution.py)

Implémente la convolution 2D **sans bibliothèque de deep learning** :
1. Charge une image en niveaux de gris (ou crée une image synthétique).
2. Applique plusieurs filtres manuellement :
   - Filtre de **détection de contours** (Sobel horizontal et vertical).
   - Filtre de **flou** (moyenne).
   - Filtre de **netteté** (Laplacien).
3. Affiche l'image originale et les résultats pour chaque filtre.

### Exercice 2 — CNN sur MNIST avec Keras (ex2_keras_mnist_convolution.py)

1. Construit un CNN simple sur MNIST (Conv2D → ReLU → MaxPool → Dense).
2. Entraîne le modèle et affiche les courbes perte/précision.
3. **Visualise les feature maps** produites par la première couche de convolution sur un exemple d'image de test.
4. Affiche les poids (filtres) appris par la première couche.

### Exercice 3 — Comparaison Max Pooling / Avg Pooling (ex3_pooling_comparison.py)

1. Prend une feature map (issue de la convolution sur une image MNIST).
2. Applique **Max Pooling** (taille 2×2, stride 2).
3. Applique **Average Pooling** (taille 2×2, stride 2).
4. Visualise côte à côte : image originale → après convolution → après Max Pool → après Avg Pool.
5. Compare quantitativement les deux méthodes.

---

## ❓ Questions & Réponses

**Q1 : Qu'est-ce qu'une convolution 2D en deep learning ?**
> La convolution 2D glisse un petit filtre (kernel) sur l'image en calculant le produit scalaire entre le filtre et chaque patch de l'image. Cela produit une *feature map* qui met en évidence certains motifs (contours, textures, etc.).

**Q2 : Pourquoi utiliser le partage de poids (weight sharing) dans les CNN ?**
> Un même filtre est appliqué à toutes les positions de l'image. Cela réduit drastiquement le nombre de paramètres (vs un réseau entièrement connecté) et confère une **invariance à la translation**.

**Q3 : Quelle est la différence entre Max Pooling et Average Pooling ?**
> - **Max Pooling** : conserve la valeur maximale dans chaque fenêtre → préserve les features les plus saillantes, robuste au bruit.
> - **Average Pooling** : calcule la moyenne → lisse la feature map, conserve davantage d'information globale.
> En pratique, Max Pooling est plus couramment utilisé dans les CNN de classification car il retient les activations les plus fortes.

**Q4 : Quel est l'effet de la taille du filtre sur les features extraites ?**
> Un filtre 3×3 détecte des motifs locaux fins (contours nets). Un filtre 5×5 ou 7×7 capture des motifs plus larges et globaux. Les filtres plus grands ont plus de paramètres mais peuvent mieux capturer le contexte.

**Q5 : Comment interpréter les feature maps visualisées ?**
> Chaque feature map correspond à la réponse d'un filtre particulier. Des zones lumineuses indiquent là où ce filtre "réagit" fortement → le réseau apprend à détecter les caractéristiques utiles pour la tâche.

---

## 🚀 Exécution

```bash
pip install -r requirements.txt
python ex1_manual_convolution.py
python ex2_keras_mnist_convolution.py
python ex3_pooling_comparison.py
```

Les graphiques sont sauvegardés dans le dossier `outputs/`.
