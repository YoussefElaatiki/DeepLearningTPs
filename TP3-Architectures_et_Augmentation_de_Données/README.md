# TP3 — Architectures et Augmentation de Données

**Cours : Deep Learning — Masters IICIA & IDA&SI 2025/2026**
**Université Ibn Tofail**

---

## 🎯 Objectifs

- Implémenter l'architecture **LeNet-5** et l'entraîner sur MNIST.
- Analyser le nombre de paramètres, la précision et le sur-apprentissage.
- Visualiser et appliquer les techniques d'**augmentation de données** (ImageDataGenerator).
- Comparer la généralisation d'un modèle sur **CIFAR-10** avec et sans augmentation.

---

## 📋 Contenu

| Fichier | Description |
|---------|-------------|
| `ex1_lenet5_mnist.py` | LeNet-5 complet sur MNIST + analyse paramètres/overfitting |
| `ex2_data_augmentation_visualization.py` | Visualisation des transformations d'augmentation sur CIFAR-10 |
| `ex3_augmentation_comparison.py` | Comparaison généralisation avec/sans augmentation sur CIFAR-10 |
| `requirements.txt` | Dépendances Python |

---

## 📝 Exercices & Résumé

### Exercice 1 — LeNet-5 sur MNIST (ex1_lenet5_mnist.py)

Implémente l'architecture originale **LeNet-5** (LeCun et al., 1998) :

```
Input (32×32×1)
  → Conv2D(6, 5×5, tanh) + AvgPool(2×2)
  → Conv2D(16, 5×5, tanh) + AvgPool(2×2)
  → Flatten
  → Dense(120, tanh)
  → Dense(84, tanh)
  → Dense(10, softmax)
```

Questions analysées :
- Nombre total de paramètres par couche.
- Précision atteinte sur MNIST.
- Détection et mitigation du sur-apprentissage.
- Impact de Dropout et de l'optimiseur.

### Exercice 2 — Visualisation de l'Augmentation de Données (ex2_data_augmentation_visualization.py)

Démontre les transformations disponibles dans `ImageDataGenerator` :
- Rotation (±20°)
- Translation horizontale/verticale (±10%)
- Zoom (±15%)
- Flip horizontal
- Cisaillement (shear)
- Combinaison de toutes les transformations

Appliqué sur des images CIFAR-10.

### Exercice 3 — Comparaison avec/sans Augmentation (ex3_augmentation_comparison.py)

Entraîne **le même modèle CNN** sur CIFAR-10 :
1. **Sans augmentation** : risque de sur-apprentissage (courbes divergentes).
2. **Avec augmentation** : meilleure généralisation (val_accuracy supérieure).

Génère une comparaison visuelle des deux courbes d'entraînement.

---

## ❓ Questions & Réponses

**Q1 : Combien de paramètres a LeNet-5 ?**
> LeNet-5 standard a environ **61 706 paramètres** :
> - Conv1 (6 filtres 5×5×1 + biais) : 156
> - Conv2 (16 filtres 5×5×6 + biais) : 2 416
> - Dense FC1 (120×400 + biais) : 48 120
> - Dense FC2 (84×120 + biais) : 10 164
> - Dense Output (10×84 + biais) : 850
> La majeure partie des paramètres est dans les couches fully connected.

**Q2 : Quelle précision atteint LeNet-5 sur MNIST ?**
> LeNet-5 atteint environ **99%** de précision sur MNIST avec quelques époques. C'est l'un des premiers CNN à démontrer l'efficacité de ce type d'architecture pour la reconnaissance de caractères.

**Q3 : Comment détecter le sur-apprentissage ?**
> Le sur-apprentissage est visible quand la courbe de `val_loss` *augmente* alors que la `train_loss` continue de *diminuer*. L'écart croissant entre `train_accuracy` et `val_accuracy` en est aussi un indicateur.

**Q4 : Comment l'augmentation de données combat-elle le sur-apprentissage ?**
> L'augmentation génère à la volée des variantes de chaque image (rotations, flips, zooms…). Cela augmente artificiellement la taille effective du jeu d'entraînement, réduisant le sur-apprentissage et améliorant la généralisation du modèle sur de nouvelles données.

**Q5 : Pourquoi CIFAR-10 est-il plus difficile que MNIST ?**
> CIFAR-10 contient des images couleur 32×32 de 10 catégories très diverses (avions, chiens, voitures…). L'intra-classe est très variable et l'inter-classe peut être similaire. MNIST contient des chiffres en niveaux de gris simples avec peu de variation.

---

## 🚀 Exécution

```bash
pip install -r requirements.txt
python ex1_lenet5_mnist.py
python ex2_data_augmentation_visualization.py
python ex3_augmentation_comparison.py
```

Les graphiques sont sauvegardés dans le dossier `outputs/`.
