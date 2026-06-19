# TP1 — Réseaux de Neurones

**Cours : Deep Learning — Masters IICIA & IDA&SI 2025/2026**
**Université Ibn Tofail**

---

## 🎯 Objectifs

- Comprendre le fonctionnement du **Perceptron** et du **Multi-Layer Perceptron (MLP)**.
- Observer l'importance de la **non-linéarité** et des **couches cachées**.
- Reproduire et analyser les expériences du **TensorFlow Playground**.
- Mettre en œuvre des réseaux feed-forward avec **Keras/TensorFlow 2**.
- Analyser l'effet du nombre de neurones, de couches, et des fonctions d'activation.

---

## 📋 Contenu

| Fichier | Description |
|---------|-------------|
| `ex1_playground_tasks.py` | Tâches équivalentes TensorFlow Playground (XOR, spirale, cercles) |
| `ex2_mlp_activation_functions.py` | Comparaison des fonctions d'activation (ReLU, Tanh, Sigmoid) |
| `ex3_hidden_layers_depth.py` | Impact du nombre de couches et de neurones sur la performance |
| `requirements.txt` | Dépendances Python |

---

## 📝 Exercices & Résumé

### Exercice 1 — TensorFlow Playground (ex1_playground_tasks.py)

Reproduce four canonical tasks from the TensorFlow Playground website:

1. **Tâche 1 — Classification linéaire (données séparables)** : Un perceptron simple (0 couche cachée) suffit pour séparer deux classes linéairement séparables.
2. **Tâche 2 — XOR (données non-linéaires)** : Montre qu'un perceptron simple échoue sur le problème XOR ; une couche cachée avec activation non-linéaire le résout.
3. **Tâche 3 — Spirale (données complexes)** : Démontre que des réseaux profonds avec plusieurs couches cachées sont nécessaires pour classifier des spirales.
4. **Tâche 4 — Ingénierie des features** : Illustre comment des features dérivées (x1², x2², sin(x1), x1·x2) améliorent la séparabilité.

### Exercice 2 — Fonctions d'activation (ex2_mlp_activation_functions.py)

- Compare ReLU, Tanh, Sigmoid et ELU sur un jeu de données synthétique.
- Trace les courbes de perte et de précision.
- **Observation** : ReLU converge généralement le plus vite ; Sigmoid souffre du problème du *vanishing gradient*.

### Exercice 3 — Profondeur du réseau (ex3_hidden_layers_depth.py)

- Étudie l'effet du nombre de couches (1, 2, 3, 4) sur la précision de classification.
- Montre le risque de sur-apprentissage avec un réseau trop profond sur un petit jeu de données.

---

## ❓ Questions & Réponses

**Q1 : Pourquoi un perceptron simple ne peut-il pas résoudre XOR ?**
> Le perceptron trace une frontière de décision **linéaire** (hyperplan). Le problème XOR n'est pas linéairement séparable : aucune droite ne peut séparer les 4 points en deux classes correctes. Une couche cachée avec activation non-linéaire crée des frontières courbes.

**Q2 : Quel est le rôle des fonctions d'activation ?**
> Sans activation non-linéaire, un MLP profond est équivalent à une régression linéaire (composition de transformations linéaires = transformation linéaire). Les activations (ReLU, Tanh…) introduisent la non-linéarité permettant d'apprendre des représentations complexes.

**Q3 : Comment la profondeur affecte-t-elle les performances ?**
> Un réseau plus profond peut apprendre des représentations hiérarchiques plus complexes, mais nécessite plus de données et est plus difficile à entraîner (vanishing gradient). Sur des données simples, un réseau trop profond peut sur-apprendre.

**Q4 : Quel est l'effet de l'ingénierie des features ?**
> Ajouter des features polynomiales ou trigonométriques (x1², x2², sin(x1), x1·x2) transforme l'espace d'entrée en un espace où les classes sont linéairement (ou plus facilement) séparables.

---

## 🚀 Exécution

```bash
pip install -r requirements.txt
python ex1_playground_tasks.py
python ex2_mlp_activation_functions.py
python ex3_hidden_layers_depth.py
```

Les graphiques sont sauvegardés dans le dossier `outputs/`.
