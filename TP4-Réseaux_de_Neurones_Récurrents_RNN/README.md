# TP4 — Réseaux de Neurones Récurrents (RNN)

**Cours : Deep Learning — Masters IICIA & IDA&SI 2025/2026**
**Université Ibn Tofail**

---

## 🎯 Objectifs

- Comprendre le principe des **Réseaux de Neurones Récurrents (RNN)** et leur capacité à modéliser des séquences.
- Implémenter **SimpleRNN** pour la prédiction de séquences numériques.
- Utiliser **SimpleRNN** pour la classification d'images MNIST vues comme séquences de lignes.
- Comparer **RNN Unidirectionnel** et **RNN Bidirectionnel** (paramètres, précision, vitesse).

---

## 📋 Contenu

| Fichier | Description |
|---------|-------------|
| `ex1_simpleRNN_sequence_prediction.py` | SimpleRNN pour la prédiction de séquences numériques (sinus) |
| `ex2_simpleRNN_mnist.py` | SimpleRNN pour la classification MNIST (28 pas de temps × 28 features) |
| `ex3_bidirectional_rnn_mnist.py` | RNN Bidirectionnel sur MNIST, comparaison avec unidirectionnel |
| `requirements.txt` | Dépendances Python |

---

## 📝 Exercices & Résumé

### Exercice 1 — SimpleRNN pour la prédiction de séquences (ex1_simpleRNN_sequence_prediction.py)

Prédit la prochaine valeur d'une **séquence sinusoïdale** :
1. Génère une séquence d'onde sinusoïdale avec bruit.
2. Transforme la séquence en paires (entrée, cible) avec une fenêtre glissante.
3. Entraîne un **SimpleRNN** pour prédire la prochaine valeur.
4. Compare différentes longueurs de séquence (seq_length = 10, 20, 50).
5. Compare les activations **tanh** vs **relu** pour le SimpleRNN.

### Exercice 2 — SimpleRNN pour MNIST (ex2_simpleRNN_mnist.py)

Traite les images MNIST comme des **séquences temporelles** :
- Chaque image 28×28 est vue comme **28 pas de temps**, chacun étant un vecteur de 28 pixels (une ligne de l'image).
- Architecture : `Input(28, 28)` → `SimpleRNN(128)` → `Dense(64)` → `Dense(10, softmax)`.
- Compare avec un MLP et un CNN pour la même tâche.

### Exercice 3 — RNN Bidirectionnel sur MNIST (ex3_bidirectional_rnn_mnist.py)

Compare **SimpleRNN unidirectionnel** vs **Bidirectionnel** :
- Le RNN bidirectionnel lit la séquence dans les **deux sens** (gauche→droite et droite→gauche).
- Analyse : nombre de paramètres, précision de classification, temps d'entraînement.
- Visualise les activations cachées des deux modèles.

---

## ❓ Questions & Réponses

**Q1 : Comment fonctionne un SimpleRNN ?**
> Un SimpleRNN maintient un **état caché** `h_t` mis à jour à chaque pas de temps :
> `h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)`
> où `x_t` est l'entrée au pas `t`, `h_{t-1}` l'état précédent. Il peut souffrir du **vanishing gradient** sur les longues séquences.

**Q2 : Quel est l'avantage d'une longue séquence pour la prédiction ?**
> Une séquence plus longue fournit plus de contexte historique pour prédire la valeur suivante. Cependant, sur les séquences très longues, un SimpleRNN peut oublier le début (vanishing gradient). Des architectures comme LSTM ou GRU y remédient.

**Q3 : Pourquoi MNIST peut-il être traité comme une séquence ?**
> Une image 28×28 peut être vue comme 28 lignes ordonnées de haut en bas. Le RNN peut apprendre à reconnaître un chiffre en lisant ces lignes séquentiellement, comme on lirait un texte de gauche à droite.

**Q4 : Quelle est la différence entre RNN unidirectionnel et bidirectionnel ?**
> - **Unidirectionnel** : lit la séquence dans un seul sens (t=1→T). L'état caché final résume toute la séquence.
> - **Bidirectionnel** : utilise **deux RNN** — un lit de t=1→T, l'autre de t=T→1. Les sorties sont concaténées, doublant la taille de l'état caché. Cela permet de capturer le contexte **passé ET futur** à chaque pas.

**Q5 : Combien de paramètres supplémentaires a un RNN Bidirectionnel ?**
> Un RNN Bidirectionnel a exactement **2 fois plus de paramètres** qu'un RNN unidirectionnel équivalent (deux copies du RNN), plus le coût de la couche Dense qui reçoit une entrée de dimension doublée.

**Q6 : Comparaison SimpleRNN vs LSTM vs GRU ?**
> - **SimpleRNN** : le plus simple, souffre du vanishing gradient sur longues séquences.
> - **GRU** : plus complexe (reset gate + update gate), mieux adapté aux longues séquences, moins de paramètres que LSTM.
> - **LSTM** : le plus complexe (input/forget/output gate + cell state), généralement meilleur sur des séquences très longues, mais plus lent.

---

## 🚀 Exécution

```bash
pip install -r requirements.txt
python ex1_simpleRNN_sequence_prediction.py
python ex2_simpleRNN_mnist.py
python ex3_bidirectional_rnn_mnist.py
```

Les graphiques sont sauvegardés dans le dossier `outputs/`.
