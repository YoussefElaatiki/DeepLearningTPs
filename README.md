# Deep Learning TPs — Masters IICIA & IDA&SI 2025/2026
**Université Ibn Tofail**

This repository contains the practical lab sessions (Travaux Pratiques) for the Deep Learning course of the Masters IICIA (Ingénierie Informatique, Connectivité, Intelligence Artificielle) and IDA&SI programmes for the academic year 2025/2026.

---

## 📂 Repository Structure

| Folder | Title | Topics |
|--------|-------|--------|
| [`TP1-Réseaux_de_Neurones`](TP1-Réseaux_de_Neurones/) | Réseaux de Neurones | Perceptron, MLP, non-linearité, TF Playground |
| [`TP2-Convolution_et_Pooling`](TP2-Convolution_et_Pooling/) | Convolution et Pooling | Convolution 2D manuelle, CNN sur MNIST, Pooling |
| [`TP3-Architectures_et_Augmentation_de_Données`](TP3-Architectures_et_Augmentation_de_Données/) | Architectures et Augmentation | LeNet-5, CIFAR-10, ImageDataGenerator |
| [`TP4-Réseaux_de_Neurones_Récurrents_RNN`](TP4-Réseaux_de_Neurones_Récurrents_RNN/) | Réseaux de Neurones Récurrents | SimpleRNN, LSTM, Bidirectional, MNIST séquentiel |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/YoussefElaatiki/DeepLearningTPs.git
cd DeepLearningTPs

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Running a TP
```bash
cd TP1-Réseaux_de_Neurones
python ex1_playground_tasks.py
```
or open the corresponding Jupyter notebook:
```bash
jupyter notebook
```

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt) at the root for the full list.  
Each TP folder also contains its own `requirements.txt` with the same pinned versions.

---

## 🎓 Learning Objectives

By completing all four TPs, students will be able to:
1. Design, train and evaluate feed-forward neural networks with Keras/TensorFlow 2.
2. Implement and understand 2D convolution and pooling from scratch and with Keras.
3. Build state-of-the-art CNN architectures (LeNet-5) and apply data augmentation.
4. Model sequential data with SimpleRNN, LSTM, and Bidirectional RNNs.

---

## 📝 Notes

- All code uses **TensorFlow / Keras 2** with the Sequential or Functional API.
- A fixed **random seed** (`SEED = 42`) is used throughout for reproducibility.
- Generated plots are saved to an `outputs/` sub-folder inside each TP directory.
- Datasets (MNIST, CIFAR-10) are downloaded automatically by Keras on first run.

---

*Université Ibn Tofail — Département Informatique — 2025/2026*
