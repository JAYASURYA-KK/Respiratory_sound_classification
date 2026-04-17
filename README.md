# 🫁 RespiCheck — Respiratory Sound Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-lightgrey?logo=onnx)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)
![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?logo=vercel)

**Machine Learning–based classification of respiratory sounds using the ICBHI 2017 dataset.**  
Achieving **99% accuracy** with KNN & SVM + MFCC feature extraction — optimized for edge deployment.

[🚀 Live Demo](https://respiratory-sound-classification.vercel.app/) · [📄 Journal Paper](https://www.iieta.org/journals/isi/paper/10.18280/isi.300211) · [📊 Dataset (ICBHI 2017)](https://bhichallenge.med.auth.gr/)

</div>

---

## 📖 Overview

Respiratory diseases account for **14.1% of global deaths annually** (WHO, 2022), underscoring the urgent need for advanced, automated diagnostic tools. This project implements a robust and computationally efficient machine learning pipeline for classifying respiratory sounds, based on the peer-reviewed research:

> **"Harmonizing Respiratory Sound Insights: Unleashing the Potential of Machine Learning Classifiers Through Hyperparameter Elegance"**  
> *Vishnu Vardhan Battu, Kalaiselvi Geetha Manoharan, Syam Prasad Gudapati*  
> Published in **Ingénierie des Systèmes d'Information (ISI)**, Vol. 30, No. 2, pp. 395–408, 2025.  
> DOI: [10.18280/isi.300211](https://doi.org/10.18280/isi.300211)

The proposed approach uses **Mel-Frequency Cepstral Coefficients (MFCCs)**, chroma, and spectral features combined with lightweight classifiers (KNN, SVM) — achieving **99% accuracy** while remaining suitable for deployment on **resource-constrained edge devices**.

---

## ✨ Key Features

- 🎯 **99% classification accuracy** using KNN & SVM with MFCC (Mean+Std) features
- 🔊 **80 audio features** extracted: MFCCs, Chroma, Spectral Centroid, ZCR, Harmonic/Percussive
- ⚖️ **Class imbalance addressed** via data augmentation (time masking, shifting, stretching)
- ⚡ **ONNX model export** for fast, framework-agnostic inference
- 🌐 **Live web demo** deployed on Vercel
- 💊 **Edge-device ready** — low computational footprint vs. deep learning alternatives

---

## 🗂️ Repository Structure

```
Respiratory_sound_classification/
│
├── data/                   # Processed feature CSVs derived from ICBHI 2017
├── journal/                # Reference paper and related literature
├── notebook/               # Jupyter notebooks for EDA, training, and evaluation
├── onnx/                   # Exported ONNX models for inference
├── outputs/                # Saved model artifacts, plots, confusion matrices
├── test_input/             # Sample audio files for quick testing
└── README.md
```

---

## 📦 Dataset — ICBHI 2017

This project uses the **ICBHI 2017 Respiratory Sound Database** (also referred to as ICHBI in the paper).

| Property | Details |
|---|---|
| Total recordings | 920 audio files |
| Subjects | 126 individuals |
| Respiratory cycles | 6,898 labeled cycles |
| Classes | Normal, Crackles, Wheezes, Crackles+Wheezes |
| Format | WAV, 16,000 Hz sampling rate |

### Class Distribution (Respiratory Cycles)

| Class | Cycles | % of Total |
|---|---|---|
| Normal / Healthy | 1,200 | 17.4% |
| Crackles (Pneumonia, COPD) | 1,720 | 25.0% |
| Wheezes (Asthma, Bronchitis) | 920 | 13.3% |
| Crackles + Wheezes (Mixed) | 3,058 | 44.3% |
| **Total** | **6,898** | **100%** |

### Class Distribution (Patient Records)

| Class | Records |
|---|---|
| COPD | 793 |
| Healthy | 35 |
| Pneumonia | 37 |
| URTI | 23 |
| Bronchiectasis | 16 |
| Bronchiolitis | 13 |
| Asthma | 1 |
| LRTI | 2 |
| **Total** | **920** |

> 📥 Download the dataset from the [official ICBHI Challenge page](https://bhichallenge.med.auth.gr/) or via [Harvard Dataverse](https://doi.org/10.7910/dvn/ht6pki).

---

## ⚙️ Methodology

### Pipeline Overview

```
Raw Audio → Normalization → Augmentation → Feature Extraction (80 features) → Feature Selection → ML Classifier → Prediction
```

### 1. Preprocessing & Segmentation

- Audio signals segmented into **4-second breath cycles** at 16,000 Hz
- **Peak normalization** applied for consistent amplitude levels

### 2. Data Augmentation (via `librosa`)

To address severe class imbalance, the following augmentations were applied to minority classes:

| Technique | Description | Parameters |
|---|---|---|
| **Time Masking** | Randomly silences audio segments | 100–200 ms duration |
| **Time Shifting** | Shifts the signal along the time axis | ±50 ms |
| **Time Stretching** | Changes duration without altering pitch | Stretch factor: 0.8–1.2 |
| **Loudness Normalization** | Adjusts amplitude uniformly | Peak normalization |

After augmentation, each minority class was expanded to approximately a **1:1 ratio** with the COPD majority class:

| Class | Selected Records | Augmentations Each | Total Records |
|---|---|---|---|
| Healthy | 35 | 30× | 1,050 |
| Bronchiectasis | 16 | 13× | 208 |
| Bronchiolitis | 13 | 16× | 208 |
| Pneumonia | 37 | 6× | 222 |
| URTI | 23 | 9× | 207 |
| COPD | 200 | 1× | 200 |
| **Total** | | | **3,140** |

### 3. Feature Extraction — 80 Features

Features are grouped into 9 sets for modular experimentation:

| Group | Features |
|---|---|
| `df_feature_all` | All 80 features combined |
| `df_feature_mel_chroma` | Mel-Spectrogram (mean, std, var) + Chroma CENS (mean, std, var) |
| `df_feature_mfcc_mean` | MFCC means (coefficients 0–12) |
| `df_feature_mfcc_std` | MFCC standard deviations (coefficients 0–12) |
| `df_feature_mfcc` | MFCC mean + std combined (26 features) |
| `df_feature_chroma_mean` | Chroma mean (12 bins) |
| `df_feature_chroma_std` | Chroma std (12 bins) |
| `df_feature_chroma` | Chroma mean + std (24 features) |
| `df_feature_csrzhp` | Spectral Centroid, Bandwidth, Roll-off, ZCR, Harmonic, Percussive |

**MFCC formula (Eq. 4):**

$$\text{MFCC}_k(t) = \sum_{m=1}^{M} \log(S_m) \cos\left[\frac{k\pi(m - 0.5)}{M}\right]$$

### 4. Machine Learning Classifiers

Eleven classifiers were evaluated across four train/test split ratios (0.20, 0.30, 0.40, 0.50):

- Decision Tree, Random Forest, Gradient Boosting, XGBoost
- AdaBoost, Extra Trees
- **K-Nearest Neighbors (KNN)** ⭐
- **Support Vector Machine (SVM)** ⭐
- Gaussian Naïve Bayes, MLP, Logistic Regression

### 5. Hyperparameter Tuning (Grid Search)

| Classifier | Best Hyperparameters | Accuracy |
|---|---|---|
| KNN | `n_neighbors=1`, `metric=euclidean`, `weights=uniform` | **100%** |
| SVM | `C=100`, `kernel=rbf`, `gamma=scale` | **100%** |
| Random Forest | `n_estimators=100`, `max_features=sqrt`, `criterion=gini` | **100%** |
| Gradient Boost | `lr=0.1`, `max_depth=7`, `n_estimators=100`, `subsample=0.5` | **100%** |
| XGBoost | `lr=0.1`, `max_depth=7`, `n_estimators=1000`, `subsample=0.5` | **100%** |
| Logistic Regression | `C=10`, `penalty=l1`, `solver=liblinear` | 74% |

---

## 📊 Results

### Best Performance — MFCC (Mean+Std) Features, Split 0.20

| Classifier | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 1.00 | 0.99 | 1.00 | 1.00 |
| Extra Trees | 1.00 | 0.99 | 1.00 | 0.99 |
| XGBoost | 1.00 | 1.00 | 1.00 | 1.00 |
| **KNN** | **0.99** | **0.99** | **1.00** | **0.99** |
| **SVM** | **0.99** | **0.99** | **1.00** | **0.98** |
| Decision Tree | 0.99 | 0.98 | 0.99 | 0.99 |

### Comparison with State-of-the-Art

| Method | Accuracy | Computational Cost |
|---|---|---|
| CNN (Aykanat et al., 2017) | 97% | High |
| Hybrid CNN-RNN | ~95–97% | Very High |
| **Proposed (KNN + SVM + MFCC)** | **99%** | **Low (Edge-ready)** |

> Misclassifications were primarily observed in overlapping classes (Crackles+Wheezes). Augmentation significantly reduced errors in minority classes.

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.8
pip install librosa numpy pandas scikit-learn xgboost onnxruntime matplotlib seaborn jupyter
```

### Installation

```bash
git clone https://github.com/JAYASURYA-KK/Respiratory_sound_classification.git
cd Respiratory_sound_classification
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook notebook/
```

Open and run the notebooks in order:
1. **Data Preparation & Augmentation** — preprocessing and augmentation pipeline
2. **Feature Extraction** — extracting 80 audio features
3. **Model Training & Evaluation** — training all classifiers with hyperparameter tuning
4. **ONNX Export** — exporting the best model for deployment

### ONNX Inference

```python
import onnxruntime as ort
import numpy as np

# Load model
sess = ort.InferenceSession("onnx/model.onnx")

# Prepare features (shape: [1, 80])
features = np.array([...], dtype=np.float32).reshape(1, -1)

# Run inference
pred = sess.run(None, {"input": features})[0]
print("Predicted class:", pred)
```

---

## 🌐 Live Demo

Try the model in your browser:

**👉 [https://respiratory-sound-classification.vercel.app/](https://respiratory-sound-classification.vercel.app/)**

The web app (RespiCheck) accepts audio input and returns a predicted respiratory disease class using the exported ONNX model.

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{battu2025respiratory,
  title     = {Harmonizing Respiratory Sound Insights: Unleashing the Potential of Machine Learning Classifiers Through Hyperparameter Elegance},
  author    = {Battu, Vishnu Vardhan and Manoharan, Kalaiselvi Geetha and Gudapati, Syam Prasad},
  journal   = {Ingénierie des Systèmes d'Information},
  volume    = {30},
  number    = {2},
  pages     = {395--408},
  year      = {2025},
  doi       = {10.18280/isi.300211},
  publisher = {IIETA}
}
```

---

## 📄 License

The journal paper is published under [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).  
The ICBHI 2017 dataset is subject to its own [usage terms](https://bhichallenge.med.auth.gr/).

---

## 🙏 Acknowledgements

- **ICBHI 2017 Challenge** organizers for the open-access respiratory sound database
- **WHO Global Health Statistics Report 2022** for epidemiological context
- Libraries: [`librosa`](https://librosa.org/), [`scikit-learn`](https://scikit-learn.org/), [`onnxruntime`](https://onnxruntime.ai/), [`XGBoost`](https://xgboost.readthedocs.io/)

---

<div align="center">
Made with ❤️ for better respiratory health diagnostics
</div>
