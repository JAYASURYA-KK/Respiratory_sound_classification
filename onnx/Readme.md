# RespiCheck — Respiratory Disease Classifier

A browser-based respiratory disease classifier that runs entirely client-side using ONNX Runtime Web. Upload a lung audio recording and get an instant prediction across 6 disease classes — no server, no backend, no data leaves your device.

---

## 🔬 What It Does

RespiCheck takes a raw audio file (WAV, MP3, FLAC, OGG, M4A), extracts librosa-equivalent MFCC features in the browser, and runs them through a KNN model exported to ONNX to classify the recording into one of 6 respiratory conditions.

---

## 🩺 Supported Classes

| Class | Description |
|---|---|
| **Bronchiectasis** | Widening / scarring of bronchial tubes |
| **Bronchiolitis** | Small airway inflammation |
| **COPD** | Chronic Obstructive Pulmonary Disease |
| **Healthy** | No respiratory disease detected |
| **Pneumonia** | Lung tissue infection / inflammation |
| **URTI** | Upper Respiratory Tract Infection |

---

## ⚙️ Signal Processing Pipeline

The MFCC extraction closely mirrors librosa's default behaviour:

| Step | Details |
|---|---|
| Sample rate | 16 000 Hz mono |
| Window length | 4 seconds → 64 000 samples (truncate or zero-pad) |
| STFT | n\_fft = 2048, hop\_length = 512, Hann window, centre = True |
| Mel filterbank | 128 bands, Slaney normalisation |
| Power → dB | `power_to_db`, top\_db = 80 |
| DCT | Type-II orthonormal → 13 MFCC coefficients |
| Feature vector | 26 floats: μ₀–μ₁₂ (means) + σ₀–σ₁₂ (std deviations) |

---

## 🤖 Model

| Property | Value |
|---|---|
| Algorithm | K-Nearest Neighbours (k = 1, Euclidean distance) |
| Preprocessing | StandardScaler — baked into the ONNX pipeline |
| Input shape | `[1, 26]` float32 |
| Export format | ONNX (scikit-learn pipeline → `sklearn-onnx`) |
| Dataset | ICBHI 2017 Respiratory Sound Database |
| Runtime | ONNX Runtime Web (browser, no server needed) |

---

## 🗂️ Project Structure

```
onnx/
├── public/
│   └── respiratory_knn.onnx   # Trained KNN model (place here)
├── src/
│   ├── App.js                  # Main React UI
│   ├── App.css                 # Styles
│   └── utils/
│       ├── mfcc.js             # librosa-equivalent MFCC extraction
│       └── inference.js        # ONNX Runtime Web loader + inference
└── package.json
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js ≥ 16
- A trained `respiratory_knn.onnx` model file

### Installation

```bash
git clone https://github.com/JAYASURYA-KK/Respiratory_sound_classification.git
cd Respiratory_sound_classification/onnx
npm install
```

### Add the Model

Place your `respiratory_knn.onnx` file inside the `public/` folder:

```
public/
└── respiratory_knn.onnx
```

### Run Locally

```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
```

The optimised build will be in the `build/` folder, ready to deploy on Vercel, Netlify, or any static host.

---

## ☁️ Deploy on Vercel

```bash
npm install -g vercel
vercel
```

Make sure `respiratory_knn.onnx` is committed inside `public/` before deploying — Vercel serves everything in `public/` as static assets.

---

## 🖥️ How It Works (step by step)

```
Audio File
    │
    ▼
Decode → Float32 PCM samples (Web Audio API)
    │
    ▼
Truncate / Zero-pad → 64 000 samples (4 s @ 16 kHz)
    │
    ▼
STFT → Mel Filterbank → power_to_db → DCT-II
    │
    ▼
26-dim feature vector  [μ₀…μ₁₂, σ₀…σ₁₂]
    │
    ▼
StandardScaler (inside ONNX graph)
    │
    ▼
KNN (k=1) → Predicted Class + Probabilities
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| UI | React 18 |
| ML Runtime | ONNX Runtime Web |
| Audio DSP | Web Audio API + custom librosa-port |
| Model training | scikit-learn (Python) |
| Model export | sklearn-onnx |
| Deployment | Vercel |

---

## 📊 Dataset

**ICBHI 2017 Respiratory Sound Database**
- 920 annotated `.wav` recordings from 126 patients
- 5.5 hours of audio across all age groups
- Labels: Bronchiectasis, Bronchiolitis, COPD, Healthy, Pneumonia, URTI
- Source: [Kaggle — Respiratory Sound Database](https://www.kaggle.com/vbookshelf/respiratory-sound-database)

---

## ⚠️ Disclaimer

RespiCheck is a research / educational tool. It is **not** a medical device and should **not** be used for clinical diagnosis. Always consult a qualified healthcare professional for medical advice.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Jayasurya KK**
[GitHub](https://github.com/JAYASURYA-KK)
