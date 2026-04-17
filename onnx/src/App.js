import React, { useState, useRef, useCallback, useEffect } from 'react';
import { extract26MfccFeatures, decodeAudioFile, truncateOrPad, SR } from './utils/mfcc';
import { loadModel, runInference, CLASSES } from './utils/inference';
import './App.css';

const DISEASE_INFO = {
  Bronchiectasis: { icon: '🫁', color: '#e07b39', desc: 'Widening/scarring of bronchial tubes' },
  Bronchiolitis:  { icon: '🌬️', color: '#d45f88', desc: 'Small airway inflammation' },
  COPD:           { icon: '💨', color: '#c0392b', desc: 'Chronic Obstructive Pulmonary Disease' },
  Healthy:        { icon: '✅', color: '#27ae60', desc: 'No respiratory disease detected' },
  Pneumonia:      { icon: '🫂', color: '#8e44ad', desc: 'Lung tissue infection/inflammation' },
  URTI:           { icon: '🤧', color: '#2980b9', desc: 'Upper Respiratory Tract Infection' },
};

const MODEL_URL = process.env.PUBLIC_URL + '/respiratory_knn.onnx';
const STAGE = { IDLE: 'idle', LOADING: 'loading', PROCESSING: 'processing', RESULT: 'result', ERROR: 'error' };

export default function App() {
  const [stage, setStage] = useState(STAGE.IDLE);
  const [statusMsg, setStatusMsg] = useState('');
  const [result, setResult] = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [audioInfo, setAudioInfo] = useState(null);
  const [features, setFeatures] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setStatusMsg('Loading ONNX model…');
    loadModel(MODEL_URL)
      .then(() => { setModelReady(true); setStatusMsg(''); })
      .catch(() => setStatusMsg('⚠️ Place respiratory_knn.onnx in /public/'));
  }, []);

  const handleFile = useCallback(async (file) => {
    if (!file || !modelReady) return;
    try {
      setStage(STAGE.LOADING); setResult(null); setFeatures(null);
      setStatusMsg('Decoding audio…');
      const raw = await decodeAudioFile(file);
      const padded = truncateOrPad(raw);
      setAudioInfo({ name: file.name, originalSamples: raw.length, dur: (raw.length / SR).toFixed(2) });

      setStage(STAGE.PROCESSING);
      setStatusMsg('Extracting MFCC features (librosa-equivalent)…');
      await new Promise(r => setTimeout(r, 30));
      const feats = extract26MfccFeatures(padded);
      setFeatures(Array.from(feats));

      setStatusMsg('Running KNN ONNX inference…');
      await new Promise(r => setTimeout(r, 30));
      const prediction = await runInference(feats);
      setResult(prediction);
      setStage(STAGE.RESULT);
      setStatusMsg('');
    } catch (err) {
      setStage(STAGE.ERROR);
      setStatusMsg('Error: ' + err.message);
    }
  }, [modelReady]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const reset = () => {
    setStage(STAGE.IDLE); setResult(null); setFeatures(null);
    setAudioInfo(null); setStatusMsg('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const busy = stage === STAGE.LOADING || stage === STAGE.PROCESSING;

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo-row">
            <span className="logo-icon">🫁</span>
            <div>
              <h1 className="logo-title">RespiCheck</h1>
              <p className="logo-sub">Respiratory Disease Classifier</p>
            </div>
          </div>
          <div className="header-right">
            <span className={`model-dot ${modelReady ? 'ready' : ''}`}>{modelReady ? '● Ready' : '○ Loading'}</span>
            <span className="tech-tag">KNN · ONNX · librosa MFCC</span>
          </div>
        </div>
      </header>

      <main className="main">
        <div
          className={`drop-zone ${busy ? 'busy' : ''} ${stage === STAGE.RESULT ? 'done' : ''}`}
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => !busy && fileInputRef.current?.click()}
        >
          <input ref={fileInputRef} type="file" accept=".wav,.mp3,.ogg,.flac,.m4a"
            style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />

          {stage === STAGE.IDLE && (
            <div className="idle-content">
              <div className="pulse-ring"><span className="drop-mic">🎙️</span></div>
              <h2>Drop Lung Audio Here</h2>
              <p className="drop-hint">WAV / MP3 / FLAC · click to browse</p>
              <p className="drop-spec">16 kHz · 4s window · 26 MFCC → KNN</p>
              {statusMsg && <p className="status-warn">{statusMsg}</p>}
            </div>
          )}

          {busy && (
            <div className="busy-content">
              <div className="loader" />
              <p className="busy-msg">{statusMsg}</p>
              <div className="steps">
                <span className={stage === STAGE.LOADING ? 'step active' : 'step done'}>① Decode</span>
                <span className="arrow">→</span>
                <span className={stage === STAGE.PROCESSING ? 'step active' : 'step'}>② MFCC</span>
                <span className="arrow">→</span>
                <span className="step">③ KNN</span>
              </div>
            </div>
          )}

          {stage === STAGE.ERROR && (
            <div className="error-content" onClick={e => e.stopPropagation()}>
              <p className="error-msg">⚠️ {statusMsg}</p>
              <button className="btn-reset" onClick={reset}>Try Again</button>
            </div>
          )}

          {stage === STAGE.RESULT && result && (
            <div className="result-content" onClick={e => e.stopPropagation()}>
              <ResultCard result={result} audioInfo={audioInfo} />
              <button className="btn-reset" onClick={reset}>↩ Analyse Another</button>
            </div>
          )}
        </div>

        {features && result && (
          <section className="feats-section">
            <h3 className="sect-title">MFCC Feature Vector <span className="dim">(26 float32 → ONNX input)</span></h3>
            <FeatureGrid features={features} />
          </section>
        )}

        <section className="info-row">
          <InfoCard icon="🔬" title="Signal Processing">
            <p>16 kHz mono · 64 000 samples (4 s)</p>
            <p>STFT: n_fft=2048, hop=512, Hann, centre=True</p>
            <p>Mel: 128 bands, Slaney norm · power_to_db top_db=80</p>
            <p>DCT-II ortho → 13 MFCC coefficients</p>
          </InfoCard>
          <InfoCard icon="🤖" title="Model">
            <p>KNeighborsClassifier (k=1, Euclidean)</p>
            <p>Input: 26 features (μ×13 + σ×13)</p>
            <p>StandardScaler baked into ONNX pipeline</p>
            <p>ICBHI Respiratory Sound Database</p>
          </InfoCard>
          <InfoCard icon="📋" title="6 Classes">
            {CLASSES.map(c => (
              <p key={c} style={{ color: DISEASE_INFO[c].color }}>
                {DISEASE_INFO[c].icon} <strong>{c}</strong>
                <span className="dim"> — {DISEASE_INFO[c].desc}</span>
              </p>
            ))}
          </InfoCard>
        </section>
      </main>
    </div>
  );
}

function ResultCard({ result, audioInfo }) {
  const { predictedClass, probabilities } = result;
  const info = DISEASE_INFO[predictedClass];
  const sorted = CLASSES.map((c, i) => ({ c, p: probabilities[i] })).sort((a, b) => b.p - a.p);

  return (
    <div className="result-card" style={{ '--accent': info.color }}>
      <div className="res-hero">
        <span className="res-icon">{info.icon}</span>
        <div>
          <div className="res-label">Predicted Condition</div>
          <div className="res-class">{predictedClass}</div>
          <div className="res-desc">{info.desc}</div>
        </div>
        <div className="res-conf">
          <div className="conf-num">{(probabilities[result.classIndex] * 100).toFixed(1)}%</div>
          <div className="conf-txt">confidence</div>
        </div>
      </div>
      {audioInfo && (
        <div className="audio-meta">
          <span>📁 {audioInfo.name}</span>
          <span>⏱ {audioInfo.dur}s → 4.0s</span>
          <span>🎵 {audioInfo.originalSamples.toLocaleString()} samples</span>
        </div>
      )}
      <div className="prob-list">
        {sorted.map(({ c, p }) => (
          <div key={c} className="prob-row">
            <span className="prob-lbl">{DISEASE_INFO[c].icon} {c}</span>
            <div className="prob-track">
              <div className="prob-fill" style={{ width: `${p * 100}%`, background: DISEASE_INFO[c].color }} />
            </div>
            <span className="prob-pct">{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function FeatureGrid({ features }) {
  return (
    <div className="feat-grid">
      {features.map((v, i) => {
        const isMean = i < 13;
        return (
          <div key={i} className={`feat-cell ${isMean ? 'is-mean' : 'is-std'}`}>
            <div className="feat-name">{isMean ? 'μ' : 'σ'}<sub>{isMean ? i : i - 13}</sub></div>
            <div className="feat-val">{v.toFixed(3)}</div>
          </div>
        );
      })}
    </div>
  );
}

function InfoCard({ icon, title, children }) {
  return (
    <div className="info-card">
      <h4 className="info-title">{icon} {title}</h4>
      <div className="info-body">{children}</div>
    </div>
  );
}