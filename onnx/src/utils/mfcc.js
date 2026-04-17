/**
 * mfcc.js — Exact JS port of librosa.feature.mfcc
 *
 * Python reference (from notebook):
 *   SR = 16000
 *   SEGMENT_LEN = 4 * SR  (64000 samples)
 *   mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
 *   mfccs_mean_i = float(np.mean(mfcc[i]))
 *   mfccs_std_i  = float(np.std(mfcc[i]))   ← population std (ddof=0)
 *
 * librosa defaults that matter:
 *   n_fft      = 2048
 *   hop_length = 512
 *   n_mels     = 128
 *   fmin       = 0.0
 *   fmax       = sr / 2
 *   power      = 2.0   (melspectrogram uses power spectrum)
 *   top_db     = 80.0  (power_to_db, ref=np.max)
 *   htk        = False (Slaney mel scale)
 *   norm       = 'slaney'
 *   n_mfcc     = 13
 */

const N_FFT = 2048;
const HOP_LENGTH = 512;
const N_MELS = 128;
const N_MFCC = 13;
export const SR = 16000;
export const SEGMENT_LEN = 4 * SR; // 64000

// ── 1. Truncate / zero-pad ───────────────────────────────
export function truncateOrPad(y, length = SEGMENT_LEN) {
  if (y.length === length) return y;
  const out = new Float32Array(length);
  out.set(y.length > length ? y.subarray(0, length) : y);
  return out;
}

// ── 2. Hann window ───────────────────────────────────────
function hannWindow(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

// ── 3. Cooley–Tukey radix-2 FFT (in-place, complex) ─────
function fftInPlace(re, im) {
  const N = re.length;
  // bit-reversal
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let uR = 1, uI = 0;
      for (let k = 0; k < len / 2; k++) {
        const vR = re[i + k + len / 2] * uR - im[i + k + len / 2] * uI;
        const vI = re[i + k + len / 2] * uI + im[i + k + len / 2] * uR;
        re[i + k + len / 2] = re[i + k] - vR;
        im[i + k + len / 2] = im[i + k] - vI;
        re[i + k] += vR; im[i + k] += vI;
        const tr = uR * wRe - uI * wIm;
        uI = uR * wIm + uI * wRe; uR = tr;
      }
    }
  }
}

// ── 4. STFT power spectrogram ────────────────────────────
// librosa centre=True → reflect-pad n_fft//2 on each side
function stftPower(y) {
  const pad = N_FFT >> 1; // 1024
  const pLen = y.length + 2 * pad;
  const padded = new Float64Array(pLen);
  for (let i = 0; i < pad; i++) padded[pad - 1 - i] = y[i];     // reflect left
  for (let i = 0; i < y.length; i++) padded[pad + i] = y[i];    // centre
  for (let i = 0; i < pad; i++) padded[pad + y.length + i] = y[y.length - 1 - i]; // reflect right

  const win = hannWindow(N_FFT);
  const nFrames = Math.floor((pLen - N_FFT) / HOP_LENGTH) + 1;
  const nBins = N_FFT / 2 + 1; // 1025

  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);
  const S = new Array(nFrames);

  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP_LENGTH;
    re.fill(0); im.fill(0);
    for (let i = 0; i < N_FFT; i++) re[i] = padded[start + i] * win[i];
    fftInPlace(re, im);
    const pow = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) pow[k] = re[k] * re[k] + im[k] * im[k];
    S[t] = pow;
  }
  return { S, nFrames };
}

// ── 5. Mel filterbank (Slaney / O'Shaughnessy, norm='slaney') ──
function hzToMelSlaney(hz) {
  const MIN_LOG_HZ = 1000.0, MIN_LOG_MEL = 15.0;
  const LOGSTEP = Math.log(6.4) / 27.0, F_SP = 200.0 / 3.0;
  return hz >= MIN_LOG_HZ
    ? MIN_LOG_MEL + Math.log(hz / MIN_LOG_HZ) / LOGSTEP
    : hz / F_SP;
}
function melToHzSlaney(mel) {
  const MIN_LOG_HZ = 1000.0, MIN_LOG_MEL = 15.0;
  const LOGSTEP = Math.log(6.4) / 27.0, F_SP = 200.0 / 3.0;
  return mel >= MIN_LOG_MEL
    ? MIN_LOG_HZ * Math.exp(LOGSTEP * (mel - MIN_LOG_MEL))
    : F_SP * mel;
}

function melFilterbank() {
  const nBins = N_FFT / 2 + 1;
  const melMin = hzToMelSlaney(0);
  const melMax = hzToMelSlaney(SR / 2);
  const melPts = new Float64Array(N_MELS + 2);
  for (let i = 0; i <= N_MELS + 1; i++)
    melPts[i] = melMin + (i / (N_MELS + 1)) * (melMax - melMin);
  const freqPts = melPts.map(melToHzSlaney);
  const binPts = freqPts.map(f => Math.floor((N_FFT + 1) * f / SR));

  const fb = [];
  for (let m = 0; m < N_MELS; m++) {
    const row = new Float64Array(nBins);
    const lo = binPts[m], cen = binPts[m + 1], hi = binPts[m + 2];
    for (let k = lo; k < cen && k < nBins; k++)
      row[k] = cen > lo ? (k - lo) / (cen - lo) : 0;
    for (let k = cen; k < hi && k < nBins; k++)
      row[k] = hi > cen ? (hi - k) / (hi - cen) : 0;
    // Slaney enorm
    const enorm = freqPts[m + 2] > freqPts[m] ? 2.0 / (freqPts[m + 2] - freqPts[m]) : 1;
    for (let k = 0; k < nBins; k++) row[k] *= enorm;
    fb.push(row);
  }
  return fb;
}

// ── 6. power_to_db (ref=np.max, top_db=80) ──────────────
function powerToDb(melPow, nFrames) {
  const TOP_DB = 80.0, amin = 1e-10;
  let globalMax = 0;
  for (let m = 0; m < N_MELS; m++)
    for (let t = 0; t < nFrames; t++)
      if (melPow[m][t] > globalMax) globalMax = melPow[m][t];
  const refVal = Math.max(amin, globalMax);
  const melDb = melPow.map(row => row.map(v => 10 * Math.log10(Math.max(amin, v) / refVal)));
  let maxDb = -Infinity;
  for (let m = 0; m < N_MELS; m++)
    for (let t = 0; t < nFrames; t++)
      if (melDb[m][t] > maxDb) maxDb = melDb[m][t];
  const thr = maxDb - TOP_DB;
  for (let m = 0; m < N_MELS; m++)
    for (let t = 0; t < nFrames; t++)
      if (melDb[m][t] < thr) melDb[m][t] = thr;
  return melDb;
}

// ── 7. DCT-II ortho (scipy.fftpack.dct norm='ortho') ────
function buildDctMatrix() {
  const D = [];
  for (let k = 0; k < N_MFCC; k++) {
    const row = new Float64Array(N_MELS);
    const fac = k === 0 ? Math.sqrt(1 / N_MELS) : Math.sqrt(2 / N_MELS);
    for (let n = 0; n < N_MELS; n++)
      row[n] = fac * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N_MELS));
    D.push(row);
  }
  return D;
}

const FB = melFilterbank();   // pre-computed once
const DCT = buildDctMatrix(); // pre-computed once

// ── 8. Main export ───────────────────────────────────────
/**
 * Returns Float32Array(26): [mfccs_mean_0..12, mfccs_std_0..12]
 * Input y: Float32Array of exactly SEGMENT_LEN samples at SR=16000
 */
export function extract26MfccFeatures(y) {
  // STFT power
  const { S, nFrames } = stftPower(y);

  // Mel spectrogram (power)
  const melPow = FB.map(filt => {
    const row = new Float64Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      let v = 0;
      for (let k = 0; k < filt.length; k++) v += filt[k] * S[t][k];
      row[t] = v;
    }
    return row;
  });

  // power_to_db
  const melDb = powerToDb(melPow, nFrames);

  // DCT → MFCC [N_MFCC][nFrames]
  const mfcc = DCT.map(dRow => {
    const row = new Float64Array(nFrames);
    for (let t = 0; t < nFrames; t++) {
      let v = 0;
      for (let m = 0; m < N_MELS; m++) v += dRow[m] * melDb[m][t];
      row[t] = v;
    }
    return row;
  });

  // mean + std (population, ddof=0) per coefficient
  const features = new Float32Array(26);
  for (let c = 0; c < N_MFCC; c++) {
    let sum = 0;
    for (let t = 0; t < nFrames; t++) sum += mfcc[c][t];
    const mean = sum / nFrames;
    let sq = 0;
    for (let t = 0; t < nFrames; t++) sq += (mfcc[c][t] - mean) ** 2;
    features[c] = mean;
    features[c + N_MFCC] = Math.sqrt(sq / nFrames);
  }
  return features;
}

// ── 9. Decode audio file → Float32Array @ SR=16000 ───────
export async function decodeAudioFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  // Request SR=16000 so Web Audio resamples for us
  const ctx = new AudioCtx({ sampleRate: SR });
  const decoded = await ctx.decodeAudioData(arrayBuffer);
  ctx.close();

  // Mix down to mono
  const ch0 = decoded.getChannelData(0);
  if (decoded.numberOfChannels === 1) return new Float32Array(ch0);
  const ch1 = decoded.getChannelData(1);
  const mono = new Float32Array(ch0.length);
  for (let i = 0; i < ch0.length; i++) mono[i] = (ch0[i] + ch1[i]) / 2;
  return mono;
}