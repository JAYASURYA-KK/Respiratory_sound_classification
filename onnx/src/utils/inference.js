/**
 * inference.js — Run ONNX model via onnxruntime-web
 *
 * Model: respiratory_knn.onnx
 *   Input : float_input    [1, 26]  float32
 *   Output: label          [1]      int64
 *           probabilities  [1, 6]   float32
 *
 * Classes (label_encoder.classes_):
 *   0: Bronchiectasis
 *   1: Bronchiolitis
 *   2: COPD
 *   3: Healthy
 *   4: Pneumonia
 *   5: URTI
 */

import * as ort from 'onnxruntime-web';

export const CLASSES = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI'];

let session = null;

export async function loadModel(modelUrl) {
  if (session) return session;
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
  });
  return session;
}

/**
 * @param {Float32Array} features - shape (26,)
 * @returns {{ predictedClass: string, classIndex: number, probabilities: number[] }}
 */
export async function runInference(features) {
  if (!session) throw new Error('Model not loaded');

  const tensor = new ort.Tensor('float32', features, [1, 26]);
  const results = await session.run({ float_input: tensor });

  // label output is int64 — BigInt64Array or Int32Array depending on runtime
  const labelData = results['label'].data;
  const classIndex = Number(labelData[0]);

  const probData = results['probabilities'].data; // Float32Array length 6
  const probabilities = Array.from(probData);

  return {
    predictedClass: CLASSES[classIndex],
    classIndex,
    probabilities,
  };
}