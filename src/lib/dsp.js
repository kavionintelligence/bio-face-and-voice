/**
 * Signal-processing primitives for the voice engine.
 *
 * The reference project ran speaker embeddings through WavLM + ECAPA ONNX models
 * on a server. Those weights are not shipped with the client, and this build has
 * no backend, so speaker features are computed here in plain JS: a standard
 * MFCC front-end (pre-emphasis -> framing -> FFT -> mel filterbank -> log -> DCT)
 * plus deltas and cepstral mean/variance normalisation.
 *
 * Everything is written against Float32Array so it stays fast on mobile.
 */

export const SAMPLE_RATE = 16000;
export const FRAME_MS = 25;
export const HOP_MS = 10;
export const FRAME_LEN = Math.round((SAMPLE_RATE * FRAME_MS) / 1000); // 400
export const HOP_LEN = Math.round((SAMPLE_RATE * HOP_MS) / 1000); // 160
export const FFT_SIZE = 512; // next power of two >= FRAME_LEN
export const N_MELS = 40;
export const N_MFCC = 20;
export const MEL_LOW_HZ = 20;
export const MEL_HIGH_HZ = 7600;
export const PRE_EMPHASIS = 0.97;

/* ============================ FFT ============================ */

const bitReverseCache = new Map();

function bitReverseTable(n) {
  if (bitReverseCache.has(n)) return bitReverseCache.get(n);
  const table = new Uint32Array(n);
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    let x = i;
    let r = 0;
    for (let b = 0; b < bits; b++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    table[i] = r;
  }
  bitReverseCache.set(n, table);
  return table;
}

const twiddleCache = new Map();

function twiddles(n) {
  if (twiddleCache.has(n)) return twiddleCache.get(n);
  const cos = new Float32Array(n / 2);
  const sin = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((-2 * Math.PI * i) / n);
    sin[i] = Math.sin((-2 * Math.PI * i) / n);
  }
  const t = { cos, sin };
  twiddleCache.set(n, t);
  return t;
}

/**
 * In-place iterative radix-2 FFT.
 * @param {Float32Array} re real part, length must be a power of two
 * @param {Float32Array} im imaginary part, same length
 */
export function fft(re, im) {
  const n = re.length;
  const rev = bitReverseTable(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  const { cos, sin } = twiddles(n);
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = i, k = 0; j < i + half; j++, k += step) {
        const wr = cos[k];
        const wi = sin[k];
        const tr = wr * re[j + half] - wi * im[j + half];
        const ti = wr * im[j + half] + wi * re[j + half];
        re[j + half] = re[j] - tr;
        im[j + half] = im[j] - ti;
        re[j] += tr;
        im[j] += ti;
      }
    }
  }
}

/* ====================== Window / filterbank ====================== */

let hammingCache = null;

function hamming(n) {
  if (hammingCache && hammingCache.length === n) return hammingCache;
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  hammingCache = w;
  return w;
}

const hzToMel = (hz) => 2595 * Math.log10(1 + hz / 700);
const melToHz = (mel) => 700 * (Math.pow(10, mel / 2595) - 1);

let melBankCache = null;

/**
 * Triangular mel filterbank over the FFT_SIZE/2+1 power-spectrum bins.
 * Stored sparsely as {start, weights} so the per-frame loop stays tight.
 */
function melFilterbank() {
  if (melBankCache) return melBankCache;
  const nBins = FFT_SIZE / 2 + 1;
  const lowMel = hzToMel(MEL_LOW_HZ);
  const highMel = hzToMel(MEL_HIGH_HZ);
  const points = new Float32Array(N_MELS + 2);
  for (let i = 0; i < N_MELS + 2; i++) {
    const mel = lowMel + ((highMel - lowMel) * i) / (N_MELS + 1);
    points[i] = Math.floor(((FFT_SIZE + 1) * melToHz(mel)) / SAMPLE_RATE);
  }
  const bank = [];
  for (let m = 1; m <= N_MELS; m++) {
    const left = points[m - 1];
    const centre = points[m];
    const right = points[m + 1];
    const start = Math.max(0, left);
    const end = Math.min(nBins - 1, right);
    const weights = new Float32Array(Math.max(0, end - start + 1));
    for (let k = start; k <= end; k++) {
      let w = 0;
      if (k >= left && k < centre && centre > left) w = (k - left) / (centre - left);
      else if (k >= centre && k <= right && right > centre) w = (right - k) / (right - centre);
      else if (k === centre) w = 1;
      weights[k - start] = w;
    }
    bank.push({ start, weights });
  }
  melBankCache = bank;
  return bank;
}

let dctCache = null;

/** Orthonormal DCT-II matrix, N_MFCC x N_MELS. */
function dctMatrix() {
  if (dctCache) return dctCache;
  const m = [];
  for (let k = 0; k < N_MFCC; k++) {
    const row = new Float32Array(N_MELS);
    const scale = k === 0 ? Math.sqrt(1 / N_MELS) : Math.sqrt(2 / N_MELS);
    for (let n = 0; n < N_MELS; n++) {
      row[n] = scale * Math.cos((Math.PI * k * (n + 0.5)) / N_MELS);
    }
    m.push(row);
  }
  dctCache = m;
  return m;
}

/* ========================= Voice activity ========================= */

/**
 * Energy-based VAD. Keeps the loudest contiguous speech region and drops
 * leading/trailing silence, which is what makes two recordings of the same
 * phrase line up under DTW.
 *
 * @param {Float32Array} signal
 * @returns {Float32Array} trimmed signal
 */
export function trimSilence(signal) {
  const win = HOP_LEN;
  const nWin = Math.floor(signal.length / win);
  if (nWin < 4) return signal;

  const energies = new Float32Array(nWin);
  for (let w = 0; w < nWin; w++) {
    let sum = 0;
    const base = w * win;
    for (let i = 0; i < win; i++) {
      const v = signal[base + i];
      sum += v * v;
    }
    energies[w] = Math.sqrt(sum / win);
  }

  const sorted = Float32Array.from(energies).sort();
  const noiseFloor = sorted[Math.floor(nWin * 0.15)] || 0;
  const peak = sorted[Math.floor(nWin * 0.97)] || 0;
  if (peak <= 1e-5) return signal;
  // Threshold sits between the noise floor and the speech peak.
  const threshold = Math.max(noiseFloor * 2.2, peak * 0.12);

  let first = -1;
  let last = -1;
  for (let w = 0; w < nWin; w++) {
    if (energies[w] >= threshold) {
      if (first === -1) first = w;
      last = w;
    }
  }
  if (first === -1 || last <= first) return signal;

  // 100 ms of padding on each side keeps plosives and trailing fricatives.
  const pad = Math.round((SAMPLE_RATE * 0.1) / win);
  const startWin = Math.max(0, first - pad);
  const endWin = Math.min(nWin - 1, last + pad);
  const start = startWin * win;
  const end = Math.min(signal.length, (endWin + 1) * win);
  if (end - start < SAMPLE_RATE * 0.4) return signal;
  return signal.slice(start, end);
}

/** Peak-normalise to +/-1 so recording gain does not shift the features. */
export function normalizeGain(signal) {
  let max = 0;
  for (let i = 0; i < signal.length; i++) {
    const v = Math.abs(signal[i]);
    if (v > max) max = v;
  }
  if (max < 1e-6) return signal;
  const out = new Float32Array(signal.length);
  const scale = 0.98 / max;
  for (let i = 0; i < signal.length; i++) out[i] = signal[i] * scale;
  return out;
}

/**
 * RMS level of a signal, used as a recording-quality gate.
 * @returns {number}
 */
export function rms(signal) {
  let sum = 0;
  for (let i = 0; i < signal.length; i++) sum += signal[i] * signal[i];
  return Math.sqrt(sum / Math.max(1, signal.length));
}

/* ============================ MFCC ============================ */

/**
 * Compute MFCC frames from a 16 kHz mono signal.
 * Output: array of Float32Array(N_MFCC) where c0 is replaced by log frame energy.
 *
 * @param {Float32Array} signal
 * @returns {Float32Array[]}
 */
export function computeMfcc(signal) {
  if (!signal || signal.length < FRAME_LEN) return [];

  // Pre-emphasis flattens the spectral tilt of the glottal source.
  const emph = new Float32Array(signal.length);
  emph[0] = signal[0];
  for (let i = 1; i < signal.length; i++) emph[i] = signal[i] - PRE_EMPHASIS * signal[i - 1];

  const window = hamming(FRAME_LEN);
  const bank = melFilterbank();
  const dct = dctMatrix();
  const nFrames = 1 + Math.floor((emph.length - FRAME_LEN) / HOP_LEN);
  const nBins = FFT_SIZE / 2 + 1;

  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);
  const power = new Float32Array(nBins);
  const logMel = new Float32Array(N_MELS);
  const frames = [];

  for (let f = 0; f < nFrames; f++) {
    const offset = f * HOP_LEN;
    re.fill(0);
    im.fill(0);
    let energy = 0;
    for (let i = 0; i < FRAME_LEN; i++) {
      const s = emph[offset + i];
      energy += s * s;
      re[i] = s * window[i];
    }

    fft(re, im);
    for (let k = 0; k < nBins; k++) power[k] = (re[k] * re[k] + im[k] * im[k]) / FFT_SIZE;

    for (let m = 0; m < N_MELS; m++) {
      const { start, weights } = bank[m];
      let acc = 0;
      for (let w = 0; w < weights.length; w++) acc += power[start + w] * weights[w];
      logMel[m] = Math.log(acc + 1e-10);
    }

    const coeffs = new Float32Array(N_MFCC);
    for (let k = 0; k < N_MFCC; k++) {
      const row = dct[k];
      let acc = 0;
      for (let n = 0; n < N_MELS; n++) acc += row[n] * logMel[n];
      coeffs[k] = acc;
    }
    // c0 tracks loudness only; log frame energy is the more stable stand-in.
    coeffs[0] = Math.log(energy / FRAME_LEN + 1e-10);
    frames.push(coeffs);
  }

  return frames;
}

/**
 * Append first and second order deltas (regression window of 2 frames).
 * @param {Float32Array[]} frames
 * @returns {Float32Array[]} frames of length 3 * N_MFCC
 */
export function addDeltas(frames) {
  const n = frames.length;
  if (n === 0) return [];
  const d = frames[0].length;
  const at = (i) => frames[Math.max(0, Math.min(n - 1, i))];

  const delta = [];
  const denom = 2 * (1 + 4); // 2 * sum(k^2) for k = 1..2
  for (let i = 0; i < n; i++) {
    const out = new Float32Array(d);
    for (let j = 0; j < d; j++) {
      out[j] = (at(i + 1)[j] - at(i - 1)[j] + 2 * (at(i + 2)[j] - at(i - 2)[j])) / denom;
    }
    delta.push(out);
  }

  const atD = (i) => delta[Math.max(0, Math.min(n - 1, i))];
  const out = [];
  for (let i = 0; i < n; i++) {
    const combined = new Float32Array(d * 3);
    combined.set(frames[i], 0);
    combined.set(delta[i], d);
    for (let j = 0; j < d; j++) {
      combined[2 * d + j] =
        (atD(i + 1)[j] - atD(i - 1)[j] + 2 * (atD(i + 2)[j] - atD(i - 2)[j])) / denom;
    }
    out.push(combined);
  }
  return out;
}

/**
 * Cepstral mean and variance normalisation, applied per utterance.
 * This is what removes microphone and channel differences between the phone
 * used at registration and the one used at verification.
 *
 * @param {Float32Array[]} frames
 * @returns {Float32Array[]} new normalised frames
 */
export function cmvn(frames) {
  const n = frames.length;
  if (n === 0) return [];
  const d = frames[0].length;
  const mean = new Float32Array(d);
  const std = new Float32Array(d);

  for (const f of frames) for (let j = 0; j < d; j++) mean[j] += f[j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  for (const f of frames) {
    for (let j = 0; j < d; j++) {
      const v = f[j] - mean[j];
      std[j] += v * v;
    }
  }
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / n) || 1;

  return frames.map((f) => {
    const out = new Float32Array(d);
    for (let j = 0; j < d; j++) out[j] = (f[j] - mean[j]) / std[j];
    return out;
  });
}
