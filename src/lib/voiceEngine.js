/**
 * Speaker verification engine (fully client side, no backend).
 *
 * Registration asks the user to speak their phone number, so verification is
 * text-dependent: the same digit sequence is spoken both times. That lets a
 * classic front-end do real work without a neural speaker-embedding model:
 *
 *   1. DTW over CMVN-normalised MFCC + delta + delta-delta frames.
 *      Time-aligns the two utterances and scores how closely the spectral
 *      trajectory matches. This carries most of the discriminative power.
 *   2. Long-term speaker statistics (per-cepstrum mean and spread).
 *      A text-independent summary of the speaker vocal tract, used as a
 *      secondary score so a mimicked cadence alone cannot pass.
 *
 * Scores fuse into [0, 1] and are compared against a threshold that is
 * calibrated per user from the self-consistency of their enrolment samples.
 */

import {
  SAMPLE_RATE,
  addDeltas,
  cmvn,
  computeMfcc,
  normalizeGain,
  rms,
  trimSilence,
} from "./dsp.js";

export const MIN_SPEECH_SEC = 1.0;
export const MAX_ANALYSIS_SEC = 6;
export const MIN_RMS = 0.012; // below this the recording is effectively silence
export const ENROLL_SAMPLES = 3;

const DTW_WEIGHT = 0.75;
const STAT_WEIGHT = 0.25;
// Distance-to-similarity decay constants, calibrated so that a genuine repeat
// lands around 0.7-0.9 and a different speaker around 0.2-0.5. See voice-test.
const DTW_SIGMA = 0.9;
const STAT_SIGMA = 0.02;
const THRESHOLD_FLOOR = 0.3;
const THRESHOLD_CEILING = 0.88;
const MAX_FRAMES = 520; // ~5.2 s of frames; bounds DTW cost on phones

/* ========================= Audio decoding ========================= */

/**
 * Decode a recorded Blob to mono Float32 PCM at SAMPLE_RATE.
 * Decoding and resampling are split because iOS Safari ignores the sampleRate
 * hint on an AudioContext used for decodeAudioData.
 *
 * @param {Blob} blob
 * @returns {Promise<Float32Array>}
 */
export async function decodeToPcm(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  if (!arrayBuffer.byteLength) throw new Error("Recording is empty");

  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) throw new Error("Web Audio is not supported in this browser");

  const decodeCtx = new Ctx();
  let decoded;
  try {
    decoded = await new Promise((resolve, reject) => {
      // Callback form for Safari, which historically lacks the promise form.
      const maybePromise = decodeCtx.decodeAudioData(arrayBuffer, resolve, reject);
      if (maybePromise && typeof maybePromise.then === "function") {
        maybePromise.then(resolve, reject);
      }
    });
  } finally {
    decodeCtx.close().catch(() => {});
  }

  // Downmix to mono.
  const channels = decoded.numberOfChannels;
  const mono = new Float32Array(decoded.length);
  for (let c = 0; c < channels; c++) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < data.length; i++) mono[i] += data[i] / channels;
  }

  if (Math.abs(decoded.sampleRate - SAMPLE_RATE) < 1) return mono;

  // Resample to 16 kHz through an offline graph.
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const targetLength = Math.max(
    1,
    Math.round((mono.length * SAMPLE_RATE) / decoded.sampleRate),
  );
  const offline = new OfflineCtx(1, targetLength, SAMPLE_RATE);
  const buffer = offline.createBuffer(1, mono.length, decoded.sampleRate);
  buffer.copyToChannel(mono, 0);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

/* ========================= Feature pipeline ========================= */

/** Uniformly subsample a frame sequence down to `max` frames. */
function capFrames(frames, max) {
  if (frames.length <= max) return frames;
  const out = [];
  for (let i = 0; i < max; i++) {
    out.push(frames[Math.floor((i * frames.length) / max)]);
  }
  return out;
}

function l2Normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

// Static cepstra used for the speaker-statistics vector. c0 is log energy, so
// it is skipped: it describes the recording level, not the speaker.
const STAT_LO = 1;
const STAT_HI = 20;

/**
 * Long-term speaker statistics: the per-dimension mean and standard deviation
 * of the static cepstra, concatenated and L2-normalised.
 *
 * These are taken before CMVN. Registration and verification happen on the same
 * device in this build, so the cepstral means carry real speaker information
 * (vocal tract resonances) rather than just channel colour.
 *
 * @param {Float32Array[]} mfcc raw MFCC frames
 * @returns {Float32Array} 38-dim unit vector
 */
function speakerStats(mfcc) {
  const dims = STAT_HI - STAT_LO;
  const n = mfcc.length;
  const vec = new Float32Array(dims * 2);
  if (n === 0) return vec;

  for (let j = 0; j < dims; j++) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += mfcc[i][STAT_LO + j];
    mean /= n;
    let variance = 0;
    for (let i = 0; i < n; i++) {
      const v = mfcc[i][STAT_LO + j] - mean;
      variance += v * v;
    }
    vec[j] = mean;
    vec[dims + j] = Math.sqrt(variance / n);
  }
  return l2Normalize(vec);
}

/**
 * Feature extraction for one utterance of 16 kHz mono PCM.
 * Split out from analyzeRecording so it can run without a DOM audio stack.
 *
 * @param {Float32Array} raw
 * @returns {{frames: Float32Array[], stats: Float32Array, durationSec: number, level: number}}
 */
export function analyzePcm(raw) {
  const level = rms(raw);
  const speech = normalizeGain(trimSilence(raw));
  const durationSec = speech.length / SAMPLE_RATE;

  const limited =
    durationSec > MAX_ANALYSIS_SEC ? speech.slice(0, MAX_ANALYSIS_SEC * SAMPLE_RATE) : speech;

  const mfcc = computeMfcc(limited);
  if (mfcc.length < 20) {
    return { frames: [], stats: new Float32Array(0), durationSec, level };
  }

  // DTW runs on fully normalised (mean and variance) features so that the
  // per-frame distance is comparable across dimensions and recordings.
  const frames = capFrames(cmvn(addDeltas(mfcc)), MAX_FRAMES);
  return { frames, stats: speakerStats(mfcc), durationSec, level };
}

/**
 * Full analysis of one recorded Blob.
 *
 * @param {Blob} blob
 * @returns {Promise<{frames: Float32Array[], stats: Float32Array, durationSec: number, level: number}>}
 */
export async function analyzeRecording(blob) {
  return analyzePcm(await decodeToPcm(blob));
}

/**
 * Reject unusable recordings before they reach the matcher.
 * @returns {string|null} an error message, or null when the sample is usable
 */
export function checkSampleQuality(analysis) {
  if (!analysis.frames.length) return "Could not hear any speech. Please record again.";
  if (analysis.level < MIN_RMS) return "Recording is too quiet. Speak closer to the mic.";
  if (analysis.durationSec < MIN_SPEECH_SEC) {
    return "Too short. Say the full number at a normal pace.";
  }
  return null;
}

/* ============================ Matching ============================ */

function cosine(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den < 1e-9 ? 0 : dot / den;
}

/**
 * Banded DTW over two CMVN frame sequences, using mean squared difference as
 * the local distance. The Sakoe-Chiba band bounds the cost and rules out
 * pathological alignments where a short utterance stretches across a long one.
 *
 * Squared Euclidean is used rather than cosine because on z-scored features it
 * spreads genuine and impostor pairs far more widely: matched frames land near
 * 0.3-0.9, unrelated frames near 2.0 (the expectation for independent
 * unit-variance vectors).
 *
 * @returns {number} mean per-frame squared distance along the optimal path
 */
function dtwDistance(a, b) {
  const n = a.length;
  const m = b.length;
  if (!n || !m) return Infinity;

  const band = Math.max(30, Math.round(Math.max(n, m) * 0.14));
  const INF = 1e9;
  // Rolling rows of (accumulated cost, path length).
  let prevCost = new Float32Array(m + 1).fill(INF);
  let prevLen = new Float32Array(m + 1);
  let curCost = new Float32Array(m + 1);
  let curLen = new Float32Array(m + 1);
  prevCost[0] = 0;

  for (let i = 1; i <= n; i++) {
    curCost.fill(INF);
    curLen.fill(0);
    const centre = Math.round((i * m) / n);
    const jStart = Math.max(1, centre - band);
    const jEnd = Math.min(m, centre + band);
    const rowA = a[i - 1];

    for (let j = jStart; j <= jEnd; j++) {
      // Mean squared difference across feature dimensions.
      const rowB = b[j - 1];
      let acc = 0;
      for (let k = 0; k < rowA.length; k++) {
        const diff = rowA[k] - rowB[k];
        acc += diff * diff;
      }
      const d = acc / rowA.length;

      let bestCost = prevCost[j - 1];
      let bestLen = prevLen[j - 1];
      if (prevCost[j] < bestCost) {
        bestCost = prevCost[j];
        bestLen = prevLen[j];
      }
      if (curCost[j - 1] < bestCost) {
        bestCost = curCost[j - 1];
        bestLen = curLen[j - 1];
      }
      if (bestCost >= INF) continue;

      curCost[j] = bestCost + d;
      curLen[j] = bestLen + 1;
    }

    const t1 = prevCost;
    prevCost = curCost;
    curCost = t1;
    const t2 = prevLen;
    prevLen = curLen;
    curLen = t2;
  }

  if (prevCost[m] >= INF || prevLen[m] === 0) return Infinity;
  return prevCost[m] / prevLen[m];
}

/**
 * Fused similarity between two analysed recordings, in [0, 1].
 *
 * Both components map a distance through exp(-d / sigma) rather than a linear
 * rescale. A linear map of cosine similarity pushed every pair above 0.9 and
 * left no usable gap between genuine and impostor; the exponential spends its
 * dynamic range on the region the distances actually occupy.
 *
 * @returns {{score: number, dtw: number, stat: number, dtwDistance: number, statDistance: number}}
 */
export function compareAnalyses(a, b) {
  const dDtw = dtwDistance(a.frames, b.frames);
  const dtw = Number.isFinite(dDtw) ? Math.exp(-dDtw / DTW_SIGMA) : 0;

  // Both stat vectors are unit length, so 1 - cos is half their squared distance.
  const dStat = 1 - cosine(a.stats, b.stats);
  const stat = Math.exp(-dStat / STAT_SIGMA);

  return {
    score: DTW_WEIGHT * dtw + STAT_WEIGHT * stat,
    dtw,
    stat,
    dtwDistance: dDtw,
    statDistance: dStat,
  };
}

/* ======================== Template storage ======================== */

// CMVN frames are z-scored, so almost all values sit inside +/-5. A scale of
// 24 covers +/-5.3 before clipping and keeps the quantisation step at 0.04 sigma.
const QUANT_SCALE = 24;

/** Quantise frames to int8 and base64-encode so templates fit in localStorage. */
export function packFrames(frames) {
  if (!frames.length) return { n: 0, d: 0, data: "" };
  const d = frames[0].length;
  const bytes = new Uint8Array(frames.length * d);
  let k = 0;
  for (const f of frames) {
    for (let j = 0; j < d; j++) {
      const q = Math.max(-127, Math.min(127, Math.round(f[j] * QUANT_SCALE)));
      bytes[k++] = q < 0 ? q + 256 : q;
    }
  }
  let binary = "";
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return { n: frames.length, d, data: btoa(binary) };
}

/** Inverse of packFrames. */
export function unpackFrames(packed) {
  if (!packed || !packed.n) return [];
  const binary = atob(packed.data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const frames = [];
  for (let i = 0; i < packed.n; i++) {
    const f = new Float32Array(packed.d);
    for (let j = 0; j < packed.d; j++) {
      const b = bytes[i * packed.d + j];
      f[j] = (b > 127 ? b - 256 : b) / QUANT_SCALE;
    }
    frames.push(f);
  }
  return frames;
}

function packStats(stats) {
  return Array.from(stats, (v) => Number(v.toFixed(5)));
}

/** Rebuild the in-memory analysis shape from a stored sample. */
function restoreSample(stored) {
  return {
    frames: unpackFrames(stored.frames),
    stats: Float32Array.from(stored.stats || []),
    durationSec: stored.durationSec || 0,
    level: stored.level || 0,
  };
}

/* ========================= Enrol and verify ========================= */

/**
 * Build a voice template from several recordings of the same phrase.
 * The acceptance threshold is derived from how consistent the samples are with
 * each other, then clamped so a lucky or unlucky enrolment cannot make the
 * system trivially loose or impossible to pass.
 *
 * @param {Array} analyses output of analyzeRecording, one per sample
 * @returns {{samples: Array, threshold: number, selfSimilarity: number}}
 */
export function buildVoiceTemplate(analyses) {
  if (analyses.length < 2) {
    throw new Error("At least two voice samples are required");
  }

  const pairScores = [];
  for (let i = 0; i < analyses.length; i++) {
    for (let j = i + 1; j < analyses.length; j++) {
      pairScores.push(compareAnalyses(analyses[i], analyses[j]).score);
    }
  }
  const mean = pairScores.reduce((a, b) => a + b, 0) / pairScores.length;
  const variance =
    pairScores.reduce((acc, s) => acc + (s - mean) ** 2, 0) / pairScores.length;
  const std = Math.sqrt(variance);

  // Sit a margin below the user own consistency. The proportional term keeps
  // the margin meaningful for a user whose samples score high, while the fixed
  // terms stop a freakishly consistent enrolment from setting an unpassable bar.
  const margin = Math.max(0.06, 2.5 * std, 0.1 * mean);
  const threshold = Math.min(THRESHOLD_CEILING, Math.max(THRESHOLD_FLOOR, mean - margin));

  return {
    samples: analyses.map((a) => ({
      frames: packFrames(a.frames),
      stats: packStats(a.stats),
      durationSec: Number(a.durationSec.toFixed(2)),
      level: Number(a.level.toFixed(4)),
    })),
    threshold: Number(threshold.toFixed(4)),
    selfSimilarity: Number(mean.toFixed(4)),
  };
}

/**
 * Score a probe recording against a stored template.
 * Uses the mean of the two best sample scores, which is more stable than the
 * single best (one lucky match) or the mean of all (one bad enrolment sample).
 *
 * @param {object} analysis output of analyzeRecording for the probe
 * @param {object} template stored voice template
 */
export function verifyAgainstTemplate(analysis, template) {
  if (!template?.samples?.length) throw new Error("No enrolled voice for this user");

  const perSample = template.samples.map((stored) =>
    compareAnalyses(analysis, restoreSample(stored)),
  );
  const sorted = [...perSample].sort((a, b) => b.score - a.score);
  const top = sorted.slice(0, Math.min(2, sorted.length));
  const score = top.reduce((acc, s) => acc + s.score, 0) / top.length;

  return {
    score,
    threshold: template.threshold,
    accepted: score >= template.threshold,
    best: sorted[0],
    perSample: perSample.map((s) => Number(s.score.toFixed(4))),
  };
}
