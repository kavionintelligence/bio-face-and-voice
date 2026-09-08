/**
 * Face recognition engine.
 *
 * Same method as the reference data-sharing client: MediaPipe face landmarks
 * drive a 5-point similarity warp to the ArcFace canonical 112x112 template,
 * then ArcFace (face.onnx, w600k_mbf) produces a 512-dim L2-normalised
 * embedding that is matched with cosine similarity.
 *
 * Preprocessing is kept byte-identical to the reference so the model behaves
 * the same way it did there:
 *   resize to 112x112 via 5-point warp -> (pixel - 127.5) / 128 -> NCHW float32
 */

// wasm-only entry: this app never uses the WebGPU/WebGL backends, and the
// full build drags in a much larger runtime.
import * as ort from "onnxruntime-web/wasm";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// Vite serves .wasm with a MIME type onnxruntime rejects; pull the binaries from a CDN.
// The version is read from the bundled runtime so the JS glue and the .wasm
// binaries can never drift apart when the dependency is bumped.
const ORT_VERSION = ort.env?.versions?.web || "1.24.2";
if (typeof ort.env?.wasm !== "undefined") {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  // Single-threaded: SharedArrayBuffer needs COOP/COEP headers that a plain
  // static deploy does not set, and multi-threaded ORT silently falls back anyway.
  ort.env.wasm.numThreads = 1;
}

const MODEL_PATH = "/face.onnx";
const VISION_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";
const LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

/** Cosine similarity above which two faces are treated as the same person. */
export const FACE_MATCH_THRESHOLD = 0.42;
export const ENROLL_FRAMES = 8;
export const VERIFY_FRAMES = 6;

// ArcFace canonical 5-point reference for 112x112 (InsightFace standard).
const ARCFACE_DST = [
  [38.2946, 51.6963], // left eye
  [73.5318, 51.5014], // right eye
  [56.0252, 71.7366], // nose tip
  [41.5493, 92.3655], // left mouth
  [70.7299, 92.2041], // right mouth
];

// Face Mesh landmark indices matching the reference alignment script.
const LM_LEFT_EYE = [33, 133];
const LM_RIGHT_EYE = [362, 263];
const LM_NOSE = 4;
const LM_LEFT_MOUTH = 57;
const LM_RIGHT_MOUTH = 287;

/* ========================= Model loading ========================= */

let sessionPromise = null;
let landmarkerPromise = null;
let lastTimestampMs = 0;

export async function getArcFaceSession() {
  if (sessionPromise) return sessionPromise;
  sessionPromise = ort.InferenceSession
    .create(MODEL_PATH, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    })
    .catch((e) => {
      sessionPromise = null;
      const msg = e?.message || String(e);
      throw new Error(`Could not load the face model (${MODEL_PATH}): ${msg}`);
    });
  return sessionPromise;
}

export async function getFaceLandmarker() {
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(VISION_WASM);
    return FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: LANDMARKER_MODEL, delegate: "CPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
  })().catch((e) => {
    landmarkerPromise = null;
    throw new Error(`Could not load face tracking: ${e?.message || e}`);
  });
  return landmarkerPromise;
}

/** Warm both models up front so the capture screen is not stuck on first frame. */
export async function preloadFaceModels(onProgress) {
  onProgress?.("Loading face tracking...");
  await getFaceLandmarker();
  onProgress?.("Loading face recognition model...");
  await getArcFaceSession();
  onProgress?.("Ready");
}

/**
 * Run the landmarker on the current video frame.
 * @param {HTMLVideoElement} video
 * @returns {Promise<Array<{x:number,y:number,z:number}>|null>}
 */
export async function detectLandmarks(video) {
  const landmarker = await getFaceLandmarker();
  if (!video || video.readyState < 2 || !video.videoWidth) return null;
  // VIDEO mode requires strictly increasing timestamps, which performance.now()
  // alone does not guarantee across paused/resumed streams.
  const now = performance.now();
  lastTimestampMs = now > lastTimestampMs ? now : lastTimestampMs + 16;
  const result = landmarker.detectForVideo(video, lastTimestampMs);
  const faces = result?.faceLandmarks;
  return faces && faces.length ? faces[0] : null;
}

/* ========================= Alignment ========================= */

/**
 * Estimate a 2x3 similarity transform (src -> dst), the closed-form equivalent
 * of cv2.estimateAffinePartial2D.
 */
function estimateSimilarityTransform(src, dst) {
  const n = src.length;
  let sx2 = 0;
  let sxy = 0;
  let sdtx = 0;
  let sx = 0;
  let sy = 0;
  let stx = 0;
  let sty = 0;
  for (let i = 0; i < n; i++) {
    const [x, y] = src[i];
    const [tx, ty] = dst[i];
    sx2 += x * x + y * y;
    sxy += x * tx + y * ty;
    sdtx += x * ty - y * tx;
    sx += x;
    sy += y;
    stx += tx;
    sty += ty;
  }
  const D = sx2 * n - (sx * sx + sy * sy);
  if (Math.abs(D) < 1e-10) return null;
  const a = (sxy * n - sx * stx - sy * sty) / D;
  const b = (sdtx * n - sx * sty + sy * stx) / D;
  const tx = (stx - a * sx + b * sy) / n;
  const ty = (sty - a * sy - b * sx) / n;
  return [a, -b, tx, b, a, ty];
}

/** Warp a source canvas by a 2x3 affine matrix into a fresh 112x112 canvas. */
function warpTo112(sourceCanvas, M) {
  const [a, b, tx, c, d, ty] = M;
  const out = document.createElement("canvas");
  out.width = 112;
  out.height = 112;
  const ctx = out.getContext("2d", { willReadFrequently: true });
  ctx.setTransform(a, c, b, d, tx, ty);
  ctx.drawImage(sourceCanvas, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return out;
}

/**
 * Pull the 5 alignment points out of a landmark set, in pixel coordinates.
 * @param {Array} landmarks normalised landmarks from the face landmarker
 */
export function extract5Points(landmarks, w, h) {
  const px = (idx) => [landmarks[idx].x * w, landmarks[idx].y * h];
  const mid = (pair) => {
    const a = px(pair[0]);
    const b = px(pair[1]);
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  };

  const src5 = [mid(LM_LEFT_EYE), mid(LM_RIGHT_EYE), px(LM_NOSE), px(LM_LEFT_MOUTH), px(LM_RIGHT_MOUTH)];

  // The front camera preview is mirrored; swap left/right pairs when it is.
  if (src5[0][0] > src5[1][0]) {
    [src5[0], src5[1]] = [src5[1], src5[0]];
    [src5[3], src5[4]] = [src5[4], src5[3]];
  }
  return src5;
}

/** NCHW float32 tensor from a 112x112 canvas, normalised (pixel - 127.5) / 128. */
function canvasToNchwTensor(canvas112) {
  const ctx = canvas112.getContext("2d", { willReadFrequently: true });
  const pixels = ctx.getImageData(0, 0, 112, 112).data;
  const N = 112 * 112;
  const tensor = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    tensor[i] = (pixels[i * 4] - 127.5) / 128.0;
    tensor[N + i] = (pixels[i * 4 + 1] - 127.5) / 128.0;
    tensor[2 * N + i] = (pixels[i * 4 + 2] - 127.5) / 128.0;
  }
  return tensor;
}

/* ========================= Quality gate ========================= */

/**
 * Geometric quality checks. Only frames that pass are embedded, which is what
 * keeps enrolment templates tight and verification stable.
 *
 * @returns {{ok: boolean, message: string, iod: number, yaw: number, pitch: number, roll: number}}
 */
export function assessQuality(landmarks, w, h) {
  const [leftEye, rightEye, nose, leftMouth, rightMouth] = extract5Points(landmarks, w, h);

  const dx = rightEye[0] - leftEye[0];
  const dy = rightEye[1] - leftEye[1];
  const iodPx = Math.hypot(dx, dy);
  const iod = iodPx / w; // inter-ocular distance as a fraction of frame width

  const roll = Math.atan2(dy, dx); // radians, 0 when the eye line is level

  const eyeMid = [(leftEye[0] + rightEye[0]) / 2, (leftEye[1] + rightEye[1]) / 2];
  const mouthMid = [(leftMouth[0] + rightMouth[0]) / 2, (leftMouth[1] + rightMouth[1]) / 2];

  // Nose offset from the eye midline, in units of inter-ocular distance.
  const yaw = iodPx > 1 ? (nose[0] - eyeMid[0]) / iodPx : 0;
  // Where the nose sits between the eye line and the mouth line; ~0.5 when level.
  const faceHeight = Math.hypot(mouthMid[0] - eyeMid[0], mouthMid[1] - eyeMid[1]);
  const pitch = faceHeight > 1 ? (nose[1] - eyeMid[1]) / faceHeight - 0.55 : 0;

  const cx = (eyeMid[0] + mouthMid[0]) / 2 / w;
  const cy = (eyeMid[1] + mouthMid[1]) / 2 / h;

  if (iod < 0.16) return fail("Move closer to the camera", iod, yaw, pitch, roll);
  if (iod > 0.42) return fail("Move a little further back", iod, yaw, pitch, roll);
  if (cx < 0.25 || cx > 0.75 || cy < 0.2 || cy > 0.8) {
    return fail("Centre your face in the circle", iod, yaw, pitch, roll);
  }
  if (Math.abs(yaw) > 0.22) return fail("Look straight at the camera", iod, yaw, pitch, roll);
  if (Math.abs(pitch) > 0.22) return fail("Keep your head level", iod, yaw, pitch, roll);
  if (Math.abs(roll) > 0.26) return fail("Straighten your head", iod, yaw, pitch, roll);

  return { ok: true, message: "Hold still", iod, yaw, pitch, roll };
}

function fail(message, iod, yaw, pitch, roll) {
  return { ok: false, message, iod, yaw, pitch, roll };
}

/* ========================= Embedding ========================= */

export function l2Normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm < 1e-10) return vec;
  return vec.map((v) => v / norm);
}

/** Cosine similarity between two L2-normalised vectors. */
export function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * Draw the current video frame into a canvas sized to the source video.
 * @returns {HTMLCanvasElement}
 */
export function videoToCanvas(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

/**
 * ArcFace 512-dim L2-normalised embedding for one frame.
 *
 * @param {HTMLCanvasElement} sourceCanvas full camera frame
 * @param {Array} landmarks normalised landmarks for that frame
 * @returns {Promise<number[]|null>}
 */
export async function getFaceEmbedding(sourceCanvas, landmarks) {
  if (!sourceCanvas || !landmarks) return null;
  const src5 = extract5Points(landmarks, sourceCanvas.width, sourceCanvas.height);
  const M = estimateSimilarityTransform(src5, ARCFACE_DST);
  if (!M) return null;

  const aligned = warpTo112(sourceCanvas, M);
  const tensor = canvasToNchwTensor(aligned);

  const session = await getArcFaceSession();
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const output = await session.run({
    [inputName]: new ort.Tensor("float32", tensor, [1, 3, 112, 112]),
  });
  return l2Normalize(Array.from(output[outputName].data));
}

/* ========================= Templates ========================= */

const EMB_SCALE = 400; // int8 quantisation; ArcFace components sit well inside +/-0.3

/** Quantise a 512-dim embedding to base64 int8 so templates stay small. */
export function packEmbedding(embedding) {
  const bytes = new Uint8Array(embedding.length);
  for (let i = 0; i < embedding.length; i++) {
    const q = Math.max(-127, Math.min(127, Math.round(embedding[i] * EMB_SCALE)));
    bytes[i] = q < 0 ? q + 256 : q;
  }
  let binary = "";
  const CHUNK = 8192;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

export function unpackEmbedding(b64) {
  const binary = atob(b64);
  const out = new Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    const b = binary.charCodeAt(i);
    out[i] = (b > 127 ? b - 256 : b) / EMB_SCALE;
  }
  return l2Normalize(out); // re-normalise away the quantisation drift
}

/** Element-wise mean of embeddings, re-normalised to unit length. */
export function meanEmbedding(embeddings) {
  const d = embeddings[0].length;
  const mean = new Array(d).fill(0);
  for (const e of embeddings) for (let i = 0; i < d; i++) mean[i] += e[i];
  for (let i = 0; i < d; i++) mean[i] /= embeddings.length;
  return l2Normalize(mean);
}

/**
 * Build a stored face template from the frames captured during registration.
 * @param {number[][]} embeddings
 */
export function buildFaceTemplate(embeddings) {
  if (!embeddings.length) throw new Error("No face frames captured");
  const mean = meanEmbedding(embeddings);
  const selfSims = embeddings.map((e) => cosineSimilarity(e, mean));
  return {
    mean: packEmbedding(mean),
    samples: embeddings.map(packEmbedding),
    threshold: FACE_MATCH_THRESHOLD,
    selfSimilarity: Number(
      (selfSims.reduce((a, b) => a + b, 0) / selfSims.length).toFixed(4),
    ),
  };
}

/**
 * Match probe frames against a stored face template.
 * Blends similarity to the enrolment centroid with the best individual sample
 * match, so a single unusual enrolment frame neither carries nor blocks a match.
 *
 * @param {number[][]} probeEmbeddings frames captured during verification
 * @param {object} template stored face template
 */
export function verifyFaceAgainstTemplate(probeEmbeddings, template) {
  if (!template?.samples?.length) throw new Error("No enrolled face for this user");
  if (!probeEmbeddings.length) throw new Error("No face frames captured");

  const storedMean = unpackEmbedding(template.mean);
  const storedSamples = template.samples.map(unpackEmbedding);
  const probeMean = meanEmbedding(probeEmbeddings);

  const meanSim = cosineSimilarity(probeMean, storedMean);
  let bestSim = -1;
  for (const probe of probeEmbeddings) {
    for (const stored of storedSamples) {
      const s = cosineSimilarity(probe, stored);
      if (s > bestSim) bestSim = s;
    }
  }

  const score = 0.6 * meanSim + 0.4 * bestSim;
  const threshold = template.threshold ?? FACE_MATCH_THRESHOLD;
  return {
    score,
    meanSim,
    bestSim,
    threshold,
    accepted: score >= threshold,
  };
}

/**
 * Search every stored template for the closest face. Used to warn about a
 * duplicate enrolment and to power identify-by-face.
 *
 * @param {number[][]} probeEmbeddings
 * @param {Array} users records that carry a face template
 */
export function findBestFaceMatch(probeEmbeddings, users) {
  let best = null;
  for (const user of users) {
    if (!user.face?.samples?.length) continue;
    const result = verifyFaceAgainstTemplate(probeEmbeddings, user.face);
    if (!best || result.score > best.score) best = { user, ...result };
  }
  return best;
}
