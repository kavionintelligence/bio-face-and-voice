# H2A Biometrics

Face and voice registration + verification that runs entirely in a mobile browser.
No backend, no uploads: every template is stored in `localStorage` on the device.
Built to deploy to Vercel as a static site.

## Flow

**Register** — name -> speak your number 3x -> capture face -> saved locally.
**Verify** — type your name to locate the record -> speak the number -> face check -> result.

Both the voice and face checks must pass for a verification to succeed.

## How the matching works

### Face

Same method as the `data-sharing-client` reference project:

1. **MediaPipe Face Landmarker** (`tasks-vision`, VIDEO mode) tracks 478 landmarks per frame.
2. A geometric quality gate accepts only frames that are well framed and near-frontal
   (inter-ocular distance, centring, yaw, pitch, roll). See `assessQuality` in
   [faceEngine.js](src/lib/faceEngine.js).
3. Five landmarks (eyes, nose, mouth corners) drive a **similarity warp** to the ArcFace
   canonical 112x112 template.
4. **ArcFace** (`public/face.onnx`, w600k_mbf) runs under `onnxruntime-web` and returns a
   512-dim L2-normalised embedding. Preprocessing is byte-identical to the reference:
   `(pixel - 127.5) / 128`, NCHW float32.
5. Registration banks 8 frames; verification captures 6 and scores
   `0.6 * cos(probeMean, enrolledMean) + 0.4 * bestFramePairCosine` against a 0.42 threshold.

### Voice

The reference project ran WavLM + ECAPA + AASIST ONNX models, but those weights ship only
with its server, and this build has no backend. Voice matching is therefore implemented from
scratch in JS ([dsp.js](src/lib/dsp.js), [voiceEngine.js](src/lib/voiceEngine.js)).

Because the user speaks the *same number* at registration and verification, the problem is
text-dependent, which a classic front-end handles well:

1. Decode to 16 kHz mono, energy VAD to trim silence, peak normalise.
2. MFCC front-end: pre-emphasis -> 25 ms / 10 ms framing -> Hamming -> 512-pt FFT ->
   40-band mel filterbank -> log -> DCT-II, giving 20 coefficients (c0 replaced by log energy),
   plus delta and delta-delta -> 60 dims per frame.
3. **DTW** with a Sakoe-Chiba band over CMVN-normalised frames, using mean squared difference
   as the local distance. Carries most of the discriminative power (weight 0.75).
4. **Speaker statistics**: per-cepstrum mean and spread of the static coefficients, L2-normalised
   (weight 0.25).
5. Both distances map through `exp(-d / sigma)` so the score spends its range where the
   distances actually sit. A linear rescale of cosine similarity pushed every pair above 0.9
   and left no usable gap.
6. The acceptance threshold is **calibrated per user** from how consistent their three
   enrolment samples were with each other, clamped to [0.30, 0.88].

Measured on a formant synthesiser (`voice-test.mjs`, speakers differing by vocal tract length
and f0):

| probe                      | score | verdict |
| -------------------------- | ----- | ------- |
| genuine repeat             | 0.61  | accept  |
| same speaker, wrong phrase | 0.33  | reject  |
| similar-sounding speaker   | 0.52  | reject  |
| different speaker          | 0.20  | reject  |

Threshold for that enrolment: 0.55.

## Storage

`localStorage` under `h2a.biometrics.users.v1`. Templates are quantised to int8 and
base64-encoded, so a full user record (8 face embeddings + 3 voice samples) is roughly
70-80 KB. Clearing site data deletes every registration.

## Running locally

```bash
npm install
npm run dev
```

Camera and microphone need a **secure context**. `localhost` counts; a plain-HTTP LAN address
does not. To test on a phone during development, either deploy to Vercel or run
`npm run dev -- --https` and accept the certificate.

### A note on dependencies

Every dependency is pinned to an **exact** version, matching the set that
`data-sharing-client` resolved to, so this tree is known to work together
(vite 7.3.1 / esbuild 0.27.3 / rollup 4.59.0 / react 19.2.4).

There is no `package-lock.json`: on the machine this was built on, `npm install` could not
fetch tarballs from the registry (it hung, then died with npm's "Exit handler never called"),
so `node_modules` was populated by copying the already-installed tree from
`data-sharing-client-main`. Exact pins keep that reproducible. On a normal network, or on
Vercel, a plain `npm install` resolves the same versions. Run `npm install` once somewhere with
working registry access and commit the lockfile it produces.

## Deploying to Vercel

The repo is a standard static Vite build; `vercel.json` adds the SPA rewrite and caches the
13 MB `face.onnx` immutably. A production build is about **13.4 MB**, almost all of it the
model: the JS bundle is 446 KB (142 KB gzipped) and the CSS 8.7 KB.

`faceEngine.js` imports `onnxruntime-web/wasm` (no WebGPU/WebGL backends) and `vite.config.js`
enables the package's `onnxruntime-web-use-extern-wasm` export condition. Without that
condition Vite emits an unused 25 MB `.wasm` asset, because the runtime is pointed at the
jsDelivr copy instead.

```bash
npx vercel --prod
```

Framework preset: **Vite**. Build command `npm run build`, output directory `dist`.
Vercel serves over HTTPS, so camera and mic work on mobile with no extra setup.

## Limitations

- **Per-device storage.** A user registered on one phone cannot verify on another. Adding a
  backend, or an export/import of the JSON record, would fix this.
- **No anti-spoofing.** The reference project ships `antispoofing_ep50.onnx` (15 MB, 10-frame
  sequence) and MiniFASNet models. They are deliberately left out here to keep the mobile
  bundle and per-frame cost down — a printed photo or a recording played back at the mic will
  pass. Add `liveness.js` from the reference client if presentation-attack detection is needed.
- **Voice matching is text-dependent.** Users must say the same number, at a similar pace, in
  reasonably quiet surroundings. It is not as strong as an ECAPA speaker embedding.
- **Thresholds are calibrated against synthetic speech**, not a real evaluation set. The face
  threshold (0.42) follows common ArcFace practice; both are single constants and easy to tune
  in `faceEngine.js` / `voiceEngine.js` once you have real pass/fail data.
