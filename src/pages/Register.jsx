import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import FaceCapture from "../components/FaceCapture.jsx";
import VoiceRecorder from "../components/VoiceRecorder.jsx";
import {
  ENROLL_SAMPLES,
  analyzeRecording,
  buildVoiceTemplate,
  checkSampleQuality,
  compareAnalyses,
} from "../lib/voiceEngine.js";
import {
  ENROLL_FRAMES,
  buildFaceTemplate,
  findBestFaceMatch,
} from "../lib/faceEngine.js";
import { findUserByName, listUsers, nameKey, saveUser } from "../lib/storage.js";

const STEPS = ["Details", "Voice", "Face", "Done"];

// A loose sanity bar. It is deliberately well below the verification threshold:
// it exists to catch silence, a different person, or a completely different
// phrase, not to police normal variation between repeats.
const ENROLL_CONSISTENCY_MIN = 0.3;

export default function Register() {
  const navigate = useNavigate();

  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [number, setNumber] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [formError, setFormError] = useState("");

  const [voiceSamples, setVoiceSamples] = useState([]); // analysed recordings
  const [voiceBusy, setVoiceBusy] = useState("");
  const [voiceError, setVoiceError] = useState("");

  const [faceError, setFaceError] = useState("");
  const [duplicateWarning, setDuplicateWarning] = useState("");
  const [saved, setSaved] = useState(null);

  /* ---------------------------- step 1: details ---------------------------- */

  const submitDetails = (e) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (trimmed.length < 2) {
      setFormError("Enter a name with at least 2 characters.");
      return;
    }
    const digits = number.replace(/\D/g, "");
    if (digits.length < 4) {
      setFormError("Enter at least 4 digits — this is the number you will speak.");
      return;
    }
    const existing = findUserByName(trimmed);
    if (existing && !overwrite) {
      setFormError(
        `"${existing.name}" is already registered. Tick the box below to replace their biometrics.`,
      );
      return;
    }
    setFormError("");
    setStep(1);
  };

  /* ----------------------------- step 2: voice ----------------------------- */

  const handleRecorded = useCallback(
    async (blob) => {
      setVoiceError("");
      setVoiceBusy("Analysing...");
      try {
        const analysis = await analyzeRecording(blob);

        const quality = checkSampleQuality(analysis);
        if (quality) {
          setVoiceError(quality);
          return;
        }

        // Every sample must resemble the ones already banked, otherwise the
        // template averages two different utterances and verification never passes.
        for (const previous of voiceSamples) {
          const { score } = compareAnalyses(analysis, previous);
          if (score < ENROLL_CONSISTENCY_MIN) {
            setVoiceError(
              "That did not match your earlier recording. Say the same number, the same way, and try again.",
            );
            return;
          }
        }

        setVoiceSamples((prev) => [...prev, analysis]);
      } catch (e) {
        setVoiceError(e?.message || "Could not process that recording.");
      } finally {
        setVoiceBusy("");
      }
    },
    [voiceSamples],
  );

  /* ------------------------------ step 3: face ------------------------------ */

  const handleFaceCaptured = useCallback(
    (embeddings) => {
      setFaceError("");
      try {
        const faceTemplate = buildFaceTemplate(embeddings);
        const voiceTemplate = buildVoiceTemplate(voiceSamples);

        // Flag an obvious re-enrolment under a second name; still allow it,
        // because twins and shared demo devices are real.
        const others = listUsers().filter((u) => u.nameKey !== nameKey(name));
        const match = findBestFaceMatch(embeddings, others);
        if (match?.accepted) {
          setDuplicateWarning(
            `This face also matches "${match.user.name}" (${(match.score * 100).toFixed(0)}%).`,
          );
        }

        const user = saveUser({
          name,
          spokenNumber: number,
          face: faceTemplate,
          voice: voiceTemplate,
        });
        setSaved({ user, faceTemplate, voiceTemplate });
        setStep(3);
      } catch (e) {
        setFaceError(e?.message || "Could not save the registration.");
      }
    },
    [voiceSamples, name, number],
  );

  /* -------------------------------- render -------------------------------- */

  return (
    <div className="stack">
      <StepBar step={step} />

      {step === 0 && (
        <form className="stack" onSubmit={submitDetails}>
          <div className="hero">
            <h2>Register</h2>
            <p>Your name identifies you. Your voice and face verify you.</p>
          </div>

          <div className="card stack">
            <div className="field">
              <label htmlFor="reg-name">Full name</label>
              <input
                id="reg-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Varun Khatri"
                autoComplete="name"
                autoCapitalize="words"
              />
              <div className="hint">You will type this name again to verify.</div>
            </div>

            <div className="field">
              <label htmlFor="reg-number">Number to speak</label>
              <input
                id="reg-number"
                value={number}
                onChange={(e) => setNumber(e.target.value)}
                placeholder="e.g. 98765 43210"
                inputMode="numeric"
                autoComplete="tel"
              />
              <div className="hint">
                You will read these digits aloud {ENROLL_SAMPLES} times now, and once again at
                verification.
              </div>
            </div>

            {findUserByName(name) && (
              <label className="muted" style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <input
                  type="checkbox"
                  checked={overwrite}
                  onChange={(e) => setOverwrite(e.target.checked)}
                  style={{ width: 18, height: 18 }}
                />
                Replace the existing registration for this name
              </label>
            )}
          </div>

          {formError && <div className="alert error">{formError}</div>}

          <button type="submit" className="btn">
            Continue
          </button>
          <button type="button" className="btn ghost" onClick={() => navigate("/")}>
            Cancel
          </button>
        </form>
      )}

      {step === 1 && (
        <div className="stack">
          <div className="hero">
            <h2>Record your voice</h2>
            <p>
              Sample {Math.min(voiceSamples.length + 1, ENROLL_SAMPLES)} of {ENROLL_SAMPLES}. Read
              the number aloud each time.
            </p>
          </div>

          <div className="progress">
            <div style={{ width: `${(voiceSamples.length / ENROLL_SAMPLES) * 100}%` }} />
          </div>

          <VoiceRecorder
            prompt={number}
            onRecorded={handleRecorded}
            disabled={!!voiceBusy || voiceSamples.length >= ENROLL_SAMPLES}
            busyLabel={voiceBusy}
          />

          {voiceError && <div className="alert error">{voiceError}</div>}

          {voiceSamples.length > 0 && (
            <div className="card tight">
              {voiceSamples.map((s, i) => (
                <div className="score-row" key={i}>
                  <span className="name">Sample {i + 1}</span>
                  <span className="val pass">{s.durationSec.toFixed(1)}s captured</span>
                </div>
              ))}
            </div>
          )}

          <div className="btn-row">
            <button
              type="button"
              className="btn secondary"
              onClick={() => {
                setVoiceSamples([]);
                setVoiceError("");
                setStep(0);
              }}
            >
              Back
            </button>
            <button
              type="button"
              className="btn"
              disabled={voiceSamples.length < ENROLL_SAMPLES}
              onClick={() => setStep(2)}
            >
              Continue
            </button>
          </div>

          {voiceSamples.length > 0 && voiceSamples.length < ENROLL_SAMPLES && (
            <button
              type="button"
              className="btn ghost"
              onClick={() => {
                setVoiceSamples((prev) => prev.slice(0, -1));
                setVoiceError("");
              }}
            >
              Redo last sample
            </button>
          )}
        </div>
      )}

      {step === 2 && (
        <div className="stack">
          <div className="hero">
            <h2>Capture your face</h2>
            <p>
              Hold steady inside the oval. We take {ENROLL_FRAMES} good frames — no photos are
              stored, only a numeric template.
            </p>
          </div>

          <FaceCapture
            count={ENROLL_FRAMES}
            onComplete={handleFaceCaptured}
            onCancel={() => setStep(1)}
            actionLabel="Capture face"
          />

          {faceError && <div className="alert error">{faceError}</div>}
        </div>
      )}

      {step === 3 && saved && (
        <div className="stack">
          <div className="card stack center">
            <div className="result-icon pass">&#10003;</div>
            <h2 style={{ margin: 0 }}>{saved.user.name} is registered</h2>
            <p className="muted">Saved in this browser. No server, no upload.</p>
          </div>

          <div className="card">
            <div className="step-label">Template quality</div>
            <div className="score-row">
              <span className="name">Face frames</span>
              <span className="val">{saved.faceTemplate.samples.length}</span>
            </div>
            <div className="score-row">
              <span className="name">Face consistency</span>
              <span className="val">{pct(saved.faceTemplate.selfSimilarity)}</span>
            </div>
            <div className="score-row">
              <span className="name">Voice samples</span>
              <span className="val">{saved.voiceTemplate.samples.length}</span>
            </div>
            <div className="score-row">
              <span className="name">Voice consistency</span>
              <span className="val">{pct(saved.voiceTemplate.selfSimilarity)}</span>
            </div>
            <div className="score-row">
              <span className="name">Voice threshold</span>
              <span className="val">{pct(saved.voiceTemplate.threshold)}</span>
            </div>
          </div>

          {saved.voiceTemplate.selfSimilarity < 0.45 && (
            <div className="alert warn">
              Your three voice samples were not very consistent with each other, so verification may
              be unreliable. Re-register somewhere quieter, saying the number the same way each time.
            </div>
          )}

          {saved.faceTemplate.selfSimilarity < 0.8 && (
            <div className="alert warn">
              The face frames varied a lot. Re-register in even lighting, holding still, for a
              tighter template.
            </div>
          )}

          {duplicateWarning && <div className="alert warn">{duplicateWarning}</div>}

          <button type="button" className="btn" onClick={() => navigate("/verify")}>
            Verify now
          </button>
          <button type="button" className="btn secondary" onClick={() => navigate("/")}>
            Back to home
          </button>
        </div>
      )}
    </div>
  );
}

function StepBar({ step }) {
  return (
    <>
      <div className="steps">
        {STEPS.map((label, i) => (
          <div
            key={label}
            className={`step ${i === step ? "active" : ""} ${i < step ? "done" : ""}`}
          />
        ))}
      </div>
      <div className="step-label">
        Step {step + 1} of {STEPS.length} &middot; {STEPS[step]}
      </div>
    </>
  );
}

function pct(v) {
  return `${(v * 100).toFixed(0)}%`;
}
