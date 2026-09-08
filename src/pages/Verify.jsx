import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import FaceCapture from "../components/FaceCapture.jsx";
import VoiceRecorder from "../components/VoiceRecorder.jsx";
import {
  analyzeRecording,
  checkSampleQuality,
  verifyAgainstTemplate,
} from "../lib/voiceEngine.js";
import { VERIFY_FRAMES, verifyFaceAgainstTemplate } from "../lib/faceEngine.js";
import { countUsers, findUserByName, logAttempt } from "../lib/storage.js";

const STEPS = ["Identify", "Voice", "Face", "Result"];

export default function Verify() {
  const navigate = useNavigate();

  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [user, setUser] = useState(null);
  const [lookupError, setLookupError] = useState("");

  const [voiceResult, setVoiceResult] = useState(null);
  const [voiceBusy, setVoiceBusy] = useState("");
  const [voiceError, setVoiceError] = useState("");
  const [voiceAttempts, setVoiceAttempts] = useState(0);

  const [faceResult, setFaceResult] = useState(null);
  const [faceError, setFaceError] = useState("");

  /* --------------------------- step 1: identify --------------------------- */

  const lookup = (e) => {
    e.preventDefault();
    const found = findUserByName(name);
    if (!found) {
      const total = countUsers();
      setLookupError(
        total === 0
          ? "No one is registered in this browser yet. Register first."
          : `No registration found for "${name.trim()}". Names must match exactly (case does not matter).`,
      );
      return;
    }
    setLookupError("");
    setUser(found);
    setStep(1);
  };

  /* ----------------------------- step 2: voice ----------------------------- */

  const handleRecorded = useCallback(
    async (blob) => {
      setVoiceError("");
      setVoiceBusy("Matching...");
      try {
        const analysis = await analyzeRecording(blob);
        const quality = checkSampleQuality(analysis);
        if (quality) {
          setVoiceError(quality);
          return;
        }
        const result = verifyAgainstTemplate(analysis, user.voice);
        setVoiceResult(result);
        setVoiceAttempts((n) => n + 1);
        if (result.accepted) setStep(2);
      } catch (e) {
        setVoiceError(e?.message || "Could not process that recording.");
      } finally {
        setVoiceBusy("");
      }
    },
    [user],
  );

  /* ------------------------------ step 3: face ------------------------------ */

  const handleFaceCaptured = useCallback(
    (embeddings) => {
      setFaceError("");
      try {
        const result = verifyFaceAgainstTemplate(embeddings, user.face);
        setFaceResult(result);
        setStep(3);
        logAttempt({
          userId: user.id,
          name: user.name,
          voiceScore: Number((voiceResult?.score ?? 0).toFixed(4)),
          voicePass: !!voiceResult?.accepted,
          faceScore: Number(result.score.toFixed(4)),
          facePass: result.accepted,
          passed: !!voiceResult?.accepted && result.accepted,
        });
      } catch (e) {
        setFaceError(e?.message || "Could not match that face.");
      }
    },
    [user, voiceResult],
  );

  const reset = () => {
    setStep(0);
    setUser(null);
    setVoiceResult(null);
    setFaceResult(null);
    setVoiceError("");
    setFaceError("");
    setVoiceAttempts(0);
  };

  const passed = !!voiceResult?.accepted && !!faceResult?.accepted;

  /* -------------------------------- render -------------------------------- */

  return (
    <div className="stack">
      <StepBar step={step} />

      {step === 0 && (
        <form className="stack" onSubmit={lookup}>
          <div className="hero">
            <h2>Verify</h2>
            <p>Enter your name, then pass the voice and face checks.</p>
          </div>

          <div className="card">
            <div className="field">
              <label htmlFor="ver-name">Registered name</label>
              <input
                id="ver-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Varun Khatri"
                autoComplete="name"
                autoCapitalize="words"
              />
              <div className="hint">{countUsers()} registration(s) stored in this browser.</div>
            </div>
          </div>

          {lookupError && <div className="alert error">{lookupError}</div>}

          <button type="submit" className="btn">
            Find me
          </button>
          <button type="button" className="btn ghost" onClick={() => navigate("/")}>
            Cancel
          </button>
        </form>
      )}

      {step === 1 && user && (
        <div className="stack">
          <div className="hero">
            <h2>Hello, {user.name}</h2>
            <p>Read your number aloud, the same way you did at registration.</p>
          </div>

          <VoiceRecorder
            prompt={user.spokenNumber || "your registered number"}
            onRecorded={handleRecorded}
            disabled={!!voiceBusy}
            busyLabel={voiceBusy}
          />

          {voiceError && <div className="alert error">{voiceError}</div>}

          {voiceResult && !voiceResult.accepted && (
            <>
              <div className="alert error">
                Voice did not match ({pct(voiceResult.score)} against a {pct(voiceResult.threshold)}{" "}
                threshold). Try again in a quieter place, at the same pace.
              </div>
              {voiceAttempts >= 2 && (
                <button type="button" className="btn secondary" onClick={() => setStep(2)}>
                  Continue to face check anyway
                </button>
              )}
            </>
          )}

          <button type="button" className="btn ghost" onClick={reset}>
            Back
          </button>
        </div>
      )}

      {step === 2 && user && (
        <div className="stack">
          <div className="hero">
            <h2>Face check</h2>
            <p>Hold steady inside the oval until {VERIFY_FRAMES} frames are captured.</p>
          </div>

          <FaceCapture
            count={VERIFY_FRAMES}
            onComplete={handleFaceCaptured}
            onCancel={() => setStep(1)}
            actionLabel="Verify face"
          />

          {faceError && <div className="alert error">{faceError}</div>}
        </div>
      )}

      {step === 3 && user && faceResult && (
        <div className="stack">
          <div className="card stack center">
            <div className={`result-icon ${passed ? "pass" : "fail"}`}>
              {passed ? "✓" : "✕"}
            </div>
            <h2 style={{ margin: 0 }}>{passed ? "Verified" : "Not verified"}</h2>
            <p className="muted">
              {passed
                ? `Voice and face both matched ${user.name}.`
                : "Both checks must pass to verify this identity."}
            </p>
          </div>

          <div className="card">
            <div className="step-label">Voice</div>
            <ScoreRow
              label="Match score"
              value={pct(voiceResult?.score ?? 0)}
              pass={voiceResult?.accepted}
            />
            <ScoreRow label="Threshold" value={pct(voiceResult?.threshold ?? 0)} />
            <ScoreRow label="Spectral alignment" value={pct(voiceResult?.best?.dtw ?? 0)} />
            <ScoreRow label="Speaker statistics" value={pct(voiceResult?.best?.stat ?? 0)} />
          </div>

          <div className="card">
            <div className="step-label">Face</div>
            <ScoreRow label="Match score" value={pct(faceResult.score)} pass={faceResult.accepted} />
            <ScoreRow label="Threshold" value={pct(faceResult.threshold)} />
            <ScoreRow label="Vs. enrolled average" value={pct(faceResult.meanSim)} />
            <ScoreRow label="Best single frame" value={pct(faceResult.bestSim)} />
          </div>

          <button type="button" className="btn" onClick={reset}>
            Verify someone else
          </button>
          <button type="button" className="btn secondary" onClick={() => navigate("/")}>
            Back to home
          </button>
        </div>
      )}
    </div>
  );
}

function ScoreRow({ label, value, pass }) {
  const cls = pass === undefined ? "" : pass ? "pass" : "fail";
  return (
    <div className="score-row">
      <span className="name">{label}</span>
      <span className={`val ${cls}`}>{value}</span>
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
  return `${(v * 100).toFixed(1)}%`;
}
