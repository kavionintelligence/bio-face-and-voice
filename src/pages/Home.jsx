import { Link } from "react-router-dom";
import { countUsers, isStorageAvailable } from "../lib/storage.js";

export default function Home() {
  const total = countUsers();
  const storageOk = isStorageAvailable();
  const secure = typeof window !== "undefined" ? window.isSecureContext : true;

  return (
    <div className="stack">
      <div className="hero">
        <h2>Face and voice verification, entirely in your browser</h2>
        <p>
          Register once with your name, your spoken number and your face. Verification then matches
          both biometrics locally — nothing is uploaded.
        </p>
      </div>

      {!secure && (
        <div className="alert warn">
          This page is not on a secure origin, so the camera and microphone are blocked. Open it over
          https:// or on localhost.
        </div>
      )}

      {!storageOk && (
        <div className="alert warn">
          Browser storage is unavailable — private browsing usually causes this. Registrations cannot
          be saved.
        </div>
      )}

      <Link to="/register" className="btn" style={{ textDecoration: "none" }}>
        Register a new user
      </Link>
      <Link to="/verify" className="btn secondary" style={{ textDecoration: "none" }}>
        Verify an identity
      </Link>

      <div className="card">
        <div className="step-label">How it works</div>
        <Step
          n="1"
          title="Name"
          body="Your name is the lookup key. Verification starts by finding your record."
        />
        <Step
          n="2"
          title="Voice"
          body="You speak your number three times. MFCC features are time-aligned with DTW and combined with speaker statistics."
        />
        <Step
          n="3"
          title="Face"
          body="MediaPipe landmarks align each frame to the ArcFace template; the model returns a 512-dim embedding matched by cosine similarity."
        />
      </div>

      <Link
        to="/users"
        className="btn ghost"
        style={{ textDecoration: "none" }}
      >
        {total === 0 ? "No users registered yet" : `Manage ${total} registered user(s)`}
      </Link>
    </div>
  );
}

function Step({ n, title, body }) {
  return (
    <div style={{ display: "flex", gap: 13, padding: "11px 0" }}>
      <div className="avatar" style={{ width: 30, height: 30, fontSize: 13 }}>
        {n}
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600, fontSize: 14.5 }}>{title}</div>
        <div className="faint" style={{ marginTop: 2 }}>
          {body}
        </div>
      </div>
    </div>
  );
}
