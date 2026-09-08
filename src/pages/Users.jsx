import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  clearAllUsers,
  clearAttempts,
  deleteUser,
  listAttempts,
  listUsers,
  storageBytes,
} from "../lib/storage.js";

export default function Users() {
  const navigate = useNavigate();
  const [users, setUsers] = useState(() => listUsers());
  const [attempts, setAttempts] = useState(() => listAttempts());
  const [confirmClear, setConfirmClear] = useState(false);

  const refresh = () => {
    setUsers(listUsers());
    setAttempts(listAttempts());
  };

  const remove = (user) => {
    deleteUser(user.id);
    refresh();
  };

  const wipe = () => {
    clearAllUsers();
    clearAttempts();
    setConfirmClear(false);
    refresh();
  };

  return (
    <div className="stack">
      <div className="hero">
        <h2>Registered users</h2>
        <p>
          Stored in this browser only, using about {(storageBytes() / 1024).toFixed(0)} KB. Clearing
          site data removes them permanently.
        </p>
      </div>

      <div className="card">
        {users.length === 0 ? (
          <div className="empty">No one is registered yet.</div>
        ) : (
          users.map((user) => (
            <div className="user-row" key={user.id}>
              <div className="avatar">{initials(user.name)}</div>
              <div className="meta">
                <div className="n">{user.name}</div>
                <div className="s">
                  {user.spokenNumber ? `${user.spokenNumber} · ` : ""}
                  {user.face?.samples?.length || 0} face · {user.voice?.samples?.length || 0} voice ·{" "}
                  {new Date(user.createdAt).toLocaleDateString()}
                </div>
              </div>
              <button type="button" className="btn danger small" onClick={() => remove(user)}>
                Delete
              </button>
            </div>
          ))
        )}
      </div>

      {attempts.length > 0 && (
        <div className="card">
          <div className="step-label">Recent attempts</div>
          {attempts.slice(0, 8).map((a, i) => (
            <div className="score-row" key={i}>
              <span className="name">
                {a.name}
                <span className="faint"> · {new Date(a.at).toLocaleTimeString()}</span>
              </span>
              <span className={`pill ${a.passed ? "pass" : "fail"}`}>
                {a.passed ? "PASS" : "FAIL"} {(a.voiceScore * 100).toFixed(0)}/
                {(a.faceScore * 100).toFixed(0)}
              </span>
            </div>
          ))}
        </div>
      )}

      <button type="button" className="btn secondary" onClick={() => navigate("/register")}>
        Register another user
      </button>

      {users.length > 0 &&
        (confirmClear ? (
          <div className="stack">
            <div className="alert warn">
              Delete every registration and the attempt log from this browser?
            </div>
            <div className="btn-row">
              <button type="button" className="btn secondary" onClick={() => setConfirmClear(false)}>
                Keep them
              </button>
              <button type="button" className="btn danger" onClick={wipe}>
                Delete all
              </button>
            </div>
          </div>
        ) : (
          <button type="button" className="btn ghost" onClick={() => setConfirmClear(true)}>
            Clear all data
          </button>
        ))}
    </div>
  );
}

function initials(name) {
  return String(name)
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((p) => p[0])
    .join("")
    .toUpperCase();
}
