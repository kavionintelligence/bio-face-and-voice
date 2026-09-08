/**
 * Browser-local user store.
 *
 * There is no backend: every enrolled user, their face template and their voice
 * template live in localStorage on the device that registered them. Templates
 * are quantised (see faceEngine.packEmbedding / voiceEngine.packFrames) so a
 * full record is a few tens of kilobytes rather than a few hundred.
 */

const STORAGE_KEY = "h2a.biometrics.users.v1";
const SCHEMA_VERSION = 1;

/** Collapse a display name to a lookup key: case and spacing insensitive. */
export function nameKey(name) {
  return String(name || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

function readRaw() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { version: SCHEMA_VERSION, users: [] };
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.users)) {
      return { version: SCHEMA_VERSION, users: [] };
    }
    return parsed;
  } catch {
    // Corrupt or unreadable storage should not brick the app.
    return { version: SCHEMA_VERSION, users: [] };
  }
}

function writeRaw(db) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(db));
    return true;
  } catch (e) {
    if (e?.name === "QuotaExceededError" || e?.code === 22) {
      throw new Error(
        "Browser storage is full. Delete an enrolled user and try again.",
      );
    }
    throw new Error(`Could not save to browser storage: ${e?.message || e}`);
  }
}

/** Every enrolled user, newest first. */
export function listUsers() {
  return readRaw().users.slice().sort((a, b) => b.createdAt - a.createdAt);
}

export function countUsers() {
  return readRaw().users.length;
}

/** Look a user up by name; matching ignores case and extra whitespace. */
export function findUserByName(name) {
  const key = nameKey(name);
  if (!key) return null;
  return readRaw().users.find((u) => u.nameKey === key) || null;
}

export function getUserById(id) {
  return readRaw().users.find((u) => u.id === id) || null;
}

/**
 * Insert or replace a user record.
 * @param {{name: string, spokenNumber?: string, face: object, voice: object}} record
 * @returns {object} the saved user
 */
export function saveUser(record) {
  const db = readRaw();
  const key = nameKey(record.name);
  if (!key) throw new Error("Name is required");

  const existingIndex = db.users.findIndex((u) => u.nameKey === key);
  const user = {
    id: existingIndex >= 0 ? db.users[existingIndex].id : newId(),
    name: String(record.name).trim(),
    nameKey: key,
    spokenNumber: record.spokenNumber ? String(record.spokenNumber).trim() : "",
    face: record.face,
    voice: record.voice,
    createdAt: existingIndex >= 0 ? db.users[existingIndex].createdAt : Date.now(),
    updatedAt: Date.now(),
  };

  if (existingIndex >= 0) db.users[existingIndex] = user;
  else db.users.push(user);

  db.version = SCHEMA_VERSION;
  writeRaw(db);
  return user;
}

export function deleteUser(id) {
  const db = readRaw();
  db.users = db.users.filter((u) => u.id !== id);
  writeRaw(db);
}

export function clearAllUsers() {
  writeRaw({ version: SCHEMA_VERSION, users: [] });
}

/** Approximate size of the store, for the storage readout on the users screen. */
export function storageBytes() {
  try {
    return (localStorage.getItem(STORAGE_KEY) || "").length;
  } catch {
    return 0;
  }
}

/** Detect private-mode / disabled storage early so we can warn instead of failing late. */
export function isStorageAvailable() {
  try {
    const probe = "__h2a_probe__";
    localStorage.setItem(probe, "1");
    localStorage.removeItem(probe);
    return true;
  } catch {
    return false;
  }
}

function newId() {
  if (crypto?.randomUUID) return crypto.randomUUID();
  return `u_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

/* ===================== Verification history ===================== */

const LOG_KEY = "h2a.biometrics.log.v1";
const LOG_LIMIT = 50;

/** Append a verification attempt to the local audit log. */
export function logAttempt(entry) {
  try {
    const raw = localStorage.getItem(LOG_KEY);
    const list = raw ? JSON.parse(raw) : [];
    list.unshift({ ...entry, at: Date.now() });
    localStorage.setItem(LOG_KEY, JSON.stringify(list.slice(0, LOG_LIMIT)));
  } catch {
    // The log is a nice-to-have; never let it break a verification.
  }
}

export function listAttempts() {
  try {
    const raw = localStorage.getItem(LOG_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function clearAttempts() {
  try {
    localStorage.removeItem(LOG_KEY);
  } catch {
    /* ignore */
  }
}
