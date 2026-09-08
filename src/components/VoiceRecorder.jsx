import { useCallback, useEffect, useRef, useState } from "react";

const MAX_SECONDS = 6;
const MIN_SECONDS = 1.2;

/**
 * Tap-to-record microphone control.
 *
 * Stops automatically at MAX_SECONDS so a forgotten recording cannot balloon,
 * and refuses anything shorter than MIN_SECONDS because the voice matcher needs
 * a full utterance to align against.
 *
 * @param {{prompt: string, onRecorded: (blob: Blob, seconds: number) => void, disabled?: boolean, busyLabel?: string}} props
 */
export default function VoiceRecorder({ prompt, onRecorded, disabled, busyLabel }) {
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const rafRef = useRef(0);
  const startedAtRef = useRef(0);
  const stopTimerRef = useRef(0);

  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState("");

  const cleanup = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    clearTimeout(stopTimerRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const stopRecording = useCallback(() => {
    clearTimeout(stopTimerRef.current);
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
  }, []);

  const startRecording = useCallback(async () => {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("This browser cannot record audio. Try Chrome or Safari.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          // Keep the browser DSP on: it is applied at registration and at
          // verification alike, so it does not bias the comparison, and it
          // markedly cleans up phone-speaker recordings.
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      // Level meter drives the ring around the mic button.
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      audioCtxRef.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const buffer = new Uint8Array(analyser.frequencyBinCount);

      const mimeType = pickMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const seconds = (performance.now() - startedAtRef.current) / 1000;
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || mimeType || "audio/webm",
        });
        cleanup();
        setRecording(false);
        setElapsed(0);
        setLevel(0);
        if (seconds < MIN_SECONDS) {
          setError("Too short. Hold on and say the whole number.");
          return;
        }
        if (!blob.size) {
          setError("Nothing was recorded. Check your microphone and try again.");
          return;
        }
        onRecorded(blob, seconds);
      };

      recorder.start();
      startedAtRef.current = performance.now();
      setRecording(true);

      const tick = () => {
        analyser.getByteTimeDomainData(buffer);
        let peak = 0;
        for (let i = 0; i < buffer.length; i++) {
          const v = Math.abs(buffer[i] - 128) / 128;
          if (v > peak) peak = v;
        }
        setLevel(peak);
        setElapsed((performance.now() - startedAtRef.current) / 1000);
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);

      stopTimerRef.current = setTimeout(stopRecording, MAX_SECONDS * 1000);
    } catch (e) {
      cleanup();
      setRecording(false);
      setError(describeMicError(e));
    }
  }, [cleanup, onRecorded, stopRecording]);

  const toggle = () => {
    if (disabled) return;
    if (recording) stopRecording();
    else startRecording();
  };

  const ringScale = 1 + Math.min(level, 1) * 0.35;

  return (
    <div className="stack">
      {prompt && (
        <div className="phrase">
          <div className="label">Say out loud</div>
          <div className="value">{prompt}</div>
        </div>
      )}

      <div className="mic">
        <button
          type="button"
          className={`mic-orb ${recording ? "recording" : ""}`}
          onClick={toggle}
          disabled={disabled}
          aria-label={recording ? "Stop recording" : "Start recording"}
        >
          {recording && <span className="ring" style={{ transform: `scale(${ringScale})` }} />}
          <MicIcon active={recording} />
        </button>

        <div className="mic-timer">
          {recording ? `${elapsed.toFixed(1)}s` : busyLabel ? busyLabel : "Tap to record"}
        </div>

        <div className="faint center">
          {recording
            ? `Tap again to stop, or it stops at ${MAX_SECONDS}s`
            : "Speak clearly at a normal pace in a quiet spot"}
        </div>
      </div>

      {error && <div className="alert error">{error}</div>}
    </div>
  );
}

function MicIcon({ active }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke={active ? "#ff5c72" : "#eaf0fb"} strokeWidth="1.8">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" strokeLinecap="round" />
      <path d="M12 18v4" strokeLinecap="round" />
    </svg>
  );
}

/** Pick a container the current browser can actually produce. */
function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4;codecs=mp4a.40.2",
    "audio/mp4", // Safari on iOS
    "audio/ogg;codecs=opus",
  ];
  for (const type of candidates) {
    if (MediaRecorder.isTypeSupported?.(type)) return type;
  }
  return "";
}

function describeMicError(e) {
  const name = e?.name || "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Microphone permission was denied. Allow mic access in your browser settings and reload.";
  }
  if (name === "NotFoundError") return "No microphone was found on this device.";
  if (name === "NotReadableError") {
    return "The microphone is in use by another app. Close it and try again.";
  }
  if (!window.isSecureContext) {
    return "Microphone access needs HTTPS. Open this page over https:// or on localhost.";
  }
  return e?.message || "Could not start recording.";
}
