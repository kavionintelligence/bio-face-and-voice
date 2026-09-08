import { useCallback, useEffect, useRef, useState } from "react";
import {
  assessQuality,
  detectLandmarks,
  getFaceEmbedding,
  preloadFaceModels,
  videoToCanvas,
} from "../lib/faceEngine.js";

const CAPTURE_INTERVAL_MS = 320; // spacing between embedded frames
const OVAL_RATIO = 0.66; // guide oval width as a fraction of the frame

/**
 * Live camera view that collects `count` good-quality ArcFace embeddings.
 *
 * The same component backs registration and verification; only the frame count
 * and the copy differ. Frames are embedded on the fly rather than stored as
 * images, so nothing leaves this component but 512-dim vectors.
 *
 * @param {{count: number, onComplete: (embeddings: number[][]) => void, onCancel?: () => void, actionLabel?: string}} props
 */
export default function FaceCapture({ count, onComplete, onCancel, actionLabel = "Start capture" }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafRef = useRef(0);
  const busyRef = useRef(false);
  const lastCaptureRef = useRef(0);
  const embeddingsRef = useRef([]);
  const capturingRef = useRef(false);
  const doneRef = useRef(false);

  const [loading, setLoading] = useState("Starting camera...");
  const [error, setError] = useState("");
  const [status, setStatus] = useState({ ok: false, message: "Looking for a face..." });
  const [captured, setCaptured] = useState(0);
  const [capturing, setCapturing] = useState(false);

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  /* ------------------------- camera + models ------------------------- */

  useEffect(() => {
    let cancelled = false;

    (async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("This browser cannot access the camera. Try Chrome or Safari.");
        setLoading("");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          video.setAttribute("playsinline", "true"); // iOS refuses fullscreen-less video without this
          await video.play().catch(() => {});
        }

        await preloadFaceModels((msg) => {
          if (!cancelled) setLoading(msg);
        });
        if (!cancelled) setLoading("");
      } catch (e) {
        if (cancelled) return;
        setLoading("");
        setError(describeCameraError(e));
      }
    })();

    return () => {
      cancelled = true;
      stop();
    };
  }, [stop]);

  /* --------------------------- frame loop --------------------------- */

  useEffect(() => {
    if (loading || error) return undefined;
    let cancelled = false;

    const tick = async () => {
      if (cancelled) return;
      const video = videoRef.current;

      if (video && video.readyState >= 2 && !busyRef.current) {
        busyRef.current = true;
        try {
          const landmarks = await detectLandmarks(video);
          if (!cancelled) {
            if (!landmarks) {
              setStatus({ ok: false, message: "No face detected" });
              drawOverlay(canvasRef.current, video, null, false);
            } else {
              const q = assessQuality(landmarks, video.videoWidth, video.videoHeight);
              setStatus({ ok: q.ok, message: q.message });
              drawOverlay(canvasRef.current, video, landmarks, q.ok);

              const now = performance.now();
              if (
                capturingRef.current &&
                q.ok &&
                !doneRef.current &&
                now - lastCaptureRef.current > CAPTURE_INTERVAL_MS
              ) {
                lastCaptureRef.current = now;
                const frame = videoToCanvas(video);
                const embedding = await getFaceEmbedding(frame, landmarks);
                if (!cancelled && embedding) {
                  embeddingsRef.current.push(embedding);
                  setCaptured(embeddingsRef.current.length);
                  if (embeddingsRef.current.length >= count) {
                    doneRef.current = true;
                    capturingRef.current = false;
                    setCapturing(false);
                    stop();
                    onComplete(embeddingsRef.current);
                    return;
                  }
                }
              }
            }
          }
        } catch (e) {
          if (!cancelled) {
            setError(e?.message || String(e));
            capturingRef.current = false;
            setCapturing(false);
          }
        } finally {
          busyRef.current = false;
        }
      }

      if (!cancelled) rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [loading, error, count, onComplete, stop]);

  const begin = () => {
    embeddingsRef.current = [];
    doneRef.current = false;
    lastCaptureRef.current = 0;
    setCaptured(0);
    capturingRef.current = true;
    setCapturing(true);
  };

  const progress = Math.round((captured / count) * 100);

  return (
    <div className="stack">
      <div className="camera">
        <video ref={videoRef} playsInline muted autoPlay />
        <canvas ref={canvasRef} className="overlay" />

        {capturing && (
          <div className="camera-badge">
            {captured} / {count}
          </div>
        )}

        {loading ? (
          <div className="camera-status pulsing">{loading}</div>
        ) : error ? null : (
          <div className={`camera-status ${status.ok ? "ok" : "warn"}`}>{status.message}</div>
        )}
      </div>

      {error && <div className="alert error">{error}</div>}

      {capturing && (
        <div className="progress">
          <div style={{ width: `${progress}%` }} />
        </div>
      )}

      {!error && (
        <div className="btn-row">
          {onCancel && (
            <button type="button" className="btn secondary" onClick={onCancel}>
              Back
            </button>
          )}
          <button type="button" className="btn" onClick={begin} disabled={!!loading || capturing}>
            {capturing ? (
              <>
                <span className="spinner" /> Capturing...
              </>
            ) : (
              actionLabel
            )}
          </button>
        </div>
      )}
    </div>
  );
}

/* ---------------------------- overlay ---------------------------- */

/**
 * Draw the framing oval plus a light landmark scatter, mirrored to match the
 * CSS-flipped video preview.
 */
function drawOverlay(canvas, video, landmarks, ok) {
  if (!canvas || !video.videoWidth) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1); // match the mirrored preview

  // The video is object-fit: cover, so replicate that mapping for landmarks.
  const scale = Math.max(w / video.videoWidth, h / video.videoHeight);
  const dw = video.videoWidth * scale;
  const dh = video.videoHeight * scale;
  const ox = (w - dw) / 2;
  const oy = (h - dh) / 2;

  if (landmarks) {
    ctx.fillStyle = ok ? "rgba(47, 208, 138, 0.55)" : "rgba(148, 163, 196, 0.4)";
    for (let i = 0; i < landmarks.length; i += 4) {
      const p = landmarks[i];
      ctx.fillRect(ox + p.x * dw - 1, oy + p.y * dh - 1, 2, 2);
    }
  }

  ctx.restore();

  // Framing oval, drawn unmirrored since it is symmetric.
  const rx = (w * OVAL_RATIO) / 2;
  const ry = rx * 1.32;
  ctx.strokeStyle = ok ? "rgba(47, 208, 138, 0.95)" : "rgba(148, 163, 196, 0.5)";
  ctx.lineWidth = 3 * dpr;
  ctx.setLineDash(ok ? [] : [10 * dpr, 9 * dpr]);
  ctx.beginPath();
  ctx.ellipse(w / 2, h * 0.46, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
}

function describeCameraError(e) {
  const name = e?.name || "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Camera permission was denied. Allow camera access in your browser settings and reload.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No front camera was found on this device.";
  }
  if (name === "NotReadableError") {
    return "The camera is already in use by another app. Close it and try again.";
  }
  if (!window.isSecureContext) {
    return "Camera access needs HTTPS. Open this page over https:// or on localhost.";
  }
  return e?.message || "Could not start the camera.";
}
