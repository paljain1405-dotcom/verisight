import { useRef, useState } from "react";

const API = import.meta.env.VITE_API_BASE || "";

/* ── Reticle mark used as the logo ──────────────────────────────── */
function Reticle() {
  return (
    <svg className="reticle" viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="11" />
      <circle cx="16" cy="16" r="3.2" className="reticle-dot" />
      <line x1="16" y1="1" x2="16" y2="7" />
      <line x1="16" y1="25" x2="16" y2="31" />
      <line x1="1" y1="16" x2="7" y2="16" />
      <line x1="25" y1="16" x2="31" y2="16" />
    </svg>
  );
}

/* ── Signature element: the authenticity gauge ──────────────────── */
function Gauge({ fakeProb }) {
  // 0 (authentic) -> needle left; 1 (fake) -> needle right.
  const angle = -90 + fakeProb * 180; // degrees, 0 = straight up
  const isFake = fakeProb >= 0.5;
  const pct = Math.round((isFake ? fakeProb : 1 - fakeProb) * 100);

  // Arc path for a 180° dial, radius 90, centre (110,110).
  const arc = (from, to) => {
    const p = (deg) => [110 + 90 * Math.cos(deg), 110 + 90 * Math.sin(deg)];
    const [x1, y1] = p((from * Math.PI) / 180);
    const [x2, y2] = p((to * Math.PI) / 180);
    return `M ${x1} ${y1} A 90 90 0 0 1 ${x2} ${y2}`;
  };

  return (
    <div className="gauge">
      <svg viewBox="0 0 220 130" aria-label={`${pct}% confidence`}>
        <path className="gauge-track real-track" d={arc(180, 270)} />
        <path className="gauge-track fake-track" d={arc(270, 360)} />
        {Array.from({ length: 21 }).map((_, i) => {
          const a = (180 + i * 9) * (Math.PI / 180);
          const r1 = i % 5 === 0 ? 74 : 80;
          return (
            <line key={i} className="tick"
              x1={110 + r1 * Math.cos(a)} y1={110 + r1 * Math.sin(a)}
              x2={110 + 88 * Math.cos(a)} y2={110 + 88 * Math.sin(a)} />
          );
        })}
        <g className="needle" style={{ transform: `rotate(${angle}deg)` }}>
          <line x1="110" y1="110" x2="110" y2="34" />
          <circle cx="110" cy="110" r="6" />
        </g>
      </svg>
      <div className="gauge-readout">
        <span className={`verdict ${isFake ? "is-fake" : "is-real"}`}>
          {isFake ? "LIKELY FAKE" : "LIKELY AUTHENTIC"}
        </span>
        <span className="confidence">{pct}<i>% confidence</i></span>
      </div>
    </div>
  );
}

/* ── Per-frame signal strip for video ───────────────────────────── */
function Timeline({ frames, peak }) {
  const max = frames.length;
  return (
    <div className="timeline">
      <div className="timeline-head">
        <span>per-frame analysis</span>
        <span className="mono">peak {Math.round(peak.fake_probability * 100)}% @ frame {peak.frame}</span>
      </div>
      <div className="strip">
        {frames.map((f, i) => (
          <span key={i}
            className="bar"
            title={`frame ${f.frame}: ${Math.round(f.fake_probability * 100)}% fake`}
            style={{
              height: `${12 + f.fake_probability * 88}%`,
              background: f.fake_probability >= 0.5 ? "var(--fake)" : "var(--real)",
              opacity: 0.35 + f.fake_probability * 0.65,
            }} />
        ))}
      </div>
      <div className="strip-axis"><span>0:00</span><span>{max} frames sampled</span></div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("image");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const pick = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch(`${API}/predict/${mode}`, { method: "POST", body });
      if (!res.ok) throw new Error((await res.json()).detail || "Analysis failed.");
      setResult(await res.json());
    } catch (e) {
      setError(e.message || "Could not reach the detector. Is the backend running on :8000?");
    } finally {
      setBusy(false);
    }
  };

  const fakeProb = result
    ? mode === "image" ? result.fake_probability : result.mean_fake_probability
    : 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Reticle />
          <div>
            <h1>VERISIGHT</h1>
            <p>media authenticity analysis</p>
          </div>
        </div>
        <div className="seg" role="tablist">
          {["image", "video"].map((m) => (
            <button key={m} role="tab" aria-selected={mode === m}
              className={mode === m ? "active" : ""}
              onClick={() => { setMode(m); setResult(null); setFile(null); setPreviewUrl(null); }}>
              {m}
            </button>
          ))}
        </div>
      </header>

      {result && result.model_trained === false && (
        <div className="notice">
          Running on an untrained backbone — scores are placeholders. Train a
          checkpoint (<code>checkpoints/detector.pt</code>) for real accuracy.
        </div>
      )}

      <main className="stage">
        <section className="panel input-panel">
          <div
            className={`drop ${previewUrl ? "has-media" : ""} ${busy ? "scanning" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files[0]); }}
          >
            {previewUrl ? (
              mode === "image"
                ? <img src={showHeatmap && result?.heatmap ? result.heatmap : previewUrl} alt="upload" />
                : <video src={previewUrl} controls muted />
            ) : (
              <div className="drop-empty">
                <Reticle />
                <p>drop a {mode} here, or click to browse</p>
                <span className="mono">{mode === "image" ? "jpg · png · webp" : "mp4 · mov · webm"}</span>
              </div>
            )}
            {busy && <div className="scanline" />}
            <input ref={inputRef} type="file" hidden
              accept={mode === "image" ? "image/*" : "video/*"}
              onChange={(e) => pick(e.target.files[0])} />
          </div>

          <div className="controls">
            <button className="run" disabled={!file || busy} onClick={analyze}>
              {busy ? "analyzing…" : `analyze ${mode}`}
            </button>
            {mode === "image" && result?.heatmap && (
              <label className="toggle">
                <input type="checkbox" checked={showHeatmap}
                  onChange={(e) => setShowHeatmap(e.target.checked)} />
                Grad-CAM overlay
              </label>
            )}
          </div>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel output-panel">
          {result ? (
            <>
              <Gauge fakeProb={fakeProb} />
              {mode === "image" ? (
                <dl className="readout">
                  <div><dt>verdict</dt><dd className="mono">{result.label}</dd></div>
                  <div><dt>fake probability</dt><dd className="mono">{(result.fake_probability * 100).toFixed(1)}%</dd></div>
                  <div><dt>method</dt><dd>CNN + Grad-CAM</dd></div>
                </dl>
              ) : (
                <>
                  <dl className="readout">
                    <div><dt>verdict</dt><dd className="mono">{result.label}</dd></div>
                    <div><dt>mean fake prob.</dt><dd className="mono">{(result.mean_fake_probability * 100).toFixed(1)}%</dd></div>
                    <div><dt>flagged frames</dt><dd className="mono">{(result.fake_frame_ratio * 100).toFixed(0)}%</dd></div>
                  </dl>
                  <Timeline frames={result.timeline} peak={result.peak_frame} />
                </>
              )}
            </>
          ) : (
            <div className="idle">
              <p>awaiting input</p>
              <span className="mono">verdict, confidence and explainability appear here</span>
            </div>
          )}
        </section>
      </main>

      <footer className="foot">
        <span className="mono">CNN backbone · face-aware frame sampling · Grad-CAM explainability</span>
      </footer>
    </div>
  );
}
