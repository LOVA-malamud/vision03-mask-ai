import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowUpLeft,
  Camera,
  ScanFace,
  Check,
  ChevronDown,
  CircleHelp,
  UserRound,
  Download,
  ImagePlus,
  Info,
  LoaderCircle,
  Moon,
  ScanEye,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Sun,
  Upload,
  X,
} from "lucide-react";
import {
  request,
  percent,
  validateFile,
  type Health,
  type Prediction,
} from "./api";


export default function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem("vision03-theme") || "dark",
  );
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const camera = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await request<Health>("/api/health"));
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);
  useEffect(() => {
    void refreshHealth();
    const timer = setInterval(() => void refreshHealth(), 15000);
    return () => {
      clearInterval(timer);
      controller.current?.abort();
    };
  }, [refreshHealth]);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("vision03-theme", theme);
  }, [theme]);
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  function selectFile(next: File | undefined) {
    if (!next) return;
    controller.current?.abort();
    setBusy(false);
    setResult(null);
    const problem = validateFile(next);
    setError(problem || "");
    setFile(problem ? null : next);
  }
  function reset() {
    controller.current?.abort();
    setFile(null);
    setResult(null);
    setError("");
    setBusy(false);
    if (input.current) input.current.value = "";
  }
  async function analyze() {
    if (!file || busy) return;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError("");
    setResult(null);
    const started = performance.now();
    const timeout = setTimeout(() => abort.abort("timeout"), 60000);
    try {
      const form = new FormData();
      form.append("file", file);
      const prediction = await request<Prediction>("/api/predict", {
        method: "POST",
        body: form,
        signal: abort.signal,
      });
      if (!abort.signal.aborted) {
        setResult(prediction);
        setElapsed((performance.now() - started) / 1000);
        setTimeout(() => resultHeading.current?.focus(), 0);
      }
    } catch (e) {
      if (!abort.signal.aborted)
        setError(e instanceof Error ? e.message : "הניתוח נכשל. נסו שוב.");
      else if (abort.signal.reason === "timeout")
        setError("הניתוח ארך יותר מהצפוי. נסו שוב.");
    } finally {
      clearTimeout(timeout);
      if (controller.current === abort) setBusy(false);
    }
  }
  const ready = !!health?.model_ready && !offline;
  return (
    <div className="app-shell">
      <header className="site-header">
        <button
          className="brand"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          aria-label="VISION — למסך הראשי"
        >
          <span className="brand-mark">
            <ScanEye size={23} />
          </span>
          <span dir="ltr">
            vision03<span className="brand-dot">.</span>
            <small>THE VISION LAB</small>
          </span>
        </button>
        <nav aria-label="ניווט ראשי"><span className="nav-link active"><ScanLine size={17} /> Project 03 · VISION</span></nav>
        <button
          className="icon-button theme-toggle"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label={theme === "dark" ? "מעבר למצב בהיר" : "מעבר למצב כהה"}
        >
          {theme === "dark" ? <Sun size={19} /> : <Moon size={19} />}
        </button>
      </header>
      <main id="main">
        <>
            <section className="hero">
              <div>
                <div className="eyebrow">
                  <span className="tiny-line" /> קצת פיקסלים. הרבה סקרנות.
                </div>
                <h1>
                  עם מסכה או בלי?
                  <br />
                  <span>בואו נגלה יחד.</span>
                  <span className="hero-asterisk" aria-hidden="true">
                    ✳
                  </span>
                </h1>
                <p className="hero-description">
                  תמונה אחת, רשת נוירונים, ושתי אפשרויות.
                  <br />
                  העלו תמונה וראו למידה עמוקה בפעולה.
                </p>
              </div>
              <div className="hero-note">
                <div className="mini-animals">
                  <span>
                    <ScanFace size={33} />
                  </span>
                  <span>
                    <UserRound size={33} />
                  </span>
                </div>
                <span className="note-kicker">A SMALL CNN. A REAL PREDICTION.</span>
                <p>
                  העיניים שלכם פוגשות
                  <br />
                  את הנוירונים של המחשב.
                </p>
                <span className="hand-arrow" aria-hidden="true">
                  ↙
                </span>
              </div>
            </section>
            <div className="workspace-grid">
              <section
                className="panel upload-panel"
                aria-labelledby="upload-title"
              >
                <div className="panel-heading">
                  <div className="heading-inline">
                    <span className="step-number">01</span>
                    <h2 id="upload-title">מתחילים בתמונה</h2>
                  </div>
                  <span className="subtle-label">JPG · PNG · WEBP</span>
                </div>
                <div
                  className={`dropzone ${dragging ? "dragging" : ""} ${preview ? "has-image" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragging(false);
                    selectFile(e.dataTransfer.files[0]);
                  }}
                >
                  {preview ? (
                    <>
                      <img
                        src={preview}
                        className="image-preview"
                        alt="התמונה שבחרתם לניתוח"
                        onError={() => {
                          setError(
                            "לא ניתן להציג את התמונה. בחרו קובץ תקין אחר.",
                          );
                          setFile(null);
                        }}
                      />
                      <button
                        className="image-clear icon-button"
                        onClick={reset}
                        aria-label="הסרת התמונה"
                      >
                        <X size={18} />
                      </button>
                      <div className="image-tag">
                        <Check size={14} /> מוכנה לסיווג
                      </div>
                    </>
                  ) : (
                    <div className="drop-content">
                      <div className="upload-art" aria-hidden="true">
                        <span className="art-grid" />
                        <span className="art-cat">
                          <ScanFace strokeWidth={1.25} size={48} />
                        </span>
                        <span className="art-dog">
                          <UserRound strokeWidth={1.25} size={47} />
                        </span>
                        <span className="art-plus">
                          <ImagePlus size={18} />
                        </span>
                      </div>
                      <h3>
                        {dragging ? "אפשר לשחרר כאן" : "בוחרים תמונת פנים"}
                      </h3>
                      <p>גררו תמונה לכאן, או בחרו אחת מהמכשיר</p>
                      <button
                        className="button upload-button"
                        onClick={() => input.current?.click()}
                      >
                        <Upload size={17} /> בחירת תמונה
                      </button>
                      <span className="file-limit">
                        עד 10 MB · עד 20 מגה־פיקסל
                      </span>
                    </div>
                  )}
                </div>
                <input
                  ref={input}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  hidden
                  aria-label="בחירת קובץ תמונה"
                  onChange={(e) => {
                    selectFile(e.target.files?.[0]);
                    e.target.value = "";
                  }}
                />
                <input
                  ref={camera}
                  type="file"
                  accept="image/*"
                  capture="user"
                  hidden
                  aria-label="צילום תמונה"
                  onChange={(e) => {
                    selectFile(e.target.files?.[0]);
                    e.target.value = "";
                  }}
                />
                <div className="upload-actions">
                  <button className="text-button" onClick={() => input.current?.click()}><ImagePlus size={16} /> גלריה / קובץ</button>
                  {file ? (
                    <span className="file-name" dir="auto">
                      {file.name}
                    </span>
                  ) : (
                    <span className="privacy-note">
                      <ShieldCheck size={15} /> התמונה אינה נשמרת בניתוח רגיל
                    </span>
                  )}
                  <button
                    className="text-button"
                    onClick={() => camera.current?.click()}
                  >
                    <Camera size={16} /> צילום תמונה
                  </button>
                </div>
                {error && (
                  <div className="error-message" role="alert">
                    <Info size={18} />
                    <span>{error}</span>
                  </div>
                )}
                <button className="button save-button" disabled={!file || !preview} onClick={() => {
                  if (!file || !preview) return;
                  const link = document.createElement("a");
                  link.href = preview;
                  link.download = file.name;
                  document.body.appendChild(link);
                  link.click();
                  link.remove();
                }}><Download size={18} /> שמור תמונה</button>
                <button
                  className="button analyze-button"
                  disabled={!file || !ready || busy}
                  onClick={() => void analyze()}
                >
                  {busy ? (
                    <>
                      <LoaderCircle className="spin" size={19} /> הרשת מסתכלת על
                      התמונה…
                    </>
                  ) : (
                    <>
                      <ScanLine size={19} /> בצע סיווג
                      <ArrowLeft size={18} />
                    </>
                  )}
                </button>
                {!ready && (
                  <p className="availability-note" role="status">
                    {offline
                      ? "השרת לא מחובר. הפעילו אותו ונסו שוב."
                      : health
                        ? "המודל אינו זמין. בדקו שקובצי המודל וההגדרות בתיקיית models והפעילו את השרת מחדש."
                        : "בודקים את החיבור למודל…"}{" "}
                    <button onClick={() => void refreshHealth()}>רענון</button>
                  </p>
                )}
              </section>
              <section
                className={`panel result-panel ${result ? "has-result" : ""}`}
                aria-labelledby="result-title"
                aria-busy={busy}
              >
                <div className="panel-heading">
                  <div className="heading-inline">
                    <span className="step-number">02</span>
                    <h2 id="result-title" ref={resultHeading} tabIndex={-1}>
                      נקודת המבט של הרשת
                    </h2>
                  </div>
                  <Sparkles size={18} className="muted-icon" />
                </div>
                <div
                  className="result-body"
                  aria-live="polite"
                  aria-atomic="true"
                >
                  {result ? (
                    <div className="prediction-content">
                      <span
                        className={`prediction-animal ${result.prediction}`}
                      >
                        {result.prediction === "with_mask" ? (
                          <ScanFace size={63} strokeWidth={1.25} />
                        ) : (
                          <UserRound size={63} strokeWidth={1.25} />
                        )}
                      </span>
                      <p className="result-eyebrow">לפי המודל, הפנים בתמונה הן</p>
                      <h3 className="prediction-title">
                        {result.prediction === "with_mask" ? "עם מסכה" : "ללא מסכה"}
                        <span>!</span>
                      </h3>
                      <div className="probability-list">
                        {(["with_mask", "without_mask"] as const).map((animal) => (
                          <div
                            key={animal}
                            className={`probability-row ${result.prediction === animal ? "winner" : ""}`}
                          >
                            <div>
                              <span>
                                {animal === "with_mask" ? (
                                  <ScanFace size={18} />
                                ) : (
                                  <UserRound size={18} />
                                )}{" "}
                                {animal === "with_mask" ? "עם מסכה" : "ללא מסכה"}
                              </span>
                              <strong dir="ltr">
                                {percent(result[`probability_${animal}`])}
                              </strong>
                            </div>
                            <div
                              className="bar-track"
                              role="meter"
                              aria-label={
                                animal === "with_mask"
                                  ? "הסתברות עם מסכה"
                                  : "הסתברות ללא מסכה"
                              }
                              aria-valuemin={0}
                              aria-valuemax={100}
                              aria-valuenow={
                                result[`probability_${animal}`] * 100
                              }
                            >
                              <div
                                style={{
                                  width: percent(
                                    result[`probability_${animal}`],
                                  ),
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="inference-meta">
                        <Check size={14} /> ניתוח CNN אמיתי <span>·</span>{" "}
                        {elapsed.toFixed(2)} שניות
                      </div>
                      <button
                        className="text-button try-again"
                        onClick={() => {
                          reset();
                          input.current?.click();
                        }}
                      >
                        <RotateCcw size={15} /> ננסה תמונה נוספת?
                      </button>
                    </div>
                  ) : (
                    <div className="empty-result">
                      <div className={`scan-orbit ${busy ? "scanning" : ""}`}>
                        <ScanLine size={35} strokeWidth={1} />
                        <i />
                        <i />
                      </div>
                      <h3>
                        {busy
                          ? "מחפשים דפוסים בפיקסלים"
                          : "קצת סבלנות, סקרנות בדרך"}
                      </h3>
                      <p>
                        {busy
                          ? `התמונה עוברת דרך שכבות ה־CNN.
עוד רגע נראה את התוצאה.`
                          : `בחרו תמונת פנים ולחצו על ״בצע סיווג״.
כאן תראו מה הרשת חושבת, ובאילו הסתברויות.`}
                      </p>
                      <span className="waiting-chip">
                        <span />
                        {busy ? "מעבד תמונה" : "ממתינים לתמונה שלכם"}
                      </span>
                    </div>
                  )}
                </div>
                <div className="result-footnote">
                  <Info size={16} />
                  <p>
                    שתי אפשרויות, לא כל האמת. המודל תמיד בוחר עם מסכה או ללא מסכה — גם
                    בתמונה של משהו אחר. הסתברות גבוהה אינה הבטחה לזיהוי נכון.
                  </p>
                </div>
              </section>
            </div>
            <section className="under-the-hood" aria-label="על המודל">
              <div className="hood-title">
                <span>מאחורי הקלעים</span>
                <span className="mono">SMALL INPUT. REAL LEARNING.</span>
              </div>
              <div className="facts-grid">
                <div className="fact">
                  <span className="fact-value" dir="ltr">
                    64 <span>×</span> 64
                  </span>
                  <span>פיקסלים בכל תמונה שהרשת רואה</span>
                </div>
                <div className="fact">
                  <span className="fact-value">
                    CNN
                    <span className="fact-dot" />
                  </span>
                  <span>רשת קונבולוציה שנבנתה מאפס</span>
                </div>
                <div className="fact">
                  <span className="fact-value" dir="ltr">
                    {health?.model ? percent(health.model.test_accuracy) : "—"}
                  </span>
                  <span>
                    {health?.model
                      ? `דיוק שנמדד על ${health.model.test_count.toLocaleString("he-IL")} תמונות מבחן`
                      : "הדיוק יוצג רק אחרי אימון אמיתי"}
                  </span>
                </div>
              </div>
            </section>
            <details className="how-it-works">
              <summary>
                <span>
                  <CircleHelp size={17} /> איך תמונה הופכת לתחזית?
                </span>
                <ChevronDown size={17} />
              </summary>
              <div className="pipeline">
                <p>
                  <b>01 · מכינים</b> השרת מתקן את כיוון התמונה, ממיר ל־RGB, משנה
                  לגודל 64×64 ומחלק את ערכי הצבע ב־255.
                </p>
                <p>
                  <b>02 · מזהים דפוסים</b> שכבות Conv2D, ReLU ו־MaxPooling
                  לומדות מסננים מתוך תמונות פנים עם מסכה וללא מסכה.
                </p>
                <p>
                  <b>03 · בוחרים</b> שכבת Sigmoid מחזירה את ההסתברות ללא מסכה (מחלקה 1).
                  ההסתברות עם מסכה (מחלקה 0) היא 1 פחות הערך הזה.
                </p>
              </div>
            </details>
          </>
      </main>
      <footer>
        <span dir="ltr">
          vision03. <span className="footer-slash">/</span> FACE MASK CLASSIFICATION
        </span>
        <span className={`model-status ${ready ? "ready" : ""}`}>
          <i />
          {ready
            ? "המודל מחובר ומוכן"
            : offline
              ? "השרת לא מחובר"
              : "ממתינים למודל"}
          <ArrowUpLeft size={13} />
        </span>
      </footer>
    </div>
  );
}
