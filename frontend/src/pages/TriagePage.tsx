import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { UploadDropzone } from "../components/UploadDropzone";
import { ResultCard } from "../components/ResultCard";
import { predict, ApiError, type PredictionResponse } from "../api/client";

export function TriagePage() {
  const { token, logout } = useAuth();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handlePredict() {
    if (!selectedFile || !token) return;
    setError(null);
    setResult(null);
    setIsSubmitting(true);
    try {
      const response = await predict(selectedFile, token);
      setResult(response);
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 401) {
          setError("Your session has expired. Please sign in again.");
          logout();
        } else {
          setError(err.message);
        }
      } else {
        setError("Unable to reach the server. Is the backend running?");
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="triage-page">
      <header className="triage-header">
        <h1>PulmoGuard</h1>
        <button className="logout-button" onClick={logout}>
          Sign out
        </button>
      </header>

      <p className="triage-intro">
        Upload a chest X-ray image to get a triage prediction. This system estimates its own
        uncertainty using Monte Carlo Dropout and will flag cases it is not confident about for
        clinician review, rather than forcing a diagnosis.
      </p>

      <UploadDropzone
        onFileSelected={(file) => {
          setSelectedFile(file);
          setResult(null);
          setError(null);
        }}
        disabled={isSubmitting}
      />

      <button
        className="predict-button"
        onClick={handlePredict}
        disabled={!selectedFile || isSubmitting}
      >
        {isSubmitting ? "Analyzing…" : "Run triage prediction"}
      </button>

      {error && <div className="triage-error">{error}</div>}
      {result && <ResultCard result={result} />}

      <footer className="triage-disclaimer">
        Research/portfolio system — not a validated medical device. Not intended for clinical use
        without regulatory clearance and clinician oversight.
      </footer>
    </div>
  );
}
