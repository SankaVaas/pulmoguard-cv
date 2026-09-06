import type { PredictionResponse } from "../api/client";

interface Props {
  result: PredictionResponse;
}

/**
 * Renders a triage prediction result. The visual hierarchy deliberately
 * puts the abstention state front and center - a red "refer to clinician"
 * banner is more prominent than the raw class label, since that is the
 * behaviorally important signal for the person using this tool.
 */
export function ResultCard({ result }: Props) {
  const confidencePct = (result.confidence * 100).toFixed(1);
  const entropyPct = (result.normalized_entropy * 100).toFixed(1);

  return (
    <div className={`result-card ${result.abstain ? "result-card--abstain" : "result-card--confident"}`}>
      {result.abstain ? (
        <div className="result-banner result-banner--warning">
          ⚠ Uncertain — recommend radiologist review
        </div>
      ) : (
        <div className="result-banner result-banner--ok">✓ Confident prediction</div>
      )}

      <div className="result-main">
        <span className="result-label">Predicted class</span>
        <span className="result-value">{result.predicted_class}</span>
      </div>

      <div className="result-grid">
        <div>
          <span className="result-label">Confidence</span>
          <span className="result-value">{confidencePct}%</span>
        </div>
        <div>
          <span className="result-label">Uncertainty (normalized entropy)</span>
          <span className="result-value">{entropyPct}%</span>
        </div>
      </div>

      <div className="result-probs">
        <span className="result-label">Class probabilities</span>
        {Object.entries(result.class_probabilities).map(([label, prob]) => (
          <div className="prob-row" key={label}>
            <span className="prob-label">{label}</span>
            <div className="prob-bar-track">
              <div className="prob-bar-fill" style={{ width: `${prob * 100}%` }} />
            </div>
            <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <p className="result-message">{result.message}</p>
    <div className="result-meta">
        <span>{result.filename}</span>
        <span>{result.mc_dropout_passes} uncertainty passes</span>
      </div>
    </div>
  );
}
