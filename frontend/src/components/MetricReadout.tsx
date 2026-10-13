import { Activity, ShieldCheck, ShieldAlert, Siren, Crosshair, BarChart2, AlertTriangle, CheckCircle, HelpCircle } from "lucide-react";
import { ModelResult } from "../types";
import { formatNumber, formatProbability, confidencePercent } from "../utils";

interface MetricReadoutProps {
  result: ModelResult | null;
  onOpenDetails?: () => void;
}

export function MetricReadout({ result, onOpenDetails }: MetricReadoutProps) {
  if (!result) return null;

  const score = result.score ?? null;
  const threshold = result.threshold ?? 38.51;
  const scoreRatio =
    typeof score === "number" && typeof threshold === "number" && threshold > 0
      ? Math.min((score / threshold) * 100, 160)
      : 0;

  const isAnomaly = result.is_anomaly ?? false;
  const isFailed =
    result.classification_status === "Failed" ||
    result.classification_status === "Classification Failed" ||
    result.verdict === "Classification Failed";
  const isUnclassified =
    !isFailed &&
    isAnomaly &&
    (result.verdict === "Anomalous — Attack Type Unclassified" ||
      result.attack_type === "Attack Type Unclassified" ||
      result.attack_type?.toLowerCase() === "benign");

  const classStatusDisplay = isFailed
    ? "Failed"
    : isUnclassified
    ? "Evaluated (Unclassified Attack)"
    : "Evaluated";

  // Predicted Attack display string
  let attackClassDisplay: string;
  if (isFailed) {
    attackClassDisplay = "Classification Failed";
  } else if (!isAnomaly) {
    attackClassDisplay = "Normal Traffic";
  } else if (isUnclassified) {
    attackClassDisplay = "Attack Type Unclassified";
  } else if (result.attack_type) {
    attackClassDisplay = result.attack_type;
  } else {
    attackClassDisplay = "Attack Type Unclassified";
  }

  // Predicted-class probability readout
  let attackProbDisplay: string;
  if (isFailed) {
    attackProbDisplay = "Failed";
  } else if (!isAnomaly) {
    attackProbDisplay = "— (Normal Traffic)";
  } else if (isUnclassified) {
    attackProbDisplay = "— (Unclassified Attack)";
  } else if (result.attack_confidence !== null && result.attack_confidence !== undefined) {
    attackProbDisplay = `${formatProbability(result.attack_confidence)} (Model Output)`;
  } else {
    attackProbDisplay = "—";
  }

  return (
    <section className="metric-readout-card">
      <div className="card-header">
        <div>
          <span className="section-kicker">Real-Time Telemetry</span>
          <h3>Threat Detection Readout</h3>
        </div>
        {onOpenDetails && (
          <button className="secondary-btn compact-btn" onClick={onOpenDetails}>
            Inspect All Features
          </button>
        )}
      </div>

      {/* Threshold Meter Bar */}
      <div className="score-meter-wrap">
        <div className="meter-labels">
          <span>Reconstruction Error: <strong>{formatNumber(score, 3)}</strong></span>
          <span>Threshold: <strong>{formatNumber(threshold, 3)}</strong></span>
        </div>
        <div className="score-track" aria-label="Score vs Threshold gauge">
          <span
            className={isAnomaly ? "meter-fill-danger" : "meter-fill-success"}
            style={{ width: `${scoreRatio}%` }}
          />
        </div>
      </div>

      {/* Classification Failure Banner */}
      {isFailed && (
        <div className="disagreement-banner" style={{ borderColor: "rgba(239, 68, 68, 0.4)", background: "rgba(239, 68, 68, 0.08)" }}>
          <AlertTriangle size={20} className="text-danger" />
          <div>
            <strong>Classification Failure Encountered:</strong>
            <p>
              The classifier encountered an inference error. Anomaly result is preserved, but attack attribution is unavailable for this flow.
            </p>
          </div>
        </div>
      )}

      {/* Classifier-Only Stealth Attack Banner */}
      {result.detection_source === "classifier_attack" && (
        <div className="disagreement-banner" style={{ borderColor: "rgba(251, 191, 36, 0.4)", background: "rgba(251, 191, 36, 0.08)" }}>
          <AlertTriangle size={20} className="text-amber" />
          <div>
            <strong>Stealth Attack Detected by Classifier:</strong>
            <p>
              Reconstruction error ({formatNumber(score, 3)}) is below the VAE baseline threshold ({formatNumber(threshold, 3)}), but the supervised 12-class classifier identified signature characteristics of {attackClassDisplay} with {formatProbability(result.attack_confidence)} classifier probability.
            </p>
          </div>
        </div>
      )}

      {/* Dual Confirmed Threat Banner */}
      {result.detection_source === "dual_confirmed" && (
        <div className="disagreement-banner" style={{ borderColor: "rgba(239, 68, 68, 0.4)", background: "rgba(239, 68, 68, 0.08)" }}>
          <ShieldAlert size={20} className="text-danger" />
          <div>
            <strong>Dual-Confirmed Threat Detected:</strong>
            <p>
              Both unsupervised VAE reconstruction error ({formatNumber(score, 3)} &gt; {formatNumber(threshold, 3)}) and supervised attack classification ({attackClassDisplay}) independently confirmed malicious activity.
            </p>
          </div>
        </div>
      )}

      {/* Unclassified Novel Anomaly Banner */}
      {(result.detection_source === "vae_anomaly" || (isUnclassified && !isFailed && result.detection_source !== "classifier_attack")) && (
        <div className="disagreement-banner">
          <AlertTriangle size={20} className="text-amber" />
          <div>
            <strong>Unclassified Anomalous Traffic Detected:</strong>
            <p>
              Reconstruction error ({formatNumber(score, 3)}) exceeds threshold ({formatNumber(threshold, 3)}), confirming anomalous behavior. The supervised classifier did not attribute this flow to a known attack class, indicating potential novel/zero-day activity or benign classifier disagreement.
            </p>
          </div>
        </div>
      )}

      {/* Granular Metric Grid distinguishing Anomaly, Classification, Confidences, and Trigger */}
      <div className="metric-grid metric-grid-6">
        <article className="metric-tile" title="Overall threat status fused from VAE reconstruction error and supervised classifier output">
          {isAnomaly ? <ShieldAlert size={22} className="text-danger" /> : <ShieldCheck size={22} className="text-success" />}
          <span>Anomaly Status</span>
          <strong className={isAnomaly ? "text-danger" : "text-success"}>
            {isAnomaly ? "Anomaly" : "Normal"}
          </strong>
        </article>

        <article
          className="metric-tile"
          title={
            result.detection_source === "dual_confirmed"
              ? "Dual Confirmed: Exceeds VAE threshold & flagged by supervised classifier"
              : result.detection_source === "classifier_attack"
              ? "Classifier Attack: Detected by supervised classifier below VAE threshold (Stealth flow)"
              : result.detection_source === "vae_anomaly"
              ? "VAE Anomaly: Exceeds VAE reconstruction threshold (Unclassified / Novel)"
              : "Clear Normal: Below VAE threshold and classified as benign"
          }
        >
          <Crosshair size={22} className={result.detection_source === "dual_confirmed" ? "text-danger" : result.detection_source === "classifier_attack" ? "text-amber" : result.detection_source === "vae_anomaly" ? "text-cyan" : "text-success"} />
          <span>Detection Trigger</span>
          <strong className={`text-sm-break ${result.detection_source === "dual_confirmed" ? "text-danger" : result.detection_source === "classifier_attack" ? "text-amber" : result.detection_source === "vae_anomaly" ? "text-cyan" : "text-success"}`}>
            {result.detection_source === "dual_confirmed"
              ? "Dual Confirmed"
              : result.detection_source === "classifier_attack"
              ? "Classifier (Stealth)"
              : result.detection_source === "vae_anomaly"
              ? "VAE (Novel/Unknown)"
              : "Clear Normal"}
          </strong>
        </article>

        <article className="metric-tile">
          {isFailed ? (
            <AlertTriangle size={22} className="text-danger" />
          ) : isUnclassified ? (
            <AlertTriangle size={22} className="text-amber" />
          ) : (
            <CheckCircle size={22} className="text-cyan" />
          )}
          <span>Classification Status</span>
          <strong className={isFailed ? "text-danger text-sm-break" : isUnclassified ? "text-amber text-sm-break" : "text-cyan text-sm-break"}>
            {classStatusDisplay}
          </strong>
        </article>

        <article className="metric-tile">
          <Siren size={22} className={isFailed ? "text-danger" : "text-indigo"} />
          <span>Predicted Attack</span>
          <strong className={`text-sm-break ${isFailed ? "text-danger" : isUnclassified ? "text-amber" : !isAnomaly ? "text-success" : ""}`}>
            {attackClassDisplay}
          </strong>
        </article>

        <article className="metric-tile">
          <Crosshair size={22} className="text-cyan" />
          <span>Verdict</span>
          <strong className={isAnomaly ? "text-danger text-sm-break" : "text-success text-sm-break"}>
            {result.verdict ?? (isAnomaly ? (isFailed ? "Classification Failed" : "Anomalous — Attack Type Unclassified") : "Normal Traffic")}
          </strong>
        </article>

        <article className="metric-tile" title="Reconstruction error distance from VAE baseline relative to threshold 38.51 (heuristic)">
          <Activity size={22} className="text-amber" />
          <span>VAE Anomaly Distance</span>
          <strong>{formatProbability(result.anomaly_confidence)}</strong>
        </article>

        <article className="metric-tile" title="Raw multi-class softmax probability from the 12-class supervised classifier (uncalibrated model output)">
          <Activity size={22} className={isFailed ? "text-danger" : "text-indigo"} />
          <span>Classifier Probability</span>
          <strong className={`text-sm-break ${isFailed ? "text-danger" : ""}`}>
            {attackProbDisplay}
          </strong>
        </article>

        <article className="metric-tile metric-tile-wide" title="Predicted-class probability for attributed attack family (uncalibrated model output)">
          <Crosshair size={22} className="text-cyan" />
          <span>Predicted-Class Probability</span>
          <strong>
            {isAnomaly && result.attack_confidence !== null && result.attack_confidence !== undefined
              ? `${formatProbability(result.attack_confidence)} (Model Output)`
              : "—"}
          </strong>
        </article>
      </div>

      {/* Top Reconstruction Error Contributors */}
      {result.top_contributors && result.top_contributors.length > 0 && (
        <div className="top-contributors-block">
          <span className="contributors-heading">
            <BarChart2 size={15} /> Top Reconstruction Error Drivers:
          </span>
          <div className="contributors-row">
            {result.top_contributors.slice(0, 5).map((item, idx) => (
              <div className="contributor-chip" key={idx}>
                <span className="chip-name">{item.feature ?? "Feature"}</span>
                <span className="chip-val">{formatNumber(item.sq_error, 2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
