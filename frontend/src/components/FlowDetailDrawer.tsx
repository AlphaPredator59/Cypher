import { useState } from "react";
import { X, ShieldAlert, ShieldCheck, Activity, Cpu, Sparkles, AlertTriangle } from "lucide-react";
import { TrafficFlowRecord } from "../types";
import { formatNumber, formatProbability, confidencePercent } from "../utils";

interface FlowDetailDrawerProps {
  flow: TrafficFlowRecord | null;
  onClose: () => void;
  apiBase: string;
}

export function FlowDetailDrawer({ flow, onClose, apiBase }: FlowDetailDrawerProps) {
  const [llmAlert, setLlmAlert] = useState<any>(null);
  const [loadingLlm, setLoadingLlm] = useState(false);
  const [llmError, setLlmError] = useState<string | null>(null);

  if (!flow) return null;

  async function requestAiSummary() {
    if (!flow) return;
    setLoadingLlm(true);
    setLlmError(null);
    try {
      const payload = {
        is_anomaly: flow.is_anomaly,
        score: flow.score,
        threshold: flow.threshold,
        anomaly_confidence: flow.anomaly_confidence,
        attack_type: flow.attack_type,
        attack_confidence: flow.attack_confidence,
        verdict: flow.verdict,
        verdict_confidence: flow.verdict_confidence,
        top_contributors: flow.top_contributors ?? [],
        input_issues: {},
      };

      const res = await fetch(`${apiBase}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          llm_payload: payload,
          model_output: { ...payload, ...flow.raw_features },
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || "LLM request failed");
      }

      const data = await res.json();
      setLlmAlert(data);
    } catch (err: any) {
      setLlmError(err.message || "Failed to generate AI summary. Check GOOGLE_API_KEY.");
    } finally {
      setLoadingLlm(false);
    }
  }

  return (
    <div className="drawer-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        <header className="drawer-header">
          <div>
            <span className="section-kicker">Telemetry Deep-Dive</span>
            <h3>Flow #{flow.row_index} Inspection</h3>
          </div>
          <button className="icon-btn" onClick={onClose} title="Close drawer">
            <X size={20} />
          </button>
        </header>

        <div className="drawer-body">
          {/* Quick Verdict Banner */}
          <div className={`verdict-banner ${flow.is_anomaly ? "anomaly" : "clear"}`}>
            {flow.is_anomaly ? <ShieldAlert size={28} /> : <ShieldCheck size={28} />}
            <div>
              <h4>{flow.verdict ?? (flow.is_anomaly ? "Anomalous — Attack Type Unclassified" : "Normal Traffic")}</h4>
              <p>
                Classifier Probability:{" "}
                <strong>
                  {flow.is_anomaly && flow.attack_confidence !== null && flow.attack_confidence !== undefined
                    ? `${formatProbability(flow.attack_confidence)} (Model Output)`
                    : "—"}
                </strong>{" "}
                | Trigger:{" "}
                <strong>
                  {flow.detection_source === "dual_confirmed"
                    ? "Dual Confirmed"
                    : flow.detection_source === "classifier_attack"
                    ? "Classifier (Stealth)"
                    : flow.detection_source === "vae_anomaly"
                    ? "VAE (Novel/Unknown)"
                    : "Clear Normal"}
                </strong>{" "}
                | VAE Score: <strong>{formatNumber(flow.score, 2)}</strong> (Threshold:{" "}
                {formatNumber(flow.threshold, 2)})
              </p>
            </div>
          </div>

          {/* Connection Metadata */}
          <section className="drawer-section">
            <h5>Connection Context</h5>
            <div className="detail-grid">
              <div className="detail-item">
                <span>Source IP</span>
                <strong>{flow.src_ip ?? "N/A"}</strong>
              </div>
              <div className="detail-item">
                <span>Destination IP</span>
                <strong>{flow.dst_ip ?? "N/A"}</strong>
              </div>
              <div className="detail-item">
                <span>Source Port</span>
                <strong>{flow.src_port ?? "N/A"}</strong>
              </div>
              <div className="detail-item">
                <span>Destination Port</span>
                <strong>{flow.dst_port ?? "N/A"}</strong>
              </div>
              <div className="detail-item">
                <span>Protocol</span>
                <strong>{flow.protocol ?? "TCP/IP"}</strong>
              </div>
              <div className="detail-item">
                <span>Duration</span>
                <strong>{formatNumber(flow.flow_duration, 0)} µs</strong>
              </div>
            </div>
          </section>

          {/* Multi-Model Decision Metrics */}
          <section className="drawer-section">
            <h5>Confidence & Model Breakdown</h5>
            {(() => {
              const isAnomaly = flow.is_anomaly ?? false;
              const isFailed =
                flow.classification_status === "Failed" ||
                flow.classification_status === "Classification Failed" ||
                flow.verdict === "Classification Failed";
              const isUnclassified =
                !isFailed &&
                isAnomaly &&
                (flow.verdict === "Anomalous — Attack Type Unclassified" ||
                  flow.attack_type === "Attack Type Unclassified" ||
                  flow.attack_type?.toLowerCase() === "benign");

              const classStatusDisplay = isFailed
                ? "Failed"
                : isUnclassified
                ? "Evaluated (Unclassified Attack)"
                : "Evaluated";

              let attackClassDisplay: string;
              if (isFailed) {
                attackClassDisplay = "Classification Failed";
              } else if (!isAnomaly) {
                attackClassDisplay = "Normal Traffic";
              } else if (isUnclassified) {
                attackClassDisplay = "Attack Type Unclassified";
              } else if (flow.attack_type) {
                attackClassDisplay = flow.attack_type;
              } else {
                attackClassDisplay = "Attack Type Unclassified";
              }

              let attackProbDisplay: string;
              if (isFailed) {
                attackProbDisplay = "Failed";
              } else if (!isAnomaly) {
                attackProbDisplay = "— (Normal Traffic)";
              } else if (isUnclassified) {
                attackProbDisplay = "— (Unclassified Attack)";
              } else if (flow.attack_confidence !== null && flow.attack_confidence !== undefined) {
                attackProbDisplay = `${formatProbability(flow.attack_confidence)} (Model Output)`;
              } else {
                attackProbDisplay = "—";
              }

              return (
                <div className="detail-grid">
                  <div className="detail-item">
                    <span>Anomaly Status</span>
                    <strong className={isAnomaly ? "text-danger" : "text-success"}>
                      {isAnomaly ? "Anomaly" : "Normal"}
                    </strong>
                  </div>
                  <div
                    className="detail-item"
                    title={
                      flow.detection_source === "dual_confirmed"
                        ? "Dual Confirmed: Exceeds VAE threshold & flagged by supervised classifier"
                        : flow.detection_source === "classifier_attack"
                        ? "Classifier Attack: Detected by supervised classifier below VAE threshold (Stealth flow)"
                        : flow.detection_source === "vae_anomaly"
                        ? "VAE Anomaly: Exceeds VAE reconstruction threshold (Unclassified / Novel)"
                        : "Clear Normal: Below VAE threshold and classified as benign"
                    }
                  >
                    <span>Detection Trigger</span>
                    <strong className={flow.detection_source === "dual_confirmed" ? "text-danger" : flow.detection_source === "classifier_attack" ? "text-amber" : flow.detection_source === "vae_anomaly" ? "text-cyan" : "text-success"}>
                      {flow.detection_source === "dual_confirmed"
                        ? "Dual Confirmed"
                        : flow.detection_source === "classifier_attack"
                        ? "Classifier (Stealth)"
                        : flow.detection_source === "vae_anomaly"
                        ? "VAE (Novel/Unknown)"
                        : "Clear Normal"}
                    </strong>
                  </div>
                  <div className="detail-item">
                    <span>Classification Status</span>
                    <strong className={isFailed ? "text-danger" : isUnclassified ? "text-amber" : "text-cyan"}>
                      {classStatusDisplay}
                    </strong>
                  </div>
                  <div className="detail-item">
                    <span>Predicted Attack</span>
                    <strong className={isFailed ? "text-danger" : isUnclassified ? "text-amber" : !isAnomaly ? "text-success" : ""}>
                      {attackClassDisplay}
                    </strong>
                  </div>
                  <div className="detail-item" title="Reconstruction error distance from VAE baseline relative to threshold 38.51 (heuristic)">
                    <span>VAE Anomaly Distance</span>
                    <strong>{formatProbability(flow.anomaly_confidence)}</strong>
                  </div>
                  <div className="detail-item" title="Raw multi-class softmax probability from the 12-class supervised classifier (uncalibrated model output)">
                    <span>Classifier Probability</span>
                    <strong className={isFailed ? "text-danger" : ""}>
                      {attackProbDisplay}
                    </strong>
                  </div>
                  <div className="detail-item" title="Predicted-class probability for attributed attack family (uncalibrated model output)">
                    <span>Predicted-Class Probability</span>
                    <strong>
                      {isAnomaly && flow.attack_confidence !== null && flow.attack_confidence !== undefined
                        ? `${formatProbability(flow.attack_confidence)} (Model Output)`
                        : "—"}
                    </strong>
                  </div>
                </div>
              );
            })()}
          </section>

          {/* Top Contributors */}
          {flow.top_contributors && flow.top_contributors.length > 0 && (
            <section className="drawer-section">
              <h5>VAE Reconstruction Error Contributors</h5>
              <div className="contributors-list">
                {flow.top_contributors.map((c, i) => (
                  <div key={i} className="contributor-row">
                    <span className="feature-name">{c.feature ?? `Feature ${i}`}</span>
                    <span className="sq-error">{formatNumber(c.sq_error, 2)} sq.err</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Gemini AI Alert Section */}
          <section className="drawer-section ai-section">
            <div className="ai-header">
              <h5>
                <Sparkles size={16} /> Google Gemini SOC Alert
              </h5>
              {!llmAlert && (
                <button
                  type="button"
                  className="secondary-btn compact-btn"
                  onClick={requestAiSummary}
                  disabled={loadingLlm}
                >
                  {loadingLlm ? "Analyzing..." : "Generate AI Brief"}
                </button>
              )}
            </div>

            {llmError && (
              <div className="alert-box error">
                <AlertTriangle size={18} />
                <span>{llmError}</span>
              </div>
            )}

            {llmAlert && (
              <div className="ai-report">
                <div className="ai-severity-badge">
                  Severity: <strong>{(llmAlert.severity || "info").toUpperCase()}</strong>
                </div>
                <p className="ai-summary">{llmAlert.summary}</p>

                {llmAlert.what_is_wrong && llmAlert.what_is_wrong.length > 0 && (
                  <div className="ai-block">
                    <strong>Findings:</strong>
                    <ul>
                      {llmAlert.what_is_wrong.map((item: string, idx: number) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {llmAlert.recommended_actions?.immediate && (
                  <div className="ai-block">
                    <strong>Immediate Countermeasures:</strong>
                    <ul>
                      {llmAlert.recommended_actions.immediate.map((item: string, idx: number) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Raw Features Drawer */}
          {flow.raw_features && Object.keys(flow.raw_features).length > 0 && (
            <section className="drawer-section">
              <h5>Sampled Flow Attributes</h5>
              <div className="raw-tags">
                {Object.entries(flow.raw_features).map(([k, v]) => (
                  <div key={k} className="raw-tag">
                    <span className="tag-key">{k}:</span>
                    <span className="tag-val">{typeof v === "number" ? formatNumber(v, 2) : String(v)}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
