import { useState, useEffect } from "react";
import { ArrowLeft, ArrowRight, Radio } from "lucide-react";
import { Scene } from "./Scene";
import { InspectionQueue } from "./components/InspectionQueue";
import { ResultsDashboard } from "./components/ResultsDashboard";
import { MetricReadout } from "./components/MetricReadout";
import { FlowDetailDrawer } from "./components/FlowDetailDrawer";
import { TrafficFlowRecord, CsvAnalysisResponse, ModelResult } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export function App() {
  const [screen, setScreen] = useState<"landing" | "detection">("landing");
  const [status, setStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [isBusy, setIsBusy] = useState(false);

  // Results state
  const [batchData, setBatchData] = useState<
    CsvAnalysisResponse | { flows: TrafficFlowRecord[]; message?: string } | null
  >(null);
  const [singleResult, setSingleResult] = useState<ModelResult | null>(null);
  const [selectedFlow, setSelectedFlow] = useState<TrafficFlowRecord | null>(null);

  // Initial connection check
  useEffect(() => {
    async function checkApiHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) setStatus("ok");
        else setStatus("error");
      } catch {
        setStatus("error");
      }
    }
    checkApiHealth();
  }, []);

  function handleBatchComplete(data: CsvAnalysisResponse | { flows: TrafficFlowRecord[]; message?: string }) {
    setBatchData(data);
    setStatus("ok");
  }

  function handleSingleResult(result: any) {
    const res: ModelResult = result.model_output ?? result;
    setSingleResult(res);
    setStatus("ok");
  }

  return (
    <>
      <Scene />

      {screen === "landing" ? (
        <main className="landing-shell">
          <section className="landing-hero">
            <p className="eyebrow">Autonomous Network Defense</p>
            <h1>Cypher IDS</h1>
            <p className="landing-copy">
              Two-stage deep learning intrusion detection: Unsupervised VAE anomaly scoring coupled with
              an 8-classifier Mixture of Experts and calibrated threat attribution.
            </p>
            <button className="launch-btn" type="button" onClick={() => setScreen("detection")}>
              Launch Inspection Console
              <ArrowRight size={20} aria-hidden="true" />
            </button>
          </section>
        </main>
      ) : (
        <main className="app-shell">
          {/* Header Bar */}
          <header className="topbar">
            <div>
              <p className="eyebrow">Cybersecurity Operations Console</p>
              <h1>Cypher IDS</h1>
            </div>

            <div className="topbar-actions">
              <button
                className="secondary-btn compact-btn"
                type="button"
                onClick={() => setScreen("landing")}
              >
                <ArrowLeft size={16} aria-hidden="true" />
                Home
              </button>

              <div className={`connection ${status}`}>
                <Radio size={16} aria-hidden="true" />
                <span>
                  {isBusy ? "Processing" : status === "ok" ? "Linked" : status === "error" ? "Offline" : "Standby"}
                </span>
              </div>
            </div>
          </header>

          {/* Main Inspection Workflow */}
          <div className="console-layout">
            {/* Top / Left Section: Inspection Queue Controls */}
            <section className="console-section">
              <InspectionQueue
                apiBase={API_BASE}
                onAnalysisComplete={handleBatchComplete}
                onSingleFlowResult={handleSingleResult}
                isBusy={isBusy}
                setIsBusy={setIsBusy}
              />
            </section>

            {/* Single Flow Live Metric Readout (if single flow was tested) */}
            {singleResult && (
              <section className="console-section">
                <MetricReadout
                  result={singleResult}
                  onOpenDetails={() => {
                    const asRecord: TrafficFlowRecord = {
                      row_index: 1,
                      score: singleResult.score ?? 0,
                      threshold: singleResult.threshold ?? 38.51,
                      is_anomaly: singleResult.is_anomaly ?? false,
                      anomaly_confidence: singleResult.anomaly_confidence,
                      classification_status: singleResult.classification_status,
                      attack_type: singleResult.attack_type ?? undefined,
                      attack_confidence: singleResult.attack_confidence,
                      verdict: singleResult.verdict ?? undefined,
                      verdict_confidence: singleResult.verdict_confidence,
                      top_contributors: singleResult.top_contributors,
                      raw_features: (singleResult as any).raw_features ?? {},
                    };
                    setSelectedFlow(asRecord);
                  }}
                />
              </section>
            )}

            {/* Structured Batch Analysis Dashboard (if CSV report or IP query completed) */}
            {batchData && (
              <section className="console-section">
                <ResultsDashboard data={batchData} onSelectFlow={(flow) => setSelectedFlow(flow)} />
              </section>
            )}
          </div>

          {/* Slide-out Feature Deep-Dive Drawer */}
          <FlowDetailDrawer
            flow={selectedFlow}
            onClose={() => setSelectedFlow(null)}
            apiBase={API_BASE}
          />
        </main>
      )}
    </>
  );
}
