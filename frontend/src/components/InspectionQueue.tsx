import { useState, useRef } from "react";
import {
  Globe,
  UploadCloud,
  FileText,
  Download,
  AlertTriangle,
  CheckCircle2,
  DatabaseZap,
  Send,
  RotateCcw,
  Search,
  Info,
} from "lucide-react";
import { isValidIp } from "../utils";
import { TrafficFlowRecord, CsvAnalysisResponse, InspectIpResponse } from "../types";

interface InspectionQueueProps {
  apiBase: string;
  onAnalysisComplete: (data: CsvAnalysisResponse | { flows: TrafficFlowRecord[]; message?: string }) => void;
  onSingleFlowResult: (result: any) => void;
  isBusy: boolean;
  setIsBusy: (busy: boolean) => void;
}

const knownAttackRow: Record<string, number> = {
  "Dst Port": 80,
  "Flow Duration": 601248,
  "Tot Fwd Pkts": 3,
  "Tot Bwd Pkts": 6,
  "TotLen Fwd Pkts": 26,
  "TotLen Bwd Pkts": 11601,
  "Fwd Pkt Len Max": 20,
  "Fwd Pkt Len Min": 0,
  "Fwd Pkt Len Mean": 8.667,
  "Bwd Pkt Len Max": 5840,
  "Bwd Pkt Len Min": 0,
  "Bwd Pkt Len Mean": 1933.5,
  "Flow Byts/s": 19338.11,
  "Flow Pkts/s": 14.97,
  "Flow IAT Mean": 75156,
  "Flow IAT Std": 210467.35,
  "Flow IAT Max": 596023,
  "Bwd IAT Tot": 601182,
  "Bwd IAT Mean": 120236.4,
  "Bwd IAT Std": 265979.5,
  "Bwd IAT Min": 160,
  "Fwd PSH Flags": 0,
  "Bwd PSH Flags": 0,
  "Fwd URG Flags": 0,
  "Bwd URG Flags": 0,
  "Pkt Len Var": 3681776.01,
  "FIN Flag Cnt": 0,
  "RST Flag Cnt": 0,
  "PSH Flag Cnt": 1,
  "ACK Flag Cnt": 0,
  "URG Flag Cnt": 0,
  "CWE Flag Count": 0,
  "Down/Up Ratio": 2,
  "Fwd Byts/b Avg": 0,
  "Fwd Pkts/b Avg": 0,
  "Fwd Blk Rate Avg": 0,
  "Bwd Byts/b Avg": 0,
  "Bwd Pkts/b Avg": 0,
  "Bwd Blk Rate Avg": 0,
  "Init Fwd Win Byts": 8192,
  "Init Bwd Win Byts": 229,
  "Fwd Act Data Pkts": 2,
  "Fwd Seg Size Min": 20,
  "Active Mean": 0,
  "Active Std": 0,
  "Active Max": 0,
  "Idle Min": 0,
};

export function InspectionQueue({
  apiBase,
  onAnalysisComplete,
  onSingleFlowResult,
  isBusy,
  setIsBusy,
}: InspectionQueueProps) {
  const [activeTab, setActiveTab] = useState<"ip" | "csv" | "sample">("ip");

  // Mode A: Inspect IP State
  const [ipInput, setIpInput] = useState("");
  const [ipError, setIpError] = useState<string | null>(null);
  const [ipNotice, setIpNotice] = useState<string | null>(null);

  // Mode B: Upload CSV State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [csvSuccess, setCsvSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Mode C: Manual Test State
  const [singlePayload, setSinglePayload] = useState(() => JSON.stringify({ row: knownAttackRow }, null, 2));

  // --- Handlers for Mode A: Inspect IP ---
  async function handleInspectIp(e?: React.FormEvent) {
    if (e) e.preventDefault();
    setIpError(null);
    setIpNotice(null);

    const trimmed = ipInput.trim();
    if (!trimmed) {
      setIpError("Please enter an IPv4 or IPv6 address to inspect.");
      return;
    }

    if (!isValidIp(trimmed)) {
      setIpError(`'${trimmed}' is not a valid IP address. Enter a valid IPv4 (e.g. 192.168.1.1) or IPv6 address.`);
      return;
    }

    setIsBusy(true);
    try {
      const res = await fetch(`${apiBase}/inspect-ip`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ip: trimmed }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail || "IP inspection request failed");
      }

      const data: InspectIpResponse = await res.json();
      if (!data.ip_metadata_available) {
        setIpNotice(data.message);
      } else {
        onAnalysisComplete({ flows: data.flows, message: data.message });
      }
    } catch (err: any) {
      setIpError(err.message || "Failed to contact inspection endpoint");
    } finally {
      setIsBusy(false);
    }
  }

  // --- Handlers for Mode B: Upload CSV Report ---
  function handleFileSelect(file: File) {
    setCsvError(null);
    setCsvSuccess(null);
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setCsvError("Invalid file type. Please select a valid .csv network flow report.");
      return;
    }
    if (file.size === 0) {
      setCsvError("Selected CSV file is empty (0 bytes).");
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setCsvError("File exceeds 50MB limit. Please upload a smaller chunk.");
      return;
    }
    setSelectedFile(file);
  }

  async function handleUploadCsv() {
    if (!selectedFile) {
      setCsvError("Please select or drop a CSV file first.");
      return;
    }

    setIsBusy(true);
    setCsvError(null);
    setCsvSuccess(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${apiBase}/analyze-csv`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail || "CSV analysis failed");
      }

      const data: CsvAnalysisResponse = await res.json();
      setCsvSuccess(`Successfully processed ${data.total_flows} flows from ${selectedFile.name}`);
      onAnalysisComplete(data);
    } catch (err: any) {
      setCsvError(err.message || "Error analyzing CSV file");
    } finally {
      setIsBusy(false);
    }
  }

  // --- Handlers for Mode C: Quick Single Flow ---
  async function handlePredictSingle() {
    setIsBusy(true);
    try {
      const parsed = JSON.parse(singlePayload);
      const res = await fetch(`${apiBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      onSingleFlowResult(data);
    } catch (err: any) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleRandomSample() {
    setIsBusy(true);
    try {
      const res = await fetch(`${apiBase}/random-analyze`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      onSingleFlowResult(data.model_output ?? data);
    } catch (err: any) {
      alert(`Random sampling failed: ${err.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <div className="inspection-card">
      {/* Mode Selector Tabs */}
      <nav className="mode-tabs" aria-label="Inspection Modes">
        <button
          type="button"
          className={`mode-tab ${activeTab === "ip" ? "active" : ""}`}
          onClick={() => setActiveTab("ip")}
        >
          <Globe size={17} />
          <span>Inspect IP</span>
        </button>

        <button
          type="button"
          className={`mode-tab ${activeTab === "csv" ? "active" : ""}`}
          onClick={() => setActiveTab("csv")}
        >
          <UploadCloud size={17} />
          <span>Upload CSV Report</span>
        </button>

        <button
          type="button"
          className={`mode-tab ${activeTab === "sample" ? "active" : ""}`}
          onClick={() => setActiveTab("sample")}
        >
          <DatabaseZap size={17} />
          <span>Quick Packet Test</span>
        </button>
      </nav>

      {/* Tab 1: Inspect IP */}
      {activeTab === "ip" && (
        <form className="tab-pane" onSubmit={handleInspectIp}>
          <div className="pane-intro">
            <h4>IP-Based Traffic Attribution</h4>
            <p>
              Query connected telemetry for ingress and egress network flows belonging to a target host or subnet.
            </p>
          </div>

          <div className="input-row">
            <div className="field-group">
              <label htmlFor="ip-input">Target IP Address (IPv4 / IPv6)</label>
              <div className="input-wrap">
                <Search size={18} className="input-icon" />
                <input
                  id="ip-input"
                  type="text"
                  placeholder="e.g. 192.168.1.105 or 2001:db8::1"
                  value={ipInput}
                  onChange={(e) => {
                    setIpInput(e.target.value);
                    setIpError(null);
                    setIpNotice(null);
                  }}
                  disabled={isBusy}
                />
              </div>
            </div>

            <button type="submit" className="primary-btn inspect-btn" disabled={isBusy}>
              <Search size={18} />
              {isBusy ? "Querying..." : "Inspect IP"}
            </button>
          </div>

          {ipError && (
            <div className="alert-box error" role="alert">
              <AlertTriangle size={18} />
              <span>{ipError}</span>
            </div>
          )}

          {ipNotice && (
            <div className="alert-box info" role="status">
              <Info size={18} />
              <div>
                <strong>Dataset Attribution Notice</strong>
                <p>{ipNotice}</p>
                <div className="alert-actions">
                  <button
                    type="button"
                    className="secondary-btn compact-btn"
                    onClick={() => setActiveTab("csv")}
                  >
                    Go to CSV Upload
                  </button>
                  <a
                    href={`${apiBase}/sample-csv-template`}
                    download="cypher_traffic_sample.csv"
                    className="secondary-btn compact-btn"
                  >
                    <Download size={14} /> Download Sample Template
                  </a>
                </div>
              </div>
            </div>
          )}
        </form>
      )}

      {/* Tab 2: Upload CSV Report */}
      {activeTab === "csv" && (
        <div className="tab-pane">
          <div className="pane-intro">
            <h4>Batch Network Flow Analysis</h4>
            <p>
              Upload captured network flows exported from CICFlowMeter, Zeek, or NetFlow sensors for high-throughput
              VAE anomaly scoring and MoE attack attribution.
            </p>
          </div>

          <div
            className={`dropzone ${dragActive ? "drag-active" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              if (e.dataTransfer.files?.[0]) {
                handleFileSelect(e.dataTransfer.files[0]);
              }
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept=".csv"
              style={{ display: "none" }}
              onChange={(e) => {
                if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
              }}
            />
            <FileText size={36} className="drop-icon" />
            <p className="drop-prompt">
              {selectedFile ? (
                <strong>{selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</strong>
              ) : (
                <>Drag and drop your network flow CSV report here, or <span>Browse Files</span></>
              )}
            </p>
            <span className="drop-subtext">Supported format: .csv (UTF-8/ASCII) up to 50MB</span>
          </div>

          <div className="drop-actions">
            <a
              href={`${apiBase}/sample-csv-template`}
              download="cypher_traffic_sample.csv"
              className="secondary-btn"
              title="Download valid CSV template with IP headers and flow features"
            >
              <Download size={16} />
              Sample Template
            </a>

            <button
              type="button"
              className="primary-btn"
              onClick={handleUploadCsv}
              disabled={isBusy || !selectedFile}
            >
              <UploadCloud size={18} />
              {isBusy ? "Analyzing Report..." : "Upload & Analyze"}
            </button>
          </div>

          {csvError && (
            <div className="alert-box error" role="alert">
              <AlertTriangle size={18} />
              <span>{csvError}</span>
            </div>
          )}

          {csvSuccess && (
            <div className="alert-box success" role="status">
              <CheckCircle2 size={18} />
              <span>{csvSuccess}</span>
            </div>
          )}
        </div>
      )}

      {/* Tab 3: Quick Packet Test */}
      {activeTab === "sample" && (
        <div className="tab-pane">
          <div className="pane-intro">
            <h4>Single-Flow Verification & Sampling</h4>
            <p>Directly test flow parameter vectors or sample random live flows from the local dataset store.</p>
          </div>

          <textarea
            className="code-textarea"
            spellCheck={false}
            value={singlePayload}
            onChange={(e) => setSinglePayload(e.target.value)}
            rows={7}
            aria-label="Raw Flow JSON"
          />

          <div className="quick-actions">
            <button
              type="button"
              className="secondary-btn"
              onClick={() => setSinglePayload(JSON.stringify({ row: knownAttackRow }, null, 2))}
              disabled={isBusy}
            >
              <RotateCcw size={16} /> Load DDoS Sample
            </button>

            <button type="button" className="secondary-btn" onClick={handleRandomSample} disabled={isBusy}>
              <DatabaseZap size={16} /> Sample Random Row
            </button>

            <button type="button" className="primary-btn" onClick={handlePredictSingle} disabled={isBusy}>
              <Send size={16} /> {isBusy ? "Evaluating..." : "Predict Flow"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
