import { useState, useMemo } from "react";
import {
  Download,
  Search,
  Filter,
  ShieldCheck,
  ShieldAlert,
  Clock,
  Layers,
  ArrowUpDown,
  Eye,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { TrafficFlowRecord, CsvAnalysisResponse } from "../types";
import { formatNumber, formatProbability, confidencePercent, exportFlowsToCsv } from "../utils";

interface ResultsDashboardProps {
  data: CsvAnalysisResponse | { flows: TrafficFlowRecord[]; message?: string };
  onSelectFlow: (flow: TrafficFlowRecord) => void;
}

function renderTriggerBadge(src?: string | null) {
  if (src === "dual_confirmed" || src === "dual_signal") {
    return (
      <span className="trigger-pill trigger-dual" title="Dual Signal: Exceeds VAE reconstruction threshold & corroborated by supervised classifier as known attack">
        Dual Signal
      </span>
    );
  }
  if (src === "classifier_attack" || src === "known_attack") {
    return (
      <span className="trigger-pill trigger-clf" title="Known Attack: Supervised classifier identified known attack family">
        Known Attack
      </span>
    );
  }
  if (src === "vae_anomaly" || src === "anomaly_candidate") {
    return (
      <span className="trigger-pill trigger-vae" title="Anomaly Candidate: Statistical outlier exceeding VAE threshold (Classifier normal/unclassified)">
        Anomaly Candidate
      </span>
    );
  }
  return null;
}

export function ResultsDashboard({ data, onSelectFlow }: ResultsDashboardProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState<"all" | "anomalies" | "benign">("all");
  const [sortField, setSortField] = useState<"row_index" | "score" | "flow_duration">("row_index");
  const [sortAsc, setSortAsc] = useState(true);
  const [page, setPage] = useState(1);
  const pageSize = 10;

  const flows: TrafficFlowRecord[] = "results" in data ? data.results : data.flows;
  const totalFlows = flows.length;
  const normalFlows = flows.filter((f) => !f.is_anomaly).length;
  const anomalousFlows = flows.filter((f) => f.is_anomaly).length;
  const processingTime = "processing_time_ms" in data ? data.processing_time_ms : null;
  const attackBreakdown = "attack_breakdown" in data ? data.attack_breakdown : {};

  const filteredFlows = useMemo(() => {
    return flows.filter((f) => {
      const matchesSearch =
        searchTerm === "" ||
        (f.src_ip && f.src_ip.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (f.dst_ip && f.dst_ip.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (f.verdict && f.verdict.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (f.attack_type && f.attack_type.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (f.protocol && f.protocol.toLowerCase().includes(searchTerm.toLowerCase()));

      const matchesFilter =
        filterType === "all" ||
        (filterType === "anomalies" && f.is_anomaly) ||
        (filterType === "benign" && !f.is_anomaly);

      return matchesSearch && matchesFilter;
    });
  }, [flows, searchTerm, filterType]);

  const sortedFlows = useMemo(() => {
    return [...filteredFlows].sort((a, b) => {
      let valA = a[sortField] ?? 0;
      let valB = b[sortField] ?? 0;
      if (valA < valB) return sortAsc ? -1 : 1;
      if (valA > valB) return sortAsc ? 1 : -1;
      return 0;
    });
  }, [filteredFlows, sortField, sortAsc]);

  const totalPages = Math.max(1, Math.ceil(sortedFlows.length / pageSize));
  const paginatedFlows = useMemo(() => {
    const start = (page - 1) * pageSize;
    return sortedFlows.slice(start, start + pageSize);
  }, [sortedFlows, page]);

  function handleSort(field: "row_index" | "score" | "flow_duration") {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
    setPage(1);
  }

  function handleExport() {
    const filename = "filename" in data && data.filename ? `cypher_${data.filename}` : "cypher_analyzed_traffic.csv";
    exportFlowsToCsv(filename, flows);
  }

  return (
    <section className="results-dashboard">
      {/* Summary Metrics Banner */}
      <div className="summary-grid">
        <article className="summary-card">
          <div className="summary-icon">
            <Layers size={22} />
          </div>
          <div>
            <span className="summary-label">Total Flows</span>
            <strong className="summary-value">{totalFlows}</strong>
          </div>
        </article>

        <article className="summary-card">
          <div className="summary-icon icon-success">
            <ShieldCheck size={22} />
          </div>
          <div>
            <span className="summary-label">Normal Traffic</span>
            <strong className="summary-value text-success">{normalFlows}</strong>
          </div>
        </article>

        <article className="summary-card">
          <div className="summary-icon icon-danger">
            <ShieldAlert size={22} />
          </div>
          <div>
            <span className="summary-label">Anomalies Detected</span>
            <strong className="summary-value text-danger">{anomalousFlows}</strong>
          </div>
        </article>

        {processingTime !== null && (
          <article className="summary-card">
            <div className="summary-icon icon-cyan">
              <Clock size={22} />
            </div>
            <div>
              <span className="summary-label">Processing Time</span>
              <strong className="summary-value text-cyan">{processingTime} ms</strong>
            </div>
          </article>
        )}
      </div>

      {/* Attack Breakdown Chips */}
      {Object.keys(attackBreakdown).length > 0 && (
        <div className="breakdown-bar">
          <span className="breakdown-label">Attack Classifications:</span>
          <div className="breakdown-chips">
            {Object.entries(attackBreakdown).map(([label, count]) => (
              <span key={label} className="breakdown-chip">
                <strong>{label}</strong>: {count}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Table Action Controls */}
      <div className="table-controls">
        <div className="search-box">
          <Search size={16} />
          <input
            type="text"
            placeholder="Search by IP, Protocol, Verdict..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setPage(1);
            }}
          />
        </div>

        <div className="filter-group">
          <Filter size={16} />
          <select
            value={filterType}
            onChange={(e) => {
              setFilterType(e.target.value as any);
              setPage(1);
            }}
          >
            <option value="all">All Verdicts ({totalFlows})</option>
            <option value="anomalies">Anomalies Only ({anomalousFlows})</option>
            <option value="benign">Normal Only ({normalFlows})</option>
          </select>
        </div>

        <button className="secondary-btn compact-btn" onClick={handleExport} title="Download CSV of results">
          <Download size={16} />
          Export CSV
        </button>
      </div>

      {/* Structured Traffic Table */}
      <div className="table-container">
        <table className="traffic-table">
          <thead>
            <tr>
              <th onClick={() => handleSort("row_index")} className="sortable">
                <span>#</span> <ArrowUpDown size={13} />
              </th>
              <th>Source IP</th>
              <th>Dest IP</th>
              <th>Port</th>
              <th>Protocol</th>
              <th onClick={() => handleSort("score")} className="sortable">
                <span>Score</span> <ArrowUpDown size={13} />
              </th>
              <th>Status</th>
              <th>Verdict</th>
              <th>Predicted Attack</th>
              <th title="Supervised classifier probability for attributed attack (uncalibrated model output)">Classifier Probability</th>
              <th>Inspect</th>
            </tr>
          </thead>
          <tbody>
            {paginatedFlows.length === 0 ? (
              <tr>
                <td colSpan={11} className="empty-table-cell">
                  No network flows match the selected criteria.
                </td>
              </tr>
            ) : (
              paginatedFlows.map((f) => (
                <tr key={f.row_index} className={f.is_anomaly ? "row-anomaly" : "row-normal"}>
                  <td className="cell-mono">{f.row_index}</td>
                  <td className="cell-ip">{f.src_ip ?? "N/A"}</td>
                  <td className="cell-ip">{f.dst_ip ?? "N/A"}</td>
                  <td className="cell-mono">{f.dst_port ?? "—"}</td>
                  <td>{f.protocol ?? "TCP/IP"}</td>
                  <td className="cell-mono">
                    <span className={f.score > f.threshold ? "score-high" : "score-normal"}>
                      {formatNumber(f.score, 2)}
                    </span>
                  </td>
                  <td>
                    <span className={`status-pill ${f.is_anomaly ? "pill-danger" : "pill-success"}`}>
                      {f.is_anomaly ? "ANOMALY" : "CLEAR"}
                    </span>
                    {renderTriggerBadge(f.alert_type ?? f.detection_source)}
                  </td>
                  <td>
                    <strong>{f.verdict ?? (f.is_anomaly ? "Anomalous — Attack Type Unclassified" : "Normal Traffic")}</strong>
                  </td>
                  <td>
                    <span className={f.is_anomaly ? (f.attack_type === "Classification Failed" ? "text-danger" : "text-amber") : "text-success"}>
                      {f.attack_type ?? (f.is_anomaly ? "Attack Type Unclassified" : "Normal Traffic")}
                    </span>
                  </td>
                  <td className="cell-mono">
                    <span className="conf-badge">
                      {f.is_anomaly && f.attack_confidence !== null && f.attack_confidence !== undefined
                        ? formatProbability(f.attack_confidence)
                        : "—"}
                    </span>
                  </td>
                  <td>
                    <button
                      className="icon-btn compact-icon"
                      onClick={() => onSelectFlow(f)}
                      title="View full flow telemetry & features"
                    >
                      <Eye size={16} />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      {totalPages > 1 && (
        <footer className="table-pagination">
          <span>
            Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, sortedFlows.length)} of{" "}
            {sortedFlows.length} flows
          </span>
          <div className="pagination-actions">
            <button
              className="icon-btn compact-icon"
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              title="Previous page"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="page-indicator">
              Page {page} / {totalPages}
            </span>
            <button
              className="icon-btn compact-icon"
              disabled={page >= totalPages}
              onClick={() => setPage(page + 1)}
              title="Next page"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </footer>
      )}
    </section>
  );
}
