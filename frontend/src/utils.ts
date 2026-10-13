import { TrafficFlowRecord } from "./types";

export function formatNumber(value: unknown, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

/**
 * Formats a predicted-class or model probability as a percentage string with 2 decimal places.
 * Retains valid bounds [0.00%, 100.00%] without arbitrary caps or suppression of high probabilities.
 * Displays "—" for missing, null, undefined, or non-finite values.
 */
export function formatProbability(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
  // Value can be 0.0 to 1.0 or already percentage (0-100)
  const pct = value <= 1.0 ? value * 100 : value;
  const clamped = Math.max(0, Math.min(100, pct));
  return `${clamped.toFixed(2)}%`;
}

/**
 * Backward-compatible alias for formatProbability.
 */
export function confidencePercent(value: unknown, fallback = "—"): string {
  return formatProbability(value, fallback);
}

export function isValidIp(ipStr: string): boolean {
  const trimmed = ipStr.trim();
  // IPv4 regex pattern
  const ipv4Pattern =
    /^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
  // Basic IPv6 regex pattern
  const ipv6Pattern =
    /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$/;
  return ipv4Pattern.test(trimmed) || ipv6Pattern.test(trimmed);
}

export function exportFlowsToCsv(filename: string, flows: TrafficFlowRecord[]) {
  if (!flows || flows.length === 0) return;

  const headers = [
    "Row Index",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Anomaly Score",
    "Threshold",
    "Is Anomaly",
    "Anomaly Confidence",
    "Classification Status",
    "Predicted Attack",
    "Classifier Probability",
    "Verdict",
    "Verdict Confidence"
  ];

  const rows = flows.map((f) => [
    f.row_index,
    `"${f.src_ip ?? "N/A"}"`,
    `"${f.dst_ip ?? "N/A"}"`,
    f.src_port ?? "",
    f.dst_port ?? "",
    `"${f.protocol ?? "TCP/IP"}"`,
    f.flow_duration ?? 0,
    f.tot_fwd_pkts ?? 0,
    f.tot_bwd_pkts ?? 0,
    f.flow_byts_s ?? 0,
    f.flow_pkts_s ?? 0,
    f.score.toFixed(3),
    f.threshold.toFixed(3),
    f.is_anomaly ? "True" : "False",
    f.anomaly_confidence !== null && f.anomaly_confidence !== undefined ? formatProbability(f.anomaly_confidence) : "",
    `"${f.classification_status ?? "Evaluated"}"`,
    `"${f.attack_type ?? (f.is_anomaly ? "Attack Type Unclassified" : "Normal Traffic")}"`,
    f.attack_confidence !== null && f.attack_confidence !== undefined ? formatProbability(f.attack_confidence) : "",
    `"${f.verdict ?? (f.is_anomaly ? "Anomalous — Attack Type Unclassified" : "Normal Traffic")}"`,
    f.verdict_confidence !== null && f.verdict_confidence !== undefined ? formatProbability(f.verdict_confidence) : ""
  ]);

  const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", filename.endsWith(".csv") ? filename : `${filename}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
