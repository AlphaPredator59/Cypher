export type TopContributor = {
  feature?: string;
  sq_error?: number;
};

export type ModelResult = {
  is_anomaly?: boolean;
  score?: number;
  threshold?: number;
  anomaly_confidence?: number | null;
  attack_type?: string | null;
  attack_confidence?: number | null;
  classifier_probability?: number | null;
  classification_status?: string | null;
  verdict?: string | null;
  verdict_confidence?: number | null;
  detection_source?: string | null;
  alert_type?: string | null;
  top_contributors?: TopContributor[];
  input_issues?: Record<string, unknown>;
};

export type TrafficFlowRecord = {
  row_index: number;
  src_ip?: string;
  dst_ip?: string;
  src_port?: number | null;
  dst_port?: number | null;
  protocol?: string;
  flow_duration?: number;
  tot_fwd_pkts?: number;
  tot_bwd_pkts?: number;
  flow_byts_s?: number;
  flow_pkts_s?: number;
  score: number;
  threshold: number;
  is_anomaly: boolean;
  anomaly_confidence?: number | null;
  attack_type?: string;
  attack_confidence?: number | null;
  classifier_probability?: number | null;
  classification_status?: string | null;
  verdict?: string;
  verdict_confidence?: number | null;
  detection_source?: string | null;
  alert_type?: string | null;
  top_contributors?: TopContributor[];
  raw_features?: Record<string, number | string>;
};

export type CsvAnalysisResponse = {
  filename: string;
  total_flows: number;
  normal_flows: number;
  anomalous_flows: number;
  attack_breakdown: Record<string, number>;
  processing_time_ms: number;
  has_ip_metadata: boolean;
  results: TrafficFlowRecord[];
};

export type InspectIpResponse = {
  query_ip: string;
  ip_metadata_available: boolean;
  matching_flows_count: number;
  message: string;
  flows: TrafficFlowRecord[];
};
