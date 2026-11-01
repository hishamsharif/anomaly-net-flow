"""
PCAP → CIC-IDS2017-compatible feature DataFrame.

Primary extractor: Python cicflowmeter (Scapy-based CICFlowMeter reimplementation).
cicflowmeter reproduces CICFlowMeter's active/idle window algorithm, giving full
coverage of all selected features including Active Mean/Std/Max/Min, Idle Std,
and Fwd/Bwd Header Length that NFStream cannot produce.

Fallback extractor: NFStream (used if cicflowmeter is unavailable).
NFStream limitation: Active Mean/Std/Max/Min, Idle Std, Fwd Header Length and
Bwd Header Length are set to zero — those 7 features are absent from NFStream's
flow statistics.

This adapter is inference-only. Extracted flows carry no labels.
"""
import numpy as np
import pandas as pd

# ── NFStream column → CIC-IDS2017 column ─────────────────────────────────────
# NFStream uses snake_case; CIC-IDS2017 uses title-cased names with spaces.
# Columns not listed here are passed through as-is; VarianceThreshold and
# NaN-imputation in the model's feature-engineering pipeline handle any gaps.
_RENAME: dict[str, str] = {
    # Identifiers / flow metadata
    "dst_port":                     "Destination Port",
    "bidirectional_duration_ms":    "Flow Duration",

    # Packet counts
    "src2dst_packets":              "Total Fwd Packets",
    "dst2src_packets":              "Total Backward Packets",

    # Byte counts
    "src2dst_bytes":                "Total Length of Fwd Packets",
    "dst2src_bytes":                "Total Length of Bwd Packets",

    # Packet size stats — bidirectional
    "bidirectional_mean_ps":        "Average Packet Size",
    "bidirectional_min_ps":         "Min Packet Length",
    "bidirectional_max_ps":         "Max Packet Length",
    "bidirectional_stddev_ps":      "Packet Length Std",

    # Packet size stats — forward direction
    "src2dst_max_ps":               "Fwd Packet Length Max",
    "src2dst_min_ps":               "Fwd Packet Length Min",
    "src2dst_mean_ps":              "Fwd Packet Length Mean",
    "src2dst_stddev_ps":            "Fwd Packet Length Std",

    # Packet size stats — backward direction
    "dst2src_max_ps":               "Bwd Packet Length Max",
    "dst2src_min_ps":               "Bwd Packet Length Min",
    "dst2src_mean_ps":              "Bwd Packet Length Mean",
    "dst2src_stddev_ps":            "Bwd Packet Length Std",

    # IAT stats — forward direction
    "src2dst_min_ms":               "Fwd IAT Min",
    "src2dst_max_ms":               "Fwd IAT Max",
    "src2dst_mean_ms":              "Fwd IAT Mean",
    "src2dst_stddev_ms":            "Fwd IAT Std",

    # IAT stats — backward direction
    "dst2src_min_ms":               "Bwd IAT Min",
    "dst2src_max_ms":               "Bwd IAT Max",
    "dst2src_mean_ms":              "Bwd IAT Mean",
    "dst2src_stddev_ms":            "Bwd IAT Std",

    # IAT stats — bidirectional
    "bidirectional_min_ms":         "Flow IAT Min",
    "bidirectional_max_ms":         "Flow IAT Max",
    "bidirectional_mean_ms":        "Flow IAT Mean",
    "bidirectional_stddev_ms":      "Flow IAT Std",

    # TCP flags — bidirectional totals
    "bidirectional_fin_packets":    "FIN Flag Count",
    "bidirectional_syn_packets":    "SYN Flag Count",
    "bidirectional_rst_packets":    "RST Flag Count",
    "bidirectional_psh_packets":    "PSH Flag Count",
    "bidirectional_ack_packets":    "ACK Flag Count",
    "bidirectional_urg_packets":    "URG Flag Count",
    "bidirectional_cwr_packets":    "CWE Flag Count",
    "bidirectional_ece_packets":    "ECE Flag Count",

    # TCP flags — per direction
    "src2dst_psh_packets":          "Fwd PSH Flags",
    "src2dst_urg_packets":          "Fwd URG Flags",
    "dst2src_psh_packets":          "Bwd PSH Flags",
    "dst2src_urg_packets":          "Bwd URG Flags",

    # TCP initial window sizes (first SYN/SYN-ACK window advertisement)
    "src2dst_init_win_bytes":       "Init_Win_bytes_forward",
    "dst2src_init_win_bytes":       "Init_Win_bytes_backward",

    # Data-bearing packets in fwd direction (TCP payload > 0)
    "src2dst_data_packets":         "act_data_pkt_fwd",

    # Flow anchors — kept temporarily for derivation, dropped before return
    "bidirectional_first_seen_ms":  "Flow Start",
    "bidirectional_last_seen_ms":   "Flow End",
}

# Columns that carry no predictive signal (identifiers, DPI metadata, anchors)
_DROP = {
    "id", "src_ip", "dst_ip", "src_port", "src_mac", "dst_mac",
    "ip_version", "vlan_id", "tunnel_id",
    "application_name", "application_category_name", "application_is_guessed",
    "requested_server_name", "client_fingerprint", "server_fingerprint",
    "user_agent", "content_type", "expiration_id",
    "Flow Start", "Flow End",
}


class PcapAdapter:
    """
    Preferred extractor: CicFlowMeterAdapter (full feature coverage).
    Falls back to NFStream if cicflowmeter is not importable.
    """

    def load(self, path: str) -> pd.DataFrame:
        """
        Extract bidirectional flows from a PCAP/PCAPNG file.
        Tries the Python cicflowmeter package first for maximum CIC-IDS2017
        feature coverage; falls back to NFStream if not available.
        """
        try:
            from inference.adapters.cicfm_adapter import CicFlowMeterAdapter
            return CicFlowMeterAdapter().load(path)
        except (ImportError, RuntimeError):
            pass  # fall through to NFStream

        # ── NFStream fallback ─────────────────────────────────────────────────
        try:
            from nfstream import NFStreamer
        except ImportError:
            raise RuntimeError(
                "No PCAP extractor available. Install cicflowmeter:\n"
                "  pip install cicflowmeter\n"
                "or nfstream (Linux/macOS only):\n"
                "  pip install nfstream"
            )

        streamer = NFStreamer(
            source=path,
            statistical_analysis=True,
            splt_analysis=0,
            n_dissections=20,
        )
        df = streamer.to_pandas()

        if df.empty:
            return df

        # 1. Rename NFStream columns to CIC-IDS2017 names
        df = df.rename(columns=_RENAME)

        # 2. Derive features that CICFlowMeter computes but NFStream names differently
        #    or that must be calculated from available columns.
        duration_s = (
            df["Flow Duration"].replace(0, 1) / 1000.0
            if "Flow Duration" in df.columns else pd.Series(1.0, index=df.index)
        )

        # Byte / packet rates
        if "Total Length of Fwd Packets" in df.columns and "Total Length of Bwd Packets" in df.columns:
            total_bytes = df["Total Length of Fwd Packets"].fillna(0) + df["Total Length of Bwd Packets"].fillna(0)
            df["Flow Bytes/s"] = total_bytes / duration_s

        if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
            total_pkts = df["Total Fwd Packets"].fillna(0) + df["Total Backward Packets"].fillna(0)
            df["Flow Packets/s"] = total_pkts / duration_s
            df["Fwd Packets/s"]  = df["Total Fwd Packets"].fillna(0) / duration_s
            df["Bwd Packets/s"]  = df["Total Backward Packets"].fillna(0) / duration_s

        # Packet length aliases — CIC-IDS2017 has redundant columns with different names
        # for the same underlying value.
        if "Average Packet Size" in df.columns:
            df["Packet Length Mean"] = df["Average Packet Size"]
        if "Fwd Packet Length Mean" in df.columns:
            df["Avg Fwd Segment Size"] = df["Fwd Packet Length Mean"]
        if "Bwd Packet Length Mean" in df.columns:
            df["Avg Bwd Segment Size"] = df["Bwd Packet Length Mean"]

        # Packet Length Variance = Std²
        if "Packet Length Std" in df.columns:
            df["Packet Length Variance"] = df["Packet Length Std"] ** 2

        # IAT totals — approximated as mean × count (exact value requires per-packet data)
        if "Fwd IAT Mean" in df.columns and "Total Fwd Packets" in df.columns:
            df["Fwd IAT Total"] = df["Fwd IAT Mean"] * df["Total Fwd Packets"].fillna(0)
        if "Bwd IAT Mean" in df.columns and "Total Backward Packets" in df.columns:
            df["Bwd IAT Total"] = df["Bwd IAT Mean"] * df["Total Backward Packets"].fillna(0)

        # Download/upload ratio
        if "Total Length of Bwd Packets" in df.columns and "Total Length of Fwd Packets" in df.columns:
            fwd_nonzero = df["Total Length of Fwd Packets"].replace(0, np.nan)
            df["Down/Up Ratio"] = df["Total Length of Bwd Packets"] / fwd_nonzero

        # min_seg_size_forward — CICFlowMeter's name for the minimum forward packet length
        if "Fwd Packet Length Min" in df.columns:
            df["min_seg_size_forward"] = df["Fwd Packet Length Min"]

        # 3. Drop identifier / non-feature columns
        df = df.drop(columns=[c for c in df.columns if c in _DROP], errors="ignore")

        # 4. Coerce everything to numeric; non-parseable cells become NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        return df
