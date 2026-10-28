"""
PCAP → CIC-IDS2017-compatible feature DataFrame via NFStream.

NFStream (https://www.nfstream.org) groups raw packets into bidirectional
flows and computes the same statistical features as CICFlowMeter (the tool
used to generate the CIC-IDS2017 dataset): duration, packet counts, byte
rates, inter-arrival time statistics, flag counts, window sizes, etc.

This adapter is inference-only. Extracted flows have no labels.
"""
import pandas as pd

# NFStream column names → CIC-IDS2017 column names (best-effort mapping).
# NFStream uses snake_case; CIC-IDS2017 uses title-cased names with spaces.
# Columns not in this map are passed through as-is (the model's VarianceThreshold
# and imputation steps handle any extras or gaps).
_RENAME: dict[str, str] = {
    "dst_port":                         "Destination Port",
    "bidirectional_duration_ms":        "Flow Duration",
    "bidirectional_packets":            "Total Fwd Packets",
    "src2dst_bytes":                    "Total Length of Fwd Packets",
    "dst2src_bytes":                    "Total Length of Bwd Packets",
    "src2dst_packets":                  "Total Fwd Packets",
    "dst2src_packets":                  "Total Backward Packets",
    "bidirectional_bytes":              "Flow Bytes/s",   # approximated below
    "bidirectional_mean_ps":            "Average Packet Size",
    "src2dst_mean_ps":                  "Avg Fwd Segment Size",
    "dst2src_mean_ps":                  "Avg Bwd Segment Size",
    "bidirectional_min_ps":             "Min Packet Length",
    "bidirectional_max_ps":             "Max Packet Length",
    "bidirectional_stddev_ps":          "Packet Length Std",
    "src2dst_min_ms":                   "Fwd IAT Min",
    "src2dst_max_ms":                   "Fwd IAT Max",
    "src2dst_mean_ms":                  "Fwd IAT Mean",
    "src2dst_stddev_ms":                "Fwd IAT Std",
    "dst2src_min_ms":                   "Bwd IAT Min",
    "dst2src_max_ms":                   "Bwd IAT Max",
    "dst2src_mean_ms":                  "Bwd IAT Mean",
    "dst2src_stddev_ms":                "Bwd IAT Std",
    "bidirectional_min_ms":             "Flow IAT Min",
    "bidirectional_max_ms":             "Flow IAT Max",
    "bidirectional_mean_ms":            "Flow IAT Mean",
    "bidirectional_stddev_ms":          "Flow IAT Std",
    "bidirectional_syn_packets":        "SYN Flag Count",
    "bidirectional_ack_packets":        "ACK Flag Count",
    "bidirectional_fin_packets":        "FIN Flag Count",
    "bidirectional_rst_packets":        "RST Flag Count",
    "bidirectional_psh_packets":        "PSH Flag Count",
    "bidirectional_urg_packets":        "URG Flag Count",
    "src2dst_syn_packets":              "Fwd PSH Flags",
    "src2dst_fin_packets":              "Fwd URG Flags",
    "src2dst_rst_packets":              "Bwd PSH Flags",
    "bidirectional_first_seen_ms":      "Flow Start",   # dropped before inference
    "bidirectional_last_seen_ms":       "Flow End",     # dropped before inference
}

# Columns that carry no predictive signal for the model (identifiers / timing anchors)
_DROP = {"id", "src_ip", "dst_ip", "src_port", "src_mac", "dst_mac",
         "ip_version", "vlan_id", "tunnel_id", "application_name",
         "application_category_name", "application_is_guessed",
         "requested_server_name", "client_fingerprint", "server_fingerprint",
         "user_agent", "content_type", "expiration_id",
         "Flow Start", "Flow End"}


class PcapAdapter:
    def load(self, path: str) -> pd.DataFrame:
        """
        Extract bidirectional flows from a PCAP/PCAPNG file using NFStream
        and return a DataFrame of CICFlowMeter-compatible features.
        """
        try:
            from nfstream import NFStreamer
        except ImportError:
            raise RuntimeError(
                "nfstream is required for PCAP inference.\n"
                "Install it with:  pip install nfstream\n"
                "Note: nfstream requires libpcap (Linux/macOS) or Npcap (Windows)."
            )

        streamer = NFStreamer(
            source=path,
            statistical_analysis=True,   # compute IAT/byte/packet stats
            splt_analysis=0,             # disable per-packet deep inspection (not needed)
            n_dissections=20,            # DPI for application-layer metadata
        )
        df = streamer.to_pandas()

        if df.empty:
            return df

        # Rename to CIC-IDS2017 column names where known
        df = df.rename(columns=_RENAME)

        # Derive Flow Bytes/s from bytes and duration (NFStream gives total bytes)
        if "Flow Duration" in df.columns and "bidirectional_bytes" not in df.columns:
            if "Total Length of Fwd Packets" in df.columns and "Total Length of Bwd Packets" in df.columns:
                total_bytes = df["Total Length of Fwd Packets"].fillna(0) + df["Total Length of Bwd Packets"].fillna(0)
                duration_s = df["Flow Duration"].replace(0, 1) / 1000.0  # ms → s
                df["Flow Bytes/s"] = total_bytes / duration_s
                df["Flow Packets/s"] = df.get("Total Fwd Packets", 0) / duration_s

        # Drop identifier / non-feature columns
        drop_cols = [c for c in df.columns if c in _DROP]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Coerce all remaining columns to numeric; non-numeric → NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        return df
