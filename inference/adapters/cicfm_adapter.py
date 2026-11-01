"""
PCAP → CIC-IDS2017-compatible feature DataFrame via the Python cicflowmeter package.

cicflowmeter (pip) is a Scapy-based reimplementation of the Java CICFlowMeter tool
that generated CIC-IDS2017.  It produces substantially more feature coverage than
NFStream because it reproduces CICFlowMeter's active/idle window algorithm:

  Active Mean / Std / Max / Min  ← computed by cicflowmeter, absent in NFStream
  Idle Max / Min / Mean / Std    ← computed by cicflowmeter, absent in NFStream
  Fwd Header Length              ← computed by cicflowmeter, absent in NFStream
  Bwd Header Length              ← computed by cicflowmeter, absent in NFStream
  Fwd IAT Total / Bwd IAT Total  ← computed by cicflowmeter
  Subflow Fwd/Bwd Packets/Bytes  ← computed as aliases in cicflowmeter

This adapter processes a PCAP/PCAPNG through cicflowmeter, writes flows to a
temporary CSV, reads them back as a DataFrame, and renames columns to their
CIC-IDS2017 equivalents.

Installation: pip install cicflowmeter  (requires scapy, already in dl_env)
"""
import os
import tempfile

import numpy as np
import pandas as pd

# ── cicflowmeter column → CIC-IDS2017 column ─────────────────────────────────
_CICFM_TO_CIC: dict[str, str] = {
    # Identifiers / timing
    "dst_port":          "Destination Port",
    "flow_duration":     "Flow Duration",

    # Rates
    "flow_byts_s":       "Flow Bytes/s",
    "flow_pkts_s":       "Flow Packets/s",
    "fwd_pkts_s":        "Fwd Packets/s",
    "bwd_pkts_s":        "Bwd Packets/s",

    # Packet counts
    "tot_fwd_pkts":      "Total Fwd Packets",
    "tot_bwd_pkts":      "Total Backward Packets",

    # Byte totals
    "totlen_fwd_pkts":   "Total Length of Fwd Packets",
    "totlen_bwd_pkts":   "Total Length of Bwd Packets",

    # Packet length — forward
    "fwd_pkt_len_max":   "Fwd Packet Length Max",
    "fwd_pkt_len_min":   "Fwd Packet Length Min",
    "fwd_pkt_len_mean":  "Fwd Packet Length Mean",
    "fwd_pkt_len_std":   "Fwd Packet Length Std",

    # Packet length — backward
    "bwd_pkt_len_max":   "Bwd Packet Length Max",
    "bwd_pkt_len_min":   "Bwd Packet Length Min",
    "bwd_pkt_len_mean":  "Bwd Packet Length Mean",
    "bwd_pkt_len_std":   "Bwd Packet Length Std",

    # Packet length — bidirectional
    "pkt_len_max":       "Max Packet Length",
    "pkt_len_min":       "Min Packet Length",
    "pkt_len_mean":      "Packet Length Mean",
    "pkt_len_std":       "Packet Length Std",
    "pkt_len_var":       "Packet Length Variance",

    # Header lengths (PRESENT — not derivable from NFStream)
    "fwd_header_len":    "Fwd Header Length",
    "bwd_header_len":    "Bwd Header Length",

    # Segment sizes
    "fwd_seg_size_min":  "min_seg_size_forward",
    "fwd_seg_size_avg":  "Avg Fwd Segment Size",
    "bwd_seg_size_avg":  "Avg Bwd Segment Size",

    # Data-bearing packets
    "fwd_act_data_pkts": "act_data_pkt_fwd",

    # IAT — bidirectional flow
    "flow_iat_mean":     "Flow IAT Mean",
    "flow_iat_max":      "Flow IAT Max",
    "flow_iat_min":      "Flow IAT Min",
    "flow_iat_std":      "Flow IAT Std",

    # IAT — forward
    "fwd_iat_tot":       "Fwd IAT Total",
    "fwd_iat_max":       "Fwd IAT Max",
    "fwd_iat_min":       "Fwd IAT Min",
    "fwd_iat_mean":      "Fwd IAT Mean",
    "fwd_iat_std":       "Fwd IAT Std",

    # IAT — backward
    "bwd_iat_tot":       "Bwd IAT Total",
    "bwd_iat_max":       "Bwd IAT Max",
    "bwd_iat_min":       "Bwd IAT Min",
    "bwd_iat_mean":      "Bwd IAT Mean",
    "bwd_iat_std":       "Bwd IAT Std",

    # TCP flags — per direction
    "fwd_psh_flags":     "Fwd PSH Flags",
    "bwd_psh_flags":     "Bwd PSH Flags",
    "fwd_urg_flags":     "Fwd URG Flags",
    "bwd_urg_flags":     "Bwd URG Flags",

    # TCP flags — bidirectional
    "fin_flag_cnt":      "FIN Flag Count",
    "syn_flag_cnt":      "SYN Flag Count",
    "rst_flag_cnt":      "RST Flag Count",
    "psh_flag_cnt":      "PSH Flag Count",
    "ack_flag_cnt":      "ACK Flag Count",
    "urg_flag_cnt":      "URG Flag Count",
    "ece_flag_cnt":      "ECE Flag Count",
    "cwr_flag_count":    "CWE Flag Count",

    # Misc ratios / averages
    "down_up_ratio":     "Down/Up Ratio",
    "pkt_size_avg":      "Average Packet Size",

    # TCP window sizes
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",

    # Active/Idle window stats (PRESENT — absent in NFStream)
    "active_max":        "Active Max",
    "active_min":        "Active Min",
    "active_mean":       "Active Mean",
    "active_std":        "Active Std",
    "idle_max":          "Idle Max",
    "idle_min":          "Idle Min",
    "idle_mean":         "Idle Mean",
    "idle_std":          "Idle Std",

    # Subflow aliases (set equal to total fwd/bwd in cicflowmeter)
    "subflow_fwd_pkts":  "Subflow Fwd Packets",
    "subflow_bwd_pkts":  "Subflow Bwd Packets",
    "subflow_fwd_byts":  "Subflow Fwd Bytes",
    "subflow_bwd_byts":  "Subflow Bwd Bytes",
}

# Metadata / identifier columns that carry no predictive signal
_DROP = {"src_ip", "dst_ip", "src_port", "protocol", "timestamp"}

_CICFM_VERSION = "0.5.0"


class CicFlowMeterAdapter:
    """
    Extract bidirectional flows from a PCAP/PCAPNG using the Python cicflowmeter
    package and return a DataFrame of CIC-IDS2017-compatible features.

    Coverage improvement over NFStream:
        39/47 selected features → 47/47 (Active*, Idle Std, Fwd/Bwd Header Length
        are now populated from cicflowmeter's activity-window algorithm).
    """

    @property
    def extractor_name(self) -> str:
        return "cicflowmeter-python"

    @property
    def extractor_version(self) -> str:
        return _CICFM_VERSION

    def load(self, path: str) -> pd.DataFrame:
        try:
            from cicflowmeter.sniffer import create_sniffer
        except ImportError:
            raise RuntimeError(
                "cicflowmeter is required for PCAP inference.\n"
                "Install it with:  pip install cicflowmeter"
            )

        tmp_csv = tempfile.mktemp(suffix=".csv")
        try:
            sniffer, session = create_sniffer(
                input_file=path,
                input_interface=None,
                output_mode="flow",
                output=tmp_csv,
                verbose=False,
            )
            sniffer.start()
            sniffer.join()
            # Flush any flows that haven't expired by end-of-capture
            try:
                session.garbage_collect(None)
            except Exception:
                pass

            if not os.path.exists(tmp_csv) or os.path.getsize(tmp_csv) == 0:
                return pd.DataFrame()

            df = pd.read_csv(tmp_csv)
        finally:
            if os.path.exists(tmp_csv):
                os.unlink(tmp_csv)

        if df.empty:
            return df

        # 1. Rename cicflowmeter columns → CIC-IDS2017 column names
        df = df.rename(columns=_CICFM_TO_CIC)

        # 2. Drop non-feature columns
        df = df.drop(columns=[c for c in df.columns if c in _DROP], errors="ignore")

        # 3. Coerce to numeric; non-parseable cells → NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        # 4. Replace inf / -inf with NaN so downstream fillna(0) handles them cleanly
        df = df.replace([np.inf, -np.inf], np.nan)

        return df
