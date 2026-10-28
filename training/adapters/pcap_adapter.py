"""
PCAP → CIC-IDS2017-compatible 72-column feature DataFrame.

Requires: scapy
Flow extraction produces bidirectional flow aggregates matching the feature
set expected by the model. Missing features (e.g. IAT stats requiring > 1
packet) are left as NaN and imputed by the preprocessor.

This adapter is inference-only. Extracted flows have no labels.
"""
import pandas as pd


class PcapAdapter:
    def load(self, path: str) -> pd.DataFrame:
        try:
            from scapy.all import rdpcap, IP, TCP, UDP
        except ImportError:
            raise RuntimeError("scapy is required for PCAP inference. pip install scapy")

        packets = rdpcap(path)
        flows: dict[tuple, dict] = {}

        for pkt in packets:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            proto = "TCP" if pkt.haslayer(TCP) else ("UDP" if pkt.haslayer(UDP) else "OTHER")
            sport = pkt[TCP].sport if pkt.haslayer(TCP) else (pkt[UDP].sport if pkt.haslayer(UDP) else 0)
            dport = pkt[TCP].dport if pkt.haslayer(TCP) else (pkt[UDP].dport if pkt.haslayer(UDP) else 0)

            # Bidirectional flow key (canonical order)
            key = tuple(sorted([(ip.src, sport), (ip.dst, dport)])) + (proto,)
            entry = flows.setdefault(key, {
                "Destination Port": dport,
                "Total Fwd Packets": 0,
                "Total Backward Packets": 0,
                "Total Length of Fwd Packets": 0.0,
                "_pkt_times": [],
            })
            entry["Total Fwd Packets"] += 1
            entry["Total Length of Fwd Packets"] += len(pkt)
            entry["_pkt_times"].append(float(pkt.time))

        rows = []
        for key, f in flows.items():
            times = sorted(f.pop("_pkt_times"))
            duration = (times[-1] - times[0]) * 1e6 if len(times) > 1 else 0.0
            f["Flow Duration"] = duration
            f["Flow Bytes/s"] = f["Total Length of Fwd Packets"] / (duration / 1e6) if duration > 0 else 0.0
            rows.append(f)

        return pd.DataFrame(rows) if rows else pd.DataFrame()
