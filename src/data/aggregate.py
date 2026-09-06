"""60-s bin aggregation: per-(host, bin, direction) feature vector.

The 5 packet-level features are derived here from per-flow packet statistics.
The rate features (byte_rate, packet_rate) use bin_size_seconds from config,
NOT a hard-coded 60.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _direction_of(src_ip: str, host_ip: str) -> str:
    """IN if src is the host; OUT if dst is the host (this is a placeholder;
    the actual direction policy is per-deployment and is set by the caller).
    """
    return "IN" if src_ip == host_ip else "OUT"


def compute_packet_features(flows: pd.DataFrame) -> dict[str, float]:
    """Derive the 5 packet-level features from per-flow packet statistics.

    For a given (host, bin, direction) group of flows, we:
      - pool all Fwd and Bwd packet sizes from per-flow statistics
      - pool the per-flow total Fwd/Bwd packet counts
      - return a dict with pkt_size_mean, pkt_size_std, pkt_size_p99,
        fwd_bwd_pkt_ratio, small_pkt_frac.
    """
    # Pool per-packet size estimates: each flow contributes Fwd Pkt Len Mean/Std/Max/Min
    # plus Bwd Pkt Len Mean/Std/Max/Min. Use the max as a proxy for the per-packet sizes.
    sizes = np.concatenate([
        flows["Fwd Pkt Len Max"].to_numpy(dtype=np.float64),
        flows["Bwd Pkt Len Max"].to_numpy(dtype=np.float64),
    ])
    sizes = sizes[~np.isnan(sizes)]

    pkt_size_mean = float(np.mean(sizes)) if sizes.size else 0.0
    pkt_size_std = float(np.std(sizes)) if sizes.size else 0.0
    pkt_size_p99 = float(np.percentile(sizes, 99)) if sizes.size else 0.0

    fwd_pkts = float(flows["Total Fwd Packet"].sum())
    bwd_pkts = float(flows["Total Bwd packet"].sum())
    fwd_bwd_pkt_ratio = fwd_pkts / max(bwd_pkts, 1.0)

    # small_pkt_frac: fraction of per-flow packet-size estimates < 64 bytes.
    small_pkt_frac = float(np.mean(sizes < 64.0)) if sizes.size else 0.0

    return {
        "pkt_size_mean": pkt_size_mean,
        "pkt_size_std": pkt_size_std,
        "pkt_size_p99": pkt_size_p99,
        "fwd_bwd_pkt_ratio": fwd_bwd_pkt_ratio,
        "small_pkt_frac": small_pkt_frac,
    }


def aggregate_per_host_per_bin(
    flows: pd.DataFrame,
    bin_seconds: int,
    hosts: list[str],
    t0: pd.Timestamp,
) -> pd.DataFrame:
    """For each (host, bin, direction), produce a row with all 18 scalar
    features per direction (IN and OUT separately).

    Returns a DataFrame indexed by (host, bin, direction) with columns:
      flow_count, total_bytes, total_packets, byte_rate, packet_rate,
      dur_mean, dur_std, dur_p99, iat_mean, iat_std, iat_max,
      unique_peer_ports, unique_peer_ips,
      pkt_size_mean, pkt_size_std, pkt_size_p99, fwd_bwd_pkt_ratio, small_pkt_frac
    """
    if flows.empty:
        return pd.DataFrame()

    # Compute bin index from timestamp
    flows = flows.copy()
    flows["bin"] = ((flows["Timestamp"] - t0).dt.total_seconds() // bin_seconds).astype(np.int64)

    rows: list[dict] = []
    for host in hosts:
        for direction, src_eq, dst_eq in [
            ("IN", lambda ip, h=host: ip == h, lambda ip, h=host: ip != h),
            ("OUT", lambda ip, h=host: ip == h, lambda ip, h=host: ip == h),
        ]:
            # For IN: this host is the SOURCE.
            # For OUT: this host is the DESTINATION.
            if direction == "IN":
                sel = flows[flows["Source IP"].apply(src_eq)]
            else:
                sel = flows[flows["Destination IP"].apply(dst_eq)]
            if sel.empty:
                continue
            for bin_idx, group in sel.groupby("bin"):
                total_bytes = int(group["Total Fwd Bytes"].sum() + group["Total Bwd Bytes"].sum())
                total_packets = int(group["Total Fwd Packet"].sum() + group["Total Bwd packet"].sum())
                row = {
                    "host": host,
                    "bin": int(bin_idx),
                    "direction": direction,
                    "flow_count": int(len(group)),
                    "total_bytes": total_bytes,
                    "total_packets": total_packets,
                    "byte_rate": float(total_bytes) / float(bin_seconds),
                    "packet_rate": float(total_packets) / float(bin_seconds),
                    "dur_mean": float(group["Flow Duration"].mean()),
                    "dur_std": float(group["Flow Duration"].std() or 0.0),
                    "dur_p99": float(np.percentile(group["Flow Duration"], 99)),
                    "iat_mean": float(group["Flow IAT Mean"].mean()),
                    "iat_std": float(group["Flow IAT Std"].mean()),
                    "iat_max": float(group["Flow IAT Max"].max()),
                    "unique_peer_ports": int(group["Destination Port"].nunique() if "Destination Port" in group.columns else 0),
                    "unique_peer_ips": int(group["Destination IP"].nunique()),
                }
                row.update(compute_packet_features(group))
                rows.append(row)
    return pd.DataFrame(rows)
