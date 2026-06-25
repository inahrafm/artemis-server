#!/usr/bin/env python3
"""
scripts/analyze_results.py
===========================
Analisis hasil eksperimen ARTEMIS v2 Topik 3.

Port dari phase_f_aggregate.py dengan adaptasi untuk:
  - Format results baru dari run_edge.py (metadata + methods)
  - Format lama dari phase_f_pi_evaluation.py (pi_id + methods)
  - Mendukung multiple Pi, multiple kondisi jaringan, multiple lokasi

CARA PAKAI:

  # Analisis semua results di folder results/
  python3 scripts/analyze_results.py \
      --results_dir results/ \
      --gt_file     data/ground_truth.json \
      --output_dir  analysis/

  # Analisis spesifik satu eksperimen (filter by experiment_id)
  python3 scripts/analyze_results.py \
      --results_dir results/ \
      --gt_file     data/ground_truth.json \
      --experiment_id rq2_urban_4g

  # Bandingkan dua kondisi jaringan
  python3 scripts/analyze_results.py \
      --results_dir results/ \
      --gt_file     data/ground_truth.json \
      --compare rq2_wifi_baseline,rq2_urban_4g,rq2_rural_3g

OUTPUT:
  analysis/
  ├── summary_all_methods.csv      ← tabel utama semua metrik
  ├── system_f1_all.csv            ← System F1 + Wilson CI
  ├── latency_breakdown.csv        ← breakdown latensi per komponen
  ├── alarm_latency.csv            ← frame ke alarm pertama
  ├── per_seq_type.csv             ← breakdown per tipe sequence
  ├── network_condition_compare.csv ← perbandingan antar kondisi jaringan
  └── full_report.json             ← semua data dalam satu file
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import DEFAULT_THRESHOLDS

log = logging.getLogger("artemis.analyze")

METHODS = [
    "device_only", "server_only",
    "static_cooperative", "adaptive_cooperative"
]
METHOD_LABELS = {
    "device_only":          "Device-Only",
    "server_only":          "Server-Only",
    "static_cooperative":   "Static Cooperative",
    "adaptive_cooperative": "Adaptive Cooperative",
}


# ── Setup logging ─────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Wilson CI ─────────────────────────────────────────────────────────────────

def wilson_ci(successes: int, total: int,
              confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score confidence interval untuk proporsi."""
    if total == 0:
        return (0.0, 0.0)
    import math
    from scipy import stats as scipy_stats
    z      = scipy_stats.norm.ppf((1 + confidence) / 2)
    p      = successes / total
    n      = total
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


# ── Data loading ──────────────────────────────────────────────────────────────

def _normalize_result(data: dict, filepath: str) -> dict:
    """
    Normalize format results ke format standar internal.

    Format LAMA (phase_f_pi_evaluation.py):
      {"pi_id": "pi5", "methods": {...}}

    Format BARU (run_edge.py):
      {"metadata": {"node_id": "pi5_real", "experiment_id": "...",
                    "location": "...", "operator": "..."}, "methods": {...}}
    """
    if "pi_id" in data:
        # Format lama
        return {
            "node_id":       data["pi_id"],
            "device_type":   data.get("pi_id", "unknown"),
            "experiment_id": "topik2_baseline",
            "location":      "lab_lan",
            "operator":      "wifi_lan",
            "methods":       data.get("methods", {}),
            "source_file":   filepath,
        }
    elif "metadata" in data:
        # Format baru
        meta = data["metadata"]
        return {
            "node_id":       meta.get("node_id", "unknown"),
            "device_type":   meta.get("device_type", "unknown"),
            "experiment_id": meta.get("experiment_id", "unknown"),
            "location":      meta.get("location", "unknown"),
            "operator":      meta.get("operator", "unknown"),
            "methods":       data.get("methods", {}),
            "source_file":   filepath,
        }
    else:
        log.warning(f"Format tidak dikenal di {filepath}, skip")
        return None


def load_results(results_dir: Path,
                 experiment_filter: Optional[str] = None) -> List[dict]:
    """
    Load semua file results dari folder.
    Support filter by experiment_id.
    """
    results = []
    patterns = ["results_*.json", "*.json"]

    files_found = set()
    for pattern in patterns:
        for jf in sorted(results_dir.glob(pattern)):
            files_found.add(jf)

    for jf in sorted(files_found):
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"Gagal baca {jf.name}: {e}")
            continue

        normalized = _normalize_result(data, str(jf))
        if normalized is None:
            continue

        # Filter by experiment_id jika diminta
        if experiment_filter:
            if experiment_filter not in normalized["experiment_id"]:
                continue

        results.append(normalized)
        log.info(f"  Loaded: {jf.name} "
                 f"(node={normalized['node_id']}, "
                 f"exp={normalized['experiment_id']}, "
                 f"loc={normalized['location']})")

    return results


def load_gt(gt_file: Path) -> dict:
    """Load ground truth per-image dari ground_truth.json."""
    with open(gt_file) as f:
        gt = json.load(f)
    has_fire  = sum(1 for v in gt.values() if v.get("has_fire"))
    has_smoke = sum(1 for v in gt.values() if v.get("has_smoke"))
    no_event  = sum(1 for v in gt.values()
                    if not v.get("has_fire") and not v.get("has_smoke"))
    log.info(f"Ground truth: {len(gt)} images "
             f"(fire={has_fire}, smoke={has_smoke}, no_event={no_event})")
    return gt


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_system_f1(frame_results: List[dict], gt: dict,
                       label: str = "all") -> dict:
    """
    Hitung System F1 dengan Wilson 95% CI.

    TP: frame adalah event (has_fire/has_smoke) DAN sistem alarm
    FN: frame adalah event tapi sistem TIDAK alarm
    FP: frame BUKAN event tapi sistem alarm
    TN: frame BUKAN event DAN sistem tidak alarm
    """
    tp = fp = fn = tn = 0
    for r in frame_results:
        fname    = r.get("filename", "")
        gt_info  = gt.get(fname, {})
        is_event = gt_info.get("has_fire", False) or gt_info.get("has_smoke", False)
        alarmed  = r.get("alarm", "NONE") != "NONE"

        if   is_event and     alarmed: tp += 1
        elif is_event and not alarmed: fn += 1
        elif not is_event and alarmed: fp += 1
        else:                          tn += 1

    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = (2 * prec * recall / (prec + recall)
              if (prec + recall) > 0 else 0.0)

    prec_ci   = wilson_ci(tp, tp + fp)
    recall_ci = wilson_ci(tp, tp + fn)

    return {
        "precision":       round(prec,   4),
        "recall":          round(recall, 4),
        "f1":              round(f1,     4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "prec_ci_lo":      round(prec_ci[0],   4),
        "prec_ci_hi":      round(prec_ci[1],   4),
        "recall_ci_lo":    round(recall_ci[0], 4),
        "recall_ci_hi":    round(recall_ci[1], 4),
        "n_frames":        len(frame_results),
        "subset":          label,
    }


def compute_latency_stats(frame_results: List[dict]) -> dict:
    """Hitung statistik latensi per komponen, steady-state only."""
    def _s(lst):
        if not lst:
            return {"avg": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "n": 0}
        arr = np.array(lst, dtype=float)
        return {
            "avg": round(float(arr.mean()), 3),
            "std": round(float(arr.std()),  3),
            "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "n":   len(lst),
        }

    steady = [r for r in frame_results if not r.get("is_warmup", False)]
    warmup = [r for r in frame_results if r.get("is_warmup",  False)]

    result = {
        "all":    _s([r["total_ms"] for r in frame_results if "total_ms" in r]),
        "steady": _s([r["total_ms"] for r in steady        if "total_ms" in r]),
        "warmup": _s([r["total_ms"] for r in warmup        if "total_ms" in r]),
    }

    # Per-komponen (steady state)
    for key in ["disk_read_ms", "preprocess_ms", "edge_inference_ms",
                "edge_total_ms", "de_ms", "network_total_ms",
                "server_inference_ms", "network_overhead_ms"]:
        vals = [r[key] for r in steady if key in r and r[key] is not None]
        if vals:
            result[key] = _s(vals)

    # Network — OFFLOAD frames only
    offload = [r for r in steady
               if r.get("de_decision") == "OFFLOAD"
               or r.get("method") == "server_only"]
    if offload:
        for key in ["network_total_ms", "server_inference_ms",
                    "network_overhead_ms"]:
            vals = [r[key] for r in offload if key in r]
            if vals:
                result[f"{key}_offload_only"] = _s(vals)

    # Network errors
    n_err = sum(1 for r in frame_results if r.get("network_error", False))
    result["network_errors"] = {
        "count": n_err,
        "rate":  round(n_err / len(frame_results), 4) if frame_results else 0.0,
    }

    return result


def compute_alarm_latency(frame_results: List[dict], gt: dict) -> dict:
    """
    Hitung alarm latency per sequence.
    = jumlah frame dari awal sequence hingga alarm TP pertama.
    """
    by_seq = defaultdict(list)
    for r in frame_results:
        by_seq[r.get("seq_id", "unknown")].append(r)

    latencies   = []
    missed_seqs = 0
    event_seqs  = 0

    for seq_id, frames in by_seq.items():
        frames_sorted = sorted(
            frames,
            key=lambda x: x.get("local_frame_idx",
                                 x.get("global_frame_idx", 0))
        )
        has_event = any(
            gt.get(r["filename"], {}).get("has_fire",  False) or
            gt.get(r["filename"], {}).get("has_smoke", False)
            for r in frames_sorted
        )
        if not has_event:
            continue
        event_seqs += 1

        first_tp = None
        for i, r in enumerate(frames_sorted):
            gt_info    = gt.get(r["filename"], {})
            is_event_f = gt_info.get("has_fire", False) or gt_info.get("has_smoke", False)
            if r.get("alarm", "NONE") != "NONE" and is_event_f:
                first_tp = i
                break

        if first_tp is not None:
            latencies.append(first_tp)
        else:
            missed_seqs += 1

    if not latencies:
        return {"event_seqs": event_seqs, "detected": 0,
                "missed": missed_seqs, "detection_rate": 0.0}

    return {
        "event_seqs":     event_seqs,
        "detected":       len(latencies),
        "missed":         missed_seqs,
        "detection_rate": round(len(latencies) / event_seqs, 4)
                          if event_seqs > 0 else 0.0,
        "mean":           round(float(np.mean(latencies)),          2),
        "median":         round(float(np.median(latencies)),        2),
        "std":            round(float(np.std(latencies)),           2),
        "p25":            round(float(np.percentile(latencies, 25)),2),
        "p75":            round(float(np.percentile(latencies, 75)),2),
        "min":            int(min(latencies)),
        "max":            int(max(latencies)),
    }


def compute_per_seq_type(frame_results: List[dict], gt: dict) -> List[dict]:
    """Breakdown System F1 dan offload rate per tipe sequence."""
    rows = []
    for seq_type in ["gradual_escalation", "confident_smoke",
                     "fire_smoke_simultaneous"]:
        sub = [r for r in frame_results if r.get("seq_type") == seq_type]
        if not sub:
            continue
        f1_data = compute_system_f1(sub, gt, label=seq_type)
        n       = len(sub) or 1
        n_off   = sum(1 for r in sub if r.get("de_decision") == "OFFLOAD")
        n_loc   = sum(1 for r in sub if r.get("de_decision") == "LOCAL")
        n_drop  = sum(1 for r in sub if r.get("de_decision") == "DROP")
        rows.append({
            "seq_type":      seq_type,
            "n_frames":      n,
            "f1":            f1_data["f1"],
            "precision":     f1_data["precision"],
            "recall":        f1_data["recall"],
            "offload_rate":  round(n_off  / n, 4),
            "local_rate":    round(n_loc  / n, 4),
            "drop_rate":     round(n_drop / n, 4),
            "alarm_count":   sum(1 for r in sub if r.get("alarm") != "NONE"),
        })
    return rows


# ── Build summary rows ────────────────────────────────────────────────────────

def build_rows(all_results: List[dict], gt: dict) -> List[dict]:
    """Build satu row per (node × method × experiment)."""
    rows = []
    for res in all_results:
        node_id   = res["node_id"]
        device    = res["device_type"]
        exp_id    = res["experiment_id"]
        location  = res["location"]
        operator  = res["operator"]

        for method_key in METHODS:
            md = res["methods"].get(method_key)
            if not md:
                continue

            fr  = md.get("frame_results", [])
            hw  = md.get("hardware", {})
            s   = md.get("summary", {})

            if not fr:
                continue

            lat    = compute_latency_stats(fr)
            f1     = compute_system_f1(fr, gt) if gt else {}
            al     = compute_alarm_latency(fr, gt) if gt else {}
            steady = lat["steady"]

            row = {
                # Identity
                "node_id":        node_id,
                "device_type":    device,
                "experiment_id":  exp_id,
                "location":       location,
                "operator":       operator,
                "method":         METHOD_LABELS.get(method_key, method_key),
                "method_key":     method_key,

                # Latency — steady state
                "latency_avg_ms": steady["avg"],
                "latency_std_ms": steady["std"],
                "latency_p50_ms": steady["p50"],
                "latency_p95_ms": steady["p95"],
                "n_steady":       steady["n"],
                "n_warmup":       lat["warmup"]["n"],

                # Latency breakdown (avg)
                "disk_read_ms":      lat.get("disk_read_ms",      {}).get("avg", 0),
                "preprocess_ms":     lat.get("preprocess_ms",     {}).get("avg", 0),
                "edge_inference_ms": lat.get("edge_inference_ms", {}).get("avg", 0),
                "de_ms":             lat.get("de_ms",             {}).get("avg", 0),
                "network_rt_ms":     lat.get("network_total_ms_offload_only",
                                            {}).get("avg", 0),
                "server_inf_ms":     lat.get("server_inference_ms_offload_only",
                                            {}).get("avg", 0),
                "net_overhead_ms":   lat.get("network_overhead_ms_offload_only",
                                            {}).get("avg", 0),
                "n_network_errors":  lat["network_errors"]["count"],
                "network_error_rate":lat["network_errors"]["rate"],

                # Routing
                "offload_rate":  round(s.get("offload_rate", 0) * 100, 1),
                "n_local":       s.get("decision_dist", {}).get("LOCAL",  0),
                "n_offload":     s.get("decision_dist", {}).get("OFFLOAD", 0),
                "n_drop":        s.get("decision_dist", {}).get("DROP",   0),
                "n_frames":      s.get("n_frames", len(fr)),
                "alarm_count":   s.get("alarm_count", 0),

                # System F1
                "precision":     f1.get("precision", 0),
                "recall":        f1.get("recall",    0),
                "f1":            f1.get("f1",        0),
                "TP":            f1.get("TP", 0),
                "FP":            f1.get("FP", 0),
                "FN":            f1.get("FN", 0),
                "TN":            f1.get("TN", 0),
                "prec_ci_lo":    f1.get("prec_ci_lo",   0),
                "prec_ci_hi":    f1.get("prec_ci_hi",   0),
                "recall_ci_lo":  f1.get("recall_ci_lo", 0),
                "recall_ci_hi":  f1.get("recall_ci_hi", 0),

                # Alarm latency (frames)
                "alarm_lat_mean":   al.get("mean",   0),
                "alarm_lat_median": al.get("median", 0),
                "alarm_lat_std":    al.get("std",    0),
                "alarm_det_rate":   al.get("detection_rate", 0),

                # Hardware
                "cpu_avg":  hw.get("cpu",    {}).get("avg", 0),
                "temp_avg": hw.get("temp_c", {}).get("avg", 0),
            }
            rows.append(row)
    return rows


# ── Console output ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*90}")
    print("SYSTEM F1 — semua node & metode")
    print(f"{'='*90}")
    cols = ["node_id", "device_type", "location", "method",
            "precision", "prec_ci_lo", "prec_ci_hi",
            "recall",    "recall_ci_lo", "recall_ci_hi",
            "f1", "TP", "FP", "FN"]
    avail = [c for c in cols if c in df.columns]
    print(df[avail].to_string(index=False))

    print(f"\n{'='*90}")
    print("LATENCY BREAKDOWN — steady-state (ms)")
    print(f"{'='*90}")
    cols = ["node_id", "device_type", "location", "method",
            "latency_avg_ms", "latency_p50_ms", "latency_p95_ms",
            "edge_inference_ms", "server_inf_ms", "network_rt_ms",
            "offload_rate", "n_network_errors"]
    avail = [c for c in cols if c in df.columns]
    print(df[avail].to_string(index=False))

    print(f"\n{'='*90}")
    print("ALARM LATENCY — frame ke deteksi TP pertama")
    print(f"{'='*90}")
    cols = ["node_id", "device_type", "location", "method",
            "alarm_lat_mean", "alarm_lat_median", "alarm_det_rate"]
    avail = [c for c in cols if c in df.columns]
    print(df[avail].to_string(index=False))


def print_network_comparison(df: pd.DataFrame):
    """Print perbandingan antar kondisi jaringan untuk RQ2."""
    locations = df["location"].unique()
    if len(locations) <= 1:
        return

    print(f"\n{'='*90}")
    print("PERBANDINGAN KONDISI JARINGAN — RQ2 Analysis")
    print(f"{'='*90}")

    for method in ["Server-Only", "Adaptive Cooperative"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        print(f"\nMethod: {method}")
        cols = ["location", "operator", "latency_avg_ms", "latency_p95_ms",
                "network_rt_ms", "f1", "n_network_errors", "offload_rate"]
        avail = [c for c in cols if c in sub.columns]
        print(sub[avail].sort_values("latency_avg_ms").to_string(index=False))

    # Crossover analysis: di kondisi mana adaptive > server_only?
    print(f"\n{'─'*60}")
    print("CROSSOVER ANALYSIS: Adaptive vs Server-Only latency")
    print(f"{'─'*60}")
    for loc in locations:
        loc_df  = df[df["location"] == loc]
        srv     = loc_df[loc_df["method"] == "Server-Only"]["latency_avg_ms"]
        adp     = loc_df[loc_df["method"] == "Adaptive Cooperative"]["latency_avg_ms"]
        if not srv.empty and not adp.empty:
            diff = srv.values[0] - adp.values[0]
            winner = "Adaptive" if diff > 0 else "Server-Only"
            print(f"  {loc:<30} diff={diff:+.1f}ms → {winner} lebih cepat")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARTEMIS v2 — Analisis Hasil Eksperimen Topik 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir",   default="results/",
                        help="Folder berisi file results JSON")
    parser.add_argument("--gt_file",       default="data/ground_truth.json",
                        help="Path ke ground_truth.json")
    parser.add_argument("--output_dir",    default="analysis/",
                        help="Folder output untuk CSV dan JSON")
    parser.add_argument("--experiment_id", default=None,
                        help="Filter by experiment_id (substring match)")
    parser.add_argument("--compare",       default=None,
                        help="Experiment IDs untuk dibandingkan, pisah koma")
    parser.add_argument("--log_level",     default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("=== ARTEMIS v2 — Results Analysis ===")

    # Load ground truth
    gt = {}
    gt_path = Path(args.gt_file)
    if gt_path.exists():
        gt = load_gt(gt_path)
    else:
        log.warning(f"ground_truth.json tidak ditemukan: {gt_path}")
        log.warning("System F1 tidak akan dihitung.")

    # Load results
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        log.error(f"results_dir tidak ditemukan: {results_dir}")
        sys.exit(1)

    # Handle --compare mode
    if args.compare:
        exp_ids = [e.strip() for e in args.compare.split(",")]
        all_results = []
        for exp_id in exp_ids:
            r = load_results(results_dir, experiment_filter=exp_id)
            all_results.extend(r)
            log.info(f"  {exp_id}: {len(r)} result files")
    else:
        all_results = load_results(results_dir, experiment_filter=args.experiment_id)

    if not all_results:
        log.error("Tidak ada results yang dimuat.")
        log.error(f"Cek folder: {results_dir}")
        sys.exit(1)

    log.info(f"\nTotal result files dimuat: {len(all_results)}")

    # Build summary rows
    log.info("\nMenghitung metrik...")
    rows = build_rows(all_results, gt)
    if not rows:
        log.error("Tidak ada rows yang dihasilkan — cek format results files")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Per-seq-type breakdown
    seq_type_rows = []
    for res in all_results:
        for mk in ["adaptive_cooperative", "static_cooperative"]:
            md = res["methods"].get(mk, {})
            fr = md.get("frame_results", [])
            if not fr:
                continue
            for r in compute_per_seq_type(fr, gt):
                seq_type_rows.append({
                    "node_id":       res["node_id"],
                    "device_type":   res["device_type"],
                    "experiment_id": res["experiment_id"],
                    "location":      res["location"],
                    "method":        METHOD_LABELS.get(mk, mk),
                    **r,
                })

    # ── Simpan output ─────────────────────────────────────────────────────────
    df.to_csv(out / "summary_all_methods.csv", index=False)
    log.info(f"  summary_all_methods.csv ({len(df)} rows)")

    # System F1
    f1_cols = ["node_id", "device_type", "experiment_id", "location",
               "operator", "method",
               "precision", "prec_ci_lo", "prec_ci_hi",
               "recall",    "recall_ci_lo", "recall_ci_hi",
               "f1", "TP", "FP", "FN", "TN", "n_frames"]
    f1_df = df[[c for c in f1_cols if c in df.columns]]
    f1_df.to_csv(out / "system_f1_all.csv", index=False)
    log.info(f"  system_f1_all.csv")

    # Latency breakdown
    lat_cols = ["node_id", "device_type", "experiment_id", "location",
                "operator", "method",
                "latency_avg_ms", "latency_std_ms",
                "latency_p50_ms", "latency_p95_ms",
                "disk_read_ms", "preprocess_ms", "edge_inference_ms",
                "de_ms", "network_rt_ms", "server_inf_ms", "net_overhead_ms",
                "offload_rate", "n_network_errors", "network_error_rate",
                "n_steady", "n_warmup"]
    lat_df = df[[c for c in lat_cols if c in df.columns]]
    lat_df.to_csv(out / "latency_breakdown.csv", index=False)
    log.info(f"  latency_breakdown.csv")

    # Alarm latency
    al_cols = ["node_id", "device_type", "experiment_id", "location",
               "operator", "method",
               "alarm_lat_mean", "alarm_lat_median", "alarm_lat_std",
               "alarm_det_rate"]
    al_df = df[[c for c in al_cols if c in df.columns]]
    al_df.to_csv(out / "alarm_latency.csv", index=False)
    log.info(f"  alarm_latency.csv")

    # Per-seq-type
    if seq_type_rows:
        seq_df = pd.DataFrame(seq_type_rows)
        seq_df.to_csv(out / "per_seq_type.csv", index=False)
        log.info(f"  per_seq_type.csv ({len(seq_type_rows)} rows)")

        # Validasi temporal DE
        adap = seq_df[seq_df["method"] == "Adaptive Cooperative"]
        if not adap.empty:
            ge = adap[adap["seq_type"] == "gradual_escalation"]["offload_rate"]
            cs = adap[adap["seq_type"] == "confident_smoke"]["offload_rate"]
            if not ge.empty and not cs.empty:
                if ge.values[0] > cs.values[0]:
                    log.info(
                        f"  ✓ Temporal DE validated: "
                        f"gradual_escalation OR={ge.values[0]:.1%} > "
                        f"confident_smoke OR={cs.values[0]:.1%}"
                    )

    # Network condition comparison (RQ2)
    locations = df["location"].unique()
    if len(locations) > 1:
        nc_cols = ["location", "operator", "method",
                   "latency_avg_ms", "latency_p95_ms",
                   "network_rt_ms", "f1", "offload_rate",
                   "n_network_errors"]
        nc_df = df[[c for c in nc_cols if c in df.columns]]
        nc_df.to_csv(out / "network_condition_compare.csv", index=False)
        log.info(f"  network_condition_compare.csv")

    # Full JSON report
    full_report = {
        "summary":          df.to_dict(orient="records"),
        "seq_type_breakdown": seq_type_rows,
        "n_result_files":   len(all_results),
        "locations":        list(df["location"].unique()),
        "methods":          list(df["method"].unique()),
    }
    with open(out / "full_report.json", "w") as f:
        json.dump(full_report, f, indent=2)
    log.info(f"  full_report.json")

    # ── Print ke terminal ─────────────────────────────────────────────────────
    print_summary(df)
    print_network_comparison(df)

    print(f"\n{'='*60}")
    print(f"Output tersimpan di: {out.resolve()}")
    print(f"{'='*60}")
    for f in sorted(out.iterdir()):
        if not f.name.startswith("."):
            size = f.stat().st_size
            print(f"  {f.name:<45} {size:>8,} bytes")


if __name__ == "__main__":
    main()