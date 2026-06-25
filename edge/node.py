"""
edge/node.py
============
Main loop orchestrator untuk edge node ARTEMIS v2.

Mengorkestrasi semua komponen edge:
  Frame Source → EdgeInference → DecisionEngine → Offloader → Logger

Mendukung 4 method evaluasi (identik dengan Topik 2 untuk komparabilitas):
  1. device_only       — semua frame diproses lokal
  2. server_only       — semua frame di-offload ke server
  3. static_cooperative — rule-based routing tanpa DE
  4. adaptive_cooperative — LightGBM DE routing

Output per method: list of frame_result dict dengan breakdown latensi lengkap.
Format output identik dengan phase_f_pi_evaluation.py untuk kompatibilitas
analisis Topik 2.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

from edge.decision_engine import DecisionEngine
from edge.inference import EdgeInference
from edge.offloader import Offloader
from shared.config_schema import EdgeNodeConfig
from shared.features import extract_frame_features

log = logging.getLogger("artemis.edge.node")


# ── Hardware telemetry ────────────────────────────────────────────────────────

class HardwareTelemetry:
    """Monitor CPU, RAM, dan suhu selama eksperimen berjalan."""

    def __init__(self, interval_ms: int = 500):
        self._interval = interval_ms / 1000
        self._data     = {"cpu": [], "ram_mb": [], "temp_c": []}
        self._running  = False
        self._thread   = None

    def start(self):
        self._data    = {"cpu": [], "ram_mb": [], "temp_c": []}
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self._running:
            try:
                self._data["cpu"].append(psutil.cpu_percent(interval=None))
                self._data["ram_mb"].append(
                    psutil.virtual_memory().used / 1024 / 1024)
                self._data["temp_c"].append(self._read_temp())
            except Exception:
                pass
            time.sleep(self._interval)

    @staticmethod
    def _read_temp() -> float:
        try:
            temps = psutil.sensors_temperatures()
            for k in ["cpu_thermal", "cpu-thermal", "thermal_zone0",
                      "coretemp", "k10temp"]:
                if k in temps and temps[k]:
                    return temps[k][0].current
            p = Path("/sys/class/thermal/thermal_zone0/temp")
            if p.exists():
                return float(p.read_text().strip()) / 1000
        except Exception:
            pass
        return 0.0

    def summary(self) -> Dict:
        def s(lst):
            return ({"avg": round(float(np.mean(lst)), 2),
                     "max": round(float(np.max(lst)),  2)}
                    if lst else {"avg": 0.0, "max": 0.0})
        return {
            "cpu":    s(self._data["cpu"]),
            "ram_mb": s(self._data["ram_mb"]),
            "temp_c": s(self._data["temp_c"]),
        }


# ── Alarm logic ───────────────────────────────────────────────────────────────

def _determine_alarm(source: str,
                     features: Dict = None,
                     server_resp: Dict = None,
                     edge_thresh: Dict = None) -> str:
    if source == "local" and features and edge_thresh:
        if features.get("confmax_fire",  0) >= edge_thresh["fire_local"]:
            return "FIRE_CRITICAL"
        if features.get("confmax_smoke", 0) >= edge_thresh["smoke_local"]:
            return "SMOKE_EARLY_WARNING"
        return "NONE"
    elif source == "server" and server_resp:
        d = server_resp.get("decision", "NONE")
        if d == "FIRE":   return "FIRE_CRITICAL"
        if d == "SMOKE":  return "SMOKE_EARLY_WARNING"
        return "NONE"
    return "NONE"


def _static_decision(feats: Dict, thresh: Dict) -> str:
    """Rule-based routing untuk static cooperative method."""
    if (feats["confmax_fire"]  >= thresh["fire_local"] or
            feats["confmax_smoke"] >= thresh["smoke_local"]):
        return "LOCAL"
    if (feats["confmax_fire"]  <  thresh["fire_drop"] and
            feats["confmax_smoke"] <  thresh["smoke_drop"]):
        return "DROP"
    return "OFFLOAD"


# ── Method runners ────────────────────────────────────────────────────────────

def run_device_only(sequences: List[Dict],
                    images_dir: Path,
                    inference: EdgeInference,
                    cfg: EdgeNodeConfig,
                    telemetry: HardwareTelemetry) -> List[Dict]:
    results = []
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists():
                continue
            t0        = time.perf_counter()
            dets, bd  = inference.infer(str(img_path))
            feats     = extract_frame_features(dets)
            alarm     = _determine_alarm("local", feats, edge_thresh=cfg.edge_thresh)
            total_ms  = (time.perf_counter() - t0) * 1000
            results.append({
                "filename": fname, "seq_id": seq["seq_id"],
                "seq_type": seq["seq_type"], "method": "device_only",
                "disk_read_ms":      bd["disk_read_ms"],
                "preprocess_ms":     bd["preprocess_ms"],
                "edge_inference_ms": bd["edge_inference_ms"],
                "edge_total_ms":     bd["edge_total_ms"],
                "network_total_ms":  0.0, "server_inference_ms": 0.0,
                "network_overhead_ms": 0.0, "de_ms": 0.0,
                "total_ms":          round(total_ms, 3),
                "alarm":             alarm,
                "confmax_fire":      round(feats["confmax_fire"],  4),
                "confmax_smoke":     round(feats["confmax_smoke"], 4),
                "is_warmup":         False, "de_decision": "LOCAL",
            })
    telemetry.stop()
    return results


def run_server_only(sequences: List[Dict],
                    images_dir: Path,
                    inference: EdgeInference,
                    offloader: Offloader,
                    telemetry: HardwareTelemetry) -> List[Dict]:
    results = []
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists():
                continue
            t0 = time.perf_counter()
            _, disk_ms = inference.read_raw(str(img_path))
            srv, net_bd = offloader.offload(str(img_path))
            total_ms    = (time.perf_counter() - t0) * 1000
            results.append({
                "filename": fname, "seq_id": seq["seq_id"],
                "seq_type": seq["seq_type"], "method": "server_only",
                "disk_read_ms":        round(disk_ms, 3),
                "preprocess_ms":       0.0, "edge_inference_ms": 0.0,
                "edge_total_ms":       0.0,
                "network_total_ms":    net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "de_ms":               0.0,
                "total_ms":            round(total_ms, 3),
                "alarm":               _determine_alarm("server", server_resp=srv),
                "confmax_fire":        srv.get("confmax_fire",  0.0),
                "confmax_smoke":       srv.get("confmax_smoke", 0.0),
                "is_warmup":           False, "de_decision": "OFFLOAD",
                "network_error":       bool(srv.get("network_error", False)),
            })
    telemetry.stop()
    return results


def run_static_cooperative(sequences: List[Dict],
                            images_dir: Path,
                            inference: EdgeInference,
                            offloader: Offloader,
                            cfg: EdgeNodeConfig,
                            telemetry: HardwareTelemetry) -> List[Dict]:
    results = []
    telemetry.start()
    for seq in sequences:
        for fname in seq["frames"]:
            img_path = images_dir / fname
            if not img_path.exists():
                continue
            t0       = time.perf_counter()
            dets, bd = inference.infer(str(img_path))
            feats    = extract_frame_features(dets)
            decision = _static_decision(feats, cfg.edge_thresh)
            alarm    = "NONE"
            net_bd   = {"network_total_ms": 0.0, "server_inference_ms": 0.0,
                        "network_overhead_ms": 0.0}
            if decision == "LOCAL":
                alarm = _determine_alarm("local", feats, edge_thresh=cfg.edge_thresh)
            elif decision == "OFFLOAD":
                srv, net_bd = offloader.offload(str(img_path))
                alarm = _determine_alarm("server", server_resp=srv)
            total_ms = (time.perf_counter() - t0) * 1000
            results.append({
                "filename": fname, "seq_id": seq["seq_id"],
                "seq_type": seq["seq_type"], "method": "static_cooperative",
                "de_decision":         decision,
                "disk_read_ms":        bd["disk_read_ms"],
                "preprocess_ms":       bd["preprocess_ms"],
                "edge_inference_ms":   bd["edge_inference_ms"],
                "edge_total_ms":       bd["edge_total_ms"],
                "network_total_ms":    net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "de_ms":               0.0,
                "total_ms":            round(total_ms, 3),
                "alarm":               alarm,
                "confmax_fire":        round(feats["confmax_fire"],  4),
                "confmax_smoke":       round(feats["confmax_smoke"], 4),
                "is_warmup":           False,
            })
    telemetry.stop()
    n_off = sum(1 for r in results if r.get("de_decision") == "OFFLOAD")
    log.info(f"[static] offload_rate={n_off/len(results):.1%}")
    return results


def run_adaptive_cooperative(sequences: List[Dict],
                              images_dir: Path,
                              inference: EdgeInference,
                              de: DecisionEngine,
                              offloader: Offloader,
                              cfg: EdgeNodeConfig,
                              telemetry: HardwareTelemetry) -> List[Dict]:
    results          = []
    global_frame_idx = 0
    telemetry.start()

    for seq in sequences:
        prev_feats = None
        for local_idx, fname in enumerate(seq["frames"]):
            img_path = images_dir / fname
            if not img_path.exists():
                global_frame_idx += 1
                continue

            t0       = time.perf_counter()
            dets, bd = inference.infer(str(img_path))
            feats    = extract_frame_features(dets, prev_feats)
            prev_feats = feats

            decision, de_ms, is_warmup = de.predict(
                feats,
                seq_id           = seq["seq_id"],
                forced_interval  = cfg.forced_offload_interval,
                global_frame_idx = global_frame_idx,
            )

            alarm  = "NONE"
            net_bd = {"network_total_ms": 0.0, "server_inference_ms": 0.0,
                      "network_overhead_ms": 0.0}
            if decision == "LOCAL":
                alarm = _determine_alarm("local", feats, edge_thresh=cfg.edge_thresh)
            elif decision == "OFFLOAD":
                srv, net_bd = offloader.offload(str(img_path))
                alarm = _determine_alarm("server", server_resp=srv)

            total_ms = (time.perf_counter() - t0) * 1000
            results.append({
                "filename": fname, "seq_id": seq["seq_id"],
                "seq_type": seq["seq_type"], "method": "adaptive_cooperative",
                "local_frame_idx":     local_idx,
                "global_frame_idx":    global_frame_idx,
                "de_decision":         decision,
                "is_forced_offload":   (cfg.forced_offload_interval > 0 and
                                        global_frame_idx % cfg.forced_offload_interval == 0),
                "is_warmup":           is_warmup,
                "disk_read_ms":        bd["disk_read_ms"],
                "preprocess_ms":       bd["preprocess_ms"],
                "edge_inference_ms":   bd["edge_inference_ms"],
                "edge_total_ms":       bd["edge_total_ms"],
                "de_ms":               de_ms,
                "network_total_ms":    net_bd["network_total_ms"],
                "server_inference_ms": net_bd["server_inference_ms"],
                "network_overhead_ms": net_bd["network_overhead_ms"],
                "total_ms":            round(total_ms, 3),
                "alarm":               alarm,
                "confmax_fire":        round(feats["confmax_fire"],  4),
                "confmax_smoke":       round(feats["confmax_smoke"], 4),
                "recent_offload_rate": round(de.recent_offload_rate, 4),
            })
            global_frame_idx += 1

    telemetry.stop()
    n_off = sum(1 for r in results if r.get("de_decision") == "OFFLOAD")
    log.info(f"[adaptive] offload_rate={n_off/len(results):.1%}")
    return results
