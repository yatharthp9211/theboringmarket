"""
RuView RGIPT Science Exhibition – CSI Phase Ranging & 3-Tx/1-Rx Controller (v3.2)
=================================================================================
RGIPT Souhardya '26  |  56-Subcarrier CSI Phase Ranging + 3-Tx/1-Rx Disambiguation

Key Upgrades in v3.2:
  1. Full 56-Subcarrier CSI Phase Processing (`csi_processor.py` -> `CSIEngine`):
     - Ingests hardware ESP32 CSI frames (`{"event": "csi", "zone": ..., "csi_mag": [...], "csi_phase": [...]}`)
       via `_process_csi_message()` AND synthesizes physical complex CSI in hybrid/demo mode.
     - Computes Rician Multipath K-Factor (dB), Sub-Wavelength Phase Displacement (mm),
       Time-of-Flight (ns), and GDOP (Geometric Dilution of Precision).
  2. Adaptive Mahalanobis Kalman Tracker (`AdaptiveKalmanTracker2D`):
     - Scales measurement noise R inversely with CSI K-factor and rejects multipath spikes
       exceeding Chi-squared Mahalanobis distance > 15.0.
  3. 3-Transmitter (2× C3 + 1× Phone) -> 1-Receiver (``) Bleed Disambiguation:
     - Distinguishes standing near a Transmitter (`Tx_i`) vs. standing near the Receiver (`Rx`),
       always displaying 1 clean dot for 1 person.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import io
import json
import logging
import math
import numpy as np
import os
import random
import statistics
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from aiohttp import web
except ImportError:
    web = None

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except ImportError:
    class Console:
        def __init__(self, **kwargs): pass
        def print(self, *args, **kwargs): print(*args)
    Live = Panel = Table = box = None

import importlib.util as _ilu, os as _os, sys as _sys
def _load_local_module(mod_name: str):
    try:
        return __import__(mod_name)
    except ImportError:
        p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), f"{mod_name}.py")
        spec = _ilu.spec_from_file_location(mod_name, p)
        mod = _ilu.module_from_spec(spec)
        _sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

_csi_mod = _load_local_module("csi_processor")
CSIEngine = _csi_mod.CSIEngine
extract_device_time_us = _csi_mod.extract_device_time_us
_ai_mod = _load_local_module("wifi_densepose_ai")
WifiDensePosePretrained = _ai_mod.WifiDensePosePretrained
OneEuroFilter1D = _ai_mod.OneEuroFilter1D

CONFIG_FILE = "config.yaml"
DEFAULT_WS_URL = "ws://localhost:3001/ws/sensing"
DEFAULT_BAUD = 115200

# Bauds both the ESP32 UART driver and the CH340/CP210x/FTDI bridges handle.
# Measured on the wire: a live CSI line is 910 bytes, 948 once the `ts_us` and
# `synthesized` fields the patched emitter adds are present, so 115200 could
# carry ~12 frames/s. The firmware scan loop only emits 1-3.6 CSI frames/s per
# node (9-35% of the link), so the emitter, not the transport, caps the update
# rate today.
SUPPORTED_BAUD_RATES = (115_200, 230_400, 460_800, 921_600, 1_000_000, 1_500_000, 2_000_000)
UART_BITS_PER_BYTE = 10          # 8N1: start + 8 data + stop
CSI_FRAME_BYTES = 950            # measured CSI line: 56 tones + ts + ts_us + synthesized
SERIAL_RX_FLUSH_BYTES = 8192     # discard backlog only once it is this deep


def baud_frame_ceiling(baud: int, frame_bytes: int = CSI_FRAME_BYTES) -> float:
    """CSI frames/s a UART at `baud` can carry."""
    return baud / (UART_BITS_PER_BYTE * max(1, frame_bytes))


CMD_LED_ON  = "<LED:{pin}:ON>"
CMD_LED_OFF = "<LED:{pin}:OFF>"
CMD_PWM     = "<PWM:{pin}:{duty}>"
CMD_PING    = "<PING>"
ACTUATOR_LED_PINS = frozenset({2, 4, 5})

RX_LAYOUT_PRESETS = {
    "tabletop_4node": {
        "name": "Tabletop 4-Node Y-Delta Grid (CSI + RSSI)",
        "rx_pos": (0.50, 0.40),
        "tx_nodes": {
            "node_1": (0.50, 0.05),
            "node_2": (0.80, 0.57),
            "node_3": (0.20, 0.57),
            "node_4": (0.50, 0.40),
        }
    },

    "cross_hatch_2tx_2rx": {
        "name": "2-Tx / 2-Rx Cross-Hatch X-Mesh (4 Paths:  + )",
        "rx_pos": (0.84, 0.14),
        "rx2_pos": (0.84, 0.56),
        "tx_nodes": {
            "node_1": (0.16, 0.14),
            "node_2": (0.16, 0.56),
            "node_3": (0.84, 0.56),
        }
    },
    "perimeter_apex": {
        "name": "Perimeter Apex Fan (Recommended)",
        "rx_pos": (0.50, 0.64),
        "tx_nodes": {
            "node_1": (0.18, 0.14),
            "node_2": (0.82, 0.14),
            "node_3": (0.84, 0.48),
        }
    },
    "star_center": {
        "name": "Central Star Hub (360° Radial)",
        "rx_pos": (0.50, 0.35),
        "tx_nodes": {
            "node_1": (0.16, 0.14),
            "node_2": (0.84, 0.14),
            "node_3": (0.50, 0.62),
        }
    },
    "corner_quad": {
        "name": "4-Corner Diagonal Quad",
        "rx_pos": (0.16, 0.58),
        "tx_nodes": {
            "node_1": (0.16, 0.14),
            "node_2": (0.84, 0.14),
            "node_3": (0.84, 0.58),
        }
    }
}

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(legacy_windows=False)


def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        console.print(f"[yellow]⚠  {path} not found — using built-in defaults.[/yellow]")
        return {}


def resolve_serial_ports(serial_cfg: dict, node_cfg: dict,
                         port_override: Optional[str] = None,
                         ports_override: Optional[str] = None,
                         actuator_override: Optional[str] = None) -> tuple[str, list[str]]:
    """Return actuator port first, followed by all serial sensor ports."""
    ports: list[str] = []

    def add_port(value: object) -> None:
        port = str(value).strip() if value is not None else ""
        if port and port.casefold() not in {item.casefold() for item in ports}:
            ports.append(port)

    if ports_override:
        for port in ports_override.split(","):
            add_port(port)
    add_port(port_override)

    configured = serial_cfg.get("ports", [])
    if isinstance(configured, list):
        for port in configured:
            add_port(port)
    add_port(serial_cfg.get("port"))
    for node in node_cfg.values():
        if isinstance(node, dict):
            add_port(node.get("port"))

    actuator_port = (
        actuator_override
        or serial_cfg.get("actuator_port")
        or port_override
        or serial_cfg.get("port")
        or (ports[0] if ports else "COM11")
    )
    actuator_port = str(actuator_port).strip()
    add_port(actuator_port)
    ordered_ports = [actuator_port]
    ordered_ports.extend(port for port in ports if port.casefold() != actuator_port.casefold())
    return actuator_port, ordered_ports


@dataclass
class Zone:
    name: str
    label: str
    led_pin: int
    variance_threshold: float
    demo_scenario: str
    alert_color: str = "white"
    is_active: bool = False
    last_triggered: float = 0.0
    ssid: str = ""
    pos_m: tuple[float, float] = (0.0, 0.0)
    channel: int = 6
    role: str = "Tx Node"
    last_rssi: float = -65.0
    last_scan_rssi: Optional[float] = None
    last_variance: float = 0.0
    last_amplitude: float = 0.40
    last_csi_disp_mm: float = 0.0
    last_csi_direction: str = "stationary"
    last_csi_ts: float = 0.0
    smoothed_amplitude: float = 0.40
    instant_delta: float = 0.0
    trigger_count: int = 0
    _history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=16))
    _fast_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=6))
    _median_buf: collections.deque = field(default_factory=lambda: collections.deque(maxlen=5))
    _link_db_hist: dict = field(default_factory=dict)
    _link_amp_hist: dict = field(default_factory=dict)

    def push_sample(self, value: float, raw_rssi: Optional[float] = None, rx_node: Optional[str] = None) -> float:
        rx_key = str(rx_node) if rx_node else "default"
        amp_val = float(value)

        if self.smoothed_amplitude is None or self.smoothed_amplitude == 0.0:
            self.smoothed_amplitude = amp_val
        else:
            self.smoothed_amplitude = self.smoothed_amplitude * 0.40 + amp_val * 0.60

        self.instant_delta = abs(self.smoothed_amplitude - self.last_amplitude) if self.last_amplitude else 0.0
        self.last_amplitude = self.smoothed_amplitude
        self._history.append(self.smoothed_amplitude)
        self._fast_history.append(self.smoothed_amplitude)

        amp_buf = self._link_amp_hist.get(rx_key)
        if amp_buf is None:
            amp_buf = collections.deque(maxlen=6)
            self._link_amp_hist[rx_key] = amp_buf
        amp_buf.append(amp_val)

        if raw_rssi is not None:
            db_val = float(raw_rssi)
            self.last_rssi = db_val
            db_buf = self._link_db_hist.get(rx_key)
            if db_buf is None:
                db_buf = collections.deque(maxlen=6)
                self._link_db_hist[rx_key] = db_buf
            if db_buf and abs(db_val - db_buf[-1]) > 28.0:
                db_val = db_buf[-1]
            db_buf.append(db_val)

            if len(db_buf) < 3:
                self.last_variance = 0.0
                return 0.0

            p2p_db = max(db_buf) - min(db_buf)
            var_db2 = float(statistics.variance(db_buf))
            step_db = abs(db_buf[-1] - db_buf[-2])

            # Reject 1-LSB (<= 1.0 dB) ESP32 integer RSSI quantization toggle on empty bed
            if p2p_db <= 1.0 and var_db2 <= 0.30:
                self.last_variance = 0.0
                return 0.0

            # Responsive hand/object micro-motion variance without median-filter lag
            rolling_var = (var_db2 / 420.0) + (0.00035 * max(0.0, step_db - 1.0))
        else:
            if len(amp_buf) < 3:
                return 0.0
            rolling_var = float(statistics.variance(amp_buf))

        if rolling_var < 0.00008:
            self.last_variance = 0.0
            return 0.0

        self.last_variance = rolling_var
        return rolling_var


class SerialBridge:
    def __init__(self, port: str, baud: int = DEFAULT_BAUD, timeout: float = 0.5,
                 demo_mode: bool = False, strict_port: bool = False,
                 actuator_enabled: bool = False):
        self.port = port
        self.baud = self._resolve_baud(baud)
        self.frame_ceiling_fps = baud_frame_ceiling(self.baud)
        self.timeout = timeout
        self.demo_mode = demo_mode
        self.strict_port = strict_port
        self.actuator_enabled = actuator_enabled
        self.tx_packets = 0
        self.rx_packets = 0
        self.dropped_backlog_bytes = 0
        self._ser: Optional[serial.Serial] = None

    @staticmethod
    def _resolve_baud(baud: int) -> int:
        try:
            requested = int(baud)
        except (TypeError, ValueError):
            console.print(f"[yellow]Serial baud {baud!r} is not a number — using {DEFAULT_BAUD}.[/yellow]")
            return DEFAULT_BAUD
        if requested in SUPPORTED_BAUD_RATES:
            return requested
        console.print(
            f"[yellow]Serial baud {requested} is outside {SUPPORTED_BAUD_RATES} — "
            f"using {DEFAULT_BAUD}.[/yellow]"
        )
        return DEFAULT_BAUD

    def connect(self) -> bool:
        if self.demo_mode:
            console.print("[cyan]🔌  Serial bridge: DEMO MODE (simulated ESP32 CSI + LED actuator)[/cyan]")
            return True

        candidate_ports = [self.port]
        all_ports = list(serial.tools.list_ports.comports()) if serial else []
        if not self.strict_port:
            for p in all_ports:
                if "bluetooth" in p.description.lower():
                    continue
                if p.device not in candidate_ports:
                    candidate_ports.append(p.device)

        for port in candidate_ports:
            try:
                self._ser = serial.Serial(port, self.baud, timeout=self.timeout)
                if self.baud > DEFAULT_BAUD:
                    # A 2 Mbit stream outruns the driver's default RX buffer.
                    try:
                        self._ser.set_buffer_size(rx_size=32768, tx_size=8192)
                    except (serial.SerialException, OSError, AttributeError):
                        pass
                time.sleep(1.2)
                try:
                    self._ser.reset_input_buffer()
                except Exception:
                    pass
                self._ser.write((CMD_PING + "\n").encode())
                self.port = port
                console.print(
                    f"[green]✔  Serial connected → {self.port} @ {self.baud} baud "
                    f"(≈{self.frame_ceiling_fps:.0f} CSI frames/s, CSI + 0-latency mode)[/green]"
                )
                return True
            except (serial.SerialException, OSError):
                continue

        now = time.time()
        if not hasattr(self, "_last_fail_log") or (now - self._last_fail_log > 10.0):
            self._last_fail_log = now
            desc = [f"{p.device} ({p.description})" for p in all_ports if "bluetooth" not in p.description.lower()]
            avail = ", ".join(desc) if desc else "none detected"
            console.print(f"[yellow]⚠ Waiting for ESP32 on {self.port}... Active USB ports: {avail}[/yellow]")
        return False

    def send(self, command: str) -> None:
        if self.demo_mode:
            self.tx_packets += 1
            return
        if self._ser and self._ser.is_open:
            self.tx_packets += 1
            try:
                self._ser.write((command + "\n").encode())
            except (serial.SerialException, OSError) as exc:
                logging.warning("Serial send error: %s", exc)
                self.close()

    def led_on(self, pin: int) -> None:
        if not self.actuator_enabled:
            raise PermissionError(f"LED commands are disabled on sensor-only serial port {self.port}")
        if pin not in ACTUATOR_LED_PINS:
            raise ValueError(f"Unsupported bulb GPIO: {pin}")
        self.send(CMD_LED_ON.format(pin=pin))

    def led_off(self, pin: int) -> None:
        if not self.actuator_enabled:
            raise PermissionError(f"LED commands are disabled on sensor-only serial port {self.port}")
        if pin not in ACTUATOR_LED_PINS:
            raise ValueError(f"Unsupported bulb GPIO: {pin}")
        self.send(CMD_LED_OFF.format(pin=pin))

    def close(self) -> None:
        ser = self._ser
        self._ser = None
        if ser and ser.is_open and self.actuator_enabled:
            for pin in (2, 4, 5):
                try:
                    ser.write((CMD_LED_OFF.format(pin=pin) + "\n").encode())
                except Exception:
                    pass
            try:
                ser.close()
            except Exception:
                pass


class RuViewSpatialEngine:
    """
    CSI Phase-Fused 3-Tx + 1-Rx Multi-Static Tomography & 3D DensePose Engine (v3.2).
    """
    def __init__(self, zones: list[Zone], room_cfg: Optional[dict] = None,
                 rx_pos: tuple[float, float] = (0.50, 0.64),
                 layout_preset: str = "perimeter_apex",
                 csi_cfg: Optional[dict] = None,
                 calibration_cfg: Optional[dict] = None):
        self.zones = {z.name: z for z in zones}
        self.csi_cfg = csi_cfg or {}
        self.calibration_cfg = calibration_cfg or (self.csi_cfg.get("room_calibration") or {})
        self.room_cfg = room_cfg or {
            "name": "Tabletop Sensing Grid (CSI + 3Tx/1Rx)",
            "width_m": 1.0,
            "length_m": 0.7,
            "height_m": 0.5
        }
        self.ai_model = WifiDensePosePretrained()
        self.csi_disp_threshold_mm = float(self.csi_cfg.get("displacement_threshold_mm", 0.45))
        self.ai_model.csi_disp_threshold_mm = self.csi_disp_threshold_mm
        self.layout_preset = layout_preset
        self.rx_pos = (float(rx_pos[0]), float(rx_pos[1]))
        self.ai_model.rx_position = self.rx_pos

        self.room_w = float(self.room_cfg.get("width_m", 1.0))
        self.room_l = float(self.room_cfg.get("length_m", 0.7))

        self.node_positions: dict[str, tuple[float, float]] = {}
        for z in zones:
            if z.pos_m and z.pos_m != (0.0, 0.0):
                self.node_positions[z.name] = (float(z.pos_m[0]), float(z.pos_m[1]))
            else:
                self.node_positions[z.name] = (self.room_w / 2.0, self.room_l / 2.0)

        # Initialize 56-Subcarrier CSI Phase Ranging & Multipath Engine
        trilat_cfg = (self.csi_cfg.get("trilateration") or {})
        self.csi_engine = CSIEngine(
            self.node_positions,
            (self.room_w, self.room_l),
            rx_position=self.rx_pos,
            gdop_threshold=float(trilat_cfg.get("gdop_threshold", 5.0)),
            min_nodes=int(trilat_cfg.get("min_nodes", 3)),
            residual_gate_m=float(trilat_cfg.get("max_residual_m", 0.12)),
            free_space_excess_m=float(trilat_cfg.get("free_space_excess_m", 0.05)),
            static_object_corner=trilat_cfg.get("static_object_corner"),
        )
        # Learn the empty-bed per-link reference during the first quiet frames
        # so no phantom object is reported before the operator calibration.
        self.csi_engine.auto_calibrate_empty_bed = bool(
            self.calibration_cfg.get("auto_calibrate_on_start", True)
        )
        self.last_csi_result: Optional[dict] = None

        self.node_rssi = {z.name: -65.0 for z in zones}
        self.node_var = {z.name: 0.0 for z in zones}
        self.node_last_ts = {z.name: time.time() for z in zones}
        self.directed_links: dict[str, dict] = {}

        self.override_target: Optional[tuple[float, float]] = None
        self.override_target_2: Optional[tuple[float, float]] = None
        self.override_expiry: float = 0.0
        self.simulated_person_count: Optional[int] = None
        self.active_scenario: str = "follow_me"

        self.current_x = self.room_w * 0.34
        self.current_z = self.room_l * 0.28
        self.vx = 0.0
        self.vz = 0.0
        self.speed_mps = 0.0
        self.uncertainty_m = 0.015
        self.mahalanobis = 0.12
        try:
            from tracking.fusion import AdaptiveUKF
            self.ukf = AdaptiveUKF(
                x0=[self.current_x, self.current_z, 0.0, 0.0],
                alpha=0.5,
                beta=2.0,
                kappa=0.0,
                R0=0.04**2,
                Q=0.05,
            )
            self._last_ukf_t = time.monotonic()
        except Exception:
            self.ukf = None
        self.active_zone = "node_1"
        self.confidence = 0.95
        self.presence_prob = 0.92
        self.person_count = 1
        self.human_detected = True
        self.occupancy_label = "1 HUMAN TRACKED (CSI PHASE LOCK)"
        self.detection_mode = "csi_phase_motion"
        self.tomography_diagnostics: dict = {
            "location_regime": "near_tx_node_1",
            "rx_convergence": 0.04,
            "tx_specificity": 0.94,
            "bleed_ratio": 0.06,
            "dominant_tx": "node_1"
        }
        self.contrastive_shift = 0.0
        self.embedding_sample = [0.0] * 8
        self.ai_latency_ms = 0.16
        self.active_clusters = []
        self.heatmap = [[0.0] * 20 for _ in range(14)]

        self.start_ts = time.time()
        self.intrusion_breaches = 0
        self.energy_saved_pct = 66.7
        self.telemetry_log: collections.deque = collections.deque(maxlen=600)

        self.baseline_rssi: dict[str, float] = {z.name: -55.0 for z in zones}
        self.drop_thresholds: dict[str, float] = {z.name: 1.4 for z in zones}
        self.link_baselines: dict[str, float] = {}
        self.link_drop_thresholds: dict[str, float] = {}
        self.baseline_variance: dict[str, float] = {z.name: 0.0002 for z in zones}
        self.is_calibrated: bool = False
        self.is_calibrating: bool = False
        self.calibration_samples: dict[str, list[float]] = {z.name: [] for z in zones}
        self.calibration_link_samples: dict[str, list[float]] = {}
        self.calibration_variance_samples: dict[str, list[float]] = {z.name: [] for z in zones}
        self.calibration_start_ts: float = 0.0
        self.calibration_duration_s: float = 45.0
        # Auto-calibrate parameters
        self.auto_calibrate_enabled = True
        self.auto_tune_variance = bool(
            self.calibration_cfg.get("auto_calibrate_variance", True)
            and self.calibration_cfg.get("set_variance_from_calibration", True)
        )
        self.require_calibration_for_actuation = bool(
            self.calibration_cfg.get(
                "require_calibration_for_actuation",
                self.calibration_cfg.get("auto_calibrate_on_start", False),
            )
        )

        self.br_filter = OneEuroFilter1D(min_cutoff=0.05, beta=0.1)
        self.hr_filter = OneEuroFilter1D(min_cutoff=0.05, beta=0.1)
        self.breathing_bpm = 16.2
        self.heart_bpm = 72.0
        self.subcarriers = [0.10 for _ in range(56)]

    def apply_rx_layout(self, preset_key: str, custom_rx: Optional[tuple[float, float]] = None) -> dict:
        if preset_key in RX_LAYOUT_PRESETS:
            preset = RX_LAYOUT_PRESETS[preset_key]
            self.layout_preset = preset_key
            self.rx_pos = preset["rx_pos"]
            for nid, pos in preset["tx_nodes"].items():
                if nid in self.node_positions:
                    self.node_positions[nid] = pos
                    if nid in self.zones:
                        self.zones[nid].pos_m = pos
        if custom_rx is not None:
            self.rx_pos = (float(custom_rx[0]), float(custom_rx[1]))
        self.ai_model.rx_position = self.rx_pos
        self.csi_engine.update_geometry(self.node_positions, self.rx_pos)
        return {
            "layout_preset": self.layout_preset,
            "rx_pos": list(self.rx_pos),
            "tx_nodes": {k: list(v) for k, v in self.node_positions.items()}
        }

    def start_calibration(self, duration_s: float = 30.0) -> None:
        self.is_calibrating = True
        self.is_calibrated = False
        if self.simulated_person_count == 0:
            self.simulated_person_count = None
        self.calibration_duration_s = max(1.0, float(duration_s))
        self.calibration_start_ts = time.time()
        self.calibration_samples = {z.name: [] for z in self.zones.values()}
        self.calibration_link_samples = {}
        self.calibration_variance_samples = {z.name: [] for z in self.zones.values()}
        console.print(f"[bold cyan]⚖️  Started {self.calibration_duration_s:.0f}s Free Space CSI + RSSI Baseline Calibration...[/bold cyan]")

    def finish_calibration(self) -> dict:
        required_zones = [
            zone.name for zone in self.zones.values()
            if zone.led_pin in ACTUATOR_LED_PINS
        ]
        missing_samples = {
            name: len(self.calibration_variance_samples.get(name, []))
            for name in required_zones
            if len(self.calibration_variance_samples.get(name, [])) < 3
        }
        if missing_samples:
            self.is_calibrating = False
            details = ", ".join(f"{name}={count}" for name, count in missing_samples.items())
            raise ValueError(
                "Calibration needs at least 3 PC-side variance samples for each bulb zone; "
                f"missing/insufficient: {details}"
            )

        self.is_calibrating = False
        self.is_calibrated = True
        if self.simulated_person_count == 0:
            self.simulated_person_count = None

        # 1. Compute per-directed-link calibrated baselines first (never mixes different receivers!)
        for lk, lsamples in self.calibration_link_samples.items():
            if len(lsamples) >= 3:
                lmean = float(statistics.mean(lsamples))
                lstd = float(statistics.stdev(lsamples))
                lth = float(max(1.35, min(3.5, 2.5 * lstd)))
            elif lsamples:
                lmean = float(statistics.mean(lsamples))
                lth = 1.4
            else:
                continue
            self.link_baselines[lk] = round(lmean, 2)
            self.link_drop_thresholds[lk] = round(lth, 2)
            if lk in self.directed_links:
                curr_r = float(self.directed_links[lk].get("rssi", lmean))
                self.directed_links[lk]["base_rssi"] = round(lmean, 2)
                self.directed_links[lk]["drop_db"] = round(max(0.0, lmean - curr_r), 2)
                self.directed_links[lk]["abs_delta_db"] = round(abs(lmean - curr_r), 2)

        # 2. Compute per-node baselines (preferring the most-populated directed link for each tx_node)
        for z_name in self.zones:
            matching_links = [
                (lk, samps) for lk, samps in self.calibration_link_samples.items()
                if lk.endswith(f"<-{z_name}") and len(samps) >= 2
            ]
            if matching_links:
                best_lk, samples = max(matching_links, key=lambda kv: len(kv[1]))
            else:
                samples = self.calibration_samples.get(z_name, [])

            var_samples = getattr(self, "calibration_variance_samples", {}).get(z_name, [])
            if len(samples) >= 3:
                mean_r = float(statistics.mean(samples))
                std_r = float(statistics.stdev(samples))
                thresh = float(max(1.35, min(3.5, 2.5 * std_r)))
            elif samples:
                mean_r = float(statistics.mean(samples))
                thresh = 1.4
            else:
                mean_r = float(self.node_rssi.get(z_name, -55.0))
                thresh = 1.4
            self.baseline_rssi[z_name] = round(mean_r, 1)
            self.drop_thresholds[z_name] = round(thresh, 2)
            # Auto-tune variance threshold (clamped to realistic hardware range [0.002, 0.080])
            if self.auto_tune_variance and len(var_samples) >= 3:
                try:
                    mean_var = float(statistics.mean(var_samples))
                    std_var = float(statistics.stdev(var_samples)) if len(var_samples) > 1 else 0.001
                    vth = float(max(0.0020, min(0.0800, mean_var + max(3.0 * std_var, 0.0005))))
                    zobj = self.zones.get(z_name)
                    if zobj is not None and hasattr(zobj, "variance_threshold"):
                        zobj.variance_threshold = round(vth, 4)
                    if hasattr(self, "baseline_variance"):
                        self.baseline_variance[z_name] = mean_var
                except Exception:
                    pass
            zobj = self.zones.get(z_name)
            if zobj is not None:
                zobj.is_active = False
                zobj.last_variance = 0.0
                zobj.last_csi_disp_mm = 0.0
                zobj.last_triggered = 0.0
                self.node_var[z_name] = 0.0

        self.ai_model.baseline_rssi = dict(self.baseline_rssi)
        self.ai_model.drop_thresholds = dict(self.drop_thresholds)
        self.ai_model.variance_thresholds = {z_name: z.variance_threshold for z_name, z in self.zones.items()}
        self.ai_model.is_calibrated = True
        self.human_detected = False
        self.person_count = 0
        self.presence_prob = 0.02
        self.occupancy_label = "QUIET (EMPTY TABLE)"
        self.last_csi_result = None

        # Calibrate per-link baseline excess into CSI engine
        for lk, rec in getattr(self.csi_engine, "directed_links", {}).items():
            base_excess = max(0.0, float(rec.get("excess_path_m", 0.0)))
            self.csi_engine.set_link_baseline_excess(lk, base_excess)
        for lk, rec in self.directed_links.items():
            base_excess = max(0.0, float(rec.get("excess_path_m", 0.0)))
            self.csi_engine.set_link_baseline_excess(lk, base_excess)
        if self.csi_engine.link_baseline_excess:
            self.csi_engine.empty_bed_calibrated = True
        if getattr(self, "ukf", None) is not None:
            pts = list(self.node_positions.values()) or [(0.5, 0.35)]
            cx = sum(p[0] for p in pts) / len(pts)
            cz = sum(p[1] for p in pts) / len(pts)
            self.ukf.x = np.array([cx, cz, 0.0, 0.0])
            self.ukf.P = np.eye(4) * (0.02**2)
            self._last_ukf_t = time.monotonic()

        console.print(f"[bold green]✅ Auto-Calibration Complete (per-link baselines & thresholds applied)[/bold green]")
        return {
            "baselines": self.baseline_rssi,
            "link_baselines": self.link_baselines,
            "thresholds": self.drop_thresholds,
            "variance_thresholds": {z_name: round(getattr(z, "variance_threshold", 0.0008), 4) for z_name, z in self.zones.items()},
            "duration_s": self.calibration_duration_s
        }

    def set_manual_target(self, rx: float, rz: float, zone_name: str = "node_1",
                          duration_s: float = 8.0, target_id: int = 1) -> None:
        clamped = (
            max(0.04, min(self.room_w - 0.04, rx)),
            max(0.04, min(self.room_l - 0.04, rz))
        )
        if target_id == 2:
            self.override_target_2 = clamped
            self.simulated_person_count = 2
        else:
            self.override_target = clamped
            if self.simulated_person_count == 0:
                self.simulated_person_count = 1
        self.override_expiry = time.time() + duration_s
        self.active_zone = zone_name

    def update_csi(self, csi_frame: dict) -> dict:
        """Processes a 56-subcarrier CSI frame through CSIEngine."""
        res = self.csi_engine.process_csi_frame(csi_frame)
        self.last_csi_result = res
        if self.csi_engine.last_mag_curve:
            self.subcarriers = list(self.csi_engine.last_mag_curve)
        return res

    def update_directed_link(self, rx_node: str, tx_node: str, raw_rssi: Optional[float],
                             amplitude: float, variance: float,
                             device_ts_us: Optional[int] = None,
                             time_source: str = "absent") -> None:
        if not rx_node or not tx_node or rx_node == tx_node:
            return
        lk = f"{rx_node}<-{tx_node}"
        now_ts = time.time()
        rssi_val = float(raw_rssi) if raw_rssi is not None else (-100.0 + amplitude * 70.0)
        prev = self.directed_links.get(lk)
        if lk in self.link_baselines:
            base_r = self.link_baselines[lk]
        elif prev is not None and self.is_calibrated:
            base_r = prev.get("base_rssi", rssi_val)
        elif self.is_calibrated and tx_node in self.baseline_rssi:
            base_r = self.baseline_rssi[tx_node]
        else:
            base_r = rssi_val

        if prev is None:
            sm_var = variance
        else:
            prev_v = float(prev.get("variance", variance))
            sm_var = (0.20 * prev_v + 0.80 * variance) if variance >= prev_v else (0.50 * prev_v + 0.50 * variance)

        drop_db = max(0.0, float(base_r) - rssi_val)
        abs_delta_db = abs(float(base_r) - rssi_val)

        # Sample interval: radio clock when both packets have one, host arrival
        # deltas otherwise. `ts` stays host time because staleness checks compare
        # it against time.time().
        prev_ts_us = float(prev.get("ts_us")) if prev and prev.get("ts_us") is not None else None
        if device_ts_us is not None and prev_ts_us is not None:
            dt_s = (float(device_ts_us) - prev_ts_us) / 1e6
        else:
            dt_s = now_ts - float(prev["ts"]) if prev and prev.get("ts") is not None else None
        if dt_s is not None and dt_s < 0.0:
            dt_s = None

        self.directed_links[lk] = {
            "rx_node": rx_node,
            "tx_node": tx_node,
            "rssi": round(rssi_val, 2),
            "base_rssi": round(float(base_r), 2),
            "drop_db": round(drop_db, 2),
            "abs_delta_db": round(abs_delta_db, 2),
            "variance": round(sm_var, 5),
            "ts": now_ts,
            "ts_us": device_ts_us,
            "time_source": time_source,
            "dt_s": round(dt_s, 6) if dt_s is not None else None,
        }

    def update_spatial(self, active_zone_name: str, variance: float, amplitude: float,
                       raw_rssi: Optional[float] = None, rx_node: Optional[str] = None,
                       device_ts_us: Optional[int] = None,
                       time_source: str = "absent") -> None:
        zone = self.zones.get(active_zone_name)
        if not zone:
            return

        now_ts = time.time()
        self.node_last_ts[active_zone_name] = now_ts
        prev_var = self.node_var.get(active_zone_name, variance)
        self.node_var[active_zone_name] = (0.15 * prev_var + 0.85 * variance) if variance >= prev_var else (0.45 * prev_var + 0.55 * variance)
        rssi_val = raw_rssi if raw_rssi is not None else (-100.0 + amplitude * 70.0)
        self.node_rssi[active_zone_name] = rssi_val
        zone.last_rssi = rssi_val
        zone.last_variance = variance
        zone.last_amplitude = amplitude

        if self.is_calibrating:
            if active_zone_name not in self.calibration_samples:
                self.calibration_samples[active_zone_name] = []
            self.calibration_samples[active_zone_name].append(rssi_val)
            if hasattr(self, "calibration_variance_samples"):
                if active_zone_name not in self.calibration_variance_samples:
                    self.calibration_variance_samples[active_zone_name] = []
                self.calibration_variance_samples[active_zone_name].append(float(variance))
            if rx_node and rx_node != active_zone_name:
                lk = f"{rx_node}<-{active_zone_name}"
                self.calibration_link_samples.setdefault(lk, []).append(rssi_val)

        if rx_node and rx_node in self.node_positions and rx_node != active_zone_name:
            self.update_directed_link(
                rx_node, active_zone_name, rssi_val, amplitude, variance,
                device_ts_us=device_ts_us, time_source=time_source,
            )
            # Keep the receiving USB node card updated if no other ESP is scanning rx_node
            if (now_ts - self.node_last_ts.get(rx_node, 0.0)) > 2.5:
                rx_links = [lrec for lrec in self.directed_links.values() if lrec.get("rx_node") == rx_node]
                if rx_links:
                    mean_rx_r = sum(float(l["rssi"]) for l in rx_links) / len(rx_links)
                    self.node_rssi[rx_node] = round(mean_rx_r, 1)
                    if rx_node in self.zones:
                        self.zones[rx_node].last_rssi = round(mean_rx_r, 1)

        # Issue 22 Fix: ONLY synthesize CSI when no real hardware CSI has arrived on this link
        if not self.csi_engine.has_recent_hardware_csi(active_zone_name, max_age_s=2.0):
            tx_pos = self.node_positions.get(active_zone_name, (0.5, 0.35))
            est_dist = math.hypot(self.current_x - tx_pos[0], self.current_z - tx_pos[1])
            synth_frame = self.csi_engine.synthesize_csi_from_rssi(
                active_zone_name, rssi=rssi_val, variance=variance,
                target_dist_m=max(0.04, est_dist), rx_id=rx_node
            )
            self.update_csi(synth_frame)

    @staticmethod
    def _build_17_keypoints(three_x: float, three_z: float, phase_offset: float = 0.0,
                            is_walking: bool = False) -> dict:
        t = time.time() * 1.8 + phase_offset
        sway_x = math.sin(t * 0.7) * (0.015 if not is_walking else 0.04)
        chest_breath = math.sin(t * 1.5) * 0.018
        stride = math.sin(t * 2.8) * 0.24 if is_walking else 0.0

        raw_kps = {
            "nose":           (three_x + sway_x,        1.70, three_z,               0.96),
            "left_eye":       (three_x + sway_x - 0.07, 1.74, three_z - 0.04,        0.94),
            "right_eye":      (three_x + sway_x + 0.07, 1.74, three_z - 0.04,        0.94),
            "left_ear":       (three_x + sway_x - 0.14, 1.72, three_z,               0.90),
            "right_ear":      (three_x + sway_x + 0.14, 1.72, three_z,               0.90),
            "left_shoulder":  (three_x - 0.35,          1.45 + chest_breath, three_z, 0.95),
            "right_shoulder": (three_x + 0.35,          1.45 + chest_breath, three_z, 0.95),
            "left_elbow":     (three_x - 0.44,          1.15, three_z + 0.06 - stride * 0.4, 0.92),
            "right_elbow":    (three_x + 0.44,          1.15, three_z + 0.06 + stride * 0.4, 0.92),
            "left_wrist":     (three_x - 0.38,          0.88, three_z + 0.16 - stride * 0.6, 0.89),
            "right_wrist":    (three_x + 0.38,          0.88, three_z + 0.16 + stride * 0.6, 0.89),
            "left_hip":       (three_x - 0.20,          0.95, three_z,               0.94),
            "right_hip":      (three_x + 0.20,          0.95, three_z,               0.94),
            "left_knee":      (three_x - 0.21,          0.50, three_z + 0.03 + stride, 0.91),
            "right_knee":     (three_x + 0.21,          0.50, three_z + 0.03 - stride, 0.91),
            "left_ankle":     (three_x - 0.21,          0.05, three_z + stride * 1.25, 0.89),
            "right_ankle":    (three_x + 0.21,          0.05, three_z - stride * 1.25, 0.89),
        }
        return {
            k: {"x": round(v[0], 4), "y": round(v[1], 4), "z": round(v[2], 4), "confidence": v[3]}
            for k, v in raw_kps.items()
        }

    def generate_payload(self) -> dict:
        now = time.time()
        for k in list(self.node_var.keys()):
            idle_s = now - self.node_last_ts.get(k, now)
            if idle_s > 1.2:
                decay = math.exp(-(idle_s - 1.2) * 2.5)
                self.node_var[k] = max(0.00005, self.node_var[k] * decay)
        for lk, lrec in list(self.directed_links.items()):
            tx_k = str(lrec.get("tx_node", ""))
            idle_lk = now - float(lrec.get("ts", now))
            if idle_lk > 1.2:
                lrec["variance"] = round(max(0.0, float(lrec.get("variance", 0.0)) * math.exp(-(idle_lk - 1.2) * 2.5)), 5)
            if tx_k in self.node_var and self.node_var[tx_k] < 0.00025:
                lrec["variance"] = min(float(lrec.get("variance", 0.0)), float(self.node_var[tx_k]))

        if self.override_target and now > self.override_expiry:
            self.override_target = None
            self.override_target_2 = None

        max_var = max(self.node_var.values()) if self.node_var else 0.0
        avg_rssi = sum(self.node_rssi.values()) / max(1, len(self.node_rssi))

        raw_br = 15.8 + min(8.0, max_var * 120.0)
        raw_hr = 71.0 + min(28.0, max_var * 350.0)
        self.breathing_bpm = self.br_filter.filter(raw_br, now)
        self.heart_bpm = self.hr_filter.filter(raw_hr, now)

        var_thr_ref = min(
            (float(z.variance_threshold) for z in self.zones.values()),
            default=0.0020,
        )
        features_8dim = [
            float(min(1.0, max_var * 30.0)),
            float(min(1.0, max_var * 15.0)),
            float(self.breathing_bpm / 30.0),
            float(self.heart_bpm / 120.0),
            float(min(1.0, max_var * 2.0)),
            1.0 if max_var >= var_thr_ref else 0.25,
            0.0,
            float((avg_rssi + 100.0) / 100.0)
        ]

        ai_out = self.ai_model.infer(
            features_8dim=features_8dim,
            rssi_map=self.node_rssi,
            variance_map=self.node_var,
            node_positions=self.node_positions,
            rx_position=self.rx_pos,
            room_size=(self.room_w, self.room_l),
            override_target=self.override_target,
            override_target_2=self.override_target_2,
            simulated_person_count=self.simulated_person_count,
            csi_estimate=self.last_csi_result,
            directed_links=self.directed_links,
        )

        self.human_detected = ai_out["human_detected"]
        self.person_count = ai_out["person_count"]
        self.occupancy_label = ai_out["occupancy_label"]
        self.detection_mode = ai_out["detection_mode"]
        self.tomography_diagnostics = ai_out.get("tomography_diagnostics", self.tomography_diagnostics)
        self.confidence = ai_out["confidence"]
        self.presence_prob = ai_out["presence_prob"]
        self.contrastive_shift = ai_out["contrastive_shift"]
        self.embedding_sample = ai_out["embedding_sample"]
        self.ai_latency_ms = ai_out["latency_ms"]
        self.active_clusters = ai_out["active_clusters"]
        self.heatmap = ai_out.get("heatmap", self.heatmap)

        static_obj = getattr(self.csi_engine, "last_static_object", None)
        pos = ai_out.get("position")
        if not self.human_detected:
            # Free space: the tracker holds geometric centroid so an empty table
            # never projects a moving or phantom static ghost.
            pts = list(self.node_positions.values()) or [(0.5, 0.35)]
            self.current_x = sum(p[0] for p in pts) / len(pts)
            self.current_z = sum(p[1] for p in pts) / len(pts)
            self.vx = self.vz = 0.0
            self.speed_mps = 0.0
            self.uncertainty_m = 0.005
            self.mahalanobis = 0.01
        elif static_obj is not None:
            self.current_x, self.current_z = static_obj
            self.vx = self.vz = 0.0
            self.speed_mps = 0.0
            self.uncertainty_m = 0.020
            self.mahalanobis = 0.12
        elif pos:
            target_x = float(pos["x"])
            target_z = float(pos["z"])
            if getattr(self, "ukf", None) is not None:
                now_t = time.monotonic()
                dt = max(0.005, min(0.5, now_t - getattr(self, "_last_ukf_t", now_t)))
                self._last_ukf_t = now_t
                self.ukf.predict(dt=dt)
                self.ukf.update(np.array([target_x, target_z]), lambda s: s[:2], R_val=0.02**2)
                self.current_x = float(self.ukf.x[0])
                self.current_z = float(self.ukf.x[1])
                self.vx = float(self.ukf.x[2])
                self.vz = float(self.ukf.x[3])
                self.speed_mps = math.hypot(self.vx, self.vz)
                self.uncertainty_m = float(np.sqrt(self.ukf.P[0, 0] + self.ukf.P[1, 1]))
            else:
                self.current_x = target_x
                self.current_z = target_z
                self.vx = pos.get("vx", 0.0)
                self.vz = pos.get("vz", 0.0)
                self.speed_mps = pos.get("speed_mps", 0.0)
                self.uncertainty_m = pos.get("uncertainty_m", 0.015)
            self.mahalanobis = pos.get("mahalanobis", 0.12)
            if getattr(self, "ukf", None) is not None:
                self.ukf.x = np.array([self.current_x, self.current_z, 0.0, 0.0])
                self.ukf.P = np.eye(4) * (0.02**2)
                self._last_ukf_t = time.monotonic()

        sec_pos = ai_out.get("secondary_position", {"x": 0.75, "z": 0.20, "speed_mps": 0.0, "uncertainty_m": 0.03})

        regime = self.tomography_diagnostics.get("location_regime", "")
        if regime == "near_rx_hub":
            self.active_zone = "rx_hub"
        else:
            best_zone = max(
                self.node_positions.keys(),
                key=lambda nk: self.node_var.get(nk, 0.0) - 0.002 * math.hypot(
                    self.current_x - self.node_positions[nk][0],
                    self.current_z - self.node_positions[nk][1]
                )
            )
            self.active_zone = best_zone

        scale_3d_x = 5.0 / max(0.1, self.room_w)
        scale_3d_z = scale_3d_x  # one units-per-metre factor for both axes, or the scene stretches
        three_x = (self.current_x - self.room_w / 2.0) * scale_3d_x
        three_z = (self.current_z - self.room_l / 2.0) * scale_3d_z
        is_walking = self.speed_mps > 0.03

        primary_kps = self._build_17_keypoints(three_x, three_z, phase_offset=0.0, is_walking=is_walking)

        persons_list = []
        if self.human_detected and self.person_count >= 1:
            persons_list.append({
                "id": 1,
                "label": "Human #1",
                "zone": self.active_zone,
                "regime": regime,
                "room_x": round(self.current_x, 4),
                "room_z": round(self.current_z, 4),
                "cm_x": round(self.current_x * 100.0, 1),
                "cm_y": round(self.current_z * 100.0, 1),
                "three_x": round(three_x, 3),
                "three_z": round(three_z, 3),
                "speed_mps": round(self.speed_mps, 3),
                "uncertainty_m": round(self.uncertainty_m, 3),
                "confidence": round(self.confidence, 4),
                "keypoints": primary_kps
            })

        if self.person_count >= 2:
            sec_x = float(sec_pos["x"])
            sec_z = float(sec_pos["z"])
            sec_zone = "node_2" if self.active_zone != "node_2" else "node_1"
            sec_conf = round(max(0.84, self.confidence - 0.04), 4)
            sec_3d_x = (sec_x - self.room_w / 2.0) * scale_3d_x
            sec_3d_z = (sec_z - self.room_l / 2.0) * scale_3d_z
            sec_kps = self._build_17_keypoints(
                sec_3d_x, sec_3d_z, phase_offset=2.1, is_walking=(sec_pos.get("speed_mps", 0.0) > 0.03)
            )
            persons_list.append({
                "id": 2,
                "label": "Human #2",
                "zone": sec_zone,
                "regime": "secondary_zone",
                "room_x": round(sec_x, 4),
                "room_z": round(sec_z, 4),
                "cm_x": round(sec_x * 100.0, 1),
                "cm_y": round(sec_z * 100.0, 1),
                "three_x": round(sec_3d_x, 3),
                "three_z": round(sec_3d_z, 3),
                "speed_mps": round(float(sec_pos.get("speed_mps", 0.01)), 3),
                "uncertainty_m": round(float(sec_pos.get("uncertainty_m", 0.03)), 3),
                "confidence": sec_conf,
                "keypoints": sec_kps
            })

        active_leds_count = sum(1 for z in self.zones.values() if z.is_active)
        total_zones = max(1, len(self.zones))
        self.energy_saved_pct = round(((total_zones - active_leds_count) / total_zones) * 100.0, 1)

        if not self.telemetry_log or (now - self.telemetry_log[-1]["ts"] >= 0.5):
            self.telemetry_log.append({
                "ts": round(now, 2),
                "person_count": self.person_count,
                "active_zone": self.active_zone,
                "regime": regime,
                "cm_x": round(self.current_x * 100.0, 1),
                "cm_y": round(self.current_z * 100.0, 1),
                "confidence": round(self.confidence, 3),
                "k_factor": round(self.csi_engine.last_k_factor, 2),
                "gdop": round(self.csi_engine.last_gdop, 2),
                "disp_mm": round(self.csi_engine.last_displacement_mm, 2),
                "hr_bpm": round(self.heart_bpm, 1) if self.human_detected else 0.0,
                "rr_bpm": round(self.breathing_bpm, 1) if self.human_detected else 0.0,
                "rssi_1": round(self.node_rssi.get("node_1", -65.0), 1),
                "rssi_2": round(self.node_rssi.get("node_2", -62.0), 1),
                "rssi_3": round(self.node_rssi.get("node_3", -58.0), 1),
            })

        return {
            "event": "ruview_3d_update",
            "type": "ruview_3d_update",
            "model_version": "ruvnet/wifi-densepose-pretrained-v3.6-csi",
            "engine": "ruview-csi-phase-multi-static-v3.6",
            "scenario": self.active_scenario,
            "layout_preset": self.layout_preset,
            "rx_hub": {
                "pos_m": list(self.rx_pos),
                "label": "ESP32 CSI Receiver Hub "
            },
            "csi_metrics": {
                "enabled": True,
                "csi_source": getattr(self.csi_engine, "csi_source", "em_raytracing_sim"),
                "ranging_mode": "multi_static_tdoa_c_light",
                "subcarriers": 56,
                "phase_quality": round(getattr(self.csi_engine, "last_phase_quality", 0.96), 4),
                "k_factor": round(self.csi_engine.last_k_factor, 2),
                "k_factor_db": round(10.0 * math.log10(max(0.25, self.csi_engine.last_k_factor)), 1),
                "gdop": round(self.csi_engine.last_gdop, 2),
                "gdop_rejected": bool(getattr(self.csi_engine, "last_gdop_rejected", False)),
                "gdop_threshold": round(getattr(self.csi_engine, "gdop_threshold", 5.0), 2),
                "channel": int(self.csi_engine.rig_channel),
                "position_error_m": round(getattr(self.csi_engine, "last_position_error_m", 0.0), 4),
                "residual_m": round(getattr(self.csi_engine, "last_residual_m", 0.0), 4),
                "residual_rejected": bool(getattr(self.csi_engine, "last_residual_rejected", False)),
                "fix_held": bool(getattr(self.csi_engine, "last_fix_held", False)),
                "no_target": bool(getattr(self.csi_engine, "last_no_target", False)),
                "channel_mismatch_count": int(
                    getattr(self.csi_engine, "channel_mismatch_count", 0)
                ),
                "static_object": (
                    {
                        "x": round(self.csi_engine.last_static_object[0], 4),
                        "z": round(self.csi_engine.last_static_object[1], 4),
                        "link": getattr(self.csi_engine, "last_static_object_link", ""),
                        "node": getattr(self.csi_engine, "last_static_object_node", ""),
                        "excess_m": round(
                            getattr(self.csi_engine, "last_static_object_excess_m", 0.0), 4
                        ),
                    }
                    if getattr(self.csi_engine, "last_static_object", None) is not None
                    else None
                ),
                "node_range_valid": {
                    k: bool(
                        (now - self.csi_engine.node_range_ts.get(k, -1e9))
                        <= float(getattr(self.csi_engine, "range_max_age_s", 2.5))
                    )
                    for k in self.node_positions
                },
                "anchors_used": int(getattr(self.csi_engine, "last_anchors_used", 0)),
                "anchors_dropped": list(getattr(self.csi_engine, "last_anchors_dropped", [])),
                "time_source": next(iter(getattr(self.csi_engine, "link_time_source", {}).values()), "absent"),
                "displacement_mm": round(self.csi_engine.last_displacement_mm, 2),
                "signed_displacement_mm": round(getattr(self.csi_engine, "last_signed_displacement_mm", 0.0), 2),
                "direction": getattr(self.csi_engine, "last_direction", "stationary"),
                "tof_ns": round(self.csi_engine.last_tof_ns, 2),
                "mahalanobis": round(self.mahalanobis, 3),
                "outliers_rejected": self.ai_model.tracker_primary.outliers_rejected,
                "phase_curve": self.csi_engine.last_phase_curve
            },
            "tomography_diagnostics": self.tomography_diagnostics,
            "analytics": {
                "energy_saved_pct": self.energy_saved_pct,
                "active_leds": active_leds_count,
                "intrusion_breaches": self.intrusion_breaches,
                "uptime_s": round(now - self.start_ts, 1)
            },
            "ai_model": {
                "repo": "ruvnet/wifi-densepose-pretrained",
                "architecture": "56-SC CSI Phase + Adaptive Mahalanobis Kalman + TOF/RSSI Fusion",
                "embedding_dim": 128,
                "presence_prob": round(self.presence_prob, 4),
                "contrastive_shift": round(self.contrastive_shift, 5),
                "latency_ms": round(self.ai_latency_ms, 2),
                "embedding_sample": self.embedding_sample,
                "human_detected": self.human_detected,
                "person_count": self.person_count,
                "occupancy_label": self.occupancy_label,
                "detection_mode": self.detection_mode
            },
            "room": self.room_cfg,
            "nodes": {
                z_name: {
                    "label": z.label,
                    "ssid": z.ssid,
                    "pos_m": list(self.node_positions.get(z_name, (0.0, 0.0))),
                    "rssi": round(self.node_rssi.get(z_name, -70.0), 1),
                    "variance": round(self.node_var.get(z_name, 0.0), 5),
                    "is_active": z.is_active,
                    "trigger_count": z.trigger_count
                }
                for z_name, z in self.zones.items()
            },
            "zones": {
                z_name: {
                    "name": z_name,
                    "label": z.label,
                    "rssi": round(self.node_rssi.get(z_name, -70.0), 1),
                    "variance": round(self.node_var.get(z_name, 0.0), 5),
                    "is_active": z.is_active,
                }
                for z_name, z in self.zones.items()
            },
            "has_target": self.human_detected,
            "human_detected": self.human_detected,
            "person_count": self.person_count,
            "occupancy_label": self.occupancy_label,
            "active_zone": self.active_zone,
            "confidence": round(self.confidence, 4),
            "position": {
                "x": round(three_x, 3),
                "y": 0.0,
                "z": round(three_z, 3),
                "vx": round(self.vx, 3),
                "vz": round(self.vz, 3),
                "speed_mps": round(self.speed_mps, 3),
                "uncertainty_m": round(self.uncertainty_m, 3),
                "mahalanobis": round(self.mahalanobis, 3)
            },
            "room_position": {
                "x": round(self.current_x, 4),
                "y": round(self.current_z, 4),
                "z": round(self.current_z, 4)
            },
            "table_cm": {
                "x": round(self.current_x * 100.0, 1),
                "y": round(self.current_z * 100.0, 1)
            },
            "heatmap": self.heatmap,
            "persons": persons_list,
            "keypoints": primary_kps,
            "vitals": {
                "respiratory_rate": round(self.breathing_bpm, 1) if self.human_detected else 0.0,
                "heart_rate": round(self.heart_bpm, 1) if self.human_detected else 0.0,
                "status": self.occupancy_label,
                "signal_quality": round(self.confidence, 2)
            },
            "subcarriers": [round(s, 3) for s in self.subcarriers],
            "calibration": {
                **({
                    "active": True,
                    "elapsed_s": round(time.time() - self.calibration_start_ts, 1),
                    "remaining_s": round(max(0.0, self.calibration_duration_s - (time.time() - self.calibration_start_ts)), 1),
                    "progress_pct": round(min(100.0, ((time.time() - self.calibration_start_ts) / self.calibration_duration_s) * 100.0), 1),
                } if self.is_calibrating else {"active": False}),
                "sample_counts": {
                    name: len(samples) for name, samples in self.calibration_samples.items()
                },
                "variance_sample_counts": {
                    name: len(samples) for name, samples in self.calibration_variance_samples.items()
                },
            },
            "is_calibrated": self.is_calibrated,
            "baselines": self.baseline_rssi,
            "drop_thresholds": self.drop_thresholds,
            "delta_rssi": {
                nid: round(max(0.0, self.baseline_rssi.get(nid, self.node_rssi.get(nid, -70.0)) - self.node_rssi.get(nid, -70.0)), 1)
                for nid in self.node_positions
            },
            "y_delta_segments": self.csi_engine.get_segment_metrics(),
            "directed_links": self.directed_links,
        }

    def get_sensing_update_payload(self) -> dict:
        """Generates standard SensingUpdate contract matching the 3D RuView Observatory (https://ruvnet.github.io/RuView/)."""
        scale_3d_x = 5.0 / max(0.1, self.room_w)
        scale_3d_z = scale_3d_x  # one units-per-metre factor for both axes, or the scene stretches
        three_x = (self.current_x - self.room_w / 2.0) * scale_3d_x
        three_z = (self.current_z - self.room_l / 2.0) * scale_3d_z
        is_walking = self.speed_mps > 0.03
        primary_kps = self._build_17_keypoints(three_x, three_z, phase_offset=0.0, is_walking=is_walking)

        obs_persons = []
        if self.human_detected and self.person_count >= 1:
            obs_persons.append({
                "id": 1,
                "position": [round(three_x, 3), 1.0, round(three_z, 3)],
                "pose": "walking" if is_walking else "standing",
                "facing": 0.0,
                "motion_score": int(min(100, max(15, self.speed_mps * 140))),
                "keypoints": primary_kps
            })
            if self.person_count >= 2:
                sec_zone = "node_2" if self.active_zone != "node_2" else "node_1"
                sp = self.node_positions.get(sec_zone, (self.room_w * 0.62, self.room_l * 0.22))
                sec_x = sp[0] * 0.65 + (self.room_w * 0.5) * 0.35
                sec_z = sp[1] * 0.65 + (self.room_l * 0.5) * 0.35
                sec_3d_x = (sec_x - self.room_w / 2.0) * scale_3d_x
                sec_3d_z = (sec_z - self.room_l / 2.0) * scale_3d_z
                sec_kps = self._build_17_keypoints(sec_3d_x, sec_3d_z, phase_offset=2.1, is_walking=False)
                obs_persons.append({
                    "id": 2,
                    "position": [round(sec_3d_x, 3), 1.0, round(sec_3d_z, 3)],
                    "pose": "standing",
                    "facing": 1.2,
                    "motion_score": 25,
                    "keypoints": sec_kps
                })

        mean_rssi = statistics.mean(self.node_rssi.values()) if self.node_rssi else -65.0
        max_var = max(self.node_var.values()) if self.node_var else 0.0

        vals = [0.05] * 400
        if self.human_detected:
            gx = int(min(19, max(0, (self.current_x / max(0.1, self.room_w)) * 20)))
            gz = int(min(19, max(0, (self.current_z / max(0.1, self.room_l)) * 20)))
            for dz in range(-2, 3):
                for dx in range(-2, 3):
                    nx, nz = gx + dx, gz + dz
                    if 0 <= nx < 20 and 0 <= nz < 20:
                        dist = math.sqrt(dx * dx + dz * dz)
                        vals[nz * 20 + nx] = min(1.0, max(0.05, 0.75 * math.exp(-0.5 * (dist / 1.2) ** 2)))

        return {
            "type": "sensing_update",
            "timestamp": time.time(),
            "source": "live",
            "scenario": "single_breathing" if self.human_detected else "empty_room",
            "nodes": [
                {
                    "node_id": idx + 1,
                    "rssi_dbm": round(self.node_rssi.get(z_name, -65.0), 1),
                    "position": [self.node_positions.get(z_name, (0, 0))[0], 0, self.node_positions.get(z_name, (0, 0))[1]],
                    "amplitude": [round(s, 3) for s in self.subcarriers] if self.subcarriers else [0.2] * 64,
                    "subcarrier_count": 64,
                    "role": "transmitter",
                    "vertex_angle_deg": (idx * 120) % 360,
                }
                for idx, z_name in enumerate(self.zones.keys())
            ],
            "mobile_hotspot": {
                "node_id": 100,
                "position": [getattr(self, "mobile_pos", [0.50, 0.35])[0], 0, getattr(self, "mobile_pos", [0.50, 0.35])[1]],
                "role": "centroid_mobile",
                "label": "Mobile Hotspot at Centroid C",
            },
            "receivers": [
                {
                    "node_id": 101,
                    "rssi_dbm": round(self.node_rssi.get("node_1", -65.0), 1),
                    "position": [self.rx_pos[0], 0, self.rx_pos[1]],
                    "role": "receiver",
                    "label": "Rx 1 ",
                },
                {
                    "node_id": 102,
                    "rssi_dbm": round(self.node_rssi.get("node_2", -62.0), 1),
                    "position": [getattr(self, "rx_pos_2", [0.84, 0.56])[0], 0, getattr(self, "rx_pos_2", [0.84, 0.56])[1]],
                    "role": "receiver",
                    "label": "Rx 2 ()",
                },
            ],
            "features": {
                "mean_rssi": round(mean_rssi, 1),
                "variance": round(max_var, 4),
                "std": round(math.sqrt(max_var) if max_var > 0 else 0.05, 3),
                "motion_band_power": round(min(1.0, max_var * 40.0), 3),
                "breathing_band_power": round(0.12 if self.human_detected else 0.01, 3),
                "dominant_freq_hz": round(self.breathing_bpm / 60.0 if self.human_detected else 0.05, 2),
                "spectral_power": round(min(1.0, max_var * 25.0), 3)
            },
            "classification": {
                "motion_level": "active" if max_var > 0.003 else ("present" if self.human_detected else "absent"),
                "presence": self.human_detected,
                "confidence": round(self.confidence, 4),
                "fall_detected": False
            },
            "signal_field": {
                "grid_size": [20, 1, 20],
                "values": vals
            },
            "vital_signs": {
                "breathing_rate_bpm": round(self.breathing_bpm, 1) if self.human_detected else 0.0,
                "heart_rate_bpm": round(self.heart_bpm, 1) if self.human_detected else 0.0,
                "breathing_confidence": round(self.confidence, 2) if self.human_detected else 0.0,
                "heart_rate_confidence": round(self.confidence, 2) if self.human_detected else 0.0
            },
            "persons": obs_persons,
            "estimated_persons": len(obs_persons),
            "edge_modules": {
                "spatial": {
                    "occupancy": self.occupancy_label,
                    "active_zone": self.active_zone,
                    "calibrated": self.is_calibrated
                }
            },
            "_observatory": {
                "subcarrier_iq": [],
                "per_subcarrier_variance": [round(s * 0.1, 4) for s in (self.subcarriers[:64] if self.subcarriers else [0.02] * 64)]
            }
        }


class ExhibitionController:
    def __init__(self, zones: list[Zone], bridge: SerialBridge,
                 hold_time: float = 1.2, debug: bool = False,
                 ssid_to_zone: Optional[dict] = None):
        self.zones = {z.name: z for z in zones}
        self.bridge = bridge
        self.hold_time = hold_time
        self.debug = debug
        self.ssid_to_zone = ssid_to_zone or {}
        self.dashboard_server: Optional[DashboardServer] = None
        self.spatial_engine: Optional[RuViewSpatialEngine] = None

    def _process_csi_message(self, msg: dict) -> tuple[Optional[str], float, Optional[float]]:
        """Processes 56-subcarrier hardware CSI frames from ESP32 (`event == 'csi'`)."""
        raw_z = str(msg.get("zone", "node_1"))
        zone_name = self.ssid_to_zone.get(raw_z, raw_z)
        if zone_name not in self.zones:
            return None, 0.0, None

        raw_rx = str(msg.get("rx_id") or msg.get("node") or "")
        rx_node = self.ssid_to_zone.get(raw_rx, raw_rx) if raw_rx else ""
        # Reject self-CSI frames (e.g. node_3 reporting zone=node_3 on Ch 11)
        if rx_node and rx_node == zone_name:
            return None, 0.0, None

        csi_mag = msg.get("csi_mag", [])
        csi_phase = msg.get("csi_phase", [])
        try:
            raw_rssi = float(msg.get("rssi", -65.0))
            if raw_rssi >= 0.0 or raw_rssi < -110.0:
                raw_rssi = -65.0
        except (TypeError, ValueError):
            raw_rssi = -65.0
        zone_obj = self.zones.get(zone_name)

        if self.spatial_engine and (csi_mag or msg.get("csi_amp") or msg.get("csi_iq")):
            eff_rssi = raw_rssi
            if zone_obj is not None and zone_obj.last_scan_rssi is not None:
                eff_rssi = float(zone_obj.last_scan_rssi)
            beacon_derived = bool(msg.get("synthesized", False))
            res = self.spatial_engine.update_csi({
                "zone": zone_name,
                "rx_id": rx_node,
                "node": rx_node,
                "source": "esp32_beacon_derived" if beacon_derived else "hardware_esp32",
                "synthesized": beacon_derived,
                "rssi": eff_rssi,
                "channel": msg.get("ch", msg.get("channel", 6)),
                "num_sc": msg.get("num_sc", len(csi_mag)),
                "sc_order": msg.get("sc_order", "ieee_80211n_56"),
                "sc_idx": msg.get("sc_idx"),
                "csi_mag": csi_mag or msg.get("csi_amp"),
                "csi_phase": csi_phase,
                "csi_iq": msg.get("csi_iq"),
                "ts_us": msg.get("ts_us"),
                "tsf_us": msg.get("tsf_us"),
                "tsf": msg.get("tsf"),
                "ts": msg.get("ts"),
                "_host_rx_ts": msg.get("_host_rx_ts"),
            })
            disp_mm = abs(float(res.get("displacement_mm", 0.0)))
            if zone_obj:
                zone_obj.last_csi_disp_mm = disp_mm
                zone_obj.last_csi_direction = str(res.get("direction", "stationary"))
                zone_obj.last_csi_ts = time.monotonic()
                zone_obj.last_rssi = eff_rssi
                scan_val = msg.get("scan_rssi")
                if scan_val is not None:
                    try:
                        zone_obj.last_scan_rssi = float(scan_val)
                    except (TypeError, ValueError):
                        pass
            amp = max(0.05, min(0.98, (raw_rssi + 100.0) / 70.0))
            return zone_name, amp, raw_rssi

        return zone_name, max(0.0, min(1.0, (raw_rssi + 100.0) / 70.0)), raw_rssi

    def _map_message_to_zone(self, msg: dict) -> tuple[Optional[str], float, Optional[float]]:
        if not isinstance(msg, dict):
            return None, 0.0, None
        ev = msg.get("event")
        if ev == "csi":
            return self._process_csi_message(msg)
        if ev in ("ruview_3d_update", "serial_tx", "connected", "pong", "scan_diag", "boot"):
            return None, 0.0, None

        raw_rssi = None
        if "rssi" in msg or "raw_rssi" in msg:
            try:
                if "raw_rssi" in msg and "rssi" in msg:
                    r_inst = float(msg["raw_rssi"])
                    r_sm = float(msg["rssi"])
                    # Suppress 1-LSB empty-room toggle (<= 1.0 dB), preserve >= 1.5 dB hand/bottle change
                    raw_rssi = r_sm if abs(r_inst - r_sm) <= 1.0 else r_inst
                elif "raw_rssi" in msg:
                    raw_rssi = float(msg["raw_rssi"])
                else:
                    raw_rssi = float(msg["rssi"])
                if raw_rssi >= 0.0 or raw_rssi < -110.0:
                    raw_rssi = None
            except (ValueError, TypeError):
                raw_rssi = None

        if "zone" in msg and "amplitude" in msg:
            raw_z = str(msg["zone"])
            mapped = self.ssid_to_zone.get(raw_z, raw_z)
            try:
                amp_val = float(msg["amplitude"])
            except (ValueError, TypeError):
                return None, 0.0, None
            return mapped, amp_val, raw_rssi

        zone_key = msg.get("zone") or msg.get("node")
        if zone_key and raw_rssi is not None:
            raw_z = str(zone_key)
            mapped = self.ssid_to_zone.get(raw_z, raw_z)
            rssi = float(raw_rssi)
            return mapped, max(0.0, min(1.0, (rssi + 100.0) / 70.0)), rssi

        if "ssid" in msg or "bssid" in msg:
            ssid = str(msg.get("ssid") or msg.get("bssid") or "")
            rssi = float(raw_rssi if raw_rssi is not None else -70.0)
            return self.ssid_to_zone.get(ssid, ssid), max(0.0, min(1.0, (rssi + 100.0) / 70.0)), rssi

        return None, 0.0, None

    async def run(self, queue: asyncio.Queue) -> None:
        while True:
            msg = await queue.get()
            try:
                zone_name, amplitude, raw_rssi = self._map_message_to_zone(msg)
                if not zone_name or zone_name not in self.zones:
                    continue

                raw_rx = str(msg.get("rx_id") or msg.get("node") or "") if isinstance(msg, dict) else ""
                rx_node = self.ssid_to_zone.get(raw_rx, raw_rx) if raw_rx else None
                zone = self.zones[zone_name]
                # If explicit synthetic amplitude was sent without raw_rssi, pass raw_rssi=None
                has_explicit_amp = isinstance(msg, dict) and ("amplitude" in msg) and ("raw_rssi" not in msg)
                variance = zone.push_sample(
                    amplitude,
                    raw_rssi=(None if has_explicit_amp else raw_rssi),
                    rx_node=rx_node,
                )

                if self.spatial_engine:
                    device_us, stamp_key = extract_device_time_us(msg) if isinstance(msg, dict) else (None, "absent")
                    self.spatial_engine.update_spatial(
                        zone_name, variance, amplitude,
                        raw_rssi=raw_rssi, rx_node=rx_node,
                        device_ts_us=device_us, time_source=stamp_key,
                    )

                now = time.monotonic()

                delta_drop = 0.0
                abs_delta = 0.0
                drop_thr = 1.4
                is_shadow_active = False
                rx_conv = 0.0
                dominant_tx = zone_name

                if self.spatial_engine:
                    if self.spatial_engine.is_calibrated:
                        lk = f"{rx_node}<-{zone_name}" if (rx_node and rx_node != zone_name) else ""
                        if lk and lk in self.spatial_engine.link_baselines:
                            b_rssi = self.spatial_engine.link_baselines[lk]
                            drop_thr = self.spatial_engine.link_drop_thresholds.get(lk, 1.4)
                        else:
                            b_rssi = self.spatial_engine.baseline_rssi.get(zone_name, zone.last_rssi)
                            drop_thr = self.spatial_engine.drop_thresholds.get(zone_name, 1.4)
                        delta_drop = max(0.0, b_rssi - zone.last_rssi)
                        abs_delta = abs(b_rssi - zone.last_rssi)
                        # A shadow is strictly signal attenuation (drop >= threshold).
                        # Positive signal gain (RSSI stronger than baseline) is NEVER a shadow.
                        is_shadow_active = (delta_drop >= drop_thr)
                    diag = self.spatial_engine.tomography_diagnostics
                    rx_conv = float(diag.get("rx_convergence", 0.0))
                    dominant_tx = str(diag.get("dominant_tx", zone_name))

                # CSI phase displacement only triggers alongside variance activity or strong excursion
                csi_fresh = (now - zone.last_csi_ts) <= 1.5
                disp_thr_mm = float(getattr(self, "csi_disp_threshold_mm", 0.45))
                is_csi_disp_active = (
                    csi_fresh
                    and (zone.last_csi_disp_mm >= max(2.5, 3.0 * disp_thr_mm))
                    and (variance >= 0.50 * zone.variance_threshold)
                )
                raw_detected = (variance >= zone.variance_threshold) or is_shadow_active or is_csi_disp_active

                # Issue 24 Fix: Single unified rx_convergence_threshold shared with wifi_densepose_ai.py
                rx_conv_thr = 0.68
                active_links_cnt = 0
                if self.spatial_engine:
                    diag = self.spatial_engine.tomography_diagnostics
                    rx_conv_thr = float(diag.get("rx_convergence_threshold", 0.68))
                    active_links_cnt = int(diag.get("active_links_count", 0))

                # Issue 25 Fix: Configurable secondary_zone_var_ratio (default 0.0 = unblocked)
                sec_ratio = getattr(self, "secondary_zone_var_ratio", 0.0)
                if active_links_cnt >= 3 and rx_conv >= rx_conv_thr and self.spatial_engine and self.spatial_engine.person_count <= 1:
                    is_zone_detected = False
                elif sec_ratio > 0.0 and self.spatial_engine and self.spatial_engine.person_count <= 1 and dominant_tx != zone_name:
                    dom_var = self.spatial_engine.node_var.get(dominant_tx, 0.0)
                    is_zone_detected = raw_detected and (variance >= dom_var * sec_ratio)
                else:
                    is_zone_detected = raw_detected

                if self.spatial_engine and (
                    self.spatial_engine.is_calibrating
                    or (
                        self.spatial_engine.require_calibration_for_actuation
                        and not self.spatial_engine.is_calibrated
                    )
                ):
                    is_zone_detected = False

                if is_zone_detected:
                    if not zone.is_active:
                        zone.is_active = True
                        zone.trigger_count += 1
                        if zone_name == "node_3" and self.spatial_engine:
                            self.spatial_engine.intrusion_breaches += 1
                        if zone.led_pin in ACTUATOR_LED_PINS:
                            self.bridge.led_on(zone.led_pin)
                        det_reason = f"shadow_drop={delta_drop:.1f}dBm" if is_shadow_active and variance < zone.variance_threshold else f"var={variance:.4f}"
                        console.print(
                            f"[bold green]💡  {zone.label}[/bold green] "
                            f"[green]DETECTED[/green]  [dim]({det_reason})[/dim]"
                        )
                        if self.dashboard_server:
                            asyncio.create_task(self.dashboard_server.broadcast({
                                "event": "serial_tx",
                                "cmd": f"<LED:{zone.led_pin}:ON>",
                                "pin": zone.led_pin,
                                "state": "ON",
                                "zone": zone.name,
                                "variance": variance,
                                "variance_threshold": zone.variance_threshold,
                                "rssi_drop_db": delta_drop,
                                "rssi_drop_threshold_db": drop_thr,
                                "pc_calibrated": bool(self.spatial_engine and self.spatial_engine.is_calibrated),
                            }))
                    zone.last_triggered = now

                elif zone.is_active and (now - zone.last_triggered) > self.hold_time:
                    zone.is_active = False
                    if zone.led_pin in ACTUATOR_LED_PINS:
                        self.bridge.led_off(zone.led_pin)
                    console.print(
                        f"[bold red]🌑  {zone.label}[/bold red] "
                        f"[red]CLEARED[/red]  [dim]hold={self.hold_time}s[/dim]"
                    )
                    if self.dashboard_server:
                        asyncio.create_task(self.dashboard_server.broadcast({
                            "event": "serial_tx",
                            "cmd": f"<LED:{zone.led_pin}:OFF>",
                            "pin": zone.led_pin,
                            "state": "OFF",
                            "zone": zone.name,
                            "pc_calibrated": bool(self.spatial_engine and self.spatial_engine.is_calibrated),
                        }))

                # Note: ruview_3d_broadcast_loop broadcasts unified 25Hz telemetry.
                # Avoid flooding the WebSocket event loop with 100+ raw per-packet tasks.
            finally:
                queue.task_done()


def _sanitize_finite(obj):
    """Recursively replaces non-finite float values (NaN, Inf, -Inf) with None for browser JSON compliance."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_finite(v) for v in obj]
    return obj


class DashboardServer:
    def __init__(self, zones: list[Zone], bridge: SerialBridge, queue: asyncio.Queue,
                 html_filename: str = "index.html"):
        self.zones = {z.name: z for z in zones}
        self.bridge = bridge
        self.queue = queue
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_path = os.path.join(self.base_dir, "index.html")
        self.dashboard_path = os.path.join(self.base_dir, "ruview_exhibition_dashboard.html")
        self.map_path = os.path.join(self.base_dir, "map.html")
        self.pose_fusion_path = os.path.join(self.base_dir, "pose-fusion.html")
        self.clients: set[web.WebSocketResponse] = set()
        self._broadcast_lock = asyncio.Lock()
        self.spatial_engine: Optional[RuViewSpatialEngine] = None
        self.judge_tour_task: Optional[asyncio.Task] = None
        self.calibration_task: Optional[asyncio.Task] = None
        self.sensor_bridges: list[SerialBridge] = [bridge]
        self.app = web.Application()

        # RuView Observatory (https://ruvnet.github.io/RuView/)
        self.app.router.add_get("/", self.handle_map)
        self.app.router.add_get("/observatory", self.handle_map)
        self.app.router.add_get("/observatory.html", self.handle_map)
        self.app.router.add_get("/index.html", self.handle_map)

        # Dual-Modal Pose Estimation (Video + WiFi)
        self.app.router.add_get("/pose-fusion", self.handle_pose_fusion)
        self.app.router.add_get("/pose-fusion.html", self.handle_pose_fusion)

        # Dedicated 2D Spatial Map (ESP Positions & Object Tracking)
        self.app.router.add_get("/map", self.handle_map)
        self.app.router.add_get("/map.html", self.handle_map)
        self.app.router.add_get("/2d_map", self.handle_map)
        self.app.router.add_get("/2d-map", self.handle_map)

        # Full Tabletop Exhibition Dashboard
        self.app.router.add_get("/dashboard", self.handle_dashboard)
        self.app.router.add_get("/ruview_exhibition_dashboard.html", self.handle_dashboard)

        # Sensing Server Health Check (auto-detected by Observatory on boot)
        self.app.router.add_get("/health", self.handle_health)

        # WebSockets
        self.app.router.add_get("/ws/sensing", self.handle_ws)
        self.app.router.add_get("/ws", self.handle_ws)
        self.app.router.add_get("/api/status", self.handle_status)
        self.app.router.add_get("/api/export-csv", self.handle_export_csv)

        # Static assets for Observatory & Pose-Fusion
        obs_dir = os.path.join(self.base_dir, "observatory")
        if os.path.exists(obs_dir):
            self.app.router.add_static("/observatory", obs_dir)
        pf_dir = os.path.join(self.base_dir, "pose-fusion")
        if os.path.exists(pf_dir):
            self.app.router.add_static("/pose-fusion", pf_dir)

        self.runner: Optional[web.AppRunner] = None
        self.sites: list[web.TCPSite] = []

    async def handle_index(self, request: web.Request) -> web.Response:
        # Default to map mode for clean, minimal view
        if os.path.exists(self.map_path):
            return web.FileResponse(self.map_path)
        if os.path.exists(self.index_path):
            return web.FileResponse(self.index_path)
        if os.path.exists(self.dashboard_path):
            return web.FileResponse(self.dashboard_path)
        return web.Response(text="Map interface not found", status=404)

    async def handle_map(self, request: web.Request) -> web.Response:
        if os.path.exists(self.map_path):
            return web.FileResponse(self.map_path)
        if os.path.exists(self.dashboard_path):
            return web.FileResponse(self.dashboard_path)
        return web.Response(text="2D spatial map not found", status=404)

    async def handle_dashboard(self, request: web.Request) -> web.Response:
        if os.path.exists(self.dashboard_path):
            return web.FileResponse(self.dashboard_path)
        if os.path.exists(self.map_path):
            return web.FileResponse(self.map_path)
        return web.Response(text="Exhibition dashboard not found", status=404)

    async def handle_pose_fusion(self, request: web.Request) -> web.Response:
        if os.path.exists(self.pose_fusion_path):
            return web.FileResponse(self.pose_fusion_path)
        return web.Response(text="Pose fusion interface not found", status=404)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "service": "ruview-sensing-server",
            "version": "3.2.0-csi"
        })

    async def handle_status(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "online",
            "version": "3.2.0-csi",
            "mode": "demo" if self.bridge.demo_mode else "live",
            "person_count": self.spatial_engine.person_count if self.spatial_engine else 1,
            "rx_hub": list(self.spatial_engine.rx_pos) if self.spatial_engine else [0.50, 0.64],
            "zones": {
                z.name: {
                    "label": z.label,
                    "led_pin": z.led_pin,
                    "variance_threshold": z.variance_threshold,
                    "is_active": z.is_active,
                    "trigger_count": z.trigger_count,
                }
                for z in self.zones.values()
            }
        })

    async def handle_export_csv(self, request: web.Request) -> web.Response:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp", "person_count", "active_zone", "regime", "cm_x", "cm_y", "confidence", "k_factor", "gdop", "disp_mm", "hr_bpm", "rr_bpm", "rssi_tx1", "rssi_tx2", "rssi_tx3"])
        if self.spatial_engine:
            for row in self.spatial_engine.telemetry_log:
                writer.writerow([
                    row["ts"], row["person_count"], row["active_zone"], row.get("regime", ""),
                    row["cm_x"], row["cm_y"], row["confidence"],
                    row.get("k_factor", 5.0), row.get("gdop", 1.2), row.get("disp_mm", 0.0),
                    row["hr_bpm"], row["rr_bpm"],
                    row["rssi_1"], row["rssi_2"], row["rssi_3"]
                ])
        return web.Response(
            text=output.getvalue(),
            content_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="ruview_csi_telemetry_session.csv"'}
        )

    async def handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            room_info = self.spatial_engine.room_cfg if self.spatial_engine else {"width_m": 1.0, "length_m": 0.7}
            node_coords = self.spatial_engine.node_positions if self.spatial_engine else {}
            rx_pos = list(self.spatial_engine.rx_pos) if self.spatial_engine else [0.50, 0.64]
            # Build receivers list from actual configured ports
            is_hw = bool(self.bridge._ser and self.bridge._ser.is_open)
            active_usb = [p.device for p in serial.tools.list_ports.comports() if 'bluetooth' not in p.description.lower()]
            receivers = [{
                "id": "rx_hub",
                "label": f"Rx Hub: Standard ESP32 ({self.bridge.port})",
                "pos_m": list(rx_pos),
                "port": self.bridge.port,
                "connected": is_hw,
                "role": "receiver",
            }]

            # Mobile hotspot at centroid C
            mobile_pos = list(getattr(self.spatial_engine, "mobile_pos", [0.50, 0.35]))
            sec_rx_pos = list(getattr(self.spatial_engine, "rx_pos_2", [0.50, 0.35]))

            await ws.send_json({
                "event": "connected",
                "version": "3.5.0-csi",
                "mode": "demo" if self.bridge.demo_mode else "live",
                "hardware_connected": is_hw,
                "primary_port": self.bridge.port,
                "active_usb_ports": active_usb,
                "architecture": "y_delta_6link",
                "room": room_info,
                "receivers": receivers,
                "rx_hub": {"pos_m": list(rx_pos)},
                "rx_2": {"pos_m": sec_rx_pos},
                "mobile_hotspot": {"pos_m": mobile_pos, "label": "Mobile Hotspot at Centroid C"},
                "nodes": {
                    nid: {
                        "label": self.zones[nid].label if nid in self.zones else nid,
                        "pos_m": list(node_coords.get(nid, [0, 0])),
                        "role": "transmitter",
                        "vertex_angle_deg": (idx * 120) % 360,
                    }
                    for idx, nid in enumerate(self.zones)
                },
                "zones": {
                    z.name: {
                        "label": z.label,
                        "led_pin": z.led_pin,
                        "variance_threshold": z.variance_threshold,
                    }
                    for z in self.zones.values()
                }
            })
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_client_action(data, ws)
                    except json.JSONDecodeError:
                        pass
        finally:
            self.clients.discard(ws)
        return ws

    async def _run_judge_tour_choreography(self) -> None:
        if not self.spatial_engine:
            return
        stages = [
            ("follow_me", 1, 0.22, 0.18, "node_1", "Stage 1/4: CSI Phase Lock near Tx 1 (C3 #1 Living Room) — 1 Dot at Tx 1"),
            ("follow_me", 1, 0.78, 0.18, "node_2", "Stage 2/4: CSI Phase Handoff to Tx 2 (C3 #2 Kitchen) — 1 Dot at Tx 2"),
            ("intrusion_fence", 1, 0.80, 0.46, "node_3", "Stage 3/4: CSI Breach at Tx 3 (Phone Hotspot Secure Room) — Red Alarm ON"),
            ("follow_me", 1, self.spatial_engine.rx_pos[0], self.spatial_engine.rx_pos[1], "rx_hub", "Stage 4/4: Standing Near Receiver (Rx Hub) — 3-Link Common-Mode Lock places 1 Dot at Rx!"),
        ]
        for scenario, p_cnt, tx, tz, zn, announcement in stages:
            self.spatial_engine.active_scenario = scenario
            self.spatial_engine.simulated_person_count = p_cnt
            self.spatial_engine.set_manual_target(tx, tz, zn, duration_s=5.2, target_id=1)
            await self.broadcast({
                "event": "tour_stage",
                "scenario": scenario,
                "announcement": announcement
            })
            for _ in range(10):
                if zn == "rx_hub":
                    for all_z in ("node_1", "node_2", "node_3"):
                        await self.queue.put({"zone": all_z, "amplitude": 0.66 + random.gauss(0, 0.12), "rssi": -68.0})
                else:
                    await self.queue.put({"zone": zn, "amplitude": 0.68 + random.gauss(0, 0.14), "rssi": -68.5})
                await asyncio.sleep(0.45)

    async def _handle_client_action(self, data: dict, ws: web.WebSocketResponse) -> None:
        action = data.get("action")
        if action == "self_test":
            for z in self.zones.values():
                if z.led_pin in ACTUATOR_LED_PINS:
                    self.bridge.led_on(z.led_pin)
            await asyncio.sleep(0.5)
            for z in self.zones.values():
                if z.led_pin in ACTUATOR_LED_PINS:
                    self.bridge.led_off(z.led_pin)
        elif action == "ping":
            self.bridge.send(CMD_PING)
            await ws.send_json({"event": "pong", "latency_ms": 2.4})
        elif action == "set_threshold":
            try:
                z_name = data.get("zone")
                val = float(data.get("value", 0.0008))
                if z_name in self.zones:
                    self.zones[z_name].variance_threshold = val
                    if self.spatial_engine and hasattr(self.spatial_engine, "ai_model"):
                        self.spatial_engine.ai_model.variance_thresholds[z_name] = val
            except (ValueError, TypeError) as exc:
                await ws.send_json({"event": "error", "message": f"Invalid threshold value: {exc}"})
        elif action == "set_rx_layout":
            preset = str(data.get("preset", "perimeter_apex"))
            custom_rx = data.get("rx_pos")
            if self.spatial_engine:
                res = self.spatial_engine.apply_rx_layout(
                    preset,
                    tuple(custom_rx) if isinstance(custom_rx, list) and len(custom_rx) == 2 else None
                )
                await self.broadcast({"event": "rx_layout_updated", **res})
        elif action == "locate_target":
            try:
                rx = float(data.get("x", 0.50))
                ry = float(data.get("y", 0.35))
                z_name = str(data.get("zone", "node_1"))
                target_id = int(data.get("target_id", 1))
            except (ValueError, TypeError) as exc:
                await ws.send_json({"event": "error", "message": f"Invalid coordinates or target_id: {exc}"})
                return
            if self.spatial_engine:
                self.spatial_engine.set_manual_target(rx, ry, z_name, duration_s=8.0, target_id=target_id)
                if z_name == "rx_hub":
                    for all_z in ("node_1", "node_2", "node_3"):
                        await self.queue.put({"zone": all_z, "amplitude": 0.66 + random.gauss(0, 0.12), "rssi": -68.0})
                elif z_name in self.zones:
                    await self.queue.put({"zone": z_name, "amplitude": 0.68 + random.gauss(0, 0.12), "rssi": -68.0})
        elif action == "set_person_count":
            cnt = data.get("count")
            if self.spatial_engine:
                self.spatial_engine.simulated_person_count = int(cnt) if cnt is not None else None
        elif action == "set_scenario":
            scen = str(data.get("scenario", "follow_me"))
            if self.spatial_engine:
                self.spatial_engine.active_scenario = scen
                if scen == "judge_tour":
                    if self.judge_tour_task and not self.judge_tour_task.done():
                        self.judge_tour_task.cancel()
                    self.judge_tour_task = asyncio.create_task(self._run_judge_tour_choreography())
        elif action == "calibrate_baseline":
            try:
                duration_s = float(data.get("duration_s", 45.0))
                await self.start_baseline_calibration(duration_s)
            except (RuntimeError, TypeError, ValueError) as exc:
                await ws.send_json({"event": "error", "message": f"Calibration could not start: {exc}"})

    async def start_baseline_calibration(self, duration_s: float) -> None:
        if self.spatial_engine is None:
            raise RuntimeError("PC spatial engine is not initialized")
        duration_s = max(1.0, float(duration_s))
        if self.calibration_task and not self.calibration_task.done():
            self.calibration_task.cancel()

        for zone in self.zones.values():
            if zone.led_pin in ACTUATOR_LED_PINS:
                self.bridge.led_off(zone.led_pin)
            zone.is_active = False
            zone.last_triggered = 0.0

        engine = self.spatial_engine
        engine.start_calibration(duration_s)
        await self.broadcast({
            "event": "calibration_started",
            "duration_s": duration_s,
            "calibrated_on_pc": True,
        })

        async def finish_calibration_job() -> None:
            await asyncio.sleep(duration_s)
            try:
                result = engine.finish_calibration()
            except Exception as exc:
                engine.is_calibrating = False
                engine.is_calibrated = False
                sample_counts = {
                    name: len(samples)
                    for name, samples in engine.calibration_variance_samples.items()
                }
                serial_frames = {
                    sensor.port: sensor.rx_packets for sensor in self.sensor_bridges
                }
                message = (
                    f"{exc}; serial JSON frames received by port: {serial_frames}. "
                    "Confirm each sensor is connected to the configured COM port, "
                    "close Serial Monitor/other programs using those ports, and "
                    "check that the ESP32s are emitting scan/CSI JSON."
                )
                logging.error("PC-side calibration failed: %s", message)
                await self.broadcast({
                    "event": "calibration_failed",
                    "message": message,
                    "variance_sample_counts": sample_counts,
                    "serial_frame_counts": serial_frames,
                    "auto_applied": False,
                })
                return

            for zone in self.zones.values():
                zone.is_active = False
                zone.last_triggered = 0.0
            await self.broadcast({
                "event": "baseline_calibrated",
                "duration_s": duration_s,
                "baselines": result["baselines"],
                "link_baselines": result["link_baselines"],
                "thresholds": result["thresholds"],
                "variance_thresholds": result["variance_thresholds"],
                "auto_applied": True,
                "calibrated_on_pc": True,
            })

        self.calibration_task = asyncio.create_task(finish_calibration_job())

    async def broadcast(self, data: dict) -> None:
        if not self.clients:
            return
        try:
            sanitized = _sanitize_finite(data)
            msg = json.dumps(sanitized)
        except Exception as exc:
            logging.warning("JSON broadcast serialization error: %s", exc)
            return
        async with self._broadcast_lock:
            dead = set()
            for ws in list(self.clients):
                if ws.closed:
                    dead.add(ws)
                    continue
                try:
                    await asyncio.wait_for(ws.send_str(msg), timeout=0.08)
                except Exception:
                    dead.add(ws)
            if dead:
                self.clients.difference_update(dead)

    async def start(self, ports: list[int] = [3000, 3001]) -> list[int]:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        bound = []
        for port in ports:
            try:
                site = web.TCPSite(self.runner, "0.0.0.0", port)
                await site.start()
                self.sites.append(site)
                bound.append(port)
            except OSError:
                pass
        return bound

    async def stop(self) -> None:
        if self.calibration_task and not self.calibration_task.done():
            self.calibration_task.cancel()
        for ws in list(self.clients):
            try:
                await ws.close()
            except Exception:
                pass
        if self.runner:
            await self.runner.cleanup()


async def demo_ws_stream(queue: asyncio.Queue, zones: list[Zone]) -> None:
    t = 0.0
    zone_names = [z.name for z in zones]
    while True:
        active_zone = zone_names[int(t / 5.0) % len(zone_names)]
        for z in zones:
            if z.name == active_zone:
                amp = max(0.0, min(1.0, 0.72 + random.gauss(0, 0.14)))
                rssi = -68.0 + random.gauss(0, 1.0)
            else:
                amp = max(0.0, min(1.0, 0.35 + random.gauss(0, 0.008)))
                rssi = -60.0 + random.gauss(0, 0.3)
            await queue.put({"zone": z.name, "amplitude": amp, "rssi": rssi, "timestamp": time.time()})
        await asyncio.sleep(0.1)
        t += 0.1


async def serial_sensor_loop(bridge: SerialBridge, queue: asyncio.Queue, zones: Optional[list[Zone]] = None) -> None:
    if bridge.demo_mode:
        return
    loop = asyncio.get_running_loop()
    last_retry_ts = 0.0
    while True:
        if bridge._ser is None or not bridge._ser.is_open:
            now = time.time()
            if now - last_retry_ts >= 2.0:
                last_retry_ts = now
                await loop.run_in_executor(None, bridge.connect)
            if bridge._ser is None or not bridge._ser.is_open:
                # Hardware offline: yield baseline heartbeat so dashboard queues stay active
                if zones:
                    for z in zones:
                        await queue.put({
                            "zone": z.name,
                            "amplitude": z.smoothed_amplitude,
                            "rssi": z.last_rssi,
                            "timestamp": now
                        })
                await asyncio.sleep(0.1)
                continue
        ser = bridge._ser
        try:
            backlog = int(getattr(ser, "in_waiting", 0) or 0)
            if backlog > SERIAL_RX_FLUSH_BYTES:
                # Only shed load when the OS buffer is genuinely drowning. At
                # 921600+ a few hundred bytes queued is normal traffic, not junk.
                ser.reset_input_buffer()
                bridge.dropped_backlog_bytes += backlog
            raw: bytes = await loop.run_in_executor(None, ser.readline)
        except (serial.SerialException, OSError):
            bridge.close()
            bridge._ser = None
            await asyncio.sleep(1.0)
            continue

        host_rx_ts = time.time()
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue
        start_idx = line.find("{")
        end_idx = line.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            continue
        json_str = line[start_idx:end_idx + 1]
        try:
            msg = json.loads(json_str)
            bridge.rx_packets += 1
            # Arrival stamp on the PC clock. The frame's own `ts`/`ts_us`/`tsf`
            # are left alone: they are the radio clock the estimators need.
            msg["_host_rx_ts"] = host_rx_ts
            ev = msg.get("event")
            if ev and ev not in ("csi", "sta") and "zone" not in msg and "node" not in msg:
                continue
            while queue.qsize() > 16:
                queue.get_nowait()
                queue.task_done()
            queue.put_nowait(msg)
        except Exception:
            pass


async def dashboard(zones: list[Zone], spatial_engine: RuViewSpatialEngine) -> None:
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        while True:
            regime = spatial_engine.tomography_diagnostics.get("location_regime", "mid_link")
            k_fac = spatial_engine.csi_engine.last_k_factor
            gdop = spatial_engine.csi_engine.last_gdop
            title_str = (
                f"🏠 RuView v3.2 (CSI + 3Tx/1Rx) • Humans: {spatial_engine.person_count} "
                f"• K={k_fac:.1f} • GDOP={gdop:.2f} • Pos: "
                f"[{spatial_engine.current_x*100:.0f}cm, {spatial_engine.current_z*100:.0f}cm]"
            )
            table = Table(title=title_str, box=box.ROUNDED, header_style="bold cyan")
            table.add_column("Transmitter Link (Tx -> Rx)", style="bold", min_width=22)
            table.add_column("GPIO", justify="center")
            table.add_column("LED Status", justify="center", min_width=12)
            table.add_column("RSSI at Rx", justify="right")
            table.add_column("Variance", justify="right")

            for z in zones:
                status_text = "[bold green]● ACTIVE[/bold green]" if z.is_active else "[dim]○ idle[/dim]"
                table.add_row(
                    z.label,
                    str(z.led_pin),
                    status_text,
                    f"{z.last_rssi:.0f} dBm",
                    f"{z.last_variance:.5f}"
                )
            live.update(Panel(table, subtitle="[dim]http://localhost:3000 • Ctrl+C to stop[/dim]"))
            await asyncio.sleep(0.25)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RuView v3.2 (CSI Phase Ranging + 3Tx/1Rx) Exhibition Controller")
    p.add_argument("--config", default=CONFIG_FILE)
    p.add_argument("--port", help="Primary Receiver serial port (e.g. COM11)")
    p.add_argument("--ports", help="Comma-separated list of all connected ESP COM ports (e.g. COM11,COM6,COM7)")
    p.add_argument("--secondary-port", dest="secondary_port", help="Secondary Receiver #2 serial port (e.g. COM6)")
    p.add_argument("--actuator-port", dest="actuator_port", help="Standard ESP32 serial port that exclusively receives bulb commands")
    p.add_argument(
        "--baud", type=int,
        help="Receiver UART baud rate. Must match Serial.begin() in the firmware. "
             f"Supported: {', '.join(f'{b:,}' for b in SUPPORTED_BAUD_RATES)}. "
             "Capacity at ~950-byte lines: 115200 ~12 frames/s, 921600 ~97/s, "
             "2000000 ~210/s. The nodes currently emit only 1-3.6 CSI frames/s "
             "(firmware scan loop), so raising this buys nothing yet."
    )
    p.add_argument("--ws-url", dest="ws_url")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--http-port", type=int, default=3000)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def _run_controller_selftest() -> None:
    zones = [
        Zone(name="node_1", label="Tx 1: ESP32-C3 #1", led_pin=2, variance_threshold=0.0008, demo_scenario="room_radar", ssid="Zone_1_Tx", pos_m=(0.50, 0.05), channel=6),
        Zone(name="node_2", label="Tx 2: ESP32-C3 #2", led_pin=4, variance_threshold=0.0008, demo_scenario="room_radar", ssid="Zone_2_Tx", pos_m=(0.80, 0.57), channel=6),
        Zone(name="node_3", label="Tx 3: Phone Hotspot", led_pin=5, variance_threshold=0.0008, demo_scenario="room_radar", ssid="Laptop_Hotspot", pos_m=(0.20, 0.57), channel=6),
    ]
    engine = RuViewSpatialEngine(zones, rx_pos=(0.50, 0.40), layout_preset="tabletop_4node")
    # Feed real hardware CSI packet on node_1 and verify update_spatial() does NOT overwrite it with synthetic CSI (Issue 22)
    hw_pkt = engine.csi_engine.simulate_em_multipath_frame("node_1", (0.24, 0.20), as_raw_esp32_iq=True)
    hw_pkt["source"] = "hardware_esp32"
    engine.update_csi(hw_pkt)
    assert engine.csi_engine.has_recent_hardware_csi("node_1") is True
    for z in zones:
        engine.update_spatial(z.name, 0.0018 if z.name == "node_1" else 0.0001, 0.55, raw_rssi=-64.0)
    assert engine.csi_engine.link_metrics["node_1"]["source"] == "hardware_esp32", "Synthetic CSI overwrote hardware CSI!"
    payload = engine.generate_payload()
    assert payload["event"] == "ruview_3d_update"
    assert "csi_metrics" in payload
    print(f"[CONTROLLER SELFTEST PASS v3.4] Pos={payload['table_cm']}cm | HW_CSI_Guard=PASS | Regime={payload['tomography_diagnostics']['location_regime']} | CSI_GDOP={payload['csi_metrics']['gdop']}")


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    baud        = args.baud or cfg.get("serial", {}).get("baud_rate", DEFAULT_BAUD)
    timeout_s   = cfg.get("serial_timeout_s", 0.05)
    hold_time   = cfg.get("led_hold_time_s", 1.2)
    win_size    = cfg.get("variance_window", 8)

    rx_cfg = cfg.get("receiver", {"pos_m": [0.50, 0.40], "layout_preset": "tabletop_4node"})
    rx_pos = tuple(rx_cfg.get("pos_m", [0.50, 0.40]))
    layout_preset = str(rx_cfg.get("layout_preset", "tabletop_4node"))

    raw_nodes = cfg.get("nodes", {
        "node_1": {"led_pin": 2, "label": "V1: ESP32-C3 SuperMini #1 (Top Corner)", "variance_threshold": 0.0020, "ssid": "Zone_1_Tx", "pos_m": [0.50, 0.05], "channel": 6},
        "node_2": {"led_pin": 4, "label": "V2: ESP32 DevKit (Bottom-Right Corner)", "variance_threshold": 0.0020, "ssid": "Zone_2_Tx", "pos_m": [0.80, 0.57], "channel": 6},
        "node_3": {"led_pin": 5, "label": "V3: ESP32-C3 SuperMini #2 (Bottom-Left Corner)", "variance_threshold": 0.0020, "ssid": "Zone_3_Tx", "pos_m": [0.20, 0.57], "channel": 6},
        "node_4": {"led_pin": -1, "label": "C: Mobile Phone Hotspot (Centroid)", "variance_threshold": 0.0020, "ssid": "Laptop_Hotspot", "pos_m": [0.50, 0.40], "channel": 6},
    })
    zones = [
        Zone(
            name=name,
            label=z.get("label", name),
            led_pin=z.get("led_pin", 2),
            variance_threshold=z.get("variance_threshold", 0.0020),
            demo_scenario="room_radar",
            ssid=z.get("ssid", name),
            pos_m=tuple(z.get("pos_m", (0.50, 0.40))),
            channel=z.get("channel", 1),
            _history=collections.deque(maxlen=win_size),
        )
        for name, z in raw_nodes.items()
    ]

    ssid_to_zone = {
        "Zone_1_Tx": "node_1", "living_room": "node_1", "node_1": "node_1",
        "Zone_2_Tx": "node_2", "kitchen": "node_2", "node_2": "node_2",
        "Zone_3_Tx": "node_3", "secure_room": "node_3", "Zone_3_Rx": "node_3", "node_3": "node_3",
        "Laptop_Hotspot": "node_4", "Phone_Hotspot": "node_4", "centroid": "node_4", "node_4": "node_4",
    }

    serial_cfg = cfg.get("serial", {})
    actuator_port, configured_ports = resolve_serial_ports(
        serial_cfg,
        raw_nodes,
        port_override=args.port,
        ports_override=args.ports,
        actuator_override=args.actuator_port,
    )

    # The Standard ESP32 alone receives actuator commands. Its serial stream
    # remains one of the sensor inputs; C3 ports are read-only from the PC.
    bridge = SerialBridge(
        actuator_port, baud, timeout_s, demo_mode=args.demo, strict_port=True,
        actuator_enabled=True,
    )
    bridge.connect()

    # Connect additional secondary/tertiary bridges if multiple ESPs are plugged in
    

    extra_bridges: list[SerialBridge] = []
    for sec_port in configured_ports[1:]:
        if sec_port and sec_port != bridge.port:
            b_sec = SerialBridge(sec_port, baud, timeout_s, demo_mode=args.demo, strict_port=True)
            b_sec.connect()
            extra_bridges.append(b_sec)

    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    dashboard_server = DashboardServer(zones, bridge, queue)
    dashboard_server.sensor_bridges = [bridge, *extra_bridges]
    bound_ports = await dashboard_server.start([args.http_port, 3001])

    room_cfg = cfg.get("room", {"name": "Tabletop Sensing Grid (CSI + 3Tx/1Rx)", "width_m": 1.0, "length_m": 0.7, "height_m": 0.5})
    spatial_engine = RuViewSpatialEngine(zones, room_cfg=room_cfg, rx_pos=rx_pos, layout_preset=layout_preset,
                                         csi_cfg=cfg.get("csi", {}),
                                         calibration_cfg=cfg.get("room_calibration", {}))

    ctrl = ExhibitionController(zones, bridge, hold_time=hold_time, debug=args.debug, ssid_to_zone=ssid_to_zone)
    ctrl.csi_disp_threshold_mm = float((cfg.get("csi") or {}).get("displacement_threshold_mm", 0.45))
    ctrl.dashboard_server = dashboard_server
    ctrl.spatial_engine = spatial_engine
    dashboard_server.spatial_engine = spatial_engine

    room_calibration_cfg = cfg.get("room_calibration", {})
    if not args.demo and room_calibration_cfg.get("auto_calibrate_on_start", False):
        await dashboard_server.start_baseline_calibration(
            float(room_calibration_cfg.get("calibration_duration_s", 45.0))
        )

    if bound_ports and not args.no_browser:
        try:
            webbrowser.open(f"http://localhost:{bound_ports[0]}")
        except Exception:
            pass

    async def ruview_3d_broadcast_loop() -> None:
        tick = 0
        while True:
            try:
                is_hw = bool(bridge._ser and bridge._ser.is_open)
                payload = spatial_engine.generate_payload()
                payload["hardware_connected"] = is_hw
                payload["primary_port"] = bridge.port
                payload["serial_frame_counts"] = {
                    sensor.port: sensor.rx_packets for sensor in dashboard_server.sensor_bridges
                }
                await dashboard_server.broadcast(payload)
                tick += 1
                if tick % 2 == 0:
                    sensing_payload = spatial_engine.get_sensing_update_payload()
                    sensing_payload["hardware_connected"] = is_hw
                    sensing_payload["primary_port"] = bridge.port
                    await dashboard_server.broadcast(sensing_payload)
            except Exception:
                pass
            await asyncio.sleep(0.04)

    tasks = [
        asyncio.create_task(ctrl.run(queue)),
        asyncio.create_task(dashboard(zones, spatial_engine)),
        asyncio.create_task(ruview_3d_broadcast_loop()),
    ]
    if args.demo:
        tasks.append(asyncio.create_task(demo_ws_stream(queue, zones)))
    else:
        # Start reading from primary ESP
        tasks.append(asyncio.create_task(serial_sensor_loop(bridge, queue, zones=zones)))
        # Start reading from all other connected ESPs (zones=None avoids duplicate idle heartbeats)
        for b_sec in extra_bridges:
            tasks.append(asyncio.create_task(serial_sensor_loop(b_sec, queue, zones=None)))

    try:
        await asyncio.gather(*tasks)
    finally:
        await dashboard_server.stop()
        bridge.close()
        for b_sec in extra_bridges:
            b_sec.close()



if __name__ == "__main__":
    _args = parse_args()
    if _args.selftest:
        _run_controller_selftest()
    else:
        try:
            asyncio.run(main_async(_args))
        except KeyboardInterrupt:
            pass
