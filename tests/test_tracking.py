"""Sprint 6 DSP / solver unit tests.

Run: python tests/test_tracking.py
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from csi_processor import (
    CARRIER_WAVELENGTH_M,
    CARRIER_FREQ_HZ,
    C_LIGHT_MPS,
    FIXED_CHANNEL,
    CSIEngine,
    CSIPhaseSanitizer,
    DeviceTimebase,
    MultiStaticTDOASolver,
    channel_to_wavelength_m,
)
from tracking.fusion import AdaptiveUKF
from tracking.phase import sanitize, slope_to_range, differential_range

K = np.array(list(range(-28, 0)) + list(range(1, 29)), dtype=float)
DF = 312_500.0
C = 299_792_458.0


def test_phase_slope():
    dist = 1.2
    slope = -2.0 * math.pi * DF * (dist / C)
    phase = slope * K + 0.7
    h = np.exp(1j * phase)
    out = sanitize(h, K)
    assert out["valid"], out
    assert abs(slope_to_range(out["slope"]) - dist) < 0.01, out
    assert abs(differential_range(slope, slope * 1.001)) > 0.0


def test_phase_sanitizer_unwraps_before_resampling():
    n_raw = 64
    amp = np.full(n_raw, 20.0)
    phase = np.angle(np.exp(1j * (0.3 + 0.1 * np.arange(n_raw))))

    clean_amp, clean_phase, *_ = CSIPhaseSanitizer().sanitize(amp, phase)

    assert clean_amp.size == 56
    assert clean_phase.size == 56
    assert np.sqrt(np.mean(clean_phase**2)) < 0.04


def test_phase_dead_signal():
    assert sanitize(np.zeros(56, complex), K)["valid"] is False
    assert sanitize(np.full(56, np.nan, complex), K)["valid"] is False
    assert sanitize(np.ones(3, complex), np.arange(3))["valid"] is False


def test_ukf_predict_holds_centroid():
    f = AdaptiveUKF()
    x0 = f.x.copy()
    for _ in range(50):
        f.predict(0.04)
    assert np.allclose(f.x, x0, atol=1e-9), f.x
    assert np.min(np.linalg.eigvalsh(f.P)) > 0.0


def test_ukf_range_update_and_psd():
    anchors = np.array([[0.50, 0.05], [0.80, 0.57], [0.20, 0.57]])
    true = np.array([0.62, 0.40])
    f = AdaptiveUKF()
    for _ in range(20):
        z = np.linalg.norm(anchors - true, axis=1)
        assert f.update_ranges(anchors, z)
        f.predict(0.04)
        assert np.min(np.linalg.eigvalsh(f.P)) > 0.0
    assert math.hypot(f.x[0] - true[0], f.x[1] - true[1]) < 0.02, f.x


def test_ukf_rejects_bad_input():
    f = AdaptiveUKF()
    x_before = f.x.copy()
    assert f.update_ranges([[0.5, 0.1], [0.8, 0.5]], [np.nan, 0.2]) is False
    assert f.update_ranges([[0.5, 0.1], [np.inf, 0.5]], [0.2, 0.3]) is False
    assert f.update_ranges(np.zeros((2, 3)), [0.2, 0.3]) is False
    assert f.update_ranges([[0.5, 0.1]], []) is False
    assert f.update_ranges([[0.5, 0.1], [0.8, 0.5]], [0.2, 0.3], R_diag=[0.01]) is False
    assert np.allclose(f.x, x_before)
    f.predict(float("nan"))
    assert np.all(np.isfinite(f.P))


def test_ukf_outlier_gate():
    f = AdaptiveUKF()
    anchors = np.array([[0.50, 0.05], [0.80, 0.57], [0.20, 0.57]])
    z = np.array([0.40, 0.40, 0.40])
    assert f.update_ranges(anchors, z, R_diag=[0.01, 0.01, 0.01], gate_chi2=0.001) is False


def test_lm_warm_matches_grid():
    solver = MultiStaticTDOASolver((1.0, 0.7))
    anchors = {"node_1": (0.50, 0.05), "node_2": (0.80, 0.57), "node_3": (0.20, 0.57)}
    true = (0.58, 0.42)
    ranges = {
        k: math.hypot(true[0] - p[0], true[1] - p[1]) for k, p in anchors.items()
    }
    weights = {k: 1.0 for k in anchors}
    cold = solver.solve(anchors, ranges, weights)
    warm = solver.solve(anchors, ranges, weights, initial_guess=(0.55, 0.45))
    for out in (cold, warm):
        assert out["converged"], out
        assert math.hypot(out["x"] - true[0], out["z"] - true[1]) < 0.01, out


def test_static_object_unit_floor():
    nodes = {
        "node_1": (0.50, 0.05),
        "node_2": (0.80, 0.57),
        "node_3": (0.20, 0.57),
        "node_4": (0.50, 0.40),
    }
    eng = CSIEngine(nodes, room_size=(1.0, 0.7), static_object_corner="node_2")
    free = [
        {"rx_id": "node_3", "tx_id": "node_1", "baseline_m": 0.52, "excess_path_m": 0.26, "rssi_dbm": -60},
        {"rx_id": "node_3", "tx_id": "node_2", "baseline_m": 0.60, "excess_path_m": 0.30, "rssi_dbm": -63},
    ]
    assert eng._static_object_estimate(free) is None
    bottle = [
        {"rx_id": "node_3", "tx_id": "node_2", "baseline_m": 0.60, "excess_path_m": 0.75, "rssi_dbm": -72},
        {"rx_id": "node_2", "tx_id": "node_3", "baseline_m": 0.60, "excess_path_m": 0.73, "rssi_dbm": -74},
    ]
    est = eng._static_object_estimate(bottle)
    assert est is not None and abs(est[0][0] - 0.65) < 0.01 and abs(est[0][1] - 0.485) < 0.01


def test_out_of_order_clock():
    tb = DeviceTimebase()
    a = tb.to_us(5_000_000, host_s=100.0)
    b = tb.to_us(5_040_000, host_s=100.04)
    c = tb.to_us(5_020_000, host_s=100.05)
    d = tb.to_us(5_080_000, host_s=100.08)
    assert (a, b, c, d) == (5_000_000, 5_040_000, 5_040_000, 5_080_000)


def test_dead_frame_does_not_create_link():
    nodes = {
        "node_1": (0.50, 0.05),
        "node_2": (0.80, 0.57),
        "node_3": (0.20, 0.57),
        "node_4": (0.50, 0.40),
    }
    eng = CSIEngine(nodes, room_size=(1.0, 0.7))
    eng.process_csi_frame(
        {
            "zone": "node_1",
            "rx_id": "node_3",
            "channel": 6,
            "rssi": -58,
            "source": "hardware_esp32",
            "csi_amp": [0.0] * 56,
            "csi_phase": [0.0] * 56,
        }
    )
    assert eng.invalid_frame_count == 1
    assert len(eng.directed_links) == 0
    assert eng.has_recent_hardware_csi("node_1") is False


def test_wavelength_follows_the_base_ap_channel():
    # FIXED_CHANNEL is only the fallback used before the first hardware frame;
    # the phone hotspot picks the real channel, so lambda is per-channel.
    assert FIXED_CHANNEL == 6
    assert abs(CARRIER_WAVELENGTH_M - C_LIGHT_MPS / CARRIER_FREQ_HZ) < 1e-6
    assert abs(channel_to_wavelength_m(6) - CARRIER_WAVELENGTH_M) < 1e-12
    assert abs(channel_to_wavelength_m(1) - C_LIGHT_MPS / 2_412_000_000.0) < 1e-12
    assert abs(channel_to_wavelength_m(11) - C_LIGHT_MPS / 2_462_000_000.0) < 1e-12
    assert channel_to_wavelength_m(1) > channel_to_wavelength_m(6) > channel_to_wavelength_m(11)


def test_engine_adopts_the_base_ap_channel():
    nodes = {
        "node_1": (0.50, 0.05),
        "node_2": (0.80, 0.57),
        "node_3": (0.20, 0.57),
        "node_4": (0.50, 0.40),
    }
    eng = CSIEngine(nodes, room_size=(1.0, 0.7))
    synth = eng.synthesize_csi_from_rssi(
        "node_1", -55.0, 0.001, target_dist_m=0.5, rx_id="node_3"
    )
    frame = dict(synth)
    frame.update({"channel": 1, "source": "hardware_esp32", "synthesized": False})
    res = eng.process_csi_frame(frame)
    # The first hardware frame of the session sets the carrier: no mismatch is
    # counted for it, and every link is evaluated at that channel.
    assert res["channel"] == 1
    assert eng.rig_channel == 1
    assert eng.channel_mismatch_count == 0
    # Reporting nodes carry the base AP channel; silent ones keep the fallback.
    assert eng.node_channels["node_1"] == 1
    assert all(rec["channel"] == 1 for rec in eng.directed_links.values())

    # A frame on another channel is a mismatch and does not move the carrier on
    # its own; the engine only follows after CHANNEL_RELOCK_FRAMES in a row.
    stray = dict(frame)
    stray["channel"] = 6
    for _ in range(3):
        eng.process_csi_frame(stray)
    assert eng.channel_mismatch_count == 3
    assert eng.rig_channel == 1


def test_reciprocal_segment_weighted_fusion():
    nodes = {
        "node_1": (0.50, 0.05),
        "node_2": (0.80, 0.57),
        "node_3": (0.20, 0.57),
        "node_4": (0.50, 0.40),
    }
    eng = CSIEngine(nodes, room_size=(1.0, 0.7))
    now = time.time()

    def rec(disp, quality, channel=6):
        return {
            "displacement_mm": disp,
            "excess_path_m": 0.30,
            "rssi_dbm": -60.0,
            "k_factor": 6.0,
            "phase_quality": quality,
            "los_weight": 0.8,
            "channel": channel,
            "ts": now,
        }

    eng.directed_links["node_2<-node_1"] = rec(4.0, 0.9)
    eng.directed_links["node_1<-node_2"] = rec(2.0, 0.1)
    edge_12 = eng.get_segment_metrics()["edge_12"]
    assert edge_12["reciprocal"] is True
    assert edge_12["directions_used"] == 2
    assert edge_12["channel"] == FIXED_CHANNEL
    assert 3.5 < edge_12["displacement_mm"] < 4.0

    eng.directed_links["node_1<-node_2"] = rec(2.0, 0.1, channel=5)
    edge_12 = eng.get_segment_metrics()["edge_12"]
    assert edge_12["reciprocal"] is False
    assert edge_12["directions_used"] == 1
    assert edge_12["displacement_mm"] == 4.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"{len(tests)} tracking/DSP tests passed")
