"""
RuView v3.6/v3.7 "Until It Breaks" Exhaustive Stress Test Suite
Simulates ESP32 firmware .ino serial responses across:
  1. Dense 2D Spatial Grid (25 target coordinates)
  2. Multi-Target Superposition (Two simultaneous scatterers)
  3. Extreme Phase Noise & Clutter Sweep (sigma = 0.05 to 2.5 rad)
  4. Link Dropout & 2-Anchor Asymmetry (Packet loss up to 80%)
  5. Microsecond Counter Rollover & Timestamp Inversion
  6. Malformed Frame / Corrupted Stream Fuzzing (Uncaught Exception Audit)
"""

import sys
import os
import math
import time
import json
import random
import traceback
import numpy as np

# Ensure root directory is on import path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from csi_processor import (
    CSIEngine,
    MultiStaticTDOASolver,
    C_LIGHT_MPS,
    CARRIER_WAVELENGTH_M,
    CARRIER_FREQ_HZ,
    FIXED_CHANNEL,
    SUBCARRIER_SPACING_HZ,
    HT20_SUBCARRIER_INDICES,
    GDOP_UNDETERMINED,
    DeviceTimebase,
)
from tracking.fusion import AdaptiveUKF
from tracking.phase import sanitize as huber_sanitize

RIG_NODES = {
    "node_1": (0.50, 0.05),
    "node_2": (0.80, 0.57),
    "node_3": (0.20, 0.57),
    "node_4": (0.50, 0.40),
}
RX_POS = RIG_NODES["node_3"]


def generate_firmware_csi_packet(
    tx_id: str,
    target_pos: tuple[float, float],
    rx_pos: tuple[float, float] = RX_POS,
    tx_nodes: dict = RIG_NODES,
    phase_noise_rad: float = 0.0,
    amp_noise: float = 0.0,
    ts_us: int = 1000000,
    is_synth: bool = False,
    second_target_pos: tuple[float, float] = None,
    motion_amp_m: float = 0.005,
) -> dict:
    """
    Generates a synthetic CSI frame matching the exact JSON emitted by
    esp32_receiver_actuator.ino / esp32_c3_transceiver.ino when a physical
    target is located at target_pos.
    """
    tx_pos = tx_nodes[tx_id]
    d_tx_tgt = math.hypot(target_pos[0] - tx_pos[0], target_pos[1] - tx_pos[1])
    
    # Path loss model to target (calibrated against rssi0 = -44.0, n = 2.15)
    rssi0_dbm = -44.0
    path_loss_n = 2.15
    rssi_dbm = rssi0_dbm - 10.0 * path_loss_n * math.log10(max(0.02, d_tx_tgt) / 0.30)
    
    # Time of flight corresponding to target range
    tau_target = d_tx_tgt / C_LIGHT_MPS
    slope_target = -2.0 * math.pi * SUBCARRIER_SPACING_HZ * tau_target

    # Temporal Doppler / displacement from target tremor or motion (2.4 Hz)
    t_sec = float(ts_us) * 1e-6
    doppler_disp_m = motion_amp_m * math.sin(2.0 * math.pi * 2.4 * t_sec)
    phi_doppler = (4.0 * math.pi / CARRIER_WAVELENGTH_M) * doppler_disp_m

    # Base amplitude
    base_mag = max(2.0, 42.0 + (rssi_dbm + 50.0) * 0.70)
    refl_coeff = 0.28

    csi_mag = []
    csi_phase = []

    # Optional 2nd target
    if second_target_pos is not None:
        d_tx2 = math.hypot(second_target_pos[0] - tx_pos[0], second_target_pos[1] - tx_pos[1])
        tau_tgt2 = d_tx2 / C_LIGHT_MPS
        slope_tgt2 = -2.0 * math.pi * SUBCARRIER_SPACING_HZ * tau_tgt2
        refl2 = 0.20
    else:
        slope_tgt2 = 0.0
        refl2 = 0.0

    for idx, sc_k in enumerate(HT20_SUBCARRIER_INDICES):
        # Target direct path
        phi_d = slope_target * sc_k + phi_doppler
        # Multipath reflection
        phi_s = (slope_target * 1.35) * sc_k + 0.12 * math.sin(0.18 * sc_k)
        
        # Superposition in complex IQ domain
        I_val = base_mag * math.cos(phi_d) + base_mag * refl_coeff * math.cos(phi_s)
        Q_val = base_mag * math.sin(phi_d) + base_mag * refl_coeff * math.sin(phi_s)

        if second_target_pos is not None:
            phi_s2 = slope_tgt2 * sc_k + phi_doppler
            I_val += base_mag * refl2 * math.cos(phi_s2)
            Q_val += base_mag * refl2 * math.sin(phi_s2)

        # Inject noise if requested
        if phase_noise_rad > 0.0:
            noise_phi = random.gauss(0.0, phase_noise_rad)
            I_rot = I_val * math.cos(noise_phi) - Q_val * math.sin(noise_phi)
            Q_rot = I_val * math.sin(noise_phi) + Q_val * math.cos(noise_phi)
            I_val, Q_val = I_rot, Q_rot

        if amp_noise > 0.0:
            I_val += random.gauss(0.0, amp_noise)
            Q_val += random.gauss(0.0, amp_noise)

        mag = math.hypot(I_val, Q_val)
        phase = math.atan2(Q_val, I_val)
        csi_mag.append(round(mag, 2))
        csi_phase.append(round(phase, 3))

    packet = {
        "event": "csi",
        "node": "node_3",
        "rx_id": "node_3",
        "ssid": f"Zone_{tx_id[-1]}_Tx",
        "zone": tx_id,
        "rssi": round(rssi_dbm, 1),
        "ch": FIXED_CHANNEL,
        "num_sc": 56,
        "csi_fmt": "espressif_htltf_v1",
        "sc_order": "ieee_80211n_56",
        "ts": int(ts_us // 1000),
        "ts_us": int(ts_us),
        "csi_mag_count": len(csi_mag),
        "csi_phase_count": len(csi_phase),
        "synthesized": is_synth,
        "csi_mag": csi_mag,
        "csi_phase": csi_phase,
    }
    return packet


def run_stress_test():
    print("=" * 80)
    print(" RUVIEW v3.6/v3.7 'UNTIL IT BREAKS' EXHAUSTIVE STRESS TEST SUITE ")
    print("=" * 80)

    total_subtests = 0
    passed_subtests = 0
    failures = []

    # Initialize Engine with 60cm rig
    engine = CSIEngine(
        node_positions=RIG_NODES,
        room_size=(1.0, 0.7),
        rx_position=RX_POS,
        gdop_threshold=3.5,
    )

    # =========================================================================
    # TEST 1: Dense 2D Spatial Grid (25 Positions Across 60cm Equilateral Rig)
    # =========================================================================
    print("\n[TEST 1] Dense 2D Spatial Grid Search (25 Positions)")
    print("-" * 80)
    
    # Generate 25 points inside and near the 60cm triangle
    test_points = [
        # Centroid & Core
        ("Centroid (0.50, 0.40)", (0.50, 0.40)),
        ("Center Core (0.50, 0.35)", (0.50, 0.35)),
        ("Center Core (0.50, 0.45)", (0.50, 0.45)),
        ("Center Left (0.40, 0.40)", (0.40, 0.40)),
        ("Center Right (0.60, 0.40)", (0.60, 0.40)),
        
        # Near Apex Node 1 (0.50, 0.05)
        ("Near Node 1 (0.50, 0.12)", (0.50, 0.12)),
        ("Near Node 1 Left (0.45, 0.15)", (0.45, 0.15)),
        ("Near Node 1 Right (0.55, 0.15)", (0.55, 0.15)),
        
        # Near Node 2 Bottom-Right (0.80, 0.57)
        ("Near Node 2 (0.75, 0.52)", (0.75, 0.52)),
        ("Near Node 2 High (0.70, 0.45)", (0.70, 0.45)),
        ("Near Node 2 Edge (0.78, 0.55)", (0.78, 0.55)),
        
        # Near Node 3 Bottom-Left (0.20, 0.57)
        ("Near Node 3 (0.25, 0.52)", (0.25, 0.52)),
        ("Near Node 3 High (0.30, 0.45)", (0.30, 0.45)),
        ("Near Node 3 Edge (0.22, 0.55)", (0.22, 0.55)),
        
        # Mid-Edges
        ("Mid Edge 1-2 (0.65, 0.31)", (0.65, 0.31)),
        ("Mid Edge 1-3 (0.35, 0.31)", (0.35, 0.31)),
        ("Mid Edge 2-3 (0.50, 0.57)", (0.50, 0.57)),
        ("Bottom Baseline Mid (0.50, 0.52)", (0.50, 0.52)),
        
        # Sectors
        ("Sector 1 (0.45, 0.25)", (0.45, 0.25)),
        ("Sector 2 (0.55, 0.25)", (0.55, 0.25)),
        ("Sector 3 (0.65, 0.48)", (0.65, 0.48)),
        ("Sector 4 (0.35, 0.48)", (0.35, 0.48)),
        ("Quarter Left (0.42, 0.38)", (0.42, 0.38)),
        ("Quarter Right (0.58, 0.38)", (0.58, 0.38)),
        ("Interior Center-Low (0.50, 0.48)", (0.50, 0.48)),
    ]

    grid_errors = []
    ts = 1000000
    for label, target_xy in test_points:
        total_subtests += 1
        # Feed all 3 active transmitters
        for tx in ("node_1", "node_2", "node_3"):
            ts += 25000  # 40 Hz
            pkt = generate_firmware_csi_packet(tx, target_xy, ts_us=ts)
            res = engine.process_csi_frame(pkt, current_pos_guess=None)
        
        est_x, est_z = res["x"], res["z"]
        err_cm = math.hypot(est_x - target_xy[0], est_z - target_xy[1]) * 100.0
        grid_errors.append(err_cm)
        
        status = "PASS" if err_cm < 5.0 else "WARN" if err_cm < 10.0 else "FAIL"
        if status == "FAIL":
            failures.append((label, f"Error {err_cm:.2f} cm exceeded 10.0 cm threshold"))
        else:
            passed_subtests += 1
            
        print(f"  [{status}] {label:<32} | True=({target_xy[0]*100:4.1f},{target_xy[1]*100:4.1f})cm "
              f"| Est=({est_x*100:4.1f},{est_z*100:4.1f})cm | Err={err_cm:4.2f}cm | GDOP={res['gdop']:.2f}")

    mean_grid_err = float(np.mean(grid_errors))
    p95_grid_err = float(np.percentile(grid_errors, 95))
    max_grid_err = float(np.max(grid_errors))
    print(f"\n>> Grid Summary: Mean Error = {mean_grid_err:.2f} cm | P95 = {p95_grid_err:.2f} cm | Max = {max_grid_err:.2f} cm")
    assert mean_grid_err < 5.0, f"Dense grid mean error {mean_grid_err:.2f}cm exceeds 5.0cm requirement"


    # =========================================================================
    # TEST 2: Multi-Target Superposition (Two Simultaneous Scatterers)
    # =========================================================================
    print("\n[TEST 2] Multi-Target Superposition Stress (Dual Objects)")
    print("-" * 80)
    
    dual_pairs = [
        ("Symmetric Left/Right", (0.35, 0.45), (0.65, 0.45)),
        ("Top-Apex & Bottom-Base", (0.50, 0.20), (0.50, 0.50)),
        ("Close Proximity (10cm)", (0.45, 0.40), (0.55, 0.40)),
    ]

    for label, obj1, obj2 in dual_pairs:
        total_subtests += 1
        expected_mid = ((obj1[0] + obj2[0]) / 2.0, (obj1[1] + obj2[1]) / 2.0)
        for tx in ("node_1", "node_2", "node_3"):
            ts += 25000
            pkt = generate_firmware_csi_packet(tx, obj1, second_target_pos=obj2, ts_us=ts)
            res = engine.process_csi_frame(pkt)
        
        est_x, est_z = res["x"], res["z"]
        # In RF superposition, solver should resolve to center-of-mass or dominant reflector without NaN
        err_mid_cm = math.hypot(est_x - expected_mid[0], est_z - expected_mid[1]) * 100.0
        no_nan = math.isfinite(est_x) and math.isfinite(est_z) and not res["residual_rejected"]
        
        if no_nan and err_mid_cm < 18.0:
            passed_subtests += 1
            print(f"  [PASS] {label:<28} | Center=({expected_mid[0]*100:.1f},{expected_mid[1]*100:.1f})cm "
                  f"| Solved=({est_x*100:.1f},{est_z*100:.1f})cm | Centroid Err={err_mid_cm:.2f}cm")
        else:
            failures.append((label, f"Dual object solver failed or deviated {err_mid_cm:.2f}cm"))
            print(f"  [FAIL] {label:<28} | Solved=({est_x*100:.1f},{est_z*100:.1f})cm")


    # =========================================================================
    # TEST 3: Extreme Phase Noise & Clutter Sweep (Find the Breaking Point)
    # =========================================================================
    print("\n[TEST 3] Extreme Phase Noise & Clutter Stress Sweep")
    print("-" * 80)
    
    target_fixed = (0.50, 0.40)
    noise_levels = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]
    
    breaking_point = None
    for sigma in noise_levels:
        total_subtests += 1
        errs = []
        for trial in range(5):
            for tx in ("node_1", "node_2", "node_3"):
                ts += 25000
                pkt = generate_firmware_csi_packet(tx, target_fixed, phase_noise_rad=sigma, amp_noise=sigma*5.0, ts_us=ts)
                res = engine.process_csi_frame(pkt)
            err = math.hypot(res["x"] - target_fixed[0], res["z"] - target_fixed[1]) * 100.0
            errs.append(err)
        
        avg_err = float(np.mean(errs))
        p95_err = float(np.percentile(errs, 95))
        
        if avg_err < 15.0 and not res.get("residual_rejected", False):
            passed_subtests += 1
            print(f"  [PASS] Phase Noise sigma={sigma:4.2f} rad ({math.degrees(sigma):5.1f} deg) "
                  f"| Avg Err: {avg_err:5.2f} cm | P95: {p95_err:5.2f} cm | Status: STABLE")
        else:
            if breaking_point is None:
                breaking_point = sigma
            if sigma <= 0.50:
                failures.append((f"Noise sigma={sigma}", f"Operational noise failed: avg err {avg_err:.2f} cm"))
                print(f"  [FAIL] Phase Noise sigma={sigma:4.2f} rad ({math.degrees(sigma):5.1f} deg) "
                      f"| Avg Err: {avg_err:5.2f} cm | Status: GATE TRIGGERED / DRIFT (Unacceptable below 0.50 rad)")
            else:
                passed_subtests += 1
                print(f"  [PASS/FRONTIER] Phase Noise sigma={sigma:4.2f} rad ({math.degrees(sigma):5.1f} deg) "
                      f"| Avg Err: {avg_err:5.2f} cm | Status: GATE TRIGGERED / DRIFT (Expected High Stress)")

    print(f">> Noise Breaking Threshold: System maintains sub-15cm tracking up to sigma = {breaking_point or '>2.0'} rad.")


    # =========================================================================
    # TEST 4: Link Dropout & Asymmetric 2-Anchor Loss
    # =========================================================================
    print("\n[TEST 4] Link Dropout & 2-Anchor Degradation Defense")
    print("-" * 80)
    
    total_subtests += 1
    # 1. Normal state
    for tx in ("node_1", "node_2", "node_3"):
        ts += 25000
        pkt = generate_firmware_csi_packet(tx, (0.50, 0.40), ts_us=ts)
        res = engine.process_csi_frame(pkt)
    
    last_known_good = (res["x"], res["z"])
    
    # 2. Complete drop of Node 1 (simulating cable unplug / RF dead zone)
    # Age out Node 1 beyond range_max_age_s (2.5s)
    engine.node_range_ts["node_1"] = time.time() - 5.0
    
    # Send only Node 2 and Node 3
    for _ in range(5):
        ts += 25000
        pkt2 = generate_firmware_csi_packet("node_2", (0.50, 0.40), ts_us=ts)
        res2 = engine.process_csi_frame(pkt2)
        pkt3 = generate_firmware_csi_packet("node_3", (0.50, 0.40), ts_us=ts)
        res3 = engine.process_csi_frame(pkt3)
        
    # The 2-anchor setup must be rejected by GDOP (undetermined) and hold last known fix!
    held_x, held_z = res3["x"], res3["z"]
    drift_cm = math.hypot(held_x - last_known_good[0], held_z - last_known_good[1]) * 100.0
    
    if res3["gdop_rejected"] or res3["fix_held"] or res3["anchors_used"] < 3:
        passed_subtests += 1
        print(f"  [PASS] 2-Anchor Fallback: GDOP Rejected={res3['gdop_rejected']}, "
              f"Fix Held={res3['fix_held']}, Anchors Used={res3['anchors_used']} (Drift={drift_cm:.2f}cm)")
    else:
        failures.append(("Link Dropout", f"Failed to reject 2-anchor ambiguous setup: anchors={res3['anchors_used']}"))
        print(f"  [FAIL] 2-Anchor setup was erroneously accepted without 3 anchors!")


    # =========================================================================
    # TEST 5: Microsecond Counter Rollover & Out-of-Order Inversion
    # =========================================================================
    print("\n[TEST 5] Microsecond Counter Rollover & Monotonic Time Protection")
    print("-" * 80)
    
    total_subtests += 1
    # 1. 32-bit counter wrap (2^32 - 1000 us -> +1500 us)
    timebase = DeviceTimebase()
    t_pre = 2**32 - 1000
    t_post = 1500
    us_pre = timebase.to_us(t_pre, host_s=100.0)
    us_post = timebase.to_us(t_post, host_s=100.0025)
    
    wrap_delta = us_post - us_pre
    assert wrap_delta == 2500, f"Rollover delta expected 2500 us, got {wrap_delta}"
    print(f"  [PASS] DeviceTimebase 32-bit Rollover: {t_pre} us -> {t_post} us yields correct dt = {wrap_delta} us")
    passed_subtests += 1

    total_subtests += 1
    # 2. Out-of-order frame arrival protection
    initial_ooo = int(sum(engine.link_out_of_order.values()))
    normal_pkt = generate_firmware_csi_packet("node_1", (0.50, 0.40), ts_us=ts + 1000000)
    engine.process_csi_frame(normal_pkt)
    stale_pkt = generate_firmware_csi_packet("node_1", (0.50, 0.40), ts_us=ts + 500000)
    res_stale = engine.process_csi_frame(stale_pkt)
    assert res_stale["out_of_order_count"] == initial_ooo + 1, (
        f"Expected out_of_order_count to increment to {initial_ooo + 1}, got {res_stale['out_of_order_count']}"
    )
    print(f"  [PASS] Out-of-Order Frame Guard: Stale frame rejected (ooo count: {initial_ooo} -> {res_stale['out_of_order_count']}).")
    passed_subtests += 1


    # =========================================================================
    # TEST 6: Malformed Frame / Corrupted Stream Fuzzing (Crash Resistance)
    # =========================================================================
    print("\n[TEST 6] Corrupted & Malformed Frame Fuzzing (Zero-Crash Audit)")
    print("-" * 80)
    
    malformed_payloads = [
        ("Empty Mag & Phase", {"event": "csi", "zone": "node_1", "csi_mag": [], "csi_phase": []}),
        ("All Zero Mag", {"event": "csi", "zone": "node_1", "csi_mag": [0.0]*56, "csi_phase": [0.1]*56}),
        ("NaN in Phase", {"event": "csi", "zone": "node_1", "csi_mag": [10.0]*56, "csi_phase": [float('nan')]*56}),
        ("Infinity in Mag", {"event": "csi", "zone": "node_1", "csi_mag": [float('inf')]*56, "csi_phase": [0.0]*56}),
        ("Non-Numeric Ch", {"event": "csi", "zone": "node_1", "ch": "invalid", "csi_mag": [10.0]*56, "csi_phase": [0.0]*56}),
        ("Mismatch Subcarrier Count", {"event": "csi", "zone": "node_1", "csi_mag": [10.0]*12, "csi_phase": [0.0]*12}),
        ("Mismatched Mag/Phase Length", {"event": "csi", "zone": "node_1", "csi_mag": [10.0]*56, "csi_phase": [0.0]*10}),
        ("Declared Subcarrier Count Mismatch", {"event": "csi", "zone": "node_1", "num_sc": 52, "csi_mag": [10.0]*56, "csi_phase": [0.0]*56}),
        ("Declared I/Q Count Mismatch", {"event": "csi", "zone": "node_1", "num_sc": 52, "csi_iq": [1.0]*112}),
        ("Invalid Subcarrier Count", {"event": "csi", "zone": "node_1", "num_sc": "invalid", "csi_mag": [10.0]*56, "csi_phase": [0.0]*56}),
        ("Odd I/Q Payload", {"event": "csi", "zone": "node_1", "csi_iq": [1.0]*17}),
        ("Negative Microsecond Stamp", {"event": "csi", "zone": "node_1", "ts_us": -99999, "csi_mag": [10.0]*56, "csi_phase": [0.0]*56}),
        ("Unknown Zone ID", {"event": "csi", "zone": "ghost_node", "csi_mag": [10.0]*56, "csi_phase": [0.0]*56}),
    ]

    fuzz_passed = 0
    for label, bad_pkt in malformed_payloads:
        total_subtests += 1
        try:
            invalid_before = engine.invalid_frame_count
            res_bad = engine.process_csi_frame(bad_pkt)
            assert res_bad is not None
            if label in {
                "Declared Subcarrier Count Mismatch",
                "Declared I/Q Count Mismatch",
                "Invalid Subcarrier Count",
                "Odd I/Q Payload",
            }:
                assert engine.invalid_frame_count == invalid_before + 1, (
                    f"{label} was accepted as a valid CSI frame"
                )
            fuzz_passed += 1
            passed_subtests += 1
            print(f"  [PASS] Fuzz: {label:<32} -> Rejected/Handled cleanly (Invalid count: {res_bad.get('invalid_frame_count', 0)})")
        except Exception as exc:
            failures.append((label, f"Unhandled exception: {traceback.format_exc()}"))
            print(f"  [FAIL] Fuzz: {label:<32} -> CRASHED: {exc}")

    print(f"\n>> Fuzzing Audit: {fuzz_passed}/{len(malformed_payloads)} corrupted payloads safely swallowed.")


    # =========================================================================
    # FINAL RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" STRESS TEST SUITE EXECUTION SUMMARY ")
    print("=" * 80)
    print(f"Total Test Cases Run: {total_subtests}")
    print(f"Passed:               {passed_subtests}")
    print(f"Failures:             {len(failures)}")
    print(f"Pass Rate:            {passed_subtests / total_subtests * 100:.1f}%")
    
    if failures:
        print("\nFAILURES DETECTED:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        return False
    else:
        print("\n>>> ALL EXHAUSTIVE STRESS TESTS PASSED WITH 0 UNHANDLED CRASHES! <<<")
        return True


if __name__ == "__main__":
    success = run_stress_test()
    sys.exit(0 if success else 1)
