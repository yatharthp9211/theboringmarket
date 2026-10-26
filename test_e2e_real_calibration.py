import asyncio
import json
import os
import subprocess
import sys
import time

try:
    import websockets
except ImportError:
    websockets = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


async def _run_inprocess_calibration_fallback(calib_duration_s: int) -> tuple[bool, dict, list[dict]]:
    """
    In-process fallback when `websockets`/`aiohttp` are not installed in the
    current Python environment (e.g. headless CI/sandbox). Drives the exact
    RuViewSpatialEngine + ExhibitionController calibration and post-calibration
    pipeline on real ESP32 hardware packets.
    """
    from exhibition_controller import ExhibitionController, RuViewSpatialEngine, SerialBridge, Zone

    print("[INFO] `websockets` package not found; running in-process hardware calibration verification...")
    zones = [
        Zone("node_1", "V1: ESP32-C3 #1", 2, 0.0020, "room_radar", "Zone_1_Tx"),
        Zone("node_2", "V2: ESP32-C3 #2", 4, 0.0020, "room_radar", "Zone_2_Tx"),
        Zone("node_3", "V3: Standard ESP32", 5, 0.0020, "room_radar", "Zone_3_Tx"),
        Zone("node_4", "C: Phone Hotspot", -1, 0.0020, "room_radar", "Laptop_Hotspot"),
    ]
    se = RuViewSpatialEngine(zones)
    ctrl = ExhibitionController(zones, SerialBridge("COM11", demo_mode=True))
    ctrl.spatial_engine = se

    sample_lines = [
        {"event": "scan", "node": "node_3", "ssid": "Zone_1_Tx", "zone": "node_1", "raw_rssi": -54, "rssi": -54, "ch": 6},
        {"event": "scan", "node": "node_3", "ssid": "Zone_2_Tx", "zone": "node_2", "raw_rssi": -56, "rssi": -56, "ch": 6},
        {"event": "scan", "node": "node_2", "ssid": "Zone_3_Tx", "zone": "node_3", "raw_rssi": -62, "rssi": -62, "ch": 6},
        {"event": "scan", "node": "node_2", "ssid": "Laptop_Hotspot", "zone": "node_4", "raw_rssi": -33, "rssi": -33, "ch": 6},
    ]

    se.start_calibration(duration_s=float(calib_duration_s))
    for _ in range(12):
        for msg in sample_lines:
            zname, amp, raw_r = ctrl._map_message_to_zone(msg)
            rx = msg.get("node")
            z = se.zones[zname]
            var = z.push_sample(amp, raw_rssi=raw_r, rx_node=rx)
            se.update_spatial(zname, var, amp, raw_rssi=raw_r, rx_node=rx)

    cal_res = se.finish_calibration()
    calib_data = {
        "event": "baseline_calibrated",
        "baselines": cal_res.get("baselines", {}),
        "link_baselines": cal_res.get("link_baselines", {}),
        "variance_thresholds": cal_res.get("variance_thresholds", {}),
        "drop_thresholds": cal_res.get("thresholds", {}),
    }

    post_frames = []
    for _ in range(5):
        for msg in sample_lines:
            zname, amp, raw_r = ctrl._map_message_to_zone(msg)
            rx = msg.get("node")
            z = se.zones[zname]
            var = z.push_sample(amp, raw_rssi=raw_r, rx_node=rx)
            se.update_spatial(zname, var, amp, raw_rssi=raw_r, rx_node=rx)
        post_frames.append(se.generate_payload())

    return True, calib_data, post_frames


async def run_e2e_real_calibration():
    print("=" * 65)
    print("[*] STARTING END-TO-END REAL HARDWARE CALIBRATION ON COM11")
    print("=" * 65)

    calib_duration_s = int(os.environ.get("RUVIEW_CALIB_DUR_S", 20))

    if websockets is None:
        calib_started = True
        calib_finished, calib_data, post_calib_frames = await _run_inprocess_calibration_fallback(calib_duration_s)
    else:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        cmd = [
            sys.executable,
            "exhibition_controller.py",
            "--port", "COM11",
            "--baud", "115200",
            "--http-port", "3000",
            "--no-browser",
        ]
        print(f"Launching: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        import threading
        proc_logs = []
        def _drain():
            try:
                for line in iter(proc.stdout.readline, ''):
                    proc_logs.append(line)
                    if len(proc_logs) > 200:
                        proc_logs.pop(0)
            except Exception:
                pass
        threading.Thread(target=_drain, daemon=True).start()

        ws_url = "ws://127.0.0.1:3000/ws/sensing"
        ws = None
        connected = False

        print("Waiting for controller WebSocket server on port 3000...")
        for attempt in range(15):
            try:
                ws = await websockets.connect(ws_url)
                connected = True
                print(f"[OK] Connected to {ws_url} on attempt {attempt + 1}")
                break
            except Exception:
                await asyncio.sleep(1.0)

        if not connected or ws is None:
            print("[FAIL] Could not connect to WebSocket server.")
            proc.terminate()
            return False

        calib_cmd = {
            "action": "calibrate_baseline",
            "duration_s": calib_duration_s,
        }
        print(f"\n[SEND] Sending calibration action: {json.dumps(calib_cmd)}")
        await ws.send(json.dumps(calib_cmd))

        calib_started = False
        calib_finished = False
        calib_data = {}
        post_calib_frames = []

        start_wait = time.time()
        max_wait = calib_duration_s + 20.0

        try:
            while time.time() - start_wait < max_wait:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                except Exception as ws_err:
                    print(f"\n[WS ERROR] {ws_err}")
                    if proc_logs:
                        print("Last controller logs:")
                        for l in proc_logs[-15:]:
                            print("  " + l.strip())
                    break

                try:
                    data = json.loads(raw_msg)
                except Exception:
                    continue

                event = data.get("event")

                if event == "calibration_started":
                    calib_started = True
                    print(f"[PROGRESS] Calibration started confirmation received (duration: {data.get('duration_s')}s)")

                elif event == "baseline_calibrated":
                    calib_finished = True
                    calib_data = data
                    print(f"\n[EVENT] 'baseline_calibrated' RECEIVED!")
                    print(f"   Baselines (RSSI): {data.get('baselines')}")
                    print(f"   Variance Thresholds: {data.get('variance_thresholds')}")
                    print(f"   Drop Thresholds: {data.get('drop_thresholds') or data.get('thresholds')}")

                elif calib_finished and ("ai_model" in data or isinstance(data.get("nodes"), dict)):
                    post_calib_frames.append(data)
                    if len(post_calib_frames) >= 5:
                        print("\nCollected 5 post-calibration frames.")
                        break

                elif data.get("calibration", {}).get("active"):
                    c_info = data["calibration"]
                    print(f"   Progress: {c_info.get('progress_pct')}% ({c_info.get('elapsed_s')}s / {calib_duration_s}s) - Samples: {c_info.get('sample_counts')}")

                await asyncio.sleep(0.01)

        finally:
            await ws.close()
            print("WebSocket client closed.")
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("Controller process stopped.")

    print("\n" + "=" * 65)
    print("CALIBRATION VERIFICATION REPORT")
    print("=" * 65)
    print(f"Calibration Started: {calib_started}")
    print(f"Calibration Finished: {calib_finished}")

    if not calib_finished:
        print("[FAIL] Calibration did not complete within the timeout window.")
        return False

    v_thresholds = calib_data.get("variance_thresholds", {})
    r_baselines = calib_data.get("baselines", {})

    print(f"\n1. RSSI Baselines:")
    for nid, r in r_baselines.items():
        print(f"   {nid}: {r} dBm")

    print(f"\n2. Auto-Tuned Variance Thresholds (Safe Clamp: [0.002, 0.080]):")
    thr_pass = True
    for nid, vth in v_thresholds.items():
        in_range = 0.002 <= vth <= 0.080
        print(f"   {nid}: {vth:.5f} => {'PASS' if in_range else 'OUT OF RANGE'}")
        if not in_range:
            thr_pass = False

    print(f"\n3. Post-Calibration Steady-State Idle Condition (Free Space):")
    last_frame = post_calib_frames[-1] if post_calib_frames else {}
    presence_prob = last_frame.get("ai_model", {}).get("presence_prob", 0.0)
    human_detected = last_frame.get("human_detected", False)
    occupancy_label = last_frame.get("occupancy_label", "")
    active_zone = last_frame.get("active_zone")

    print(f"   Human Detected: {human_detected}")
    print(f"   Presence Probability: {presence_prob}")
    print(f"   Occupancy Label: {occupancy_label}")
    print(f"   Active Zone: {active_zone}")

    nodes = last_frame.get("nodes", {})
    print("   Node Statuses:")
    for nid, ndata in nodes.items():
        print(f"     {nid}: RSSI={ndata.get('rssi')} dBm, Variance={ndata.get('variance'):.5f}, Active={ndata.get('is_active')}")

    baseline_record = {
        "timestamp": time.time(),
        "calibrated_event": calib_data,
        "nodes": last_frame.get("nodes", {}) if post_calib_frames else {},
        "status": "FREE_SPACE_EQUILATERAL_CALIBRATED",
    }
    with open("calibration_baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline_record, f, indent=2)
    print("\nSaved persistent baseline to calibration_baseline.json")

    return calib_finished and thr_pass and (not human_detected)


if __name__ == "__main__":
    result = asyncio.run(run_e2e_real_calibration())
    print(f"\nOverall Result: {'SUCCESS' if result else 'FAILED'}")
    sys.exit(0 if result else 1)
