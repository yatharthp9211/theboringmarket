#!/usr/bin/env python3
"""
Master Test Runner for RuView v3.6/v3.7
Sequentially executes and benchmarks all test suites across the project:
  1. tests/test_tracking.py (Phase & Fusion Unit Tests)
  2. csi_processor.py --selftest (56-SC CSI Bi-Static Solver)
  3. wifi_densepose_ai.py --selftest (Tomography & Joseph Kalman AI)
  4. exhibition_controller.py --selftest (Controller & Guard Rails)
  5. tests/test_stress_until_break.py (Exhaustive 'Until It Breaks' Stress Test)
"""

import sys
import subprocess
import time

SUITES = [
    ("Tracking & DSP Unit Tests", [sys.executable, "-m", "pytest", "-q", "tests/test_tracking.py"]),
    ("CSI Bi-Static Solver Self-Test", [sys.executable, "csi_processor.py", "--selftest"]),
    ("DensePose & AI Tomography Self-Test", [sys.executable, "wifi_densepose_ai.py", "--selftest"]),
    ("Exhibition Controller Guard Self-Test", [sys.executable, "exhibition_controller.py", "--selftest"]),
    ("Exhaustive 'Until It Breaks' Stress Suite", [sys.executable, "tests/test_stress_until_break.py"]),
]

def main():
    print("=" * 80)
    print(" RUVIEW v3.6/v3.7 MASTER INTEGRATION & STRESS TEST RUNNER ")
    print("=" * 80)
    
    total_start = time.time()
    results = []
    all_passed = True
    
    for name, cmd in SUITES:
        print(f"\n[RUNNING] {name}...")
        print("-" * 80)
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        
        passed = (res.returncode == 0)
        if passed:
            print(f">> PASS ({dt:.2f}s)")
            # Print last few lines of stdout
            lines = [line for line in res.stdout.strip().split("\n") if line.strip()]
            for line in lines[-4:]:
                print(f"   | {line}")
        else:
            all_passed = False
            print(f">> FAIL ({dt:.2f}s) — Exit Code {res.returncode}")
            print(res.stdout[-1000:] if res.stdout else "")
            print(res.stderr[-1000:] if res.stderr else "")
            
        results.append((name, passed, dt))

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(" MASTER SUITE EXECUTION SUMMARY ")
    print("=" * 80)
    for name, passed, dt in results:
        status_str = "PASS" if passed else "FAIL"
        print(f"  [{status_str}] {name:<45} ({dt:5.2f}s)")
    print("-" * 80)
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Overall Result:       {'ALL SUITES PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 80)
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
