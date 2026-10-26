"""
serial_rssi_bridge.py — Bidirectional Serial <-> WebSocket RSSI & Actuator Bridge (v3.0)
========================================================================================
Upgrades in v3.0 Ultra Edition:
  1. Full Bidirectional Communication:
     - Reads JSON RSSI/CSI frames from the ESP32 USB serial port and broadcasts them to
       all connected WebSocket clients (`ws://localhost:3001/ws/sensing`).
     - Receives LED/PWM/PING actuation commands (`<LED:PIN:ON>`, `<PWM:PIN:DUTY>`, `<PING>`)
       from WebSocket clients and writes them directly to the ESP32 serial port!
  2. Auto-Port Discovery & Resilient Reconnection:
     - Automatically scans available USB-UART bridges (CH340, CP210x, FTDI) if the
       configured port is unavailable, and seamlessly reconnects if unplugged.
  3. Built-in `--simulate` Fallback Mode:
     - Allows full testing of the WebSocket pipeline without physical hardware attached.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import sys
import time
from typing import Optional, Set


def _sanitize_finite(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_finite(v) for v in obj]
    return obj

import serial
import serial.tools.list_ports
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Bauds the ESP32 UART driver and CH340/CP210x/FTDI bridges both handle. A live
# CSI line measures 910 bytes, 948 with the emitter's new `ts_us` and
# `synthesized` fields, so 115200 could carry ~12 frames/s -- but the firmware
# scan loop only emits 1-3.6 CSI frames/s per node, so the emitter caps the
# sensing rate, not the transport.
SUPPORTED_BAUD_RATES = (115_200, 230_400, 460_800, 921_600, 1_000_000, 1_500_000, 2_000_000)
DEFAULT_BAUD = 115_200
UART_BITS_PER_BYTE = 10          # 8N1: start + 8 data + stop
CSI_FRAME_BYTES = 950            # measured CSI line: 56 tones + ts + ts_us + synthesized


def validate_baud(baud: int) -> int:
    try:
        requested = int(baud)
    except (TypeError, ValueError):
        logging.warning("Baud %r is not a number — using %d", baud, DEFAULT_BAUD)
        return DEFAULT_BAUD
    if requested in SUPPORTED_BAUD_RATES:
        return requested
    nearest = min(SUPPORTED_BAUD_RATES, key=lambda b: abs(b - requested))
    logging.warning(
        "Baud %d is not supported by the ESP32 UART/USB bridge pairing — using %d. "
        "Supported: %s",
        requested, nearest, ", ".join(f"{b:,}" for b in SUPPORTED_BAUD_RATES),
    )
    return nearest


def frame_ceiling_fps(baud: int, frame_bytes: int = CSI_FRAME_BYTES) -> float:
    """CSI frames/s this UART can carry."""
    return baud / (UART_BITS_PER_BYTE * max(1, frame_bytes))


class BidirectionalSerialBridge:
    # A dashboard tab that stops draining its socket (backgrounded, frozen by a
    # long synchronous handler, or simply wedged) makes websockets send() wait on
    # the TCP write buffer for as long as that buffer takes to fill — measured at
    # 18 s. 80 ms is two frames of telemetry at 25 Hz; past that the client is no
    # use to the show and must not be allowed to stall the healthy ones.
    SEND_TIMEOUT_S = 0.08

    def __init__(self, port: str, baud: int = DEFAULT_BAUD, simulate: bool = False):
        self.port = port
        self.baud = validate_baud(baud)
        self.frame_ceiling_fps = frame_ceiling_fps(self.baud)
        self.simulate = simulate
        self.ser: Optional[serial.Serial] = None
        self.clients: Set = set()
        self.rx_count = 0
        self.tx_count = 0
        self.send_timeouts = 0
        self._closing: set = set()

    def try_open_serial(self) -> bool:
        if self.simulate:
            return True
        candidates = [self.port]
        for p in serial.tools.list_ports.comports():
            if "bluetooth" not in p.description.lower() and p.device not in candidates:
                candidates.append(p.device)

        for candidate in candidates:
            try:
                self.ser = serial.Serial(candidate, self.baud, timeout=0.1)
                if self.baud > DEFAULT_BAUD:
                    # A 2 Mbit stream outruns the driver's default RX buffer.
                    try:
                        self.ser.set_buffer_size(rx_size=32768, tx_size=8192)
                    except (serial.SerialException, OSError, AttributeError):
                        pass
                self.port = candidate
                logging.info(
                    "Connected to ESP32 Serial on %s @ %d baud (~%.0f CSI frames/s)",
                    self.port, self.baud, self.frame_ceiling_fps,
                )
                return True
            except (serial.SerialException, OSError):
                continue
        return False

    def write_serial_command(self, cmd: str) -> None:
        cmd = self._sanitize_command(cmd)
        if not cmd:
            logging.warning("Ignoring empty serial command")
            return
        self.tx_count += 1
        if self.simulate:
            logging.info("[SIM-TX] -> ESP32: %s", cmd.strip())
            return
        if self.ser and self.ser.is_open:
            try:
                if not cmd.endswith("\n"):
                    cmd += "\n"
                self.ser.write(cmd.encode("utf-8"))
            except (serial.SerialException, OSError) as exc:
                logging.warning("Serial write failed: %s", exc)

    @staticmethod
    def _sanitize_command(cmd: object) -> str:
        return str(cmd).replace("\r", "").replace("\n", "").strip()

    async def serial_reader_loop(self, queue: asyncio.Queue) -> None:
        if self.simulate:
            t = 0.0
            sim_boot_s = time.monotonic()
            zones = [("Zone_1_Tx", "node_1", 6), ("Zone_2_Tx", "node_2", 6), ("Laptop_Hotspot", "node_3", 6)]
            logging.info(
                "Running in --simulate mode (synthetic 25 Hz ESP32 RSSI stream with a "
                "wrapping 32-bit ts_us device clock)"
            )
            while True:
                active_idx = int(t / 4.5) % len(zones)
                for idx, (ssid, zone, ch) in enumerate(zones):
                    rssi = int(-68 + random.gauss(0, 3.2)) if idx == active_idx else int(-61 + random.gauss(0, 0.4))
                    msg = {
                        "ssid": ssid,
                        "zone": zone,
                        "rssi": rssi,
                        "raw_rssi": rssi,
                        "ch": ch,
                        "ts": int(time.time() * 1000),
                        # esp_timer_get_time() shape: uint32 microseconds since boot.
                        "ts_us": int((time.monotonic() - sim_boot_s) * 1e6) & 0xFFFFFFFF,
                    }
                    await queue.put(msg)
                await asyncio.sleep(0.08)
                t += 0.08

        loop = asyncio.get_running_loop()
        while True:
            if self.ser is None or not self.ser.is_open:
                if not self.try_open_serial():
                    await asyncio.sleep(2.0)
                    continue
            try:
                raw_line = await loop.run_in_executor(None, self.ser.readline)
            except (serial.SerialException, OSError):
                logging.warning("Serial connection lost on %s — reconnecting...", self.port)
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                await asyncio.sleep(1.0)
                continue

            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line == "PONG":
                await queue.put({"event": "pong", "ts": int(time.time() * 1000)})
                continue
            try:
                msg = json.loads(line)
                self.rx_count += 1
                # Host arrival only: the frame's own ts/ts_us stay the device clock.
                if isinstance(msg, dict):
                    msg["_host_rx_ts"] = time.time()
                await queue.put(msg)
            except json.JSONDecodeError:
                logging.debug("Non-JSON serial line: %s", line)

    async def ws_handler(self, ws) -> None:
        self.clients.add(ws)
        logging.info("WebSocket client connected (%d active)", len(self.clients))
        try:
            await ws.send(json.dumps({
                "event": "connected",
                "bridge": "serial_rssi_bridge_v3.0",
                "port": "SIMULATED" if self.simulate else self.port
            }))
            async for raw_msg in ws:
                raw_str = str(raw_msg).strip()
                sanitized = self._sanitize_command(raw_str)
                if sanitized.startswith("<") and sanitized.endswith(">"):
                    self.write_serial_command(sanitized)
                    continue
                try:
                    data = json.loads(raw_str)
                except json.JSONDecodeError:
                    continue
                try:
                    if not isinstance(data, dict):
                        logging.warning("Ignoring non-object WS command payload: %r", data)
                        continue
                    if "cmd" in data:
                        self.write_serial_command(str(data["cmd"]))
                    elif data.get("action") == "ping":
                        self.write_serial_command("<PING>")
                    elif data.get("action") == "led":
                        pin = int(data.get("pin", 2))
                        if not 0 <= pin <= 48:
                            logging.warning("Ignoring out-of-range LED pin: %s", pin)
                            continue
                        state = "ON" if data.get("state") in (True, "ON", 1) else "OFF"
                        self.write_serial_command(f"<LED:{pin}:{state}>")
                except (ValueError, TypeError, AttributeError) as exc:
                    logging.warning("Ignoring malformed WS command: %s", exc)
        finally:
            self.clients.discard(ws)
            logging.info("WebSocket client disconnected (%d active)", len(self.clients))

    async def broadcast_loop(self, queue: asyncio.Queue) -> None:
        while True:
            msg = await queue.get()
            try:
                try:
                    payload = json.dumps(msg, separators=(",", ":"))
                    if "NaN" in payload or "Infinity" in payload:
                        payload = json.dumps(_sanitize_finite(msg), separators=(",", ":"))
                except (TypeError, ValueError) as exc:
                    logging.warning("Dropping non-serializable broadcast message: %s", exc)
                    continue
                clients = list(self.clients)
                if not clients:
                    continue
                results = await asyncio.gather(
                    *(asyncio.wait_for(c.send(payload), timeout=self.SEND_TIMEOUT_S)
                      for c in clients),
                    return_exceptions=True)
                for c, res in zip(clients, results):
                    if not isinstance(res, BaseException):
                        continue
                    self.clients.discard(c)
                    if isinstance(res, asyncio.TimeoutError):
                        self.send_timeouts += 1
                        logging.warning(
                            "WebSocket client stalled (>%.0f ms to drain a frame) - "
                            "dropped, %d client(s) left",
                            self.SEND_TIMEOUT_S * 1000, len(self.clients))
                    self._close_quietly(c)
            finally:
                queue.task_done()

    def _close_quietly(self, ws) -> None:
        """Evict a wedged connection without making the next frame wait for it."""
        task = asyncio.create_task(self._close(ws))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _close(self, ws) -> None:
        try:
            await ws.close(code=1011, reason="send timeout")
        except Exception as exc:
            logging.debug("Evicted client close failed: %s", exc)


async def main_async(args: argparse.Namespace) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=512)
    bridge = BidirectionalSerialBridge(args.port, args.baud, simulate=args.simulate)
    server = await websockets.serve(bridge.ws_handler, args.host, args.ws_port)
    logging.info("Bidirectional Serial<->WS Bridge listening on ws://%s:%d/ws/sensing", args.host, args.ws_port)
    await asyncio.gather(
        server.wait_closed(),
        bridge.serial_reader_loop(queue),
        bridge.broadcast_loop(queue),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Bidirectional Serial<->WebSocket RSSI & Actuator Bridge v3.0")
    p.add_argument("--port", default="COM8", help="ESP32 USB serial port (default: COM8)")
    p.add_argument(
        "--baud", type=int, default=DEFAULT_BAUD,
        help="Serial baud rate; must match Serial.begin() in the firmware. "
             f"Supported: {', '.join(f'{b:,}' for b in SUPPORTED_BAUD_RATES)}. "
             f"Default {DEFAULT_BAUD:,} carries ~{frame_ceiling_fps(DEFAULT_BAUD):.0f} CSI "
             f"frames/s, 921,600 ~{frame_ceiling_fps(921_600):.0f}/s, "
             f"2,000,000 ~{frame_ceiling_fps(2_000_000):.0f}/s."
    )
    p.add_argument("--host", default="localhost", help="WebSocket bind host (default: localhost)")
    p.add_argument("--ws-port", dest="ws_port", type=int, default=3001, help="WebSocket port (default: 3001)")
    p.add_argument("--simulate", action="store_true", help="Simulate ESP32 RSSI stream without hardware")
    args = p.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logging.info("Bridge stopped cleanly.")


if __name__ == "__main__":
    main()
