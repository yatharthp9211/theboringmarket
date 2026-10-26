"""PC calibration and Standard ESP32 actuator routing tests."""

import asyncio
from pathlib import Path

import pytest

from exhibition_controller import (
    ExhibitionController,
    RuViewSpatialEngine,
    DashboardServer,
    SerialBridge,
    Zone,
    load_config,
    resolve_serial_ports,
)


ROOT = Path(__file__).resolve().parents[1]


class RecordingBridge:
    def __init__(self):
        self.commands = []

    def led_on(self, pin):
        self.commands.append(("ON", pin))

    def led_off(self, pin):
        self.commands.append(("OFF", pin))


def test_standard_esp32_is_actuator_independent_of_sensor_port_order():
    actuator, ports = resolve_serial_ports(
        {"actuator_port": "COM11", "ports": ["COM6", "COM7", "COM11"]},
        {},
        ports_override="COM6,COM7",
    )

    assert actuator == "COM11"
    assert ports == ["COM11", "COM6", "COM7"]


def test_actuator_override_is_explicit_and_kept_first():
    actuator, ports = resolve_serial_ports(
        {"actuator_port": "COM11", "ports": ["COM11", "COM6"]},
        {},
        actuator_override="COM8",
    )

    assert actuator == "COM8"
    assert ports == ["COM8", "COM11", "COM6"]


def test_legacy_primary_port_does_not_redirect_bulb_commands():
    actuator, ports = resolve_serial_ports(
        {"actuator_port": "COM11", "ports": ["COM11", "COM6"]},
        {},
        port_override="COM6",
    )

    assert actuator == "COM11"
    assert ports == ["COM11", "COM6"]


def test_sensor_only_bridge_cannot_send_bulb_commands():
    sensor_bridge = SerialBridge("COM6", demo_mode=True)

    with pytest.raises(PermissionError, match="sensor-only"):
        sensor_bridge.led_on(2)


def test_config_routes_three_bulbs_to_standard_esp32_and_pc_calibration():
    config = load_config(str(ROOT / "config.example.yaml"))

    assert config["serial"]["actuator_port"] == "COM11"
    assert [config["nodes"][name]["led_pin"] for name in ("node_1", "node_2", "node_3")] == [2, 4, 5]
    assert config["room_calibration"]["require_calibration_for_actuation"] is True


def test_calibration_needs_samples_for_each_bulb_zone():
    config = load_config(str(ROOT / "config.example.yaml"))
    zones = [
        Zone(
            name=name,
            label=node["label"],
            led_pin=node["led_pin"],
            variance_threshold=node["variance_threshold"],
            demo_scenario="room_radar",
            ssid=node["ssid"],
            pos_m=tuple(node["pos_m"]),
        )
        for name, node in config["nodes"].items()
    ]
    engine = RuViewSpatialEngine(
        zones,
        room_cfg=config["room"],
        rx_pos=tuple(config["receiver"]["pos_m"]),
        layout_preset=config["receiver"]["layout_preset"],
        csi_cfg=config["csi"],
        calibration_cfg=config["room_calibration"],
    )

    engine.start_calibration(1)
    with pytest.raises(ValueError, match="at least 3 PC-side variance samples"):
        engine.finish_calibration()
    assert not engine.is_calibrated


def test_pc_calibration_applies_thresholds_to_the_zone_objects():
    config = load_config(str(ROOT / "config.example.yaml"))
    zones = [
        Zone(
            name=name,
            label=node["label"],
            led_pin=node["led_pin"],
            variance_threshold=node["variance_threshold"],
            demo_scenario="room_radar",
            ssid=node["ssid"],
            pos_m=tuple(node["pos_m"]),
        )
        for name, node in config["nodes"].items()
    ]
    engine = RuViewSpatialEngine(
        zones,
        room_cfg=config["room"],
        rx_pos=tuple(config["receiver"]["pos_m"]),
        layout_preset=config["receiver"]["layout_preset"],
        csi_cfg=config["csi"],
        calibration_cfg=config["room_calibration"],
    )

    engine.start_calibration(1)
    for name in ("node_1", "node_2", "node_3"):
        for variance in (0.0001, 0.0002, 0.0003):
            engine.update_spatial(name, variance, 0.5, raw_rssi=-60.0)

    result = engine.finish_calibration()

    assert engine.is_calibrated
    for name in ("node_1", "node_2", "node_3"):
        assert engine.zones[name].variance_threshold == result["variance_thresholds"][name]


def test_dashboard_calibration_uses_pc_samples_before_arming_bulbs():
    config = load_config(str(ROOT / "config.example.yaml"))
    zones = [
        Zone(
            name=name,
            label=node["label"],
            led_pin=node["led_pin"],
            variance_threshold=node["variance_threshold"],
            demo_scenario="room_radar",
            ssid=node["ssid"],
            pos_m=tuple(node["pos_m"]),
        )
        for name, node in config["nodes"].items()
    ]
    engine = RuViewSpatialEngine(
        zones,
        room_cfg=config["room"],
        rx_pos=tuple(config["receiver"]["pos_m"]),
        layout_preset=config["receiver"]["layout_preset"],
        csi_cfg=config["csi"],
        calibration_cfg=config["room_calibration"],
    )
    bridge = SerialBridge("COM11", demo_mode=True, actuator_enabled=True)
    dashboard = DashboardServer(zones, bridge, asyncio.Queue())
    dashboard.spatial_engine = engine

    async def run_calibration():
        await dashboard.start_baseline_calibration(1)
        assert engine.is_calibrating
        for name in ("node_1", "node_2", "node_3"):
            for variance in (0.0001, 0.0002, 0.0003):
                engine.update_spatial(name, variance, 0.5, raw_rssi=-60.0)
        await dashboard.calibration_task

    asyncio.run(run_calibration())

    assert engine.is_calibrated
    assert bridge.tx_packets == 3


def test_calibration_collects_variance_from_real_scan_messages():
    config = load_config(str(ROOT / "config.example.yaml"))
    zones = [
        Zone(
            name=name,
            label=node["label"],
            led_pin=node["led_pin"],
            variance_threshold=node["variance_threshold"],
            demo_scenario="room_radar",
            ssid=node["ssid"],
            pos_m=tuple(node["pos_m"]),
        )
        for name, node in config["nodes"].items()
    ]
    engine = RuViewSpatialEngine(
        zones,
        room_cfg=config["room"],
        rx_pos=tuple(config["receiver"]["pos_m"]),
        layout_preset=config["receiver"]["layout_preset"],
        csi_cfg=config["csi"],
        calibration_cfg=config["room_calibration"],
    )
    controller = ExhibitionController(
        zones,
        SerialBridge("COM11", demo_mode=True, actuator_enabled=True),
        ssid_to_zone={"node_1": "node_1", "node_2": "node_2", "node_3": "node_3"},
    )
    controller.spatial_engine = engine
    engine.start_calibration(1)

    for zone_name, rssi in (("node_1", -55), ("node_2", -58), ("node_3", -61)):
        message = {
            "event": "scan",
            "node": "node_3",
            "zone": zone_name,
            "raw_rssi": rssi,
            "rssi": rssi,
        }
        mapped_zone, amplitude, raw_rssi = controller._map_message_to_zone(message)
        variance = engine.zones[mapped_zone].push_sample(
            amplitude, raw_rssi=raw_rssi, rx_node="node_3"
        )
        engine.update_spatial(
            mapped_zone, variance, amplitude, raw_rssi=raw_rssi, rx_node="node_3"
        )

    assert {
        name: len(samples)
        for name, samples in engine.calibration_variance_samples.items()
    } == {"node_1": 1, "node_2": 1, "node_3": 1, "node_4": 0}


def test_pc_detection_selects_zone_bulb_using_calibrated_zone_mapping():
    config = load_config(str(ROOT / "config.example.yaml"))
    zones = [
        Zone(
            name=name,
            label=node["label"],
            led_pin=node["led_pin"],
            variance_threshold=node["variance_threshold"],
            demo_scenario="room_radar",
            ssid=node["ssid"],
            pos_m=tuple(node["pos_m"]),
        )
        for name, node in config["nodes"].items()
    ]
    engine = RuViewSpatialEngine(
        zones,
        room_cfg=config["room"],
        rx_pos=tuple(config["receiver"]["pos_m"]),
        layout_preset=config["receiver"]["layout_preset"],
        csi_cfg=config["csi"],
        calibration_cfg=config["room_calibration"],
    )
    engine.is_calibrated = True
    bridge = RecordingBridge()
    controller = ExhibitionController(zones, bridge, ssid_to_zone={"node_2": "node_2"})
    controller.spatial_engine = engine
    queue = asyncio.Queue()

    async def feed_motion():
        task = asyncio.create_task(controller.run(queue))
        try:
            for amplitude in (0.0, 1.0, 0.0):
                await queue.put({"zone": "node_2", "amplitude": amplitude})
            await asyncio.wait_for(queue.join(), timeout=2.0)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(feed_motion())

    assert bridge.commands == [("ON", 4)]
