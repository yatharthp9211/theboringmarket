"""End-to-end checks for the single-carrier tabletop sensing pipeline.

The carrier is the one the base AP (phone hotspot) is serving; the engine takes
it from the hardware frames instead of a build constant.
"""

import math
from pathlib import Path

import pytest

from exhibition_controller import (
    ExhibitionController,
    RuViewSpatialEngine,
    SerialBridge,
    Zone,
    load_config,
)
from csi_processor import Y_DELTA_SEGMENTS


ROOT = Path(__file__).resolve().parents[1]
CHANNEL = 6
CHANNEL_FREQUENCY_HZ = 2_437_000_000


def make_pipeline(static_object_corner=None):
    config = load_config(str(ROOT / "config.example.yaml"))
    if static_object_corner is not None:
        config["csi"]["trilateration"]["static_object_corner"] = static_object_corner
    nodes = config["nodes"]
    zones = [
        Zone(
            name=node_id,
            label=node["label"],
            led_pin=node["led_pin"],
            variance_threshold=node["variance_threshold"],
            demo_scenario="room_radar",
            ssid=node["ssid"],
            pos_m=tuple(node["pos_m"]),
            channel=node["channel"],
        )
        for node_id, node in nodes.items()
    ]
    rx_pos = tuple(config["receiver"]["pos_m"])
    spatial = RuViewSpatialEngine(
        zones,
        room_cfg=config["room"],
        rx_pos=rx_pos,
        layout_preset=config["receiver"]["layout_preset"],
        csi_cfg=config["csi"],
    )
    controller = ExhibitionController(
        zones,
        SerialBridge("COM11", demo_mode=True),
        ssid_to_zone={node_id: node_id for node_id in nodes},
    )
    controller.spatial_engine = spatial
    return config, spatial, controller


def packet_message(spatial, tx_id, rx_id, rssi_dbm, ts_us):
    frame = spatial.csi_engine.synthesize_csi_from_rssi(
        tx_id,
        rssi=rssi_dbm,
        variance=0.0,
        target_dist_m=math.hypot(
            spatial.node_positions[tx_id][0] - spatial.node_positions[rx_id][0],
            spatial.node_positions[tx_id][1] - spatial.node_positions[rx_id][1],
        ),
        rx_id=rx_id,
    )
    return {
        "event": "csi",
        "zone": tx_id,
        "rx_id": rx_id,
        "rssi": rssi_dbm,
        "channel": CHANNEL,
        "csi_mag": frame["csi_mag"],
        "csi_phase": frame["csi_phase"],
        "ts_us": ts_us,
    }


def test_rig_runs_on_one_carrier_adopted_from_the_base_ap():
    config, spatial, controller = make_pipeline()
    positions = config["nodes"]
    assert math.dist(
        positions["node_1"]["pos_m"], positions["node_2"]["pos_m"]
    ) == pytest.approx(0.60, abs=0.001)
    assert math.dist(
        positions["node_2"]["pos_m"], positions["node_3"]["pos_m"]
    ) == pytest.approx(0.60, abs=0.001)
    assert math.dist(
        positions["node_3"]["pos_m"], positions["node_1"]["pos_m"]
    ) == pytest.approx(0.60, abs=0.001)
    assert positions["node_4"]["pos_m"] == [0.50, 0.40]
    stream = []
    timestamp = 1_000_000
    for _segment, (node_a, node_b, _kind) in Y_DELTA_SEGMENTS.items():
        for tx_id, rx_id in ((node_a, node_b), (node_b, node_a)):
            stream.append(packet_message(spatial, tx_id, rx_id, -58.0, timestamp))
            timestamp += 25_000

    assert all(node["channel"] == CHANNEL for node in config["nodes"].values() if node["id"] != "node_4")
    assert config["nodes"]["node_4"]["ssid"] == "YOUR_HOTSPOT_SSID"
    assert CHANNEL_FREQUENCY_HZ == 2_407_000_000 + 5_000_000 * CHANNEL
    assert len(stream) == 12
    for message in stream:
        controller._process_csi_message(message)

    # The base AP owns the channel: the engine adopts it from the frames, so
    # every link, displacement and payload rides one carrier with no mismatch.
    assert spatial.csi_engine.rig_channel == CHANNEL
    assert spatial.csi_engine.channel_mismatch_count == 0
    assert all(spatial.csi_engine.node_channels[node_id] == CHANNEL for node_id in config["nodes"])
    assert all(message["channel"] == CHANNEL for message in stream)
    assert spatial.generate_payload()["csi_metrics"]["channel"] == CHANNEL


def test_bidirectional_reciprocity_fuses_without_channel_offset():
    _config, spatial, controller = make_pipeline()
    streams = []
    timestamp = 2_000_000
    for _segment, (node_a, node_b, _kind) in Y_DELTA_SEGMENTS.items():
        for tx_id, rx_id in ((node_a, node_b), (node_b, node_a)):
            streams.append(packet_message(spatial, tx_id, rx_id, -58.0, timestamp))
            timestamp += 25_000
    for message in streams:
        controller._process_csi_message(message)

    metrics = spatial.csi_engine.get_segment_metrics()
    for segment_id, (node_a, node_b, _kind) in Y_DELTA_SEGMENTS.items():
        forward = spatial.csi_engine.directed_links[f"{node_b}<-{node_a}"]
        reverse = spatial.csi_engine.directed_links[f"{node_a}<-{node_b}"]
        assert forward["source"] == reverse["source"] == "hardware_esp32"
        reciprocity_offset_error = abs(
            forward["excess_path_m"] - reverse["excess_path_m"]
        )
        assert reciprocity_offset_error == pytest.approx(0.0, abs=1e-12)
        expected_fusion = (forward["excess_path_m"] + reverse["excess_path_m"]) / 2
        assert metrics[segment_id]["excess_path_m"] == pytest.approx(expected_fusion, abs=1e-4)


def test_csi_uses_packet_rssi_before_a_scan_sample_exists():
    _config, spatial, controller = make_pipeline()
    message = packet_message(spatial, "node_1", "node_4", -48.0, 2_500_000)

    controller._process_csi_message(message)

    assert spatial.csi_engine.directed_links["node_4<-node_1"]["rssi_dbm"] == pytest.approx(-48.0)


def test_csi_prefers_measured_scan_rssi_over_packet_rssi():
    _config, spatial, controller = make_pipeline()
    controller.zones["node_1"].last_scan_rssi = -61.0
    message = packet_message(spatial, "node_1", "node_4", -48.0, 2_600_000)

    controller._process_csi_message(message)

    assert spatial.csi_engine.directed_links["node_4<-node_1"]["rssi_dbm"] == pytest.approx(-61.0)


def test_free_space_stays_idle_and_holds_geometric_centroid():
    _config, spatial, controller = make_pipeline()
    timestamp = 3_000_000
    for tx_id in ("node_1", "node_2", "node_3"):
        rx_id = "node_4"
        baseline = math.dist(spatial.node_positions[tx_id], spatial.node_positions[rx_id])
        rssi_dbm = -44.0 - 21.5 * math.log10(baseline / 0.30)
        spatial.zones[tx_id].last_rssi = rssi_dbm
        spatial.zones[tx_id].last_scan_rssi = rssi_dbm
        spatial.node_rssi[tx_id] = rssi_dbm
        controller._process_csi_message(
            packet_message(spatial, tx_id, rx_id, rssi_dbm, timestamp)
        )
        timestamp += 25_000

    payload = spatial.generate_payload()
    assert spatial.csi_engine.last_no_target is True
    assert payload["human_detected"] is False
    assert payload["ai_model"]["presence_prob"] == 0.0
    assert payload["csi_metrics"]["channel_mismatch_count"] == 0
    assert spatial.current_x == pytest.approx(0.50, abs=0.005)
    assert spatial.current_z == pytest.approx(0.40, abs=0.005)
    assert spatial.csi_engine.last_anchors_used == 3
    assert "node_4" not in spatial.csi_engine.node_range_ts
    assert "node_4" not in spatial.csi_engine.last_anchors_dropped


def test_single_link_does_not_claim_a_static_obstacle():
    config, spatial, controller = make_pipeline(static_object_corner="node_2")
    assert config["csi"]["trilateration"]["static_object_corner"] == "node_2"
    message = packet_message(spatial, "node_4", "node_2", -55.0, 4_000_000)

    controller._process_csi_message(message)

    assert spatial.last_csi_result["static_object"] is None


def test_channel_mismatch_counter_tracks_out_of_band_packet():
    _config, spatial, _controller = make_pipeline()
    frame = spatial.csi_engine.synthesize_csi_from_rssi(
        "node_1", rssi=-58.0, variance=0.0, rx_id="node_4"
    )
    frame["channel"] = 11

    result = spatial.update_csi(frame)

    assert result["channel_mismatch_count"] == 1
    assert spatial.csi_engine.channel_mismatch_count == 1
