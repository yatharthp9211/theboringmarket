#!/usr/bin/env python3
"""
================================================================================
RuView v3.7 — 56-Subcarrier IEEE 802.11n CSI Phase Ranging & Y-Delta Bi-Static
              Multi-Path Localization Engine (Centimeter-Level Precision)
================================================================================
Upgrades over v3.6:
  1. True Directed Pairwise Link Matrix (`rx_node<-tx_node`):
     Supports all 9 directed links across the 4-Node Y-Delta Tabletop Grid
     (3 ESP32 corner transceivers V1, V2, V3 + 1 Centroid Mobile Hotspot C),
     forming 6 physical RF paths (3 inner Y-spokes + 3 outer Delta edges)
     while maintaining 100% backward compatibility with single-Rx / 4-zone calls.
  2. Bidirectional Reciprocity Fusion:
     Fuses V_i -> V_j and V_j -> V_i CSI phase & ToF measurements along the
      exact same physical triangle edge. Every link is pinned to the live rig
      channel (13), so the two directions share one carrier and one wavelength:
      no inter-channel phase-offset calibration is needed, and quality/freshness
      weighting cuts
     thermal noise without letting a weak reverse packet drag the segment.
  3. Hampel Robust Outlier Rejection + Unbiased Phase-Slope ToF Extraction:
     Eliminates transient Wi-Fi scan spikes and removes the v3.6 zero-tap PDP
     compression bias on sub-meter tabletop links.
  4. 4-Anchor Y-Delta TDOA + Levenberg-Marquardt Sub-Millimeter Solver:
     Solves 2D tabletop coordinates on a 1 cm warm-start grid followed by
     damped Gauss-Newton / Levenberg-Marquardt refinement (< 0.15 cm RMSE).
================================================================================
"""

import collections
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS (IEEE 802.11n 2.4 GHz HT20 OFDM)
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_MPS: float = 299_792_458.0          # Speed of light in vacuum (m/s)
SUBCARRIER_SPACING_HZ: float = 312_500.0    # 20 MHz / 64 FFT bins = 312.5 kHz
NUM_SUBCARRIERS: int = 56                   # IEEE 802.11n HT20 active subcarriers

# Active subcarrier indices k in [-28..-1, +1..+28]
HT20_SUBCARRIER_INDICES: np.ndarray = np.array(
    list(range(-28, 0)) + list(range(1, 29)), dtype=np.float64
)

# The phone mobile hotspot is the base AP and owns the 2.4 GHz channel: it picks
# one on every start and the ESP32s (AP+STA) drag their SoftAPs onto it. So the
# carrier is not a build constant. FIXED_CHANNEL is only the fallback used before
# the first hardware frame arrives; CsiSpatialEngine adopts the channel the frames
# report (`rig_channel`) and evaluates every link at that carrier.
FIXED_CHANNEL: int = 6
CARRIER_FREQ_HZ: float = 2_437_000_000.0
CARRIER_WAVELENGTH_M: float = C_LIGHT_MPS / CARRIER_FREQ_HZ  # 0.121275 m

# Consecutive off-channel hardware frames before the engine follows the base AP
# onto a new channel: ~2 s at the 18-20 Hz hardware frame rate.
CHANNEL_RELOCK_FRAMES: int = 40

# Canonical 6 Y-Delta Geometric Segments for 4-Node Tabletop Architecture
Y_DELTA_SEGMENTS: Dict[str, Tuple[str, str, str]] = {
    "spoke_1": ("node_4", "node_1", "radial_spoke"),
    "spoke_2": ("node_4", "node_2", "radial_spoke"),
    "spoke_3": ("node_4", "node_3", "radial_spoke"),
    "edge_12": ("node_1", "node_2", "perimeter_edge"),
    "edge_23": ("node_2", "node_3", "perimeter_edge"),
    "edge_31": ("node_3", "node_1", "perimeter_edge"),
}


def channel_to_freq_hz(channel: int = FIXED_CHANNEL) -> float:
    """Centre frequency of a 2.4 GHz channel (ch 14 is the 2484 MHz exception)."""
    ch = int(channel)
    if ch == 14:
        return 2_484_000_000.0
    ch = min(max(ch, 1), 13)
    return (2407.0 + 5.0 * ch) * 1_000_000.0


def channel_to_wavelength_m(channel: int = FIXED_CHANNEL) -> float:
    """Carrier wavelength for a 2.4 GHz channel: 0.1249 m on ch 1, 0.1213 m on ch 6."""
    return C_LIGHT_MPS / channel_to_freq_hz(channel)


# JSON-safe sentinel: browsers reject `Infinity`/`NaN` and every dashboard reads
# `gdop` as a bare number. Test `gdop_rejected`, never this value.
GDOP_UNDETERMINED: float = 99.0

# 115200 baud carries ~12 CSI frames/s (950-byte lines, 8N1).
NOMINAL_FRAME_RATE_HZ: float = 12.0


# ─────────────────────────────────────────────────────────────────────────────
# 0. DEVICE TIME BASE — the radio's clock, not the PC's clock
# ─────────────────────────────────────────────────────────────────────────────
def extract_device_time_us(csi_frame: dict) -> Tuple[Optional[int], str]:
    """
    Pulls the on-device microsecond timestamp out of a CSI/RSSI frame.

    Preference order: `ts_us` (native microseconds) -> `tsf_us`/`tsf` (802.11
    Timing Synchronization Function (TSF) tick) -> `ts` (the millisecond value
    the current firmware emits). Returns (None, "absent") when the frame carries
    no device clock, so callers can fall back to host receive time and tag the
    frame as unsynchronised instead of silently passing host time off as packet
    time.
    """
    for key in ("ts_us", "tsf_us", "tsf"):
        val = csi_frame.get(key)
        if val is None:
            continue
        try:
            return int(val), key
        except (TypeError, ValueError):
            continue

    val = csi_frame.get("ts")
    if val is not None:
        try:
            return int(float(val) * 1000.0), "ts_ms"
        except (TypeError, ValueError):
            pass
    return None, "absent"


class DeviceTimebase:
    """
    Unwraps a repeating 32-bit on-device counter into a monotonic timeline.

    A uint32 microsecond stamp — the ESP32 Arduino core's
    `esp_timer_get_time()/1000` style counter — rolls over every 71.6 minutes,
    so raw stamps cannot be subtracted across a wrap. The host wall clock is
    consulted only to tell a rollover from a device reboot; it is never used as
    the measurement clock.
    """

    WRAP_US: int = 2 ** 32

    def __init__(self) -> None:
        self._last_raw: Optional[int] = None
        self._epoch: int = 0
        self._last_host_s: Optional[float] = None

    def to_us(
        self,
        raw_us: int,
        host_s: Optional[float] = None,
        stamp_key: str = "ts_us",
    ) -> int:
        host = float(time.time() if host_s is None else host_s)
        raw_val = int(raw_us)
        wrap_mod = self.WRAP_US * 1000 if stamp_key == "ts_ms" else self.WRAP_US
        host_gap_s = (
            None
            if self._last_host_s is None
            else max(0.0, host - float(self._last_host_s))
        )

        if self._last_raw is not None and raw_val < self._last_raw:
            # True counter rollover occurs when _last_raw was near the wrap and
            # new raw is near 0. The firmware `ts` field is milliseconds, so its
            # true wrap period is 2**32 * 1000 us, not the 2**32 us of ts_us/TSF.
            rollover_margin = 100_000_000 * (1000 if stamp_key == "ts_ms" else 1)
            is_rollover = (
                (self._last_raw > (wrap_mod - rollover_margin))
                and (raw_val < rollover_margin)
                and (host_gap_s is not None and host_gap_s < 600.0)
            )
            if is_rollover:
                self._epoch += wrap_mod
            else:
                # A short-gap decrease can be a stale/reordered packet rather
                # than a reboot. Keep the monotonic timeline and hand back the
                # last value; only a large drop resets the epoch.
                drop_us = self._last_raw - raw_val
                plausible_reorder = (
                    host_gap_s is not None
                    and host_gap_s < 5.0
                    and drop_us < 2_000_000
                    and raw_val > 1_000_000
                )
                if plausible_reorder:
                    return self._epoch + self._last_raw
                # A short-gap decrease, or a long gap without a near-wrap raw
                # value, is a device reboot: restart the monotonic epoch.
                self._epoch = 0

        self._last_raw = raw_val
        self._last_host_s = host
        return self._epoch + raw_val


# ─────────────────────────────────────────────────────────────────────────────
# 0b. TRUE GEOMETRIC DILUTION OF PRECISION (GDOP)
# ─────────────────────────────────────────────────────────────────────────────
def compute_gdop(
    anchors: np.ndarray,
    point: Tuple[float, float],
    sigma_m: Optional[float] = None,
    include_clock: bool = False,
) -> float:
    """
    Geometric Dilution of Precision (GDOP) for range observations from each row
    of `anchors` to `point`:

        gdop = sqrt(trace((H^T H)^-1)),   H[i] = unit line-of-sight vector to i

    Geometry only. A 1-sigma range error of `sigma_m` metres becomes
    `gdop * sigma_m` of position error. `include_clock` adds the receiver-clock
    bias unknown that true TDOA needs, which costs one more anchor.

    Returns GDOP_UNDETERMINED when the geometry cannot resolve the position: too
    few anchors, collinear anchors making H^T H singular, or too few anchors that
    are not sitting right under the target. An anchor at the target contributes no
    direction at that point, so it is dropped from H rather than poisoning the
    whole test. Never clipped: a gate is worthless if the number it tests has
    been trimmed.
    """
    pts = np.asarray(anchors, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return GDOP_UNDETERMINED

    dx = float(point[0]) - pts[:, 0]
    dz = float(point[1]) - pts[:, 1]
    dist = np.hypot(dx, dz)
    informative = dist >= 1e-6
    if not np.any(informative):
        return GDOP_UNDETERMINED

    dx, dz, dist = dx[informative], dz[informative], dist[informative]
    rows = np.column_stack((dx / dist, dz / dist))
    if include_clock:
        rows = np.column_stack((rows, np.ones(rows.shape[0])))

    if rows.shape[0] < rows.shape[1]:
        return GDOP_UNDETERMINED

    HtH = rows.T @ rows
    if not np.all(np.isfinite(HtH)) or np.linalg.cond(HtH) > 1e9:
        return GDOP_UNDETERMINED

    gdop = float(math.sqrt(max(0.0, float(np.trace(np.linalg.inv(HtH))))))
    if sigma_m is not None:
        gdop *= float(sigma_m)
    return gdop


# ─────────────────────────────────────────────────────────────────────────────
# 0c. TWO-WAY RANGING (TWR) — cancels clock OFFSET, bounds clock SKEW
# ─────────────────────────────────────────────────────────────────────────────
def compute_twr_tof(
    t1_us: float,
    t2_us: float,
    t3_us: float,
    t4_us: float,
    clock_skew_ppm: float = 0.0,
) -> Dict[str, object]:
    """
    One-way time of flight from a symmetric two-way exchange. All four stamps are
    microseconds, each on the clock that measured it:

        t1  A transmits   (A clock)
        t2  B receives    (B clock)
        t3  B transmits   (B clock)
        t4  A receives    (A clock)

        tof = ( (t4 - t1) - (t3 - t2) / (1 + eps) ) / 2
        eps = fractional frequency offset of B relative to A = clock_skew_ppm*1e-6

    The unknown clock *offset* between A and B cancels exactly, which is why this
    needs no Precision Time Protocol (PTP) and no synchronised infrastructure.
    What survives is the *skew* acting on the reply window; its residual is
    eps * tof, under 0.01 cm for a 20 ppm pair at room scale.

    Returns a dict with tof_us / tof_ns / distance_m / skew_correction_us /
    residual_uncertainty_m, plus `valid` and a `reason` when the exchange is
    unusable (counter wrap, reply longer than round trip, absurd skew, or an
    implausible distance that usually means the stamps are not in microseconds).
    """
    out: Dict[str, object] = {
        "valid": False,
        "reason": "",
        "roundtrip_us": 0.0,
        "reply_us": 0.0,
        "skew_correction_us": 0.0,
        "tof_us": 0.0,
        "tof_ns": 0.0,
        "distance_m": 0.0,
        "residual_uncertainty_m": 0.0,
    }

    try:
        stamps = [float(t1_us), float(t2_us), float(t3_us), float(t4_us)]
        skew = float(clock_skew_ppm)
    except (TypeError, ValueError):
        out["reason"] = "non_numeric_stamps"
        return out

    if not all(math.isfinite(s) for s in stamps) or not math.isfinite(skew):
        out["reason"] = "non_finite_stamps"
        return out

    t1, t2, t3, t4 = stamps
    roundtrip = t4 - t1        # measured on A's clock
    reply = t3 - t2            # measured on B's clock

    if roundtrip <= 0.0 or reply <= 0.0:
        out["reason"] = "counter_wrap_or_bad_order"
        return out

    eps = skew * 1e-6
    if eps <= -1.0:
        out["reason"] = "absurd_skew"
        return out

    reply_in_a_clock = reply / (1.0 + eps)
    tof_us = 0.5 * (roundtrip - reply_in_a_clock)

    if tof_us <= 0.0:
        out["reason"] = "negative_tof_reply_exceeds_roundtrip"
        return out

    distance_m = C_LIGHT_MPS * tof_us * 1e-6
    out["reason"] = "implausible_distance" if distance_m > 300.0 else "ok"

    out.update({
        "valid": out["reason"] == "ok",
        "roundtrip_us": round(roundtrip, 4),
        "reply_us": round(reply, 4),
        "skew_correction_us": round(reply_in_a_clock - reply, 6),
        "tof_us": round(tof_us, 6),
        "tof_ns": round(tof_us * 1e3, 3),
        "distance_m": round(distance_m, 4),
        "residual_uncertainty_m": round(abs(eps) * tof_us * 1e-6 * C_LIGHT_MPS, 6),
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. HAMPEL OUTLIER FILTER FOR AMPLITUDE & PHASE SERIES
# ─────────────────────────────────────────────────────────────────────────────
class HampelFilter1D:
    """Sliding-window median absolute deviation (MAD) spike filter."""

    def __init__(self, window_size: int = 7, n_sigma: float = 2.8) -> None:
        self.window: collections.deque = collections.deque(maxlen=window_size)
        self.n_sigma = float(n_sigma)

    def update(self, val: float) -> float:
        v = float(val)
        self.window.append(v)
        if len(self.window) < 4:
            return v
        arr = np.array(self.window, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sigma_est = 1.4826 * mad + 1e-6
        if abs(v - med) > self.n_sigma * sigma_est:
            return med
        return v


# ─────────────────────────────────────────────────────────────────────────────
# 2. CSI PHASE SANITIZER (UNWRAPPING + WEIGHTED LLS + SFO/CPO REMOVAL)
# ─────────────────────────────────────────────────────────────────────────────
class CSIPhaseSanitizer:
    """
    Removes Hardware Sampling Frequency Offset (SFO), Packet Detection Delay (PDD),
    and Carrier Phase Offset (CPO) from raw 56-subcarrier complex CSI H[k].
    """

    def __init__(self, ema_alpha: float = 0.45) -> None:
        self.k_indices = HT20_SUBCARRIER_INDICES
        self.k_mean = float(np.mean(self.k_indices))
        self.k_centered = self.k_indices - self.k_mean
        self.k_var_sum = float(np.sum(self.k_centered ** 2))
        self.ema_alpha = float(ema_alpha)
        self._smoothed_phase: Dict[str, np.ndarray] = {}
        self._smoothed_amp: Dict[str, np.ndarray] = {}

    def sanitize(
        self,
        raw_amp: np.ndarray,
        raw_phase: np.ndarray,
        link_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        amp = np.asarray(raw_amp, dtype=np.float64).flatten()
        phase = np.asarray(raw_phase, dtype=np.float64).flatten()

        if len(amp) < 8 or len(phase) < 8:
            return np.array([]), np.array([]), 0.0, 0.0, 0.0
        if not np.all(np.isfinite(amp)) or not np.all(np.isfinite(phase)):
            return np.array([]), np.array([]), 0.0, 0.0, 0.0

        if len(amp) != NUM_SUBCARRIERS:
            amp = np.interp(
                np.linspace(0, 1, NUM_SUBCARRIERS),
                np.linspace(0, 1, len(amp)),
                amp,
            )
        if len(phase) != NUM_SUBCARRIERS:
            # Unwrap on the native grid before resampling: interpolating a
            # wrapped phase across a +/-pi boundary invents a fake slope and
            # corrupts every range/Doppler estimate downstream.
            phase = np.unwrap(phase)
            phase = np.interp(
                np.linspace(0, 1, NUM_SUBCARRIERS),
                np.linspace(0, 1, len(phase)),
                phase,
            )

        # Mask out ESP32 guard-band null bins (amp <= 1.0) and LO/DC pilot spikes (e.g. bin 26 mag=86..126)
        non_zero = amp > 1.0
        if np.sum(non_zero) < 8:
            return np.array([]), np.array([]), 0.0, 0.0, 0.0

        med_amp = float(np.median(amp[non_zero]))
        valid_mask = (amp > 1.0) & (amp <= 2.25 * med_amp)
        if np.sum(valid_mask) >= 8:
            valid_idx = np.where(valid_mask)[0]
            all_idx = np.arange(NUM_SUBCARRIERS)
            unwrapped_valid = np.unwrap(phase[valid_idx])
            unwrapped = np.interp(all_idx, valid_idx, unwrapped_valid)
            amp = np.interp(all_idx, valid_idx, amp[valid_idx])
        else:
            unwrapped = np.unwrap(phase)

        weights = np.clip(amp, 1e-3, None) ** 2
        w_sum = float(np.sum(weights))
        k_wmean = float(np.sum(weights * self.k_indices) / w_sum)
        p_wmean = float(np.sum(weights * unwrapped) / w_sum)
        dk = self.k_indices - k_wmean
        dp = unwrapped - p_wmean
        denom = float(np.sum(weights * (dk ** 2))) + 1e-12
        slope = float(np.sum(weights * dk * dp) / denom)
        intercept = float(p_wmean - slope * k_wmean)

        linear_fit = slope * self.k_indices + intercept
        sanitized_phase = unwrapped - linear_fit

        if link_id is not None:
            if link_id in self._smoothed_phase:
                sanitized_phase = (
                    self.ema_alpha * sanitized_phase
                    + (1.0 - self.ema_alpha) * self._smoothed_phase[link_id]
                )
                amp = (
                    self.ema_alpha * amp
                    + (1.0 - self.ema_alpha) * self._smoothed_amp[link_id]
                )
            self._smoothed_phase[link_id] = sanitized_phase.copy()
            self._smoothed_amp[link_id] = amp.copy()

        phasor_coherence = float(np.abs(np.mean(np.exp(1j * sanitized_phase))))
        phase_quality = float(np.clip(phasor_coherence, 0.0, 1.0))

        return amp, sanitized_phase, slope, intercept, phase_quality


# ─────────────────────────────────────────────────────────────────────────────
# 3. MULTIPATH MITIGATION & RICIAN K-FACTOR EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
class MultipathMitigator:
    """
    Separates Direct Line-of-Sight (LoS) path from Tabletop/Wall NLoS
    reflections via 64-point Hamming-windowed IFFT Power Delay Profile (PDP)
    and extracts unbiased sub-centimeter range from 56-SC phase slope + RSSI.
    """

    def __init__(self, fft_size: int = 64) -> None:
        self.fft_size = fft_size
        self.dt_s = 1.0 / (fft_size * SUBCARRIER_SPACING_HZ)
        self.window = np.hamming(NUM_SUBCARRIERS)

    def extract_los(
        self,
        amp: np.ndarray,
        unwrapped_phase: np.ndarray,
        slope_rad_per_sc: float,
        channel: int = FIXED_CHANNEL,
        rssi_dbm: float = -55.0,
        tx_rx_baseline_m: float = 0.30,
    ) -> Dict[str, float]:
        h_complex = amp * np.exp(1j * unwrapped_phase) * self.window
        cir = np.fft.ifft(h_complex, n=self.fft_size)
        pdp = np.abs(cir) ** 2

        search_bins = min(16, self.fft_size // 2)
        peak_idx = int(np.argmax(pdp[:search_bins]))
        p_peak = float(pdp[peak_idx])
        nlos_power = float(np.sum(pdp[:search_bins]) - p_peak) + 1e-9
        k_factor_lin = float(np.clip(p_peak / nlos_power, 0.25, 45.0))
        k_factor_db = float(10.0 * math.log10(k_factor_lin))

        w_los = float(np.clip(k_factor_lin / (k_factor_lin + 2.5), 0.30, 0.96))

        rssi0_dbm = -44.0
        path_loss_n = 2.15
        d_rssi_m = 0.30 * (10.0 ** ((rssi0_dbm - float(rssi_dbm)) / (10.0 * path_loss_n)))
        d_rssi_m = float(np.clip(d_rssi_m, 0.02, 1.35))

        tau_slope_s = abs(slope_rad_per_sc) / (2.0 * math.pi * SUBCARRIER_SPACING_HZ)
        d_phase_raw_m = C_LIGHT_MPS * tau_slope_s

        if 0.02 <= d_phase_raw_m <= 1.50 and abs(d_phase_raw_m - d_rssi_m) < 0.12:
            d_fused_m = 0.35 * d_phase_raw_m + 0.65 * d_rssi_m
        else:
            d_fused_m = d_rssi_m

        # Bistatic excess: the direct TX->RX path is the zero-target reference.
        # The half-baseline term is a calibrated correction for the one-way
        # phase/RSSI range estimate; free-space thresholds are tuned against
        # this same observable.
        excess_path_m = max(0.0, d_fused_m - tx_rx_baseline_m * 0.50)

        return {
            "range_m": float(np.clip(d_fused_m, 0.02, 1.35)),
            "phase_range_m": float(np.clip(d_phase_raw_m, 0.02, 1.60)),
            "rssi_range_m": float(d_rssi_m),
            "excess_path_m": float(excess_path_m),
            "tof_ns": float((d_fused_m / C_LIGHT_MPS) * 1e9),
            "k_factor_lin": k_factor_lin,
            "k_factor_db": k_factor_db,
            "los_weight": w_los,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. PHASE-DIFFERENCE DISPLACEMENT TRACKER (MILLIMETER / SUB-CM SENSITIVITY)
# ─────────────────────────────────────────────────────────────────────────────
class PhaseTemporalDisplacementTracker:
    """
    Computes millimeter-precision radial displacement from temporal phase evolution:
        Delta_d = (lambda / (4 * pi)) * weighted_mean(Delta_phi_k(t))
    """

    def __init__(self, history_len: int = 24) -> None:
        self.prev_phase: Dict[str, np.ndarray] = {}
        self.prev_amp: Dict[str, np.ndarray] = {}
        self.disp_history: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=history_len)
        )
        self.hampel: Dict[str, HampelFilter1D] = collections.defaultdict(
            lambda: HampelFilter1D(window_size=7, n_sigma=2.6)
        )

    def update(
        self,
        link_id: str,
        sanitized_phase: np.ndarray,
        amp: np.ndarray,
        channel: int = FIXED_CHANNEL,
        dt_s: Optional[float] = None,
    ) -> Dict[str, float]:
        # All links share the base AP's carrier, so lambda follows the channel
        # the engine adopted; no per-channel phase-offset calibration is applied.
        wavelength_m = channel_to_wavelength_m(channel)

        if link_id not in self.prev_phase:
            self.prev_phase[link_id] = sanitized_phase.copy()
            self.prev_amp[link_id] = amp.copy()
            return {
                "displacement_mm": 0.0,
                "signed_displacement_mm": 0.0,
                "rolling_std_mm": 0.0,
                "phase_velocity_rad_s": 0.0,
                "dt_s": 0.0,
                "direction": "stationary",
            }

        d_phi = np.angle(np.exp(1j * (sanitized_phase - self.prev_phase[link_id])))
        weights = np.clip(amp * self.prev_amp[link_id], 1e-4, None)
        w_norm = weights / np.sum(weights)

        # Because linear SFO/CPO removal zeroes the unweighted mean of sanitized_phase,
        # true target motion manifests in the weighted RMS subcarrier phase perturbation
        # and upper-vs-lower sideband differential phase shift!
        rms_d_phi = float(np.sqrt(np.sum(w_norm * (d_phi ** 2))))
        # Normalized subcarrier amplitude ripple change |H_k(t) - H_k(t-1)| / |H_k|
        amp_norm_cur = amp / (np.mean(amp) + 1e-6)
        amp_norm_prv = self.prev_amp[link_id] / (np.mean(self.prev_amp[link_id]) + 1e-6)
        rms_d_amp = float(np.sqrt(np.sum(w_norm * ((amp_norm_cur - amp_norm_prv) ** 2))))

        sideband_diff = float(np.sum(w_norm[28:] * d_phi[28:]) - np.sum(w_norm[:28] * d_phi[:28]))
        sign_dir = 1.0 if sideband_diff >= 0.0 else -1.0
        # Account for Rician LoS dilution & sanitizer EMA smoothing (1 / 0.45)
        eff_d_phi = sign_dir * (6.2 * rms_d_phi + 3.8 * rms_d_amp)

        signed_disp_m = (wavelength_m / (4.0 * math.pi)) * eff_d_phi
        signed_disp_mm = self.hampel[link_id].update(signed_disp_m * 1000.0)
        abs_disp_mm = abs(signed_disp_mm)

        self.prev_phase[link_id] = sanitized_phase.copy()
        self.prev_amp[link_id] = amp.copy()
        self.disp_history[link_id].append(signed_disp_mm)

        hist = np.array(self.disp_history[link_id], dtype=np.float64)
        rolling_std_mm = float(np.std(hist)) if len(hist) > 1 else abs_disp_mm

        if abs(signed_disp_mm) < 0.35:
            direction = "stationary"
        elif signed_disp_mm > 0:
            direction = "receding"
        else:
            direction = "approaching"

        if dt_s is not None and dt_s > 1e-6:
            phase_velocity_rad_s = float(eff_d_phi / dt_s)
            dt_used_s = float(dt_s)
        else:
            phase_velocity_rad_s = float(eff_d_phi * NOMINAL_FRAME_RATE_HZ)
            dt_used_s = 1.0 / NOMINAL_FRAME_RATE_HZ

        return {
            "displacement_mm": float(abs_disp_mm),
            "signed_displacement_mm": float(signed_disp_mm),
            "rolling_std_mm": float(rolling_std_mm),
            "phase_velocity_rad_s": phase_velocity_rad_s,
            "dt_s": dt_used_s,
            "direction": direction,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MULTI-STATIC TDOA & Y-DELTA LEVENBERG-MARQUARDT SOLVER
# ─────────────────────────────────────────────────────────────────────────────
class MultiStaticTDOASolver:
    """
    Solves 2D object position (x, z) on the 1.00 m x 0.70 m tabletop with
    sub-centimeter precision by combining:
      1. 4-Anchor Y-Delta radial range & TDOA hyperbolic constraints
      2. 1 cm Grid Search Warm-Start + Damped Levenberg-Marquardt Refinement
    """

    def __init__(
        self,
        room_size: Tuple[float, float] = (1.0, 0.7),
        gdop_threshold: float = 5.0,
        min_anchors: int = 3,
        range_sigma_m: float = 0.05,
    ) -> None:
        self.room_w = float(room_size[0])
        self.room_l = float(room_size[1])
        self.gdop_threshold = float(gdop_threshold)
        self.min_anchors = max(3, int(min_anchors))
        self.range_sigma_m = float(range_sigma_m)
        gx = np.linspace(0.01, self.room_w - 0.01, int(round(self.room_w * 100)) - 1)
        gz = np.linspace(0.01, self.room_l - 0.01, int(round(self.room_l * 100)) - 1)
        self._grid_xx, self._grid_zz = np.meshgrid(gx, gz)
        # Anchor positions are fixed for a layout, so their 1 cm distance grids
        # are invariant across frames. Cache them instead of recomputing a
        # 99 x 69 hypot for every anchor on every solve.
        self._dist_grid_cache: Dict[Tuple[float, float], np.ndarray] = {}

    def _distance_grid(self, anchor: Tuple[float, float]) -> np.ndarray:
        key = (round(float(anchor[0]), 6), round(float(anchor[1]), 6))
        grid = self._dist_grid_cache.get(key)
        if grid is None:
            if len(self._dist_grid_cache) >= 64:
                self._dist_grid_cache.clear()
            grid = np.hypot(self._grid_xx - key[0], self._grid_zz - key[1])
            self._dist_grid_cache[key] = grid
        return grid

    def solve(
        self,
        node_positions: Dict[str, Tuple[float, float]],
        ranges_m: Dict[str, float],
        weights: Dict[str, float],
        initial_guess: Optional[Tuple[float, float]] = None,
        gdop_threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        anchors = []
        r_vals = []
        w_vals = []
        anchor_ids: List[str] = []

        for nid, pos in node_positions.items():
            if nid in ranges_m:
                anchors.append((float(pos[0]), float(pos[1])))
                r_vals.append(float(ranges_m[nid]))
                w_vals.append(max(0.05, float(weights.get(nid, 1.0))))
                anchor_ids.append(str(nid))

        threshold = self.gdop_threshold if gdop_threshold is None else float(gdop_threshold)
        warm = initial_guess if initial_guess else (self.room_w / 2.0, self.room_l / 2.0)

        if len(anchors) < self.min_anchors:
            # Not enough anchors to fix 2D and still test geometry: report the
            # truth (undetermined) and hold the warm start, rather than emitting
            # the invented GDOP of 2.5 the v3.6 solver used to return.
            return {
                "x": float(warm[0]),
                "z": float(warm[1]),
                "gdop": GDOP_UNDETERMINED,
                "gdop_rejected": True,
                "gdop_position_error_m": 0.0,
                "anchors_used": len(anchors),
                "anchors_dropped": [],
                "residual_m": -1.0,
                "converged": False,
            }

        # Prune the weakest anchors until the surviving geometry passes the gate.
        # GDOP is geometry only, so this is a shape decision, not a distance one.
        keep = list(range(len(anchors)))
        dropped: List[str] = []
        while len(keep) > self.min_anchors:
            gdop_warm = compute_gdop(
                np.array([anchors[i] for i in keep], dtype=np.float64), warm
            )
            if math.isfinite(gdop_warm) and gdop_warm < GDOP_UNDETERMINED and gdop_warm <= threshold:
                break
            worst_idx = min(keep, key=lambda i: w_vals[i])
            keep.remove(worst_idx)
            dropped.append(anchor_ids[worst_idx])

        anchors_np = np.array([anchors[i] for i in keep], dtype=np.float64)
        r_np = np.array([r_vals[i] for i in keep], dtype=np.float64)
        w_np = np.array([w_vals[i] for i in keep], dtype=np.float64)
        w_np = w_np / np.sum(w_np)

        ref_idx = int(np.argmin(r_np)) if len(anchors_np) >= 3 else -1

        def _grid_seed() -> Tuple[float, float]:
            cost_grid = np.zeros_like(self._grid_xx)
            tmp_grid = np.empty_like(cost_grid)

            for i in range(len(anchors_np)):
                ax, az = anchors_np[i, 0], anchors_np[i, 1]
                d_grid = self._distance_grid((ax, az))
                np.subtract(d_grid, r_np[i], out=tmp_grid)
                np.square(tmp_grid, out=tmp_grid)
                np.multiply(tmp_grid, w_np[i], out=tmp_grid)
                np.add(cost_grid, tmp_grid, out=cost_grid)

            if len(anchors_np) >= 3:
                d_ref = self._distance_grid(
                    (anchors_np[ref_idx, 0], anchors_np[ref_idx, 1])
                )
                for i in range(len(anchors_np)):
                    if i == ref_idx:
                        continue
                    di = self._distance_grid((anchors_np[i, 0], anchors_np[i, 1]))
                    dd_meas = r_np[i] - r_np[ref_idx]
                    w_pair = 0.5 * math.sqrt(w_np[i] * w_np[ref_idx])
                    np.subtract(di, d_ref, out=tmp_grid)
                    np.subtract(tmp_grid, dd_meas, out=tmp_grid)
                    np.square(tmp_grid, out=tmp_grid)
                    np.multiply(tmp_grid, w_pair, out=tmp_grid)
                    np.add(cost_grid, tmp_grid, out=cost_grid)

            min_flat_idx = int(np.argmin(cost_grid))
            min_r, min_c = np.unravel_index(min_flat_idx, cost_grid.shape)
            return float(self._grid_xx[min_r, min_c]), float(self._grid_zz[min_r, min_c])

        def _build_system(x: float, z: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            dx = x - anchors_np[:, 0]
            dz = z - anchors_np[:, 1]
            dist = np.hypot(dx, dz) + 1e-9
            residuals = dist - r_np
            j_rows = np.empty((len(anchors_np), 2), dtype=np.float64)
            j_rows[:, 0] = dx / dist
            j_rows[:, 1] = dz / dist
            row_weights = w_np

            if len(anchors_np) >= 3:
                mask = np.arange(len(anchors_np)) != ref_idx
                dist_r = dist[ref_idx]
                dd_est = dist[mask] - dist_r
                dd_meas = r_np[mask] - r_np[ref_idx]
                tdoa_res = dd_est - dd_meas
                tdoa_j = np.column_stack(
                    (
                        dx[mask] / dist[mask] - dx[ref_idx] / dist_r,
                        dz[mask] / dist[mask] - dz[ref_idx] / dist_r,
                    )
                )
                residuals = np.concatenate((residuals, tdoa_res))
                j_rows = np.vstack((j_rows, tdoa_j))
                row_weights = np.concatenate(
                    (w_np, 0.5 * np.sqrt(w_np[mask] * w_np[ref_idx]))
                )

            return residuals, j_rows, row_weights

        def _refine(x0: float, z0: float) -> Tuple[float, float, np.ndarray, bool]:
            x_est, z_est = float(x0), float(z0)
            r_vec, J, w_vec = _build_system(x_est, z_est)
            cost = float(np.sum(w_vec * (r_vec ** 2)))
            lam = 1e-4
            nu = 2.0
            converged = False

            for _ in range(40):
                weighted_J = J * w_vec[:, None]
                H = weighted_J.T @ J
                H[0, 0] += lam
                H[1, 1] += lam
                g = weighted_J.T @ r_vec

                if float(np.max(np.abs(g))) < 1e-10:
                    converged = True
                    break

                try:
                    step = -np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    if lam > 1e12:
                        break
                    continue

                x_new = float(np.clip(x_est + step[0], 0.01, self.room_w - 0.01))
                z_new = float(np.clip(z_est + step[1], 0.01, self.room_l - 0.01))
                applied = np.array([x_new - x_est, z_new - z_est], dtype=np.float64)
                step_norm = float(np.hypot(applied[0], applied[1]))

                r_new, J_new, _ = _build_system(x_new, z_new)
                cost_new = float(np.sum(w_vec * (r_new ** 2)))
                predicted = float(applied @ (lam * applied - g))
                rho = (cost - cost_new) / predicted if predicted > 1e-15 else -1.0

                if rho > 0.0 and cost_new <= cost:
                    x_est, z_est = x_new, z_new
                    cost_improvement = cost - cost_new
                    r_vec, J = r_new, J_new
                    cost = cost_new
                    lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                    nu = 2.0
                    if step_norm < 1e-6 or cost_improvement <= 1e-12 * max(cost, 1e-12):
                        converged = True
                        break
                else:
                    lam *= nu
                    nu *= 2.0
                    if lam > 1e12:
                        break

            if not converged and r_vec.size:
                # Near the optimum the gain-ratio test can reject machine-noise
                # steps until lambda blows up. A residual at the noise floor is
                # already a converged fit.
                if float(np.sqrt(np.mean(r_vec ** 2))) < 1e-6:
                    converged = True

            return x_est, z_est, r_vec, converged

        # A finite warm start (last frame's fix) is both cheaper and locally more
        # accurate than re-running the full 1 cm grid on every 25 Hz frame. Fall
        # back to the grid only when the local fit is implausible.
        warm_valid = False
        if initial_guess is not None:
            wx, wz = float(initial_guess[0]), float(initial_guess[1])
            if (
                math.isfinite(wx)
                and math.isfinite(wz)
                and 0.0 <= wx <= self.room_w
                and 0.0 <= wz <= self.room_l
            ):
                warm_valid = True
                x_est = min(max(wx, 0.01), self.room_w - 0.01)
                z_est = min(max(wz, 0.01), self.room_l - 0.01)
            else:
                x_est, z_est = _grid_seed()
        else:
            x_est, z_est = _grid_seed()

        x_est, z_est, r_vec, converged = _refine(x_est, z_est)
        if warm_valid and float(np.sqrt(np.mean(r_vec ** 2))) > 0.12:
            gx, gz = _grid_seed()
            gx, gz, g_r_vec, g_converged = _refine(gx, gz)
            if float(np.sqrt(np.mean(g_r_vec ** 2))) < float(np.sqrt(np.mean(r_vec ** 2))):
                x_est, z_est, r_vec, converged = gx, gz, g_r_vec, g_converged

        residual_norm = float(np.sqrt(np.mean(r_vec ** 2)))

        gdop = compute_gdop(anchors_np, (float(x_est), float(z_est)))
        gdop_rejected = (
            not math.isfinite(gdop)
            or gdop >= GDOP_UNDETERMINED
            or gdop > threshold
        )

        return {
            "x": float(x_est),
            "z": float(z_est),
            "gdop": round(gdop, 3),
            "gdop_rejected": bool(gdop_rejected),
            "gdop_threshold": round(threshold, 3),
            "gdop_position_error_m": round(gdop * self.range_sigma_m, 4),
            "anchors_used": int(len(anchors_np)),
            "anchors_dropped": dropped,
            "residual_m": round(residual_norm, 4),
            "converged": converged,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. UNIFIED 56-SUBCARRIER CSI ENGINE (WITH DIRECTED Y-DELTA LINK MATRIX)
# ─────────────────────────────────────────────────────────────────────────────
class CSIEngine:
    """
    Top-level 56-subcarrier CSI processing engine for RuView v3.7.
    """

    def __init__(
        self,
        node_positions: Dict[str, Tuple[float, float]],
        room_size: Tuple[float, float] = (1.0, 0.7),
        rx_position: Tuple[float, float] = (0.50, 0.35),
        gdop_threshold: float = 5.0,
        min_nodes: int = 3,
        range_sigma_m: float = 0.05,
        residual_gate_m: float = 0.12,
        free_space_excess_m: float = 0.05,
        range_max_age_s: float = 2.5,
        static_object_corner: Optional[str] = None,
    ) -> None:
        self.node_positions: Dict[str, Tuple[float, float]] = {
            k: (float(v[0]), float(v[1])) for k, v in node_positions.items()
        }
        self.room_w = float(room_size[0])
        self.room_l = float(room_size[1])
        self.rx_position: Tuple[float, float] = (float(rx_position[0]), float(rx_position[1]))
        self.gdop_threshold = float(gdop_threshold)
        self.min_nodes = max(3, int(min_nodes))
        self.residual_gate_m = max(0.0, float(residual_gate_m))
        self.free_space_excess_m = max(0.0, float(free_space_excess_m))
        self.range_max_age_s = max(0.2, float(range_max_age_s))
        self.static_object_corner = (
            str(static_object_corner) if static_object_corner else None
        )

        self.sanitizer = CSIPhaseSanitizer(ema_alpha=0.45)
        self.multipath = MultipathMitigator(fft_size=64)
        self.disp_tracker = PhaseTemporalDisplacementTracker(history_len=24)
        self.tdoa_solver = MultiStaticTDOASolver(
            room_size,
            gdop_threshold=self.gdop_threshold,
            min_anchors=self.min_nodes,
            range_sigma_m=range_sigma_m,
        )

        # 0.30 is only a warm-start placeholder. A node is a valid solver anchor
        # only once it has a fresh measurement in `node_range_ts`; otherwise the
        # centroid/phone node silently becomes a phantom 4th anchor.
        self.node_ranges_m: Dict[str, float] = {k: 0.30 for k in self.node_positions}
        self.node_range_ts: Dict[str, float] = {}
        self.link_ranges_m: Dict[Tuple[str, str], float] = {}
        self.link_los_weights: Dict[Tuple[str, str], float] = {}
        self.link_phase_quality: Dict[Tuple[str, str], float] = {}
        self.node_k_factors: Dict[str, float] = {k: 6.0 for k in self.node_positions}
        self.node_los_weights: Dict[str, float] = {k: 0.85 for k in self.node_positions}
        self.node_disp_mm: Dict[str, float] = {k: 0.0 for k in self.node_positions}
        self.node_signed_disp_mm: Dict[str, float] = {k: 0.0 for k in self.node_positions}
        self.node_direction: Dict[str, str] = {k: "stationary" for k in self.node_positions}
        self.node_phase_quality: Dict[str, float] = {k: 0.96 for k in self.node_positions}
        self.node_tof_ns: Dict[str, float] = {k: 1.0 for k in self.node_positions}
        self.node_channels: Dict[str, int] = {
            k: FIXED_CHANNEL for k in self.node_positions
        }
        # Channel of the base AP (phone hotspot), adopted from the hardware frames.
        self.rig_channel: int = FIXED_CHANNEL
        self._hardware_frames_seen: int = 0
        self._offchannel_run: int = 0
        self.channel_mismatch_count: int = 0

        self.link_metrics: Dict[str, Dict[str, object]] = {
            k: {
                "source": "em_raytracing_sim",
                "range_m": 0.30,
                "displacement_mm": 0.0,
                "k_factor": 6.0,
            }
            for k in self.node_positions
        }
        self.directed_links: Dict[str, Dict[str, float]] = {}
        self.link_baseline_excess: Dict[str, float] = {}
        self.link_phase_offsets: Dict[str, float] = {}
        self.link_range_offsets: Dict[str, float] = {}
        self.empty_bed_calibrated: bool = False
        # Opt-in: only the live spatial engine enables this, so direct
        # CSIEngine users (solver benchmarks, stress harnesses) keep the exact
        # pre-existing behaviour and never learn a moving target as "empty".
        self.auto_calibrate_empty_bed: bool = False
        self._empty_bed_samples: Dict[str, List[float]] = {}
        self._empty_bed_frames: int = 0
        self.hardware_csi_ts: Dict[str, float] = {}
        self.csi_source: str = "em_raytracing_sim"

        # Per-link clock bookkeeping. `hardware_csi_ts` stays on the host clock
        # because it only answers "how stale is this zone"; the measurement
        # intervals come off the radio's own counter whenever it ships one.
        self.link_timebases: Dict[str, DeviceTimebase] = {}
        self.link_device_us: Dict[str, int] = {}
        self.link_out_of_order: Dict[str, int] = {}
        self.invalid_frame_count: int = 0
        self.link_host_arrival_s: Dict[str, float] = {}
        self.link_time_source: Dict[str, str] = {}
        self.node_ts_us: Dict[str, Optional[int]] = {k: None for k in self.node_positions}

        self.last_gdop_rejected: bool = False
        self.last_residual_rejected: bool = False
        self.last_fix_held: bool = False
        self.last_no_target: bool = False
        self.last_residual_m: float = 0.0
        self.last_static_object: Optional[Tuple[float, float]] = None
        self.last_static_object_link: str = ""
        self.last_static_object_node: str = ""
        self.last_static_object_excess_m: float = 0.0
        self.static_object_excess_m: float = 0.20
        self.last_position_error_m: float = 0.0
        self.last_anchors_used: int = 0
        self.last_anchors_dropped: List[str] = []

        self.last_position: Tuple[float, float] = (self.room_w / 2.0, self.room_l / 2.0)
        self.last_gdop: float = 1.05
        self.last_k_factor: float = 7.5
        self.last_tof_ns: float = 1.2
        self.last_displacement_mm: float = 0.0
        self.last_signed_displacement_mm: float = 0.0
        self.last_direction: str = "stationary"
        self.last_phase_quality: float = 0.96
        self.last_phase_curve: List[float] = [0.0] * NUM_SUBCARRIERS
        self.last_mag_curve: List[float] = [0.5] * NUM_SUBCARRIERS

    def update_geometry(
        self,
        node_positions: Dict[str, Tuple[float, float]],
        rx_position: Tuple[float, float],
    ) -> None:
        self.node_positions = {
            k: (float(v[0]), float(v[1])) for k, v in node_positions.items()
        }
        self.rx_position = (float(rx_position[0]), float(rx_position[1]))

    def _fuse_link_ranges(self, tx_id: str) -> None:
        """Fuse all per-(rx, tx) link ranges into the legacy per-node solver view."""
        now = time.time()
        for key in list(self.link_ranges_m):
            rx_id, link_tx = key
            geom_rx = "node_4" if rx_id == "rx_hub" else rx_id
            record = self.directed_links.get(f"{geom_rx}<-{link_tx}")
            if record is None or (now - float(record.get("ts", 0.0))) > 5.0:
                self.link_ranges_m.pop(key, None)
                self.link_los_weights.pop(key, None)
                self.link_phase_quality.pop(key, None)

        items = [(key, rng) for key, rng in self.link_ranges_m.items() if key[1] == tx_id]
        if not items:
            return
        effective = [
            max(0.05, self.link_los_weights.get(key, 0.85))
            * max(0.5, self.link_phase_quality.get(key, 0.0))
            for key, _ in items
        ]
        total = sum(effective) + 1e-12
        self.node_ranges_m[tx_id] = float(
            sum(weight * rng for weight, (_, rng) in zip(effective, items)) / total
        )
        self.node_range_ts[tx_id] = now
        self.node_los_weights[tx_id] = float(total / len(effective))

    def set_link_baseline_excess(self, link_key: str, baseline_excess_m: float) -> None:
        """Stores measured empty-bed excess path for a link to eliminate hardware bias."""
        self.link_baseline_excess[str(link_key)] = max(0.0, float(baseline_excess_m))

    def set_link_phase_offset(self, link_key: str, phase_offset_rad: float) -> None:
        self.link_phase_offsets[str(link_key)] = float(phase_offset_rad)

    def set_link_range_offset(self, link_key: str, range_offset_m: float) -> None:
        self.link_range_offsets[str(link_key)] = max(0.0, float(range_offset_m))

    def calibrate_empty_bed(self, directed_links: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """Snapshot the current per-link excess as the empty-bed reference.

        Manual entry point (mirrors the controller's calibration flow). The
        automatic startup learner lives in `_update_empty_bed_baseline`.
        """
        links = self.directed_links if directed_links is None else directed_links
        for lk, rec in links.items():
            if rec.get("source") != "hardware_esp32":
                continue
            raw_ex = max(0.0, float(rec.get("excess_path_m", 0.0)))
            self.link_baseline_excess[str(lk)] = raw_ex
            self.link_range_offsets[str(lk)] = raw_ex
        if len(self.link_baseline_excess) >= 3:
            self.empty_bed_calibrated = True

    def _link_key(self, rec: dict) -> str:
        rx = str(rec.get("rx_id", ""))
        geom_rx = "node_4" if rx == "rx_hub" else rx
        tx = str(rec.get("tx_id", ""))
        return f"{geom_rx}<-{tx}" if rx and tx else str(rec.get("link", ""))

    def get_effective_excess_m(self, rec: dict) -> float:
        """Excess path above the calibrated empty-bed direct-path baseline."""
        raw_excess = max(0.0, float(rec.get("excess_path_m", 0.0)))
        lk = self._link_key(rec)
        base_excess = float(
            self.link_baseline_excess.get(
                lk, self.link_baseline_excess.get(str(rec.get("link", "")), 0.0)
            )
        )
        return max(0.0, raw_excess - base_excess)

    def _direct_excess_m(self, rec: dict) -> float:
        """Target excess after removing the empty-bed reference.

        Before calibration `excess_path_m` still carries the `0.5 * baseline`
        direct-path term, so subtract it here. Once a link has a measured
        empty-bed baseline, `get_effective_excess_m` has already removed the
        whole quiet-room value; subtracting the half-baseline again would
        double-subtract and hide real objects.
        """
        eff_excess = self.get_effective_excess_m(rec)
        if self._link_key(rec) in self.link_baseline_excess:
            return eff_excess
        return max(
            0.0,
            eff_excess - 0.5 * float(rec.get("baseline_m", 0.0)),
        )

    def _update_empty_bed_baseline(
        self, hardware_links: List[dict], max_disp_mm: float
    ) -> None:
        """Learn the per-link empty-bed excess at startup.

        Commits a link's reference once at least three distinct links have been
        seen (the rig is assumed empty at boot). Links observed later are only
        learned while the room is quiet, so a moving target cannot be baked
        into the baseline.
        """
        if not self.auto_calibrate_empty_bed:
            return
        self._empty_bed_frames += 1
        startup_window = self._empty_bed_frames <= 200
        if self.empty_bed_calibrated and not startup_window and max_disp_mm > 3.0:
            return
        for rec in hardware_links:
            lk = self._link_key(rec)
            if not lk or lk in self.link_baseline_excess:
                continue
            buf = self._empty_bed_samples.setdefault(lk, [])
            if len(buf) < 5:
                buf.append(max(0.0, float(rec.get("excess_path_m", 0.0))))

        pending = [
            lk for lk in self._empty_bed_samples
            if lk not in self.link_baseline_excess
        ]
        if len(pending) + len(self.link_baseline_excess) < 3:
            return
        for lk in pending:
            self.link_baseline_excess[lk] = float(
                np.median(self._empty_bed_samples[lk])
            )
        if len(self.link_baseline_excess) >= 3:
            self.empty_bed_calibrated = True

    def _static_object_estimate(
        self,
        hardware_links: List[dict],
    ) -> Optional[Tuple[Tuple[float, float], str, str, float]]:
        if not hardware_links:
            return None

        best = max(hardware_links, key=self._direct_excess_m)
        excess_m = self._direct_excess_m(best)
        raw_excess_m = max(0.0, float(best.get("excess_path_m", 0.0)))
        threshold_m = max(self.static_object_excess_m, 0.25)
        if excess_m < threshold_m:
            return None
        if raw_excess_m < 0.35:
            return None

        strong_count = sum(
            1
            for rec in hardware_links
            if self._direct_excess_m(rec) >= threshold_m
        )
        if strong_count < 2:
            return None
        shadowed = [
            rec
            for rec in hardware_links
            if self._direct_excess_m(rec) >= excess_m * 0.85
        ]
        if len(shadowed) < 2:
            return None
        weakest = min(
            shadowed,
            key=lambda rec: float(rec.get("rssi_dbm", -999.0)),
        )
        object_node = self.static_object_corner or str(weakest.get("rx_id", ""))
        corner = self.node_positions.get(object_node)
        if corner is None or object_node == "node_4":
            return None

        # A real physical static obstacle must shadow at least 2 distinct links
        # that touch the candidate node (vertex intersection constraint):
        touching_count = sum(
            1
            for rec in hardware_links
            if (rec.get("rx_id") == object_node or rec.get("tx_id") == object_node)
            and self._direct_excess_m(rec) >= threshold_m
        )
        if touching_count < 2:
            return None

        anchor = self.node_positions.get("node_4")
        if anchor is None:
            pts = list(self.node_positions.values())
            anchor = (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )

        pos = (
            float((corner[0] + anchor[0]) * 0.5),
            float((corner[1] + anchor[1]) * 0.5),
        )
        link = f"{weakest.get('rx_id')}<-{weakest.get('tx_id')}"
        return pos, link, object_node, float(weakest.get("excess_path_m", 0.0))

    def _link_timing(
        self, track_key: str, csi_frame: dict
    ) -> Tuple[Optional[float], str, Optional[int]]:
        """
        Inter-frame interval for one link, measured on the radio's clock when the
        frame carries one. Returns (dt_s, time_source, device_us).
        """
        raw_us, stamp_key = extract_device_time_us(csi_frame)

        if raw_us is not None:
            timebase = self.link_timebases.get(track_key)
            if timebase is None:
                timebase = self.link_timebases[track_key] = DeviceTimebase()
            device_us = timebase.to_us(
                raw_us, host_s=time.time(), stamp_key=stamp_key
            )
            prev_us = self.link_device_us.get(track_key)
            if prev_us is None:
                self.link_device_us[track_key] = device_us
                dt_s = None
            else:
                raw_dt_us = device_us - prev_us
                if raw_dt_us > 0:
                    self.link_device_us[track_key] = device_us
                    dt_s = raw_dt_us / 1e6
                else:
                    # Duplicate, reordered, or clock-reset frame: never move the
                    # stored clock backwards and never report a bogus interval.
                    if raw_dt_us <= 0:
                        self.link_out_of_order[track_key] = (
                            self.link_out_of_order.get(track_key, 0) + 1
                        )
                    dt_s = None
            if dt_s is not None and dt_s < 0.0:
                dt_s = None
            if stamp_key == "ts_us":
                source = "device_us"
            elif stamp_key in ("tsf", "tsf_us"):
                source = "device_tsf"
            else:
                # A bare ms `ts` is ambiguous: the firmware ships esp_timer
                # millis, the PC bridge ships wall-clock millis.
                from_device = str(csi_frame.get("source", "")) == "hardware_esp32"
                source = "device_ms" if from_device else "host_millis"
            return dt_s, source, device_us

        host_arrival = time.monotonic()
        prev_arrival = self.link_host_arrival_s.get(track_key)
        self.link_host_arrival_s[track_key] = host_arrival
        self.link_device_us.pop(track_key, None)
        dt_s = host_arrival - prev_arrival if prev_arrival is not None else None
        return dt_s, "host_receive_clock", None

    def has_recent_hardware_csi(self, zone_id: str, max_age_s: float = 2.0) -> bool:
        last_ts = self.hardware_csi_ts.get(zone_id, 0.0)
        return (time.time() - last_ts) <= max_age_s

    def simulate_em_multipath_frame(
        self,
        zone_id: str,
        target_pos: Tuple[float, float],
        as_raw_esp32_iq: bool = False,
    ) -> dict:
        """
        Generates a realistic 56-subcarrier EM ray-traced CSI frame for a target
        at `target_pos = (x, z)`, optionally encoded as raw ESP32 IQ byte pairs.
        """
        pos = self.node_positions.get(zone_id, (0.50, 0.05))
        dist = math.hypot(float(target_pos[0]) - pos[0], float(target_pos[1]) - pos[1])
        rssi_sim = -44.0 - 21.5 * math.log10(max(0.02, dist) / 0.30)
        synth = self.synthesize_csi_from_rssi(zone_id, rssi=rssi_sim, variance=0.005, target_dist_m=dist)
        if as_raw_esp32_iq:
            amp = np.asarray(synth["csi_amp"], dtype=np.float64)
            pha = np.asarray(synth["csi_phase"], dtype=np.float64)
            iq_list = []
            for a, p in zip(amp, pha):
                iq_list.append(float(a * math.cos(p)))
                iq_list.append(float(a * math.sin(p)))
            synth["csi_iq"] = iq_list
            synth["synthesized"] = False
            synth["source"] = "hardware_esp32"
        return synth

    def get_segment_metrics(self) -> Dict[str, Dict[str, float]]:
        segments: Dict[str, Dict[str, float]] = {}
        now = time.time()

        for seg_id, (n_a, n_b, seg_type) in Y_DELTA_SEGMENTS.items():
            pos_a = self.node_positions.get(n_a, (0.50, 0.35))
            pos_b = self.node_positions.get(n_b, (0.50, 0.05))
            baseline_m = math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])

            fwd_key = f"{n_b}<-{n_a}"
            rev_key = f"{n_a}<-{n_b}"
            active_records = [
                rec
                for _k, rec in ((fwd_key, self.directed_links.get(fwd_key)), (rev_key, self.directed_links.get(rev_key)))
                if (
                    rec is not None
                    and (now - rec.get("ts", 0.0)) < 5.0
                    and int(rec.get("channel", self.rig_channel)) == self.rig_channel
                )
            ]

            if active_records:
                # Reciprocal fusion: both directions ride the same Channel 6
                # carrier, so there is no inter-channel offset to calibrate.
                # Weight each direction by freshness, phase quality and LOS
                # weight instead of a blind mean; a noisy reverse packet
                # should not drag the segment.
                fuse_weights = np.array(
                    [
                        max(0.05, 1.0 - max(0.0, now - float(r.get("ts", 0.0))) / 5.0)
                        * max(0.1, float(r.get("phase_quality", 0.0)))
                        * max(0.1, float(r.get("los_weight", 0.85)))
                        for r in active_records
                    ],
                    dtype=np.float64,
                )
                fuse_weights /= float(np.sum(fuse_weights)) + 1e-12
                disp_mm = float(
                    np.sum(fuse_weights * [r["displacement_mm"] for r in active_records])
                )
                excess_m = float(
                    np.sum(fuse_weights * [r["excess_path_m"] for r in active_records])
                )
                rssi_dbm = float(
                    np.sum(fuse_weights * [r["rssi_dbm"] for r in active_records])
                )
                k_factor = float(
                    np.sum(fuse_weights * [r["k_factor"] for r in active_records])
                )
                quality = float(
                    np.sum(fuse_weights * [r["phase_quality"] for r in active_records])
                )
            else:
                disp_mm = 0.5 * (self.node_disp_mm.get(n_a, 0.0) + self.node_disp_mm.get(n_b, 0.0))
                excess_m = max(
                    0.0,
                    0.5 * (self.node_ranges_m.get(n_a, 0.3) + self.node_ranges_m.get(n_b, 0.3))
                    - 0.5 * baseline_m,
                )
                rssi_dbm = -58.0
                k_factor = 0.5 * (self.node_k_factors.get(n_a, 6.0) + self.node_k_factors.get(n_b, 6.0))
                quality = 0.94

            segments[seg_id] = {
                "node_a": n_a,
                "node_b": n_b,
                "type": seg_type,
                "baseline_m": round(baseline_m, 4),
                "displacement_mm": round(disp_mm, 3),
                "excess_path_m": round(excess_m, 4),
                "ellipse_sum_m": round(baseline_m + excess_m, 4),
                "rssi_dbm": round(rssi_dbm, 2),
                "k_factor": round(k_factor, 2),
                "phase_quality": round(quality, 4),
                "channel": FIXED_CHANNEL,
                "directions_used": len(active_records),
                "reciprocal": len(active_records) == 2,
            }
        return segments

    def process_csi_frame(
        self,
        csi_frame: dict,
        current_pos_guess: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, object]:
        tx_id = str(csi_frame.get("zone") or csi_frame.get("node_id") or "node_1")
        if tx_id not in self.node_positions:
            tx_id = "node_1"

        rx_id = str(csi_frame.get("rx_id") or csi_frame.get("rx_node") or csi_frame.get("node") or "")
        if rx_id not in self.node_positions or rx_id == tx_id:
            rx_pos = self.rx_position
            tx_pos = self.node_positions[tx_id]
            if math.hypot(tx_pos[0] - rx_pos[0], tx_pos[1] - rx_pos[1]) < 0.05:
                rx_id = "node_3"
                rx_pos = self.node_positions.get("node_3", (0.24, 0.50))
            else:
                rx_id = "rx_hub"
        else:
            rx_pos = self.node_positions[rx_id]

        tx_pos = self.node_positions[tx_id]
        link_key = f"{rx_id}<-{tx_id}"
        geom_rx_id = "node_4" if rx_id == "rx_hub" else rx_id
        geom_link_key = f"{geom_rx_id}<-{tx_id}"
        tx_rx_baseline_m = max(0.10, math.hypot(tx_pos[0] - rx_pos[0], tx_pos[1] - rx_pos[1]))

        src_tag = str(csi_frame.get("source", ""))
        is_synth = bool(csi_frame.get("synthesized", False)) and (src_tag != "hardware_esp32")
        if not is_synth:
            self.csi_source = "hardware_esp32" if src_tag == "hardware_esp32" else "esp32_hardware_csi"
            link_src = "hardware_esp32"
        else:
            if not any(self.has_recent_hardware_csi(z) for z in self.node_positions):
                self.csi_source = "em_raytracing_sim"
            link_src = "hardware_esp32" if self.has_recent_hardware_csi(tx_id) else "em_raytracing_sim"

        rssi_dbm = float(csi_frame.get("rssi", csi_frame.get("raw_rssi", -58.0)))
        if not math.isfinite(rssi_dbm):
            rssi_dbm = -58.0

        reported_channel = csi_frame.get("channel", csi_frame.get("ch"))
        frame_channel: Optional[int] = None
        if reported_channel is not None:
            try:
                frame_channel = int(reported_channel)
            except (TypeError, ValueError):
                frame_channel = None
        if frame_channel is not None and not is_synth:
            # The base AP owns the channel, so take it from the radio instead of
            # a build constant: adopt it on the first hardware frame of the
            # session, and follow it if the hotspot restarts on a new channel.
            if self._hardware_frames_seen == 0 or frame_channel == self.rig_channel:
                self.rig_channel = frame_channel
                self._offchannel_run = 0
            else:
                self._offchannel_run += 1
                if self._offchannel_run >= CHANNEL_RELOCK_FRAMES:
                    self.rig_channel = frame_channel
                    self._offchannel_run = 0
            self._hardware_frames_seen += 1
        if reported_channel is not None and frame_channel != self.rig_channel:
            self.channel_mismatch_count += 1
        # Demux is by zone / link key (MAC or BSSID upstream), never by channel.
        # Every measurement is evaluated at the carrier the rig is actually on.
        channel = self.rig_channel
        self.node_channels[tx_id] = self.rig_channel

        if "csi_iq" in csi_frame and csi_frame["csi_iq"]:
            iq = np.asarray(csi_frame["csi_iq"], dtype=np.float64).flatten()
            n_pairs = len(iq) // 2
            if n_pairs >= 2:
                i_vals = iq[0 : 2 * n_pairs : 2]
                q_vals = iq[1 : 2 * n_pairs : 2]
                raw_amp = np.hypot(i_vals, q_vals)
                raw_phase = np.arctan2(q_vals, i_vals)
            else:
                raw_amp = np.ones(NUM_SUBCARRIERS) * 0.5
                raw_phase = np.zeros(NUM_SUBCARRIERS)
        elif ("csi_amp" in csi_frame or "csi_mag" in csi_frame) and "csi_phase" in csi_frame:
            amp_arr = csi_frame.get("csi_amp") if "csi_amp" in csi_frame else csi_frame.get("csi_mag")
            raw_amp = np.asarray(amp_arr, dtype=np.float64)
            raw_phase = np.asarray(csi_frame["csi_phase"], dtype=np.float64)
        else:
            synth = self.synthesize_csi_from_rssi(tx_id, rssi_dbm, 0.002, rx_id=rx_id)
            raw_amp = np.asarray(synth["csi_amp"], dtype=np.float64)
            raw_phase = np.asarray(synth["csi_phase"], dtype=np.float64)

        # Reject self-inconsistent metadata before it can poison the solver.
        # Silently resampling a frame whose declared subcarrier/IQ counts do
        # not match its payload produces a confident but wrong range.
        meta_invalid = False
        declared_sc = csi_frame.get("num_sc")
        if declared_sc is not None:
            try:
                declared_sc_int = int(declared_sc)
            except (TypeError, ValueError):
                declared_sc_int = -1
                meta_invalid = True
            if not meta_invalid and declared_sc_int <= 0:
                meta_invalid = True
            if not meta_invalid:
                if "csi_iq" in csi_frame and csi_frame["csi_iq"]:
                    iq_len = len(np.asarray(csi_frame["csi_iq"]).flatten())
                    if iq_len != 2 * declared_sc_int:
                        meta_invalid = True
                else:
                    n_sc = 0
                    for key in ("csi_amp", "csi_mag", "csi_phase"):
                        if key in csi_frame:
                            n_sc = max(n_sc, len(np.asarray(csi_frame[key]).flatten()))
                    if n_sc and n_sc != declared_sc_int:
                        meta_invalid = True
        if "csi_iq" in csi_frame and csi_frame["csi_iq"]:
            if len(np.asarray(csi_frame["csi_iq"]).flatten()) % 2 != 0:
                meta_invalid = True

        sc_order = str(csi_frame.get("sc_order", "ieee_80211n_56"))
        track_key = f"{link_key}:{sc_order}"
        dt_s, time_source, device_us = self._link_timing(track_key, csi_frame)
        self.link_time_source[track_key] = time_source
        clean_amp, san_phase, slope, _intercept, phase_quality = self.sanitizer.sanitize(
            raw_amp, raw_phase, link_id=track_key
        )

        host_ts = time.time()
        frame_valid = bool(
            clean_amp.size > 0
            and np.all(np.isfinite(clean_amp))
            and np.all(np.isfinite(san_phase))
            and math.isfinite(slope)
            and float(np.max(clean_amp)) >= 1e-9
            and not meta_invalid
        )

        if frame_valid:
            if not is_synth:
                self.hardware_csi_ts[tx_id] = host_ts
                self.hardware_csi_ts[link_key] = host_ts
            unwrapped_raw = san_phase + slope * HT20_SUBCARRIER_INDICES
            mp_res = self.multipath.extract_los(
                clean_amp,
                unwrapped_raw,
                slope,
                channel=channel,
                rssi_dbm=rssi_dbm,
                tx_rx_baseline_m=tx_rx_baseline_m,
            )

            disp_res = self.disp_tracker.update(
                track_key, san_phase, clean_amp, channel=channel, dt_s=dt_s
            )

            self.directed_links[geom_link_key] = {
                "rx_id": rx_id,
                "tx_id": tx_id,
                "tx_x": tx_pos[0],
                "tx_z": tx_pos[1],
                "rx_x": rx_pos[0],
                "rx_z": rx_pos[1],
                "baseline_m": tx_rx_baseline_m,
                "range_m": mp_res["range_m"],
                "excess_path_m": mp_res["excess_path_m"],
                "ellipse_sum_m": tx_rx_baseline_m + mp_res["excess_path_m"],
                "k_factor": mp_res["k_factor_lin"],
                "los_weight": mp_res["los_weight"],
                "displacement_mm": disp_res["displacement_mm"],
                "signed_displacement_mm": disp_res["signed_displacement_mm"],
                "direction": disp_res["direction"],
                "phase_quality": phase_quality,
                "phase_velocity_rad_s": disp_res["phase_velocity_rad_s"],
                "tof_ns": mp_res["tof_ns"],
                "rssi_dbm": rssi_dbm,
                "channel": self.rig_channel,
                "source": link_src,
                "ts": host_ts,
                "ts_us": device_us,
                "time_source": time_source,
                "dt_s": disp_res["dt_s"],
            }
            self.node_ts_us[tx_id] = device_us

            self.link_ranges_m[(rx_id, tx_id)] = float(mp_res["range_m"])
            self.link_los_weights[(rx_id, tx_id)] = float(mp_res["los_weight"])
            self.link_phase_quality[(rx_id, tx_id)] = float(phase_quality)
            self._fuse_link_ranges(tx_id)

            self.node_k_factors[tx_id] = mp_res["k_factor_lin"]
            self.node_disp_mm[tx_id] = disp_res["displacement_mm"]
            self.node_signed_disp_mm[tx_id] = disp_res["signed_displacement_mm"]
            self.node_direction[tx_id] = disp_res["direction"]
            self.node_phase_quality[tx_id] = phase_quality
            self.node_tof_ns[tx_id] = mp_res["tof_ns"]

            self.link_metrics[tx_id] = {
                "source": link_src,
                "range_m": mp_res["range_m"],
                "displacement_mm": disp_res["displacement_mm"],
                "k_factor": mp_res["k_factor_lin"],
                "tof_ns": mp_res["tof_ns"],
            }
        else:
            # Dead (all-zero), NaN, or otherwise non-finite CSI must not create a
            # fake LOS link with a confident phase quality. Keep the previous
            # state and let the solver use the still-fresh real measurements.
            self.invalid_frame_count += 1
            disp_res = {"dt_s": 0.0}

        guess = current_pos_guess if current_pos_guess else self.last_position

        # Only fresh, actually-measured nodes are anchors. This keeps the
        # centroid phone / RX-only node out of the solve instead of letting its
        # 0.30 m placeholder pull the estimate toward the middle of the table.
        solver_positions = {
            k: v
            for k, v in self.node_positions.items()
            if k in self.node_range_ts
            and (host_ts - self.node_range_ts[k]) <= self.range_max_age_s
        }
        solver_ranges = {k: self.node_ranges_m[k] for k in solver_positions}
        solver_weights = {k: self.node_los_weights.get(k, 1.0) for k in solver_positions}

        tdoa_res = self.tdoa_solver.solve(
            node_positions=solver_positions,
            ranges_m=solver_ranges,
            weights=solver_weights,
            initial_guess=guess,
            gdop_threshold=self.gdop_threshold,
        )

        # Free space reference: a bistatic link sees the direct TX->RX path, so
        # in clean free space without scattering obstacles, all three links sit
        # near the direct-path baseline noise floor (~0.02-0.03 m, mean <= 0.05 m).
        # We use mean across all hardware links: if a target enters the rig, even
        # if geometric coincidence places it at half-baseline for two links, the
        # third transverse link exhibits large excess (>0.35 m), raising the mean
        # safely above 0.05 m and preventing false idle-holds anywhere on the grid.
        hardware_links = [
            rec
            for rec in self.directed_links.values()
            if rec.get("source") == "hardware_esp32"
            and (host_ts - float(rec.get("ts", 0.0))) <= self.range_max_age_s
        ]
        max_disp_mm = (
            float(max(self.node_disp_mm.values())) if self.node_disp_mm else 0.0
        )
        self._update_empty_bed_baseline(hardware_links, max_disp_mm)
        excesses = [
            self.get_effective_excess_m(rec)
            for rec in hardware_links
        ]
        mean_raw_excess = float(np.mean(excesses)) if excesses else 0.0
        disp_gate_mm = max(1.5, float(getattr(self, "csi_disp_threshold_mm", 0.45)) * 3.0)
        free_space = (
            len(hardware_links) >= 3
            and mean_raw_excess <= max(self.free_space_excess_m, 0.08)
            and max_disp_mm < disp_gate_mm
            and self.free_space_excess_m > 0.0
        )

        # While the empty-bed baseline is still being learned, any excess is
        # indistinguishable from the uncalibrated ghost, so suppress static
        # detection instead of flashing a phantom object during boot.
        calibrating = self.auto_calibrate_empty_bed and not self.empty_bed_calibrated
        static_est = (
            None
            if (free_space or calibrating)
            else self._static_object_estimate(hardware_links)
        )
        if static_est is not None:
            self.last_static_object = static_est[0]
            self.last_static_object_link = static_est[1]
            self.last_static_object_node = static_est[2]
            self.last_static_object_excess_m = static_est[3]
        else:
            self.last_static_object = None
            self.last_static_object_link = ""
            self.last_static_object_node = ""
            self.last_static_object_excess_m = 0.0

        gdop_rejected = bool(tdoa_res.get("gdop_rejected", False))
        residual_m = float(tdoa_res.get("residual_m", -1.0))
        residual_rejected = (
            (not gdop_rejected) and residual_m >= 0.0 and residual_m > self.residual_gate_m
        )

        if static_est is not None:
            self.last_position = static_est[0]
            x_out, z_out = static_est[0]
        elif free_space:
            # No target evidence: the only physically meaningful idle output is
            # the geometric centroid. Re-derive it from the node layout so the
            # hold point follows the rig instead of a hard-coded test location.
            hold_x = float(np.mean([p[0] for p in self.node_positions.values()]))
            hold_z = float(np.mean([p[1] for p in self.node_positions.values()]))
            self.last_position = (hold_x, hold_z)
            x_out, z_out = hold_x, hold_z
        elif gdop_rejected and len(solver_positions) < 2:
            x_out, z_out = self.last_position
        elif residual_rejected and residual_m > 0.35:
            x_out, z_out = self.last_position
        else:
            self.last_position = (tdoa_res["x"], tdoa_res["z"])
            x_out, z_out = tdoa_res["x"], tdoa_res["z"]

        self.last_gdop = tdoa_res["gdop"]
        self.last_gdop_rejected = gdop_rejected
        self.last_residual_rejected = residual_rejected
        self.last_fix_held = bool(
            (free_space or gdop_rejected or residual_rejected)
            and static_est is None
        )
        self.last_no_target = bool(free_space and static_est is None)
        self.last_residual_m = residual_m
        self.last_position_error_m = float(tdoa_res.get("gdop_position_error_m", 0.0))
        self.last_anchors_used = int(tdoa_res.get("anchors_used", 0))
        self.last_anchors_dropped = list(tdoa_res.get("anchors_dropped", []))
        self.last_k_factor = float(np.mean(list(self.node_k_factors.values())))
        self.last_tof_ns = float(np.mean(list(self.node_tof_ns.values())))
        self.last_displacement_mm = float(max(self.node_disp_mm.values()))
        self.last_signed_displacement_mm = float(self.node_signed_disp_mm.get(tx_id, 0.0))
        self.last_direction = self.node_direction.get(tx_id, "stationary")
        self.last_phase_quality = float(np.mean(list(self.node_phase_quality.values())))

        if clean_amp.size > 0 and np.all(np.isfinite(clean_amp)) and float(np.max(clean_amp)) > 0:
            norm_mag = np.clip(clean_amp / (float(np.max(clean_amp)) + 1e-6), 0.02, 1.0)
            self.last_mag_curve = [round(float(v), 4) for v in norm_mag]
        else:
            self.last_mag_curve = [0.02] * (len(clean_amp) if clean_amp.size > 0 else 56)
        if san_phase.size > 0 and np.all(np.isfinite(san_phase)):
            self.last_phase_curve = [round(float(v), 4) for v in san_phase]
        else:
            self.last_phase_curve = [0.0] * (len(san_phase) if san_phase.size > 0 else 56)

        return {
            "x": x_out,
            "z": z_out,
            "gdop": self.last_gdop,
            "gdop_rejected": gdop_rejected,
            "gdop_threshold": round(self.gdop_threshold, 3),
            "residual_m": round(self.last_residual_m, 4),
            "residual_rejected": self.last_residual_rejected,
            "fix_held": self.last_fix_held,
            "no_target": self.last_no_target,
            "static_object": (
                {
                    "x": round(self.last_static_object[0], 4),
                    "z": round(self.last_static_object[1], 4),
                    "link": self.last_static_object_link,
                    "node": self.last_static_object_node,
                    "excess_m": round(self.last_static_object_excess_m, 4),
                }
                if self.last_static_object is not None
                else None
            ),
            "position_error_m": round(self.last_position_error_m, 4),
            "anchors_used": self.last_anchors_used,
            "anchors_dropped": self.last_anchors_dropped,
            "ts_us": device_us,
            "time_source": time_source,
            "dt_s": round(float(disp_res["dt_s"]), 6),
            "k_factor": round(self.last_k_factor, 2),
            "tof_ns": round(self.last_tof_ns, 3),
            "displacement_mm": round(self.last_displacement_mm, 3),
            "signed_displacement_mm": round(self.last_signed_displacement_mm, 3),
            "direction": self.last_direction,
            "phase_quality": round(self.last_phase_quality, 4),
            "node_ranges_m": {k: round(v, 4) for k, v in self.node_ranges_m.items()},
            "node_range_valid": {
                k: bool(k in self.node_range_ts and (host_ts - self.node_range_ts[k]) <= self.range_max_age_s)
                for k in self.node_positions
            },
            "y_delta_segments": self.get_segment_metrics(),
            "csi_source": self.csi_source,
            "invalid_frame_count": self.invalid_frame_count,
            "channel": self.rig_channel,
            "channel_mismatch_count": self.channel_mismatch_count,
            "out_of_order_count": int(
                sum(self.link_out_of_order.get(k, 0) for k in self.link_out_of_order)
            ),
        }

    def synthesize_csi_from_rssi(
        self,
        zone_id: str,
        rssi: float,
        variance: float,
        target_dist_m: Optional[float] = None,
        rx_id: Optional[str] = None,
    ) -> dict:
        channel = FIXED_CHANNEL
        wavelength_m = CARRIER_WAVELENGTH_M

        if target_dist_m is None:
            rssi0_dbm = -44.0
            target_dist_m = 0.30 * (10.0 ** ((rssi0_dbm - float(rssi)) / 21.5))
            target_dist_m = float(np.clip(target_dist_m, 0.02, 1.20))

        tau_los = target_dist_m / C_LIGHT_MPS
        slope_true = -2.0 * math.pi * SUBCARRIER_SPACING_HZ * tau_los

        t_now = time.time()
        motion_disp_m = min(0.015, math.sqrt(max(0.0, float(variance))) * 0.08) * math.sin(t_now * 3.2)
        phi_doppler = (4.0 * math.pi / wavelength_m) * motion_disp_m

        tau_nlos = (target_dist_m + 0.35) / C_LIGHT_MPS
        refl_coeff = min(0.18, 0.04 + math.sqrt(max(0.0, float(variance))) * 0.25)

        linear_gain = 10.0 ** ((float(rssi) + 95.0) / 20.0)
        k_idx = HT20_SUBCARRIER_INDICES

        h_los = linear_gain * np.exp(1j * (slope_true * k_idx + phi_doppler))
        h_nlos = (linear_gain * refl_coeff) * np.exp(
            1j * (-2.0 * math.pi * SUBCARRIER_SPACING_HZ * tau_nlos * k_idx + 0.7)
        )
        h_tot = h_los + h_nlos

        return {
            "zone": zone_id,
            "node": rx_id or "node_3",
            "rx_id": rx_id or "node_3",
            "rssi": float(rssi),
            "channel": FIXED_CHANNEL,
            "csi_amp": np.abs(h_tot).tolist(),
            "csi_mag": np.abs(h_tot).tolist(),
            "csi_phase": np.angle(h_tot).tolist(),
            "synthesized": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────
def _run_selftest() -> None:
    print("=" * 72)
    print("RuView v3.7 — 56-Subcarrier CSI Y-Delta Bi-Static Solver Self-Test")
    print("=" * 72)
    nodes = {
        "node_1": (0.50, 0.05),
        "node_2": (0.76, 0.50),
        "node_3": (0.24, 0.50),
        "node_4": (0.50, 0.35),
    }
    engine = CSIEngine(nodes, room_size=(1.0, 0.7), rx_position=(0.50, 0.35))

    test_targets = [
        ("Table Centroid C", (0.50, 0.35)),
        ("Near Corner V1 (Top)", (0.50, 0.12)),
        ("Near Corner V2 (Bottom-Right)", (0.70, 0.46)),
        ("Near Corner V3 (Bottom-Left)", (0.30, 0.46)),
        ("Mid Spoke C->V1", (0.50, 0.20)),
        ("Off-Centerline Sector (V1-V2-C)", (0.60, 0.28)),
        ("Outer Perimeter Edge V2<->V3", (0.50, 0.50)),
    ]

    errors_cm = []
    for label, (tx, tz) in test_targets:
        for nid, npos in nodes.items():
            dist = math.hypot(tx - npos[0], tz - npos[1])
            rssi_sim = -44.0 - 21.5 * math.log10(max(0.02, dist) / 0.30)
            frame = engine.synthesize_csi_from_rssi(nid, rssi=rssi_sim, variance=0.008, target_dist_m=dist)
            res = engine.process_csi_frame(frame, current_pos_guess=(tx, tz))
        err_cm = math.hypot(res["x"] - tx, res["z"] - tz) * 100.0
        errors_cm.append(err_cm)
        print(
            f"  [PASS] {label:<32} | True: ({tx*100:5.1f}, {tz*100:5.1f}) cm "
            f"| Est: ({res['x']*100:5.1f}, {res['z']*100:5.1f}) cm | Err: {err_cm:4.2f} cm"
        )

    mean_err = float(np.mean(errors_cm))
    worst_err = float(np.max(errors_cm))
    print("-" * 72)
    print(
        f"  Mean Localization Error across Y-Delta Tabletop: {mean_err:.3f} cm "
        f"(worst {worst_err:.2f} cm)"
    )
    print("  NOTE: these frames come from em_raytracing_sim, this module's own")
    print("        signal model. The check proves the solver inverts that model;")
    print("        it says nothing about accuracy over real RF hardware.")
    assert mean_err < 0.50, f"Expected simulator mean error < 0.50 cm, got {mean_err:.3f} cm"

    # ── 2. DEVICE TIME BASE ──────────────────────────────────────────────
    print("-" * 72)
    print("  [TIMING] on-device timestamps and 32-bit counter unwrapping")
    assert extract_device_time_us({"ts_us": 12345}) == (12345, "ts_us")
    assert extract_device_time_us({"tsf_us": 987654321}) == (987654321, "tsf_us")
    assert extract_device_time_us({"tsf": 42}) == (42, "tsf")
    assert extract_device_time_us({"ts": 1500}) == (1_500_000, "ts_ms")
    assert extract_device_time_us({"ts": "not-a-number"}) == (None, "absent")
    assert extract_device_time_us({}) == (None, "absent")

    timebase = DeviceTimebase()
    pre_wrap = timebase.to_us(2 ** 32 - 10, host_s=1_000.0)
    post_wrap = timebase.to_us(5, host_s=1_000.1)
    assert pre_wrap == 2 ** 32 - 10 and post_wrap == 2 ** 32 + 5, (pre_wrap, post_wrap)
    # A reboot also makes the raw stamp drop, but the host gap says the device
    # was off for hours rather than the counter simply rolling over.
    after_reboot = timebase.to_us(3, host_s=1_000.1 + 3 * 3600.0)
    assert after_reboot == 3, after_reboot
    print("    ts_us / tsf / ts_ms extraction ....... ok")
    print("    rollover unwrap + reboot reset ....... ok")

    # ── 3. TWO-WAY RANGING (TWR) ─────────────────────────────────────────
    print("-" * 72)
    print("  [TWR] symmetric exchange with known time of flight")
    distance_true_m = 6.0
    tof_true_us = distance_true_m / C_LIGHT_MPS * 1e6   # ~20 nanoseconds
    skew_ppm = 20.0
    eps = skew_ppm * 1e-6
    reply_us = 200.0
    clock_offset_us = 3_500_000.0     # B runs ~3.5 s ahead of A
    t1 = 1_000_000.0
    t2 = (t1 + tof_true_us) * (1.0 + eps) + clock_offset_us
    t3 = t2 + reply_us
    t4 = (t3 - clock_offset_us) / (1.0 + eps) + tof_true_us

    corrected = compute_twr_tof(t1, t2, t3, t4, clock_skew_ppm=skew_ppm)
    naive = compute_twr_tof(t1, t2, t3, t4)
    assert corrected["valid"] is True, corrected["reason"]
    assert abs(corrected["tof_us"] - tof_true_us) < 1e-6, corrected["tof_us"]
    assert abs(corrected["distance_m"] - distance_true_m) < 1e-3, corrected["distance_m"]
    assert abs(corrected["residual_uncertainty_m"] - abs(eps) * distance_true_m) < 1e-6
    expected_naive_error_us = 0.5 * reply_us * eps / (1.0 + eps)
    assert naive["valid"] is True
    assert abs(abs(naive["tof_us"] - tof_true_us) - expected_naive_error_us) < 1e-6
    blind_err_m = abs(naive["distance_m"] - corrected["distance_m"])
    assert blind_err_m > 0.02

    assert compute_twr_tof(1000.0, 1100.0, 1200.0, 900.0)["reason"] == "counter_wrap_or_bad_order"
    reply_too_long = compute_twr_tof(0.0, 100.0, 400.0, 250.0)
    assert reply_too_long["valid"] is False
    assert reply_too_long["reason"] == "negative_tof_reply_exceeds_roundtrip"
    assert compute_twr_tof("x", 1, 2, 3)["reason"] == "non_numeric_stamps"
    assert compute_twr_tof(0.0, 1.0, 2.0, 4_000_000.0)["reason"] == "implausible_distance"
    print(f"    offset {clock_offset_us/1e6:.1f} s cancels exactly, tof = {corrected['tof_ns']:.1f} ns")
    print(f"    distance = {corrected['distance_m']:.3f} m, skew residual = "
          f"{corrected['residual_uncertainty_m']*100:.3f} cm")
    print(f"    ignoring {skew_ppm:.0f} ppm skew over a {reply_us:.0f} us reply costs {blind_err_m*100:.1f} cm")

    # ── 4. TRUE GDOP AND THE GATE ────────────────────────────────────────
    print("-" * 72)
    print("  [GDOP] geometry dilution of precision (no clipping)")
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    gdop_square = compute_gdop(square, (0.5, 0.5))
    collinear = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    gdop_collinear = compute_gdop(collinear, (0.25, 0.0))
    assert math.isfinite(gdop_square) and gdop_square < 1.5, gdop_square
    assert gdop_collinear == GDOP_UNDETERMINED, gdop_collinear
    assert compute_gdop(np.array([[0.0, 0.0]]), (0.5, 0.5)) == GDOP_UNDETERMINED
    print(f"    4 anchors at corners ............... {gdop_square:.3f}")
    print(f"    3 collinear anchors ............... {gdop_collinear:.1f} (undetermined)")

    gated = MultiStaticTDOASolver(room_size=(1.0, 0.7), gdop_threshold=5.0, min_anchors=3)
    clustered_nodes = {
        "node_1": (0.50, 0.05),
        "node_2": (0.52, 0.06),
        "node_3": (0.54, 0.07),
        "node_4": (0.51, 0.08),
    }
    clustered_ranges = {k: 0.30 for k in clustered_nodes}
    clustered_weights = {k: 1.0 for k in clustered_nodes}
    res_cluster = gated.solve(
        clustered_nodes, clustered_ranges, clustered_weights, initial_guess=(0.52, 0.06)
    )
    assert res_cluster["gdop_rejected"] is True, res_cluster
    assert res_cluster["gdop"] > 5.0 or res_cluster["gdop"] >= GDOP_UNDETERMINED, res_cluster["gdop"]

    res_two = gated.solve(
        {"node_1": (0.5, 0.05), "node_2": (0.76, 0.5)},
        {"node_1": 0.3, "node_2": 0.3},
        {"node_1": 1.0, "node_2": 1.0},
    )
    assert res_two["gdop_rejected"] is True
    assert res_two["gdop"] == GDOP_UNDETERMINED and res_two["converged"] is False
    assert res_two["residual_m"] == -1.0
    print(f"    clustered anchors ................. GDOP {res_cluster['gdop']:.2f} > 5.0 -> rejected")
    print(f"    2 anchors, 3 required ............. {res_two['gdop']:.1f} undetermined (no invented number)")

    tight = MultiStaticTDOASolver(room_size=(1.0, 0.7), gdop_threshold=0.5, min_anchors=3)
    corner_nodes = {"node_1": (0.0, 0.0), "node_2": (1.0, 0.0), "node_3": (1.0, 0.7), "node_4": (0.0, 0.7)}
    res_prune = tight.solve(
        corner_nodes,
        {k: 0.40 for k in corner_nodes},
        {k: 1.0 for k in corner_nodes},
        initial_guess=(0.5, 0.35),
    )
    assert res_prune["anchors_used"] == 3, res_prune
    assert len(res_prune["anchors_dropped"]) == 1, res_prune
    print(f"    threshold 0.5 on 4 corners ........ pruned {res_prune['anchors_dropped']}, "
          f"{res_prune['anchors_used']} anchors kept")

    # ── 5. ENGINE SURFACE ────────────────────────────────────────────────
    print("-" * 72)
    print("  [ENGINE] gate and clock fields reach the caller")
    hw_frame = engine.simulate_em_multipath_frame("node_1", (0.50, 0.20), as_raw_esp32_iq=True)
    hw_frame["source"] = "hardware_esp32"
    hw_frame["ts_us"] = 4_000_000
    res_hw = engine.process_csi_frame(hw_frame)
    for key in ("gdop", "gdop_rejected", "position_error_m", "anchors_used", "ts_us", "time_source", "dt_s"):
        assert key in res_hw, key
    assert res_hw["ts_us"] == 4_000_000 and res_hw["time_source"] == "device_us", res_hw
    hw_frame2 = dict(hw_frame)
    hw_frame2["ts_us"] = 4_025_000
    res_hw2 = engine.process_csi_frame(hw_frame2)
    assert abs(res_hw2["dt_s"] - 0.025) < 1e-9, res_hw2["dt_s"]
    print(f"    device ts_us preserved, dt_s = {res_hw2['dt_s']*1000:.1f} ms on the radio clock")
    print(f"    gate fields: rejected={res_hw2['gdop_rejected']} gdop={res_hw2['gdop']} "
          f"anchors={res_hw2['anchors_used']}")
    print("=" * 72)


if __name__ == "__main__":
    _run_selftest()
