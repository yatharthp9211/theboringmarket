/**
 * LiveDataProvider — RuView Observatory (v3.5)
 *
 * Replaces DemoDataGenerator. Always returns the latest WebSocket frame.
 * No auto-cycling, no scenarios, no synthetic data.
 * Falls back to a quiet empty-room frame when no live data has arrived.
 */

const EMPTY_FRAME = {
  type: 'sensing_update',
  timestamp: 0,
  source: 'live',
  scenario: 'empty_room',
  nodes: [],
  features: { mean_rssi: 0, variance: 0, std: 0, motion_band_power: 0, breathing_band_power: 0, dominant_freq_hz: 0, spectral_power: 0 },
  classification: { motion_level: 'absent', presence: false, confidence: 0, fall_detected: false },
  signal_field: { grid_size: [20, 1, 20], values: new Array(400).fill(0.05) },
  vital_signs: { breathing_rate_bpm: 0, heart_rate_bpm: 0, breathing_confidence: 0, heart_rate_confidence: 0 },
  persons: [],
  estimated_persons: 0,
  edge_modules: {},
  _observatory: { subcarrier_iq: [], per_subcarrier_variance: new Array(64).fill(0.02) },
};

export class LiveDataProvider {
  constructor() {
    this._latestFrame = null;
    this._autoMode = false;
    this._paused = false;
    this._scenario = 'empty_room';
    this._cycleDuration = 30;
  }

  get currentScenario() {
    if (this._latestFrame && this._latestFrame.scenario) {
      return this._latestFrame.scenario;
    }
    return this._scenario;
  }

  get paused() { return this._paused; }
  set paused(v) { this._paused = v; }

  setLiveFrame(frame) {
    if (frame && typeof frame === 'object') {
      this._latestFrame = frame;
    }
  }

  setScenario(name) { this._scenario = name || 'empty_room'; }
  setCycleDuration(seconds) { this._cycleDuration = Math.max(5, seconds); }
  cycleScenario() {}

  update(dt) {
    if (this._paused) {
      return this._latestFrame || EMPTY_FRAME;
    }
    return this._latestFrame || EMPTY_FRAME;
  }
}
