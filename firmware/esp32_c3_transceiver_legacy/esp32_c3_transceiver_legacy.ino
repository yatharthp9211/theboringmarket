/*
 * ============================================================================
 * RuView Y-Delta Transceiver Firmware (AP + STA + Hardware CSI) — ESP32-C3 (v2.4)
 * ============================================================================
 * Target Hardware : ESP32-C3 SuperMini (Also auto-guards if compiled for ESP32)
 * Operating Mode  : Simultaneous SoftAP Beacon + Non-Blocking STA Scanner
 *                   + 56-Subcarrier Hardware CSI Extraction (4-Slot Ring Buffer)
 * Serial Output   : 115200 baud JSON lines (Zero external library dependencies):
 *   1. Scan JSON : {"event":"scan","node":"node_1","ssid":"Zone_2_Tx","zone":"node_2","raw_rssi":-54,"rssi":-55,"ch":6}
 *   2. CSI JSON  : {"event":"csi","node":"node_1","rx_id":"node_1","ssid":"Zone_2_Tx","zone":"node_2","rssi":-54,"ch":6,"num_sc":56,"csi_fmt":"espressif_htltf_v1","sc_order":"ieee_80211n_56","ts":12345,"ts_us":12345678,"csi_mag":[...],"csi_phase":[...]}
 * ============================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
#include <esp_timer.h>
#include <math.h>
#include <string.h>
#include "secrets.h"

// ----------------------------------------------------------------------------
// 1. NODE IDENTITY & RF CONFIGURATION
//    Set NODE_ID = 1 for C3 #1 (Zone_1_Tx, Ch 1)
//    Set NODE_ID = 2 for C3 #2 (Zone_2_Tx, Ch 6)
// ----------------------------------------------------------------------------
#define NODE_ID               2

#if NODE_ID == 1
  #define ZONE_SSID           "Zone_1_Tx"
  #define SELF_ZONE           "node_1"
  #define CHANNEL             6
#elif NODE_ID == 2
  #define ZONE_SSID           "Zone_2_Tx"
  #define SELF_ZONE           "node_2"
  #define CHANNEL             6
#elif NODE_ID == 3
  #define ZONE_SSID           "Zone_3_Tx"
  #define SELF_ZONE           "node_3"
  #define CHANNEL             6
#else
  #define ZONE_SSID           "Zone_1_Tx"
  #define SELF_ZONE           "node_1"
  #define CHANNEL             6
#endif

#define ZONE_PASS             ""

// ----------------------------------------------------------------------------
// CSI TRAFFIC PUMP
// Promiscuous CSI can only be computed from OFDM data frames. Each node's own
// SoftAP emits nothing but 802.11b beacons, so with no associated client the
// radio is silent and csi_rx_cb never fires. Every node therefore joins the
// laptop hotspot and pumps a 25 Hz UDP packet at the gateway it hands out: that
// is what puts real 56-subcarrier CSI on the wire, and the access point
// retransmits each frame so all three boards see the same over-the-air link.
// ----------------------------------------------------------------------------
// No fixed gateway address: a laptop hands out its own subnet (Windows uses
// 192.168.137.1) and every ESP32 SoftAP defaults to 192.168.4.1, so any
// hard-coded destination is either wrong or this board's own interface.
#define CSI_PUMP_PORT         42029
#define CSI_PUMP_INTERVAL_MS  40UL
#define CSI_STA_RETRY_MS      5000UL
WiFiUDP csiPump;
unsigned long last_csi_pump_ms = 0;
unsigned long last_sta_attempt_ms = 0;
unsigned long last_sta_report_ms  = 0;
// -999 never ran, -2 timed out, -1 the driver refused it, >=0 networks found.
int           last_scan_results   = -999;
#define MAX_CONN              4

// Hardware safety guard: GPIO 8 is onboard LED on ESP32-C3, but SPI Flash on Standard ESP32
#if defined(CONFIG_IDF_TARGET_ESP32C3)
  #define LED_PIN             8
  #define LED_ACTIVE_LOW      1
#else
  #define LED_PIN             2
  #define LED_ACTIVE_LOW      0
#endif

#define LED_ON_MS             150UL
#define LED_OFF_MS            850UL

#define SCAN_DWELL_MS         45
// One channel to learn now, so rescanning every 10 ms after a 45 ms dwell
// kept the radio in scan state ~80% of the time and starved CSI capture.
#define SCAN_COOLDOWN_MS      3000UL
#define SCAN_TIMEOUT_MS       1500UL

#define MAX_SUBCARRIERS       56
#define CSI_RING_SIZE         4
#define CSI_EMIT_INTERVAL_MS  25UL

// ----------------------------------------------------------------------------
// 2. TARGET NETWORKS, BSSID TABLE & 3-POINT MEDIAN FILTER
// ----------------------------------------------------------------------------
struct ScanTarget {
  const char* ssid;
  const char* zone;
  uint8_t     bssid[6];
  volatile bool hasBssid;
};

ScanTarget SCAN_TARGETS[] = {
  { "Zone_1_Tx",      "node_1", {0}, false },
  { "Zone_2_Tx",      "node_2", {0}, false },
  { "Zone_3_Tx",      "node_3", {0}, false },
  { "YOUR_HOTSPOT_SSID",      "node_4", {0}, false },   // base AP = phone hotspot
};
const int SCAN_TARGET_COUNT = sizeof(SCAN_TARGETS) / sizeof(SCAN_TARGETS[0]);

struct FilteredRSSI {
  int           buf[3];
  int           bufIdx;
  bool          filled;
  int           lastStableRssi;
  unsigned long lastSeenMs;
  unsigned long lastHwCsiMs;
};
FilteredRSSI FILTERS[SCAN_TARGET_COUNT];

static inline int median3(int a, int b, int c) {
  if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
  if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
  return c;
}

static inline void resetFilters() {
  for (int i = 0; i < SCAN_TARGET_COUNT; i++) {
    FILTERS[i].buf[0]     = -70;
    FILTERS[i].buf[1]     = -70;
    FILTERS[i].buf[2]     = -70;
    FILTERS[i].bufIdx         = 0;
    FILTERS[i].filled         = false;
    FILTERS[i].lastStableRssi = -70;
    FILTERS[i].lastSeenMs     = 0;
    FILTERS[i].lastHwCsiMs    = 0;
  }
}

// ----------------------------------------------------------------------------
// 3. LOCK-FREE 4-SLOT HARDWARE CSI RING BUFFER & 64-BIN FFT UNSCRAMBLER
// ----------------------------------------------------------------------------
struct CSISnapshot {
  volatile bool ready;
  uint8_t       mac[6];
  int8_t        rssi;
  uint8_t       channel;
  int8_t        target_idx;
  uint16_t      num_subcarriers;
  bool          fft_reordered;
  int8_t        iq_buf[MAX_SUBCARRIERS * 2];
  uint32_t      ts_ms;
  // Low 32 bits of esp_timer_get_time(), read at the same instant as ts_ms.
  // Wraps every 2^32 us = 71.58 min; csi_processor.DeviceTimebase unwraps it.
  uint32_t      ts_us;
};

static CSISnapshot        csi_ring[CSI_RING_SIZE];
static volatile uint8_t   csi_write_idx = 0;
static volatile uint8_t   csi_read_idx  = 0;
static volatile uint8_t   csi_count     = 0;
static portMUX_TYPE       csi_mux       = portMUX_INITIALIZER_UNLOCKED;
static unsigned long      last_csi_emit_ms = 0;

// Match incoming CSI frame MAC against learned target BSSIDs
static inline int8_t matchTargetByMac(const uint8_t* mac) {
  for (int t = 0; t < SCAN_TARGET_COUNT; t++) {
    if (SCAN_TARGETS[t].hasBssid && memcmp(SCAN_TARGETS[t].bssid, mac, 6) == 0) {
      return (int8_t)t;
    }
  }
  // An associated client transmits with its own STA MAC, which is its AP BSSID
  // with bit 1 of the last octet set. Scans only ever learn AP BSSIDs, so
  // without this the pump's uplink frames are dropped and hardware CSI is
  // replaced by the beacon-derived fabrication. Background routers whose BSSID
  // was never learned still return -2 and are ignored.
  for (int t = 0; t < SCAN_TARGET_COUNT; t++) {
    if (!SCAN_TARGETS[t].hasBssid) continue;
    const uint8_t* b = SCAN_TARGETS[t].bssid;
    bool same_prefix = true;
    for (int i = 0; i < 5; i++) {
      if (b[i] != mac[i]) { same_prefix = false; break; }
    }
    if (same_prefix && ((b[5] ^ mac[5]) & 0xFDU) == 0x00) {
      return (int8_t)t;
    }
  }
  return -2;
}

// Scanning only has to run until every zone's BSSID is known; after that the
// radio should stay on the operating channel and capture CSI.
static inline bool allTargetsLearned() {
  for (int t = 0; t < SCAN_TARGET_COUNT; t++) {
    if (!SCAN_TARGETS[t].hasBssid) return false;
  }
  return true;
}

// Index of the access point this board joins for its traffic pump, or -1.
static inline int hotspotTargetIdx() {
  for (int t = 0; t < SCAN_TARGET_COUNT; t++) {
    if (strcasecmp(SCAN_TARGETS[t].ssid, CSI_STA_SSID) == 0) return t;
  }
  return -1;
}

void IRAM_ATTR csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len < 8) return;

  int8_t t_idx = matchTargetByMac(info->mac);
  if (t_idx == -2) return; // Ignore unrelated background routers once targets are learned

  portENTER_CRITICAL_ISR(&csi_mux);
  if (csi_count >= CSI_RING_SIZE) {
    // Ring full: advance read pointer to overwrite oldest frame
    csi_read_idx = (csi_read_idx + 1) % CSI_RING_SIZE;
    csi_count--;
  }
  uint8_t slot_idx = csi_write_idx;
  csi_write_idx = (csi_write_idx + 1) % CSI_RING_SIZE;
  csi_count++;
  portEXIT_CRITICAL_ISR(&csi_mux);

  CSISnapshot* slot = &csi_ring[slot_idx];
  memcpy((void*)slot->mac, info->mac, 6);
  slot->rssi       = info->rx_ctrl.rssi;
  slot->channel    = info->rx_ctrl.channel;
  slot->target_idx = t_idx;

  // Handle ESP32 hardware bug where first 4 bytes of CSI buffer are invalid
  int offset = info->first_word_invalid ? 4 : 0;
  int avail_pairs = (info->len - offset) / 2;
  const int8_t *src = (const int8_t*)info->buf + offset;

  // Uncramble 64-bin raw ESP32 FFT order -> 56 monotonic IEEE 802.11n HT20 subcarriers (-28..-1, +1..+28)
  if (avail_pairs >= 64) {
    slot->num_subcarriers = 56;
    slot->fft_reordered   = true;
    // Negative subcarriers (-28 to -1) live at FFT bins 36..63
    for (int k = 0; k < 28; k++) {
      int fft_bin = 36 + k;
      slot->iq_buf[k * 2]     = src[fft_bin * 2];
      slot->iq_buf[k * 2 + 1] = src[fft_bin * 2 + 1];
    }
    // Positive subcarriers (+1 to +28) live at FFT bins 1..28 (skipping DC bin 0)
    for (int k = 0; k < 28; k++) {
      int fft_bin = 1 + k;
      int dst_idx = 28 + k;
      slot->iq_buf[dst_idx * 2]     = src[fft_bin * 2];
      slot->iq_buf[dst_idx * 2 + 1] = src[fft_bin * 2 + 1];
    }
    // Sanitize ESP32 hardware pilot/LO spike at index 26 and guard nulls at 52..55
    slot->iq_buf[26 * 2]     = (int8_t)(((int)slot->iq_buf[25 * 2] + (int)slot->iq_buf[27 * 2]) / 2);
    slot->iq_buf[26 * 2 + 1] = (int8_t)(((int)slot->iq_buf[25 * 2 + 1] + (int)slot->iq_buf[27 * 2 + 1]) / 2);
    for (int k = 52; k < 56; k++) {
      slot->iq_buf[k * 2]     = slot->iq_buf[51 * 2];
      slot->iq_buf[k * 2 + 1] = slot->iq_buf[51 * 2 + 1];
    }
  } else {
    if (avail_pairs > MAX_SUBCARRIERS) avail_pairs = MAX_SUBCARRIERS;
    slot->num_subcarriers = avail_pairs > 0 ? avail_pairs : 0;
    slot->fft_reordered   = false;
    for (int i = 0; i < slot->num_subcarriers * 2; i++) {
      slot->iq_buf[i] = src[i];
    }
  }

  const int64_t rx_time_us = esp_timer_get_time();
  slot->ts_us = (uint32_t)rx_time_us;
  slot->ts_ms = (uint32_t)(rx_time_us / 1000ULL);
  slot->ready = true;
}

static void setupCsiExtraction() {
  wifi_csi_config_t csi_config;
  memset(&csi_config, 0, sizeof(csi_config));
  csi_config.lltf_en           = true;
  csi_config.htltf_en          = true;
  csi_config.stbc_htltf2_en    = true;
  csi_config.ltf_merge_en      = true;
  csi_config.channel_filter_en = false; // Raw 64-bin FFT for true linear phase slope
  csi_config.manu_scale        = false;
  csi_config.shift             = false;

  wifi_promiscuous_filter_t filt;
  filt.filter_mask = WIFI_PROMIS_CTRL_FILTER_MASK_ALL;
  esp_wifi_set_promiscuous_filter(&filt);
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_protocol(WIFI_IF_AP, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&csi_rx_cb, NULL);
  esp_wifi_set_csi(true);
}

// Guaranteed 56-subcarrier CSI emitter when SoftAPs broadcast 802.11b beacons
static void emitBeaconDerivedCsiPacket(int t_idx, int raw_rssi, int smooth_rssi, int ch, unsigned long nowMs) {
  const float PI_F = 3.14159265f;
  float base_mag = fmaxf(2.0f, 38.0f + (float)(raw_rssi + 50) * 0.65f);
  float delta_db = (float)(raw_rssi - smooth_rssi);
  float phase_shift = delta_db * 0.22f;

  Serial.print("{\"event\":\"csi\",\"node\":\"");
  Serial.print(SELF_ZONE);
  Serial.print("\",\"rx_id\":\"");
  Serial.print(SELF_ZONE);
  Serial.print("\",\"ssid\":\"");
  Serial.print(SCAN_TARGETS[t_idx].ssid);
  Serial.print("\",\"zone\":\"");
  Serial.print(SCAN_TARGETS[t_idx].zone);
  Serial.print("\",\"rssi\":");
  Serial.print(raw_rssi);
  Serial.print(",\"ch\":");
  Serial.print(ch);
  Serial.print(",\"num_sc\":56,\"csi_fmt\":\"espressif_htltf_v1\",\"sc_order\":\"ieee_80211n_56\",\"ts\":");
  Serial.print((uint32_t)nowMs);
  // Beacon-derived frames need the same radio-clock stamp as the
  // hardware path, or the host falls back to the nominal frame rate.
  Serial.print(",\"ts_us\":");
  Serial.print((uint32_t)esp_timer_get_time());
  // These 56 tones are an analytic curve fitted to RSSI, not radio
  // measurements: declare it so the host does not count them as hardware CSI.
  Serial.print(",\"synthesized\":true");

  Serial.print(",\"csi_mag\":[");
  for (int i = 0; i < 56; i++) {
    if (i > 0) Serial.print(",");
    int sc_k = (i < 28) ? (-28 + i) : (1 + (i - 28));
    float ripple = 1.6f * cosf(0.18f * (float)sc_k + phase_shift) + 0.45f * delta_db * sinf(0.11f * (float)sc_k);
    float mag = fmaxf(1.0f, base_mag + ripple);
    Serial.print(mag, 2);
  }

  Serial.print("],\"csi_phase\":[");
  float slope = -0.055f - 0.004f * (float)(ch - 1);
  for (int i = 0; i < 56; i++) {
    if (i > 0) Serial.print(",");
    int sc_k = (i < 28) ? (-28 + i) : (1 + (i - 28));
    float pha = slope * (float)sc_k + phase_shift + 0.14f * sinf(0.18f * (float)sc_k + phase_shift);
    while (pha > PI_F)  pha -= 2.0f * PI_F;
    while (pha < -PI_F) pha += 2.0f * PI_F;
    Serial.print(pha, 3);
  }
  Serial.println("]}");
}

// Re-assert promiscuous + CSI after WiFi.scanDelete() so scans never disable CSI capture
static inline void ensureCsiActive() {
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_csi(true);
}

static void emitPendingCsiFrames(unsigned long nowMs) {
  while (csi_count > 0 && (nowMs - last_csi_emit_ms >= CSI_EMIT_INTERVAL_MS)) {
    CSISnapshot local_snap;
    bool has_frame = false;

    portENTER_CRITICAL(&csi_mux);
    if (csi_count > 0) {
      local_snap = csi_ring[csi_read_idx];
      csi_ring[csi_read_idx].ready = false;
      csi_read_idx = (csi_read_idx + 1) % CSI_RING_SIZE;
      csi_count--;
      has_frame = true;
    }
    portEXIT_CRITICAL(&csi_mux);

    if (!has_frame) break;
    last_csi_emit_ms = nowMs;

    int8_t t_idx = local_snap.target_idx;
    const char* ssid_str = "Unknown_Tx";
    const char* zone_str = "node_4";

    if (t_idx >= 0 && t_idx < SCAN_TARGET_COUNT) {
      ssid_str = SCAN_TARGETS[t_idx].ssid;
      zone_str = SCAN_TARGETS[t_idx].zone;
      FILTERS[t_idx].lastHwCsiMs = nowMs;
    } else {
      // Fallback by channel before BSSID table is populated
      if (local_snap.channel == 1)      { ssid_str = "Zone_1_Tx"; zone_str = "node_1"; }
      else if (local_snap.channel == 6) { ssid_str = "Zone_2_Tx"; zone_str = "node_2"; }
      else if (local_snap.channel == 11){ ssid_str = "Zone_3_Tx"; zone_str = "node_3"; }
    }

    uint16_t n_sc = local_snap.num_subcarriers;
    if (n_sc > MAX_SUBCARRIERS) n_sc = MAX_SUBCARRIERS;

    Serial.print("{\"event\":\"csi\",\"node\":\"");
    Serial.print(SELF_ZONE);
    Serial.print("\",\"rx_id\":\"");
    Serial.print(SELF_ZONE);
    Serial.print("\",\"ssid\":\"");
    Serial.print(ssid_str);
    Serial.print("\",\"zone\":\"");
    Serial.print(zone_str);
    Serial.print("\",\"rssi\":");
    Serial.print((int)local_snap.rssi);
    Serial.print(",\"ch\":");
    Serial.print((int)local_snap.channel);
    Serial.print(",\"num_sc\":");
    Serial.print((int)n_sc);
    Serial.print(",\"csi_fmt\":\"espressif_htltf_v1\",\"sc_order\":\"");
    Serial.print(local_snap.fft_reordered ? "ieee_80211n_56" : "esp32_raw");
    Serial.print("\",\"ts\":");
    Serial.print(local_snap.ts_ms);
    Serial.print(",\"ts_us\":");
    Serial.print(local_snap.ts_us);

    Serial.print(",\"csi_mag\":[");
    for (uint16_t i = 0; i < n_sc; i++) {
      if (i > 0) Serial.print(",");
      int8_t imag_v = local_snap.iq_buf[i * 2];
      int8_t real_v = local_snap.iq_buf[i * 2 + 1];
      float mag = sqrtf((float)(real_v * real_v + imag_v * imag_v));
      Serial.print(mag, 2);
    }

    Serial.print("],\"csi_phase\":[");
    for (uint16_t i = 0; i < n_sc; i++) {
      if (i > 0) Serial.print(",");
      int8_t imag_v = local_snap.iq_buf[i * 2];
      int8_t real_v = local_snap.iq_buf[i * 2 + 1];
      float pha = atan2f((float)imag_v, (float)real_v);
      Serial.print(pha, 3);
    }
    Serial.println("]}");
    Serial.flush();
    nowMs = millis();
  }
}

// ----------------------------------------------------------------------------
// 4. NON-BLOCKING SCANNER & HEARTBEAT LED STATE
// ----------------------------------------------------------------------------
// Sprint 6 Single-Channel Coherence: Channel is locked to Channel 6 (2437 MHz).
// Radio stays on Channel 6 100% of the time with 0 hopping penalty.
const uint8_t SCAN_CHANNELS[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
const int NUM_SCAN_CHANNELS = sizeof(SCAN_CHANNELS) / sizeof(SCAN_CHANNELS[0]);

int           current_chan_idx   = 0;
unsigned long last_scan_done_ms  = 0;
unsigned long scan_started_ms    = 0;
bool          scan_running       = false;

bool          led_is_on          = false;
unsigned long led_last_toggle_ms = 0;

static inline void setLed(bool turnOn) {
  led_is_on = turnOn;
#if LED_ACTIVE_LOW
  digitalWrite(LED_PIN, turnOn ? LOW : HIGH);
#else
  digitalWrite(LED_PIN, turnOn ? HIGH : LOW);
#endif
}

static inline void updateHeartbeatLed(unsigned long nowMs) {
  unsigned long interval = led_is_on ? LED_ON_MS : LED_OFF_MS;
  if (nowMs - led_last_toggle_ms >= interval) {
    led_last_toggle_ms = nowMs;
    setLed(!led_is_on);
  }
}

static void boostSoftApBeaconRate() {
  wifi_config_t ap_cfg;
  if (esp_wifi_get_config(WIFI_IF_AP, &ap_cfg) == ESP_OK) {
    ap_cfg.ap.beacon_interval = 30; // 30 TU = 30.72 ms (~32.5 Hz)
    esp_wifi_set_config(WIFI_IF_AP, &ap_cfg);
  }
}

static void handleSerialCommands() {
  while (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("<PING>") || cmd.equalsIgnoreCase("PING")) {
      Serial.print("{\"event\":\"pong\",\"node\":\"");
      Serial.print(SELF_ZONE);
      Serial.print("\",\"ssid\":\"");
      Serial.print(ZONE_SSID);
      Serial.print("\",\"ch\":");
      Serial.print(CHANNEL);
      Serial.println("}");
      Serial.flush();
    } else if (cmd.equalsIgnoreCase("<RESET>")) {
      resetFilters();
      Serial.println("{\"event\":\"status\",\"msg\":\"filters_reset\"}");
      Serial.flush();
    }
  }
}

// ----------------------------------------------------------------------------
// 5. SETUP
// ----------------------------------------------------------------------------
void setup() {
  setCpuFrequencyMhz(80);
  Serial.begin(115200);

  // USB CDC Handshake: wait up to 2 seconds for USB serial port enumeration
  unsigned long usb_wait_start = millis();
  while (!Serial && (millis() - usb_wait_start < 2000)) {
    delay(10);
  }

  pinMode(LED_PIN, OUTPUT);
  setLed(true);
  led_last_toggle_ms = millis();

  resetFilters();

  delay(200);
  Serial.println();
  Serial.println("=== RuView Y-Delta C3 Transceiver (AP + STA + CSI) v3.0 ===");
  Serial.print("Node   : "); Serial.println(SELF_ZONE);
  Serial.print("SSID   : "); Serial.println(ZONE_SSID);
  Serial.print("Channel: "); Serial.println(WiFi.channel());
  Serial.print("Joins  : "); Serial.println(CSI_STA_SSID);

  WiFi.persistent(false);
  WiFi.mode(WIFI_AP_STA);
  delay(150);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  // 8.5 dBm (~7 mW): smooth current draw, avoids brownout spikes on USB power
  WiFi.setTxPower(WIFI_POWER_8_5dBm);

  bool ok = WiFi.softAP(ZONE_SSID, ZONE_PASS, CHANNEL, 0, MAX_CONN);
  if (ok) {
    boostSoftApBeaconRate();
    setupCsiExtraction();
    csiPump.begin(CSI_PUMP_PORT);
    // Join the base hotspot without naming a channel: the station scans
    // for it, so it works whatever channel the phone hotspot booted on.
    WiFi.begin(CSI_STA_SSID, CSI_STA_PASS);
    last_scan_done_ms = millis() - SCAN_COOLDOWN_MS;
    Serial.print("AP IP  : ");
    Serial.println(WiFi.softAPIP());
    Serial.println("Transceiver ON - AP broadcasting + STA on base AP (auto channel) + 56-subcarrier CSI active.");
  } else {
    Serial.println("{\"event\":\"error\",\"msg\":\"softAP_failed\"}");
  }
  Serial.flush();
}

// ----------------------------------------------------------------------------
// 6. MAIN LOOP (100% Non-Blocking)
// ----------------------------------------------------------------------------
void loop() {
  unsigned long now = millis();

  // 0. Keep OFDM data frames in the air so the CSI callback has real traffic
  //    to compute a channel impulse response from.
  if (now - last_csi_pump_ms >= CSI_PUMP_INTERVAL_MS) {
    last_csi_pump_ms = now;
    if (WiFi.status() != WL_CONNECTED) {
      if (now - last_sta_attempt_ms >= CSI_STA_RETRY_MS) {
        last_sta_attempt_ms = now;
        WiFi.begin(CSI_STA_SSID, CSI_STA_PASS);
      }
    } else {
      // Send to whatever gateway the hotspot handed out, and only once there is
      // a route: transmitting into a down interface returned ENOMEM (error 12)
      // 25 times a second. The access point retransmits each frame, so every
      // board measures the same over-the-air link.
      IPAddress gw = WiFi.gatewayIP();
      if (gw[0] != 0) {
        csiPump.beginPacket(gw, CSI_PUMP_PORT);
        csiPump.write((const uint8_t *)"CSI-PUMP", 8);
        csiPump.endPacket();
      }
    }
  }

  // 0b. Station link report. A board that is associated but idle and a board
  //     that cannot see the access point both print nothing at all, so name the
  //     difference: status 6 = still looking, 1 = SSID not found, 3 = linked.
  if (now - last_sta_report_ms >= 1000UL) {
    last_sta_report_ms = now;

    // The station already knows the access point it joined, so attribute that
    // link without waiting for a sweep: sweeps were completing with zero
    // networks, which left every hardware frame unattributed and dropped.
    int ht = hotspotTargetIdx();
    if (ht >= 0 && WiFi.status() == WL_CONNECTED && !SCAN_TARGETS[ht].hasBssid) {
      const uint8_t* joined = WiFi.BSSID();
      if (joined) {
        memcpy(SCAN_TARGETS[ht].bssid, joined, 6);
        SCAN_TARGETS[ht].hasBssid = true;
        FILTERS[ht].lastSeenMs     = now;
        FILTERS[ht].lastStableRssi = (int)WiFi.RSSI();
        FILTERS[ht].buf[0] = FILTERS[ht].buf[1] = FILTERS[ht].buf[2] = FILTERS[ht].lastStableRssi;
        Serial.print("{\"event\":\"learned\",\"node\":\"");
        Serial.print(SELF_ZONE);
        Serial.print("\",\"zone\":\"");
        Serial.print(SCAN_TARGETS[ht].zone);
        Serial.print("\",\"bssid\":\"");
        Serial.print(WiFi.BSSIDstr());
        Serial.println("\"}");
      }
    }
    Serial.print("{\"event\":\"sta\",\"node\":\"");
    Serial.print(SELF_ZONE);
    Serial.print("\",\"status\":");
    Serial.print((int)WiFi.status());
    Serial.print(",\"rssi\":");
    Serial.print(WiFi.RSSI());
    Serial.print(",\"ip\":\"");
    Serial.print(WiFi.localIP());
    Serial.print("\",\"gw\":\"");
    Serial.print(WiFi.gatewayIP());
    Serial.print("\",\"scan_results\":");
    Serial.print(last_scan_results);
    Serial.print(",\"learned\":");
    int learned_n = 0;
    for (int t = 0; t < SCAN_TARGET_COUNT; t++) {
      if (SCAN_TARGETS[t].hasBssid) learned_n++;
    }
    Serial.print(learned_n);
    Serial.println("}");
  }

  // 1. Non-blocking 1 Hz heartbeat blink (150ms ON, 850ms OFF)
  updateHeartbeatLed(now);

  // 2. Emit any queued 56-subcarrier CSI frames from the 4-slot ring buffer
  emitPendingCsiFrames(now);

  // 3. Non-blocking Serial command handler
  handleSerialCommands();

  // 4. Launch next async single-channel scan after cooldown
  if (!scan_running && !allTargetsLearned() && (now - last_scan_done_ms >= SCAN_COOLDOWN_MS)) {
    // Full-band sweep: the hotspot sits on channel 13, outside the old
    // 1/6/11 rotation, so the per-channel form could never find a peer.
    // Capture is deliberately left on; switching it off made the driver refuse
    // the scan outright (WIFI_SCAN_FAILED) instead of returning zero results.
    int16_t ret = WiFi.scanNetworks(
      /*async=*/true,
      /*show_hidden=*/false,
      /*passive=*/false,
      /*max_ms_per_chan=*/SCAN_DWELL_MS
    );

    if (ret != WIFI_SCAN_FAILED) {
      scan_running    = true;
      scan_started_ms = now;
      current_chan_idx = (current_chan_idx + 1) % NUM_SCAN_CHANNELS;
    } else {
      last_scan_results = -1;
      last_scan_done_ms = now;
    }
  }

  // 5. Poll async scan completion
  if (scan_running) {
    int16_t n = WiFi.scanComplete();

    if (n == WIFI_SCAN_RUNNING) {
      if (now - scan_started_ms > SCAN_TIMEOUT_MS) {
        WiFi.scanDelete();
        ensureCsiActive();
        scan_running      = false;
        last_scan_results = -2;
        last_scan_done_ms = now;
      }
      return;
    }

    scan_running      = false;
    last_scan_done_ms = now;

    last_scan_results = (int)n;
    if (n < 0) {
      WiFi.scanDelete();
      ensureCsiActive();
      return;
    }

    bool emitted_in_batch[SCAN_TARGET_COUNT] = { false };

    for (int i = 0; i < n; i++) {
      String found_ssid_str = WiFi.SSID(i);
      const char* found_ssid = found_ssid_str.c_str();

      if (found_ssid[0] == '\0' || strcasecmp(found_ssid, ZONE_SSID) == 0) {
        continue;
      }

      for (int t = 0; t < SCAN_TARGET_COUNT; t++) {
        if (!emitted_in_batch[t] && strcasecmp(found_ssid, SCAN_TARGETS[t].ssid) == 0) {
          emitted_in_batch[t] = true;

          // Record BSSID so the hardware CSI callback can tag frames to this exact link
          const uint8_t* bssid_ptr = WiFi.BSSID(i);
          if (bssid_ptr) {
            memcpy(SCAN_TARGETS[t].bssid, bssid_ptr, 6);
            SCAN_TARGETS[t].hasBssid = true;
          }

          int raw_rssi = WiFi.RSSI(i);
          int32_t ch   = WiFi.channel(i);

          if (FILTERS[t].lastSeenMs > 0 && (now - FILTERS[t].lastSeenMs > 5000UL)) {
            FILTERS[t].filled = false;
            FILTERS[t].bufIdx = 0;
          }
          FILTERS[t].lastSeenMs = now;

          FILTERS[t].buf[FILTERS[t].bufIdx] = raw_rssi;
          FILTERS[t].bufIdx = (FILTERS[t].bufIdx + 1) % 3;
          if (FILTERS[t].bufIdx == 0) {
            FILTERS[t].filled = true;
          }

          int filtered_rssi = raw_rssi;
          if (!FILTERS[t].filled && FILTERS[t].bufIdx == 1) {
            FILTERS[t].lastStableRssi = raw_rssi;
          } else {
            int diff = abs(raw_rssi - FILTERS[t].lastStableRssi);
            if (diff <= 1) {
              // Suppress 1-LSB ADC quantization toggle on empty calibrated bed
              filtered_rssi = FILTERS[t].lastStableRssi;
            } else if (diff <= 28) {
              // Immediate zero-lag response to >= 2 dB human hand wave or steel/water bottle
              FILTERS[t].lastStableRssi = raw_rssi;
              filtered_rssi = raw_rssi;
            } else {
              // Reject impossible >28 dB single-frame RF glitch
              filtered_rssi = FILTERS[t].lastStableRssi;
            }
          }

          Serial.print("{\"event\":\"scan\",\"node\":\"");
          Serial.print(SELF_ZONE);
          Serial.print("\",\"ssid\":\"");
          Serial.print(SCAN_TARGETS[t].ssid);
          Serial.print("\",\"zone\":\"");
          Serial.print(SCAN_TARGETS[t].zone);
          Serial.print("\",\"raw_rssi\":");
          Serial.print(raw_rssi);
          Serial.print(",\"rssi\":");
          Serial.print(filtered_rssi);
          Serial.print(",\"ch\":");
          Serial.print(ch);
          Serial.println("}");

          // Guarantee 56-subcarrier CSI frame even when SoftAPs emit 802.11b beacons
          if (now - FILTERS[t].lastHwCsiMs > 250UL) {
            emitBeaconDerivedCsiPacket(t, raw_rssi, filtered_rssi, ch, now);
          }
          break;
        }
      }
    }

    Serial.flush();
    WiFi.scanDelete();
    ensureCsiActive();
  }
}
