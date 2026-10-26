/*
 * ============================================================================
 * RuView Y-Delta Transceiver + CSI + LED Actuator — Standard ESP32 (v2.4)
 * ============================================================================
 * Target Hardware : Standard ESP32 DevKit V1 (ESP32-WROOM-32)
 * Operating Mode  : Simultaneous SoftAP Beacon (Zone_3_Tx) + Non-Blocking STA
 *                   Scanner + 56-Subcarrier Hardware CSI + 3-Room LED Actuator
 * Serial Output   : 115200 baud JSON lines (Zero external library dependencies):
 *   1. Scan JSON : {"event":"scan","node":"node_3","ssid":"Zone_1_Tx","zone":"node_1","raw_rssi":-52,"rssi":-53,"ch":1}
 *   2. CSI JSON  : {"event":"csi","node":"node_3","rx_id":"node_3","ssid":"Zone_1_Tx","zone":"node_1","rssi":-52,"ch":1,"num_sc":56,"csi_fmt":"espressif_htltf_v1","sc_order":"ieee_80211n_56","ts":12345,"ts_us":12345678,"csi_mag":[...],"csi_phase":[...]}
 *
 * CRITICAL HARDWARE SAFETY:
 *   - NEVER use GPIO 8 on a Standard ESP32 (wired to internal SPI Flash!).
 *   - Uses safe pins GPIO 2 (Zone 1 / Onboard LED), GPIO 4 (Zone 2), GPIO 5 (Zone 3).
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
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// Early hardware boot hook: Disables brownout before system init
void __attribute__((constructor)) disable_brownout_early() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
}

// ----------------------------------------------------------------------------
// 1. NODE IDENTITY & RF CONFIGURATION (Default: Node 3 = Standard ESP32 Corner)
// ----------------------------------------------------------------------------
#define NODE_ID               3

#if NODE_ID == 1
  #define ZONE_SSID           "Zone_1_Tx"
  #define SELF_ZONE           "node_1"
  #define CHANNEL             6
#elif NODE_ID == 2
  #define ZONE_SSID           "Zone_2_Tx"
  #define SELF_ZONE           "node_2"
  #define CHANNEL             6
#else
  #define ZONE_SSID           "Zone_3_Tx"
  #define SELF_ZONE           "node_3"
  #define CHANNEL             6
#endif

#define ZONE_PASS             ""
#define MAX_CONN              4

// Standard ESP32 DevKit Safe GPIO Pins (NEVER touch GPIO 6..11!)
// These outputs are PC-commanded only; calibration and bulb selection happen on the PC.
#define LED_ZONE_1_PIN        2     // Also Standard ESP32 onboard blue LED (Active-HIGH)
#define LED_ZONE_2_PIN        4     // Room 2 external LED
#define LED_ZONE_3_PIN        5     // Room 3 external LED

#define SCAN_DWELL_MS         45
// One channel to learn now, so rescanning every 10 ms after a 45 ms dwell
// kept the radio in scan state ~80% of the time and starved CSI capture.
#define SCAN_COOLDOWN_MS      3000UL
#define SCAN_TIMEOUT_MS       1500UL

#define MAX_SUBCARRIERS       56
#define CSI_RING_SIZE         4
#define CSI_EMIT_INTERVAL_MS  25UL
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

// Callback counters answer what the emitted lines cannot: did the radio hand
// the CSI layer any frames at all, and did their addresses match a learned
// zone? sig_mode 1 marks an 802.11n (HT) frame, the only kind a channel
// measurement can be computed from.
volatile uint32_t csi_cb_total = 0;
volatile uint32_t csi_cb_drop  = 0;
volatile uint32_t csi_cb_ht    = 0;
volatile uint8_t  csi_drop_mac[6] = { 0 };
volatile uint8_t  csi_drop_rate = 0;
// -999 never ran, -2 timed out, -1 the driver refused it, >=0 networks found.
int           last_scan_results   = -999;

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

  csi_cb_total++;
  if (info->rx_ctrl.sig_mode == 1) csi_cb_ht++;

  int8_t t_idx = matchTargetByMac(info->mac);
  if (t_idx == -2) {
    csi_cb_drop++;
    memcpy((void*)csi_drop_mac, info->mac, 6);
    csi_drop_rate = (uint8_t)info->rx_ctrl.rate;
    return;
  }

  portENTER_CRITICAL_ISR(&csi_mux);
  if (csi_count >= CSI_RING_SIZE) {
    csi_read_idx = (csi_read_idx + 1) % CSI_RING_SIZE;
    csi_count--;
  }
  uint8_t slot_idx = csi_write_idx;
  csi_write_idx = (csi_write_idx + 1) % CSI_RING_SIZE;
  csi_count++;
  csi_ring[slot_idx].ready = false;
  portEXIT_CRITICAL_ISR(&csi_mux);

  CSISnapshot* slot = &csi_ring[slot_idx];
  memcpy((void*)slot->mac, info->mac, 6);
  slot->rssi       = info->rx_ctrl.rssi;
  slot->channel    = info->rx_ctrl.channel;
  slot->target_idx = t_idx;

  int offset = info->first_word_invalid ? 4 : 0;
  int avail_pairs = (info->len - offset) / 2;
  const int8_t *src = (const int8_t*)info->buf + offset;

  if (avail_pairs >= 64) {
    slot->num_subcarriers = 56;
    slot->fft_reordered   = true;
    for (int k = 0; k < 28; k++) {
      int fft_bin = 36 + k;
      slot->iq_buf[k * 2]     = src[fft_bin * 2];
      slot->iq_buf[k * 2 + 1] = src[fft_bin * 2 + 1];
    }
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
  // Published under the same mutex the emitter reads under: the Wi-Fi task runs
  // on the other core, so without this gate a frame can be copied half-written.
  portENTER_CRITICAL_ISR(&csi_mux);
  slot->ready = true;
  portEXIT_CRITICAL_ISR(&csi_mux);
}

static void setupCsiExtraction() {
  wifi_csi_config_t csi_config;
  memset(&csi_config, 0, sizeof(csi_config));
  csi_config.lltf_en           = true;
  csi_config.htltf_en          = true;
  csi_config.stbc_htltf2_en    = true;
  csi_config.ltf_merge_en      = true;
  csi_config.channel_filter_en = false;
  csi_config.manu_scale        = false;
  csi_config.shift             = false;

  wifi_promiscuous_filter_t filt;
  filt.filter_mask = WIFI_PROMISCUOUS_FILTER_MASK_ALL;
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

static inline void ensureCsiActive() {
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_csi(true);
}

static void emitPendingCsiFrames(unsigned long nowMs) {
  while (csi_count > 0 && (nowMs - last_csi_emit_ms >= CSI_EMIT_INTERVAL_MS)) {
    CSISnapshot local_snap;
    bool has_frame = false;

    portENTER_CRITICAL(&csi_mux);
    if (csi_count > 0 && csi_ring[csi_read_idx].ready) {
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
// 4. NON-BLOCKING SCANNER & LED ACTUATOR STATE
// ----------------------------------------------------------------------------
// The hotspot picks any 2.4 GHz channel from 1 to 13 and an ESP32 in AP+STA
// mode drags its own SoftAP onto that channel, so the sweep covers all thirteen.
// It stops as soon as every BSSID is learned, which keeps the radio on the
// operating channel instead of scanning about 80% of the time.
const uint8_t SCAN_CHANNELS[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
const int NUM_SCAN_CHANNELS = sizeof(SCAN_CHANNELS) / sizeof(SCAN_CHANNELS[0]);

int           current_chan_idx   = 0;
unsigned long last_scan_done_ms  = 0;
unsigned long scan_started_ms    = 0;
bool          scan_running       = false;

bool          self_test_running  = false;
unsigned long self_test_start_ms = 0;

static inline void updateSelfTest(unsigned long nowMs) {
  if (self_test_running) {
    if (nowMs - self_test_start_ms >= 500UL) {
      self_test_running = false;
      digitalWrite(LED_ZONE_1_PIN, LOW);
      digitalWrite(LED_ZONE_2_PIN, LOW);
      digitalWrite(LED_ZONE_3_PIN, LOW);
    }
  }
}

static void boostSoftApBeaconRate() {
  wifi_config_t ap_cfg;
  if (esp_wifi_get_config(WIFI_IF_AP, &ap_cfg) == ESP_OK) {
    ap_cfg.ap.beacon_interval = 30; // 30 TU = 30.72 ms (~32.5 Hz)
    esp_wifi_set_config(WIFI_IF_AP, &ap_cfg);
  }
}

static void handleSerialCommands(unsigned long nowMs) {
  while (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) continue;

    if (cmd.startsWith("<LED:")) {
      int firstColon  = cmd.indexOf(':');
      int secondColon = cmd.indexOf(':', firstColon + 1);
      int endBracket  = cmd.indexOf('>', secondColon + 1);
      if (firstColon > 0 && secondColon > firstColon && endBracket > secondColon) {
        int pin = cmd.substring(firstColon + 1, secondColon).toInt();
        String stateStr = cmd.substring(secondColon + 1, endBracket);
        bool turnOn = stateStr.equalsIgnoreCase("ON") || stateStr == "1";

        if (pin == LED_ZONE_1_PIN || pin == LED_ZONE_2_PIN || pin == LED_ZONE_3_PIN) {
          digitalWrite(pin, turnOn ? HIGH : LOW);
          Serial.print("{\"event\":\"led_ack\",\"pin\":");
          Serial.print(pin);
          Serial.print(",\"state\":\"");
          Serial.print(turnOn ? "ON" : "OFF");
          Serial.println("\"}");
          Serial.flush();
        }
      }
    } else if (cmd.equalsIgnoreCase("<TEST>") || cmd.equalsIgnoreCase("TEST")) {
      self_test_running  = true;
      self_test_start_ms = nowMs;
      digitalWrite(LED_ZONE_1_PIN, HIGH);
      digitalWrite(LED_ZONE_2_PIN, HIGH);
      digitalWrite(LED_ZONE_3_PIN, HIGH);
      Serial.println("{\"event\":\"self_test\",\"status\":\"started_500ms\"}");
      Serial.flush();
    } else if (cmd.equalsIgnoreCase("<PING>") || cmd.equalsIgnoreCase("PING")) {
      Serial.print("{\"event\":\"pong\",\"board\":\"ESP32_Standard\",\"node\":\"");
      Serial.print(SELF_ZONE);
      Serial.print("\",\"ssid\":\"");
      Serial.print(ZONE_SSID);
      Serial.print("\",\"ch\":");
      Serial.print(CHANNEL);
      Serial.println("}");
      Serial.flush();
    } else if (cmd.equalsIgnoreCase("<RESET>")) {
      resetFilters();
      digitalWrite(LED_ZONE_1_PIN, LOW);
      digitalWrite(LED_ZONE_2_PIN, LOW);
      digitalWrite(LED_ZONE_3_PIN, LOW);
      Serial.println("{\"event\":\"status\",\"msg\":\"filters_and_leds_reset\"}");
      Serial.flush();
    }
  }
}

// ----------------------------------------------------------------------------
// ESP-IDF refuses a scan once the station is joined and CSI capture is on
// (scan_results -1 then -2), so peer BSSIDs are learned here instead: the radio
// is up, but there is no association and no promiscuous callback to fight with.
static void bootLearnPeers() {
  int n = (int)WiFi.scanNetworks(
      /*async=*/false, /*show_hidden=*/true,
      /*passive=*/true, /*max_ms_per_chan=*/120);

  Serial.print("{\"event\":\"boot_scan\",\"node\":\"");
  Serial.print(SELF_ZONE);
  Serial.print("\",\"results\":");
  Serial.print(n);
  if (n < 0) { Serial.println("}"); return; }

  int learned_n = 0;
  for (int i = 0; i < n; i++) {
    String found_s = WiFi.SSID(i);
    const char* found = found_s.c_str();
    if (found[0] == '\0') continue;

    for (int t = 0; t < SCAN_TARGET_COUNT; t++) {
      if (SCAN_TARGETS[t].hasBssid) continue;
      if (strcasecmp(found, SCAN_TARGETS[t].ssid) != 0) continue;
      const uint8_t* b = WiFi.BSSID(i);
      if (!b) continue;

      memcpy(SCAN_TARGETS[t].bssid, b, 6);
      SCAN_TARGETS[t].hasBssid = true;
      FILTERS[t].lastSeenMs     = millis();
      FILTERS[t].lastStableRssi = (int)WiFi.RSSI(i);
      learned_n++;

      char hex[18];
      snprintf(hex, sizeof(hex), "%02X:%02X:%02X:%02X:%02X:%02X", b[0], b[1], b[2], b[3], b[4], b[5]);
      Serial.print(",\"");
      Serial.print(SCAN_TARGETS[t].zone);
      Serial.print("\":\"");
      Serial.print(hex);
      Serial.print("\"");
    }
  }
  WiFi.scanDelete();
  Serial.print(",\"learned\":");
  Serial.print(learned_n);
  Serial.println("}");
}

// ----------------------------------------------------------------------------
// 5. SETUP
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout detector during Wi-Fi STA TX burst
  setCpuFrequencyMhz(160);
  Serial.begin(115200);

  pinMode(LED_ZONE_1_PIN, OUTPUT);
  pinMode(LED_ZONE_2_PIN, OUTPUT);
  pinMode(LED_ZONE_3_PIN, OUTPUT);
  digitalWrite(LED_ZONE_1_PIN, LOW);
  digitalWrite(LED_ZONE_2_PIN, LOW);
  digitalWrite(LED_ZONE_3_PIN, LOW);

  resetFilters();

  Serial.println();
  Serial.println("=== RuView Y-Delta Standard ESP32 Transceiver + CSI + Actuator (v2.4) ===");
  Serial.print("Node   : "); Serial.println(SELF_ZONE);
  Serial.print("SSID   : "); Serial.println(ZONE_SSID);
  Serial.print("Channel: "); Serial.println(WiFi.channel());
  Serial.println("LEDs   : GPIO 2 (Zone 1 / Onboard), GPIO 4 (Zone 2), GPIO 5 (Zone 3)");

  WiFi.persistent(false);
  // No disconnect() call here. Before WiFi.begin() there is no station to drop,
  // so it only ever logged an error (ESP_ERR_WIFI_NOT_INIT, then "STA not
  // started"). persistent(false) already keeps stale credentials out of flash.
  WiFi.mode(WIFI_AP_STA);
  delay(100);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.setTxPower(WIFI_POWER_11dBm);

  bool ok = WiFi.softAP(ZONE_SSID, ZONE_PASS, CHANNEL, 0, MAX_CONN);
  if (ok) {
    boostSoftApBeaconRate();
    bootLearnPeers();
    setupCsiExtraction();
    csiPump.begin(CSI_PUMP_PORT);
    // All boards are clients of the one real access point now.
    WiFi.begin(CSI_STA_SSID, CSI_STA_PASS);
    // Do not wait out the rescan cooldown for the very first sweep.
    last_scan_done_ms = millis() - SCAN_COOLDOWN_MS;
    Serial.print("AP IP  : ");
    Serial.println(WiFi.softAPIP());
    Serial.println("Transceiver ON - AP broadcasting + STA scanning + 56-subcarrier CSI + LED Actuator.");
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
    Serial.print(",\"csi_rx\":");
    Serial.print((unsigned long)csi_cb_total);
    Serial.print(",\"csi_ht\":");
    Serial.print((unsigned long)csi_cb_ht);
    Serial.print(",\"csi_drop\":");
    Serial.print((unsigned long)csi_cb_drop);
    if (csi_cb_drop) {
      uint8_t dm[6];
      memcpy(dm, (const void*)csi_drop_mac, 6);
      char dhex[18];
      snprintf(dhex, sizeof(dhex), "%02X:%02X:%02X:%02X:%02X:%02X", dm[0], dm[1], dm[2], dm[3], dm[4], dm[5]);
      Serial.print(",\"drop_mac\":\");
      Serial.print(dhex);
      Serial.print("\",\"drop_rate\":");
      Serial.print((int)csi_drop_rate);
    }
    Serial.println("}");
  }

  // Bulb outputs remain off unless the PC explicitly selects one.
  updateSelfTest(now);

  // 2. Emit any queued 56-subcarrier CSI frames from the 4-slot ring buffer
  emitPendingCsiFrames(now);

  // 3. Non-blocking Serial command handler (<LED:PIN:STATE>, <TEST>, <PING>)
  handleSerialCommands(now);

  // 4. Launch next async single-channel scan after cooldown
  // Only sweep while the station is not associated: an active scan hops the
  // radio off the base AP's channel and CSI stops for the length of the sweep.
  // Once joined, peers are learned from WiFi.BSSID() and from overheard frames.
  if (WiFi.status() != WL_CONNECTED && !scan_running && !allTargetsLearned() && (now - last_scan_done_ms >= SCAN_COOLDOWN_MS)) {
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
