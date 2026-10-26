/*
 * ============================================================================
 * RuView Y-Delta Transceiver + CSI + LED Actuator — Standard ESP32 (v2.2)
 * ============================================================================
 * Target Hardware : Standard ESP32 DevKit V1 (ESP32-WROOM-32)
 * Operating Mode  : Simultaneous SoftAP Beacon (Zone_3_Tx) + Non-Blocking STA
 *                   Scanner + 56-Subcarrier Hardware CSI + 3-Room LED Actuator
 * Serial Output   : 115200 baud JSON lines (Zero external library dependencies):
 *   1. Scan JSON : {"event":"scan","node":"node_3","ssid":"Zone_1_Tx","zone":"node_1","raw_rssi":-52,"rssi":-53,"ch":6}
 *   2. CSI JSON  : {"event":"csi","node":"node_3","rx_id":"node_3","ssid":"Zone_1_Tx","zone":"node_1","rssi":-52,"ch":6,"num_sc":56,"csi_fmt":"espressif_htltf_v1","sc_order":"ieee_80211n_56","ts":12345,"csi_mag":[...],"csi_phase":[...]}
 *
 * CRITICAL HARDWARE SAFETY:
 *   - NEVER use GPIO 8 on a Standard ESP32 (wired to internal SPI Flash!).
 *   - Uses safe pins GPIO 2 (Zone 1 / Onboard LED), GPIO 4 (Zone 2), GPIO 5 (Zone 3).
 * ============================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_timer.h>
#include <math.h>
#include <string.h>
#include "secrets.h"

// ----------------------------------------------------------------------------
// 1. NODE IDENTITY & RF CONFIGURATION (Default: Node 3 = Standard ESP32 Corner)
// ----------------------------------------------------------------------------
#define NODE_ID               3

#if NODE_ID == 1
  #define ZONE_SSID           "Zone_1_Tx"
  #define SELF_ZONE           "node_1"
#elif NODE_ID == 2
  #define ZONE_SSID           "Zone_2_Tx"
  #define SELF_ZONE           "node_2"
#else
  #define ZONE_SSID           "Zone_3_Tx"
  #define SELF_ZONE           "node_3"
#endif

#define CHANNEL               6
#define ZONE_PASS             ""
#define MAX_CONN              4

// Base AP: this board joins the phone hotspot so its own SoftAP is dragged
// onto the same channel. Without the join the SoftAP would sit alone on
// CHANNEL and no peer could see it.

// Standard ESP32 DevKit Safe GPIO Pins (NEVER touch GPIO 6..11!)
// These outputs are PC-commanded only; calibration and bulb selection happen on the PC.
#define LED_ZONE_1_PIN        2     // Also Standard ESP32 onboard blue LED (Active-HIGH)
#define LED_ZONE_2_PIN        4     // Room 2 external LED
#define LED_ZONE_3_PIN        5     // Room 3 external LED

#define SCAN_DWELL_MS         45
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
  { "YOUR_HOTSPOT_SSID", "node_4", {0}, false },
};
const int SCAN_TARGET_COUNT = sizeof(SCAN_TARGETS) / sizeof(SCAN_TARGETS[0]);

struct FilteredRSSI {
  int           buf[3];
  int           bufIdx;
  bool          filled;
  unsigned long lastSeenMs;
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
    FILTERS[i].bufIdx     = 0;
    FILTERS[i].filled     = false;
    FILTERS[i].lastSeenMs = 0;
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
};

static CSISnapshot        csi_ring[CSI_RING_SIZE];
static volatile uint8_t   csi_write_idx = 0;
static volatile uint8_t   csi_read_idx  = 0;
static volatile uint8_t   csi_count     = 0;
static portMUX_TYPE       csi_mux       = portMUX_INITIALIZER_UNLOCKED;
static unsigned long      last_csi_emit_ms = 0;

static inline int8_t matchTargetByMac(const uint8_t* mac) {
  bool any_learned = false;
  for (int t = 0; t < SCAN_TARGET_COUNT; t++) {
    if (SCAN_TARGETS[t].hasBssid) {
      any_learned = true;
      if (memcmp(SCAN_TARGETS[t].bssid, mac, 6) == 0) {
        return (int8_t)t;
      }
    }
  }
  return any_learned ? -2 : -1;
}

void IRAM_ATTR csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len < 8) return;

  int8_t t_idx = matchTargetByMac(info->mac);
  if (t_idx == -2) return;

  portENTER_CRITICAL_ISR(&csi_mux);
  if (csi_count >= CSI_RING_SIZE) {
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
  } else {
    if (avail_pairs > MAX_SUBCARRIERS) avail_pairs = MAX_SUBCARRIERS;
    slot->num_subcarriers = avail_pairs > 0 ? avail_pairs : 0;
    slot->fft_reordered   = false;
    for (int i = 0; i < slot->num_subcarriers * 2; i++) {
      slot->iq_buf[i] = src[i];
    }
  }

  slot->ts_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);
  slot->ready = true;
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

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_csi_config(&csi_config);
  esp_wifi_set_csi_rx_cb(&csi_rx_cb, NULL);
  esp_wifi_set_csi(true);
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
// 5. SETUP
// ----------------------------------------------------------------------------
void setup() {
  setCpuFrequencyMhz(80);
  Serial.begin(115200);

  pinMode(LED_ZONE_1_PIN, OUTPUT);
  pinMode(LED_ZONE_2_PIN, OUTPUT);
  pinMode(LED_ZONE_3_PIN, OUTPUT);

  digitalWrite(LED_ZONE_1_PIN, LOW);
  digitalWrite(LED_ZONE_2_PIN, LOW);
  digitalWrite(LED_ZONE_3_PIN, LOW);

  resetFilters();

  Serial.println();
  Serial.println("=== RuView Y-Delta Standard ESP32 Transceiver + CSI + Actuator (v2.2) ===");
  Serial.print("Node   : "); Serial.println(SELF_ZONE);
  Serial.print("SSID   : "); Serial.println(ZONE_SSID);
  Serial.print("Channel: "); Serial.println(WiFi.channel());
  Serial.println("LEDs   : GPIO 2 (Zone 1 / Onboard), GPIO 4 (Zone 2), GPIO 5 (Zone 3)");

  WiFi.persistent(false);
  WiFi.disconnect(true, true);
  delay(100);

  WiFi.mode(WIFI_AP_STA);
  WiFi.setSleep(false);
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);

  bool ok = WiFi.softAP(ZONE_SSID, ZONE_PASS, CHANNEL, 0, MAX_CONN);
  if (ok) {
    boostSoftApBeaconRate();
    setupCsiExtraction();
    // No channel argument: the station scans for the base AP on whatever
    // channel the phone hotspot booted on, and the SoftAP follows it.
    WiFi.begin(CSI_STA_SSID, CSI_STA_PASS);
    Serial.print("AP IP  : ");
    Serial.println(WiFi.softAPIP());
    Serial.println("Transceiver ON - AP broadcasting + STA scanning + 56-subcarrier CSI + LED Actuator.");
  } else {
    Serial.println("{\"event\":\"error\",\"msg\":\"softAP_failed\"}");
  }
  last_scan_done_ms = millis() - SCAN_COOLDOWN_MS;
  Serial.flush();
}

// ----------------------------------------------------------------------------
// 6. MAIN LOOP (100% Non-Blocking)
// ----------------------------------------------------------------------------
void loop() {
  unsigned long now = millis();

  // 1. Non-blocking heartbeat blink & 500ms self-test timer
  updateSelfTest(now);

  // 2. Emit any queued 56-subcarrier CSI frames from the 4-slot ring buffer
  emitPendingCsiFrames(now);

  // 3. Non-blocking Serial command handler (<LED:PIN:STATE>, <TEST>, <PING>)
  handleSerialCommands(now);

  // 4. Launch next async single-channel scan after cooldown
  if (!scan_running && (now - last_scan_done_ms >= SCAN_COOLDOWN_MS)) {
    uint8_t target_chan = SCAN_CHANNELS[current_chan_idx];
    int16_t ret = WiFi.scanNetworks(
      /*async=*/true,
      /*show_hidden=*/false,
      /*passive=*/false,
      /*max_ms_per_chan=*/SCAN_DWELL_MS,
      /*channel=*/target_chan
    );

    if (ret != WIFI_SCAN_FAILED) {
      scan_running    = true;
      scan_started_ms = now;
      current_chan_idx = (current_chan_idx + 1) % NUM_SCAN_CHANNELS;
    } else {
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
        last_scan_done_ms = now;
      }
      return;
    }

    scan_running      = false;
    last_scan_done_ms = now;

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

          int filtered_rssi = FILTERS[t].filled
            ? median3(FILTERS[t].buf[0], FILTERS[t].buf[1], FILTERS[t].buf[2])
            : raw_rssi;

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
          break;
        }
      }
    }

    Serial.flush();
    WiFi.scanDelete();
    ensureCsiActive();
  }
}
