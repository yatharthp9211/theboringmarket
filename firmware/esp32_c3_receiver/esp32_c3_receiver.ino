 /*
 * RuView RGIPT Exhibition – ESP32-C3 Secondary Receiver #2 (`rx_2` on COM4) (v3.5.3)
 * ================================================================================
 * Fixed: CSI setup deferred to loop(), LED heartbeat added for visibility.
 */

#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_timer.h>
#include <ArduinoJson.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"


#define RX_NODE_ID       "rx_2"
#define MAX_SUBCARRIERS  56
#define CSI_RING_SIZE    4
#define LED_PIN          8

struct BeaconState {
  const char* ssid;
  const char* zone;
  const char* tx_id;
  uint8_t bssid[6];
  bool hasBssid;
  int rssiBuf[3];
  int bufIdx;
  bool filled;
  float kalmanX;
  float kalmanP;
  unsigned long sampleCount;
};

BeaconState BEACONS[] = {
  { "YOUR_HOTSPOT_SSID", "node_4", "tx_laptop", {0}, false, {-60, -60, -60}, 0, false, -60.0f, 1.0f, 0 },
  { "Zone_1_Tx",      "node_1", "tx_c3_1",  {0}, false, {-64, -64, -64}, 0, false, -64.0f, 1.0f, 0 },
  { "Zone_2_Tx",      "node_2", "tx_c3_2",  {0}, false, {-64, -64, -64}, 0, false, -64.0f, 1.0f, 0 },
};
const int BEACON_COUNT = sizeof(BEACONS) / sizeof(BEACONS[0]);

struct CSISnapshot {
  volatile bool ready;
  uint8_t mac[6];
  int8_t rssi;
  uint8_t channel;
  uint16_t num_subcarriers;
  bool fft_reordered;
  int8_t iq_buf[MAX_SUBCARRIERS * 2];
  uint32_t ts_ms;
  // Low 32 bits of esp_timer_get_time(), read at the same instant as ts_ms.
  // Wraps every 2^32 us = 71.58 min; csi_processor.DeviceTimebase unwraps it.
  uint32_t ts_us;
};

static CSISnapshot csi_ring[CSI_RING_SIZE];
static volatile uint8_t csi_write_idx = 0;
static volatile uint8_t csi_read_idx  = 0;
static volatile uint8_t csi_count     = 0;
static volatile uint32_t csi_dropped  = 0;
static portMUX_TYPE csi_mux = portMUX_INITIALIZER_UNLOCKED;

static unsigned long last_csi_emit_ms = 0;
static unsigned long last_heartbeat_ms = 0;
#define CSI_EMIT_INTERVAL_MS 20
#define HEARTBEAT_INTERVAL_MS 1000

#define SCAN_CHANNEL          6
#define SCAN_COOLDOWN_MS      3000UL
#define CHANNEL_DWELL_MS      20
unsigned long last_scan_ms = 0;
bool scan_running = false;

static inline int median3(int a, int b, int c) {
  if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
  if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
  return c;
}

static inline float updateKalman1D(BeaconState &b, float measurement) {
  const float Q = 0.08f;
  const float R = 0.42f;
  if (b.sampleCount == 0) {
    b.kalmanX = measurement;
    b.kalmanP = 1.0f;
    return measurement;
  }
  b.kalmanP += Q;
  float K = b.kalmanP / (b.kalmanP + R);
  b.kalmanX = b.kalmanX + K * (measurement - b.kalmanX);
  b.kalmanP = (1.0f - K) * b.kalmanP;
  return b.kalmanX;
}

const char* mac_to_zone(const uint8_t* mac, uint8_t ch) {
  for (int b = 0; b < BEACON_COUNT; b++) {
    if (BEACONS[b].hasBssid && memcmp(BEACONS[b].bssid, mac, 6) == 0) {
      return BEACONS[b].zone;
    }
  }
  if (ch <= 3) return "node_1";
  if (ch >= 4 && ch <= 8) return "node_2";
  return "node_3";
}

const char* mac_to_tx_id(const uint8_t* mac, uint8_t ch) {
  for (int b = 0; b < BEACON_COUNT; b++) {
    if (BEACONS[b].hasBssid && memcmp(BEACONS[b].bssid, mac, 6) == 0) {
      return BEACONS[b].tx_id;
    }
  }
  if (ch <= 3) return "tx_laptop";
  if (ch >= 4 && ch <= 8) return "tx_c3_1";
  return "tx_c3_2";
}

void IRAM_ATTR csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len < 8) return;

  portENTER_CRITICAL_ISR(&csi_mux);
  if (csi_count >= CSI_RING_SIZE) {
    csi_read_idx = (csi_read_idx + 1) % CSI_RING_SIZE;
    csi_count--;
    csi_dropped++;
  }
  uint8_t slot_idx = csi_write_idx;
  csi_write_idx = (csi_write_idx + 1) % CSI_RING_SIZE;
  csi_count++;

  CSISnapshot* slot = &csi_ring[slot_idx];
  memcpy((void*)slot->mac, info->mac, 6);
  slot->rssi = info->rx_ctrl.rssi;
  slot->channel = info->rx_ctrl.channel;

  int offset = (info->len >= 120) ? 4 : 0;
  int avail_pairs = (info->len - offset) / 2;
  const int8_t *src = (const int8_t*)info->buf + offset;

  if (avail_pairs >= 64) {
    slot->num_subcarriers = 56;
    slot->fft_reordered = true;
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
    slot->num_subcarriers = avail_pairs;
    slot->fft_reordered = false;
    for (int i = 0; i < avail_pairs * 2; i++) {
      slot->iq_buf[i] = src[i];
    }
  }

  const int64_t rx_time_us = esp_timer_get_time();
  slot->ts_us = (uint32_t)rx_time_us;
  slot->ts_ms = (uint32_t)(rx_time_us / 1000ULL);
  slot->ready = true;
  portEXIT_CRITICAL_ISR(&csi_mux);
}

bool setup_csi_extraction() {
  wifi_csi_config_t csi_config;
  memset(&csi_config, 0, sizeof(csi_config));
  csi_config.lltf_en    = true;
  csi_config.htltf_en   = true;

  esp_err_t err = esp_wifi_set_csi_config(&csi_config);
  if (err != ESP_OK) {
    Serial.println("{\"event\":\"error\",\"msg\":\"csi_config_failed\",\"code\":" + String(err) + "}");
    Serial.flush();
    return false;
  }

  esp_wifi_set_csi_rx_cb(&csi_rx_cb, NULL);
  esp_wifi_set_csi(true);
  return true;
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout detector
  setCpuFrequencyMhz(80);                    // Clock down to 80 MHz to prevent current surges
  Serial.begin(115200);
  delay(200);
  Serial.flush();

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("{\"event\":\"boot\",\"msg\":\"serial_ready\",\"rx_id\":\"rx_2\"}");
  Serial.flush();

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_ps(WIFI_PS_NONE);

  Serial.println("{\"event\":\"boot\",\"msg\":\"wifi_ready\"}");
  Serial.flush();
}

void emit_heartbeat() {
  if (millis() - last_heartbeat_ms < HEARTBEAT_INTERVAL_MS) return;
  last_heartbeat_ms = millis();
  JsonDocument doc;
  doc["event"] = "rx_heartbeat";
  doc["rx_id"] = RX_NODE_ID;
  doc["uptime_ms"] = millis();
  doc["free_heap"] = ESP.getFreeHeap();
  doc["csi_ring_count"] = csi_count;
  doc["csi_dropped"] = csi_dropped;
  serializeJson(doc, Serial);
  Serial.println();
}

void emit_pending_csi_frames() {
  while (csi_count > 0 && (millis() - last_csi_emit_ms >= CSI_EMIT_INTERVAL_MS)) {
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
    last_csi_emit_ms = millis();

    const char* zone_str = mac_to_zone(local_snap.mac, local_snap.channel);
    const char* tx_id    = mac_to_tx_id(local_snap.mac, local_snap.channel);
    uint16_t n_sc        = local_snap.num_subcarriers;

    JsonDocument doc;
    doc["event"]    = "csi";
    doc["rx_id"]    = RX_NODE_ID;
    doc["tx_id"]    = tx_id;
    doc["zone"]     = zone_str;
    doc["rssi"]     = (int)local_snap.rssi;
    doc["ch"]       = (int)local_snap.channel;
    doc["num_sc"]   = (int)n_sc;
    doc["csi_fmt"]  = "espressif_htltf_v1";
    doc["sc_order"] = local_snap.fft_reordered ? "ieee_80211n_56" : "esp32_raw";
    doc["ts"]       = local_snap.ts_ms;
    doc["ts_us"]    = local_snap.ts_us;

    JsonArray mag = doc["csi_mag"].to<JsonArray>();
    JsonArray pha = doc["csi_phase"].to<JsonArray>();

    for (uint16_t i = 0; i < n_sc && i < MAX_SUBCARRIERS; i++) {
      int8_t imag_val = local_snap.iq_buf[i * 2];
      int8_t real_val = local_snap.iq_buf[i * 2 + 1];
      float magnitude = sqrtf((float)(real_val * real_val + imag_val * imag_val));
      float phase     = atan2f((float)imag_val, (float)real_val);
      mag.add(serialized(String(magnitude, 2)));
      pha.add(serialized(String(phase, 3)));
    }

    serializeJson(doc, Serial);
    Serial.println();
  }
}

void loop() {
  // LED heartbeat: blink every 500ms — visible even if serial fails
  static bool led_state = false;
  static unsigned long last_blink = 0;
  if (millis() - last_blink > 500) {
    last_blink = millis();
    led_state = !led_state;
    digitalWrite(LED_PIN, led_state ? HIGH : LOW);
  }

  emit_heartbeat();
  emit_pending_csi_frames();

  // Defer CSI setup to loop() so serial output appears first
  static bool csi_initialized = false;
  static unsigned long last_csi_attempt = 0;
  if (!csi_initialized && (millis() - last_csi_attempt > 3000)) {
    last_csi_attempt = millis();
    Serial.println("{\"event\":\"attempt\",\"msg\":\"initializing_csi\"}");
    Serial.flush();
    
    bool ok = setup_csi_extraction();
    csi_initialized = true;
    
    Serial.println("{\"event\":\"attempt\",\"msg\":\"csi_done\",\"ok\":" + String(ok ? "true" : "false") + "}");
    Serial.flush();
  }

  if (!scan_running && (millis() - last_scan_ms >= SCAN_COOLDOWN_MS)) {
    int16_t scan_result = WiFi.scanNetworks(
      /*async=*/true,
      /*show_hidden=*/true,
      /*passive=*/true,
      /*max_ms_per_chan=*/CHANNEL_DWELL_MS,
      /*channel=*/SCAN_CHANNEL
    );
    last_scan_ms = millis();
    if (scan_result != WIFI_SCAN_FAILED) {
      scan_running = true;
    }
  }

  if (scan_running) {
    int n = WiFi.scanComplete();
    if (n == WIFI_SCAN_RUNNING) return;
    scan_running = false;
    for (int i = 0; i < n; i++) {
      String found_ssid = WiFi.SSID(i);
      found_ssid.trim();
      for (int b = 0; b < BEACON_COUNT; b++) {
        if (found_ssid.equalsIgnoreCase(BEACONS[b].ssid)) {
          const uint8_t* bssid_ptr = WiFi.BSSID(i);
          if (bssid_ptr) {
            memcpy(BEACONS[b].bssid, bssid_ptr, 6);
            BEACONS[b].hasBssid = true;
          }
          int raw_rssi = WiFi.RSSI(i);
          BEACONS[b].rssiBuf[BEACONS[b].bufIdx] = raw_rssi;
          BEACONS[b].bufIdx = (BEACONS[b].bufIdx + 1) % 3;
          if (BEACONS[b].bufIdx == 0) BEACONS[b].filled = true;
          int med_rssi = BEACONS[b].filled
            ? median3(BEACONS[b].rssiBuf[0], BEACONS[b].rssiBuf[1], BEACONS[b].rssiBuf[2])
            : raw_rssi;
          float kalman_rssi = updateKalman1D(BEACONS[b], (float)med_rssi);
          BEACONS[b].sampleCount++;

          JsonDocument doc;
          doc["rx_id"]    = RX_NODE_ID;
          doc["tx_id"]    = BEACONS[b].tx_id;
          doc["ssid"]     = BEACONS[b].ssid;
          doc["zone"]     = BEACONS[b].zone;
          doc["rssi"]     = serialized(String(kalman_rssi, 2));
          doc["med_rssi"] = med_rssi;
          doc["raw_rssi"] = raw_rssi;
          doc["ch"]       = WiFi.channel(i);
          doc["ts"]       = millis();
          doc["ts_us"]    = (uint32_t)esp_timer_get_time();
          serializeJson(doc, Serial);
          Serial.println();
          break;
        }
      }
    }
    WiFi.scanDelete();
  }
}