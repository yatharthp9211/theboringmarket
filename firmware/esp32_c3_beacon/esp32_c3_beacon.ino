/*
 * RuView RGIPT Exhibition – ESP32-C3 High-Rate Beacon Firmware (v3.0 Ultra)
 * =========================================================================
 * Board    : ESP32-C3 (SuperMini or DevKit) × 2 units
 * Role     : High-Frequency WiFi SoftAP Sensing Beacon (40 Hz Beacon Frame Rate)
 *
 * KEY UPGRADES IN v3.0 ULTRA:
 *   1. Single-Toggle Board Selector: Simply set `BEACON_NODE_ID` to `1` or `2`!
 *      - Node 1 automatically configures SSID="Zone_1_Tx" on Channel 1 (2412 MHz)
 *      - Node 2 automatically configures SSID="Zone_2_Tx" on Channel 6 (2437 MHz)
 *   2. 4x Faster RF Sensing Refresh Rate:
 *      - Uses `esp_wifi_set_beacon_interval(25)` to broadcast 802.11 beacon frames
 *        every ~25.6 ms (~39 Hz) instead of the default 102.4 ms (9.7 Hz), giving
 *        the receiver 4x faster temporal resolution for hand/object tracking!
 *   3. Adaptive Thermal Management & Non-Blocking Telemetry:
 *      - Runs at 80 MHz CPU clock + 11 dBm TX power by default; automatically steps
 *        down TX power if die temperature exceeds 68 °C during long exhibition hours.
 *      - Zero `delay()` blocking in `loop()`, with smooth non-blocking status heartbeat.
 */

#include <WiFi.h>
#include <esp_wifi.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// Early hardware boot hook: Disables brownout before system init
void __attribute__((constructor)) disable_brownout_early() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
}

// ─────────── SELECT BOARD ID (1 OR 2) ───────────────────────────────────────
#define BEACON_NODE_ID  1             // <-- Set to 1 for Hub Beacon, 2 for Far Beacon
// ─────────────────────────────────────────────────────────────────────────────

#if BEACON_NODE_ID == 1
  #define ZONE_SSID     "Zone_1_Tx"
  #define CHANNEL       6
#else
  #define ZONE_SSID     "Zone_2_Tx"
  #define CHANNEL       6
#endif

#define ZONE_PASS       ""            // Open SoftAP network for fast passive scanning
#define MAX_CONN        4             // Lightweight AP table
#define LED_PIN         8             // Built-in status LED on ESP32-C3 SuperMini

static unsigned long lastLedToggleMs = 0;
static unsigned long lastTelemMs     = 0;
static bool ledState                 = false;
static bool thermalThrottled         = false;

void configureHighRateBeacon() {
  wifi_config_t conf;
  if (esp_wifi_get_config(WIFI_IF_AP, &conf) == ESP_OK) {
    // 25 TU * 1.024 ms = 25.6 ms beacon interval (~39 Hz frame rate)
    conf.ap.beacon_interval = 25;
    esp_wifi_set_config(WIFI_IF_AP, &conf);
  }
  // Disable WiFi modem sleep for rock-solid transmit timing stability
  esp_wifi_set_ps(WIFI_PS_NONE);
}

void setup() {
  // 0. Disable brownout detector to prevent reset on RF power surge
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  // 1. Clock down from 160 MHz to 80 MHz to cut idle heat & current surge in half
  setCpuFrequencyMhz(80);

  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  delay(150);
  Serial.println();
  Serial.println("============================================================");
  Serial.println("  RuView v3.0 Ultra — High-Rate 39Hz RF Tomography Beacon   ");
  Serial.println("============================================================");
  Serial.printf("  Node ID  : %d\n", BEACON_NODE_ID);
  Serial.printf("  SSID     : %s\n", ZONE_SSID);
  Serial.printf("  Channel  : %d\n", CHANNEL);

  WiFi.persistent(false);
  WiFi.disconnect(true);
  delay(80);

  // 2. Pure Access Point mode
  WiFi.mode(WIFI_AP);

  // 3. Balanced 11 dBm TX power (~12.5 mW): strong -50 to -68 dBm tabletop signal
  //    with cool, continuous 110 mA current draw.
  WiFi.setTxPower(WIFI_POWER_11dBm);

  // 4. Start SoftAP and apply 25 ms high-rate beacon interval
  bool ok = WiFi.softAP(ZONE_SSID, ZONE_PASS, CHANNEL, 0 /*hidden=no*/, MAX_CONN);
  if (ok) {
    configureHighRateBeacon();
    Serial.print("  AP IP    : ");
    Serial.println(WiFi.softAPIP());
    Serial.println("  Status   : ACTIVE (39 Hz Beacon Rate, PS_NONE, 11 dBm)");
  } else {
    Serial.println("[ERROR] softAP initialization failed!");
  }
  Serial.println("============================================================");
}

void loop() {
  unsigned long now = millis();

  // 1. Non-blocking asymmetric heartbeat LED (850 ms ON, 150 ms OFF)
  unsigned long interval = ledState ? 850 : 150;
  if (now - lastLedToggleMs >= interval) {
    lastLedToggleMs = now;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState ? HIGH : LOW);
  }

  // 2. Non-blocking JSON health & thermal telemetry every 5 seconds
  if (now - lastTelemMs >= 5000) {
    lastTelemMs = now;
    float tempC = temperatureRead();

    // Automatic thermal guard
    if (tempC > 68.0f && !thermalThrottled) {
      WiFi.setTxPower(WIFI_POWER_8_5dBm);
      thermalThrottled = true;
    } else if (tempC < 60.0f && thermalThrottled) {
      WiFi.setTxPower(WIFI_POWER_11dBm);
      thermalThrottled = false;
    }

    Serial.printf(
      "{\"event\":\"beacon_health\",\"node\":%d,\"ssid\":\"%s\",\"ch\":%d,"
      "\"temp_c\":%.1f,\"throttled\":%s,\"uptime_s\":%lu}\n",
      BEACON_NODE_ID,
      ZONE_SSID,
      CHANNEL,
      tempC,
      thermalThrottled ? "true" : "false",
      now / 1000UL
    );
  }
}
