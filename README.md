# RuView Wi-Fi Sensing

Python-based Wi-Fi channel-state-information (CSI) sensing and visualization
with ESP32 firmware and PC-controlled GPIO outputs.

## Setup

1. Install Python 3.10 or newer.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Copy `config.example.yaml` to `config.yaml` and set the serial ports, room
   layout, and the exact 2.4 GHz hotspot SSID used by your boards.
4. In each firmware sketch folder you plan to flash, copy
   `secrets.example.h` to `secrets.h` and enter your local hotspot name and
   password. `secrets.h` is ignored by Git.
5. Open and flash the needed sketch folder in Arduino IDE with the ESP32 board
   support installed. Close Arduino Serial Monitor before running the PC app.
6. Run the controller: `python exhibition_controller.py`.

The Standard ESP32 actuator is configured separately as `serial.actuator_port`
in `config.yaml` (default example: `COM11`). It alone receives bulb commands.
The PC performs calibration and detection; the ESP32 only switches the
requested output. The configured example maps `node_1`, `node_2`, and `node_3`
to GPIO 2, 4, and 5 respectively. Use a suitable driver for loads that exceed
GPIO current limits.

Calibration starts on controller launch when enabled in `room_calibration`.
Keep the sensing area still during calibration. Bulb outputs remain disarmed
until the PC collects enough measurements for each bulb zone. If calibration
fails, check the dashboard's per-zone sample counts and serial frame counts,
and verify every configured COM port is available.

## Checks

Run focused tests with `python -m pytest tests -q`, or run the project checks
with `python run_all_tests.py`. Hardware-dependent checks require the connected
ESP32 boards and a configured `config.yaml`.

## Firmware folders

Each folder under `firmware/` is an independent Arduino sketch. Do not compile
multiple sketches together. The legacy and beacon/receiver sketches are
alternative firmware variants, not required for the default three-board setup.

## Configuration files

Do not commit `config.yaml`, `firmware/*/secrets.h`, calibration captures, or
live hardware logs. Update the example files without adding personal Wi-Fi
credentials.
