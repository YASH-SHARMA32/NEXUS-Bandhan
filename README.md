<div align="center">

<!-- BANNER IMAGE (replace with your own) -->
<img src="docs/images/banner.png" width="100%" alt="NEXUS Banner" style="border-radius:12px;">

# 🤖 NEXUS (बंधन)

### *One Brain. Three Brothers. Infinite Motion.*

### Wireless Vision‑Based Tendon‑Driven Bionic Hand

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge&logo=open-source-initiative&logoColor=white" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ESP32--S3-Firmware-red?style=for-the-badge&logo=espressif&logoColor=white" alt="ESP32‑S3">
  <img src="https://img.shields.io/badge/Arduino-Mega-00979D?style=for-the-badge&logo=arduino&logoColor=white" alt="Arduino Mega">
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Hand_Tracking-blue?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="Status">
</p>

*“Three different controllers from three different companies, somehow sharing one brain.”*

<!-- DEMO GIF -->
<img src="docs/images/demo.gif" width="90%" alt="Demo" style="border-radius:8px;">

</div>

---

# 🚀 Overview

**NEXUS (बंधन)** is a low‑cost, open‑source wireless bionic hand that mirrors human finger movements in real time using computer vision, embedded systems, and tendon‑driven mechanics.  
Unlike traditional robotic hands that rely on expensive sensors or custom hardware, NEXUS uses:

- 📷 A standard laptop webcam  
- 🧠 MediaPipe AI hand tracking  
- 📡 ESP32‑S3 wireless communication  
- 💪 Arduino Mega servo controller  
- 🪝 Fishing‑line tendons  
- 📦 Cardboard mechanics  

The complete cyber‑physical pipeline runs from your real hand to a robotic hand in **~80‑120 ms**:

---

# 💡 Why "बंधन"?

**बंधन (Bandhan)** means **bond** or **connection** in Hindi.  
It represents the unlikely bond between three completely different microcontrollers, and the bond between a human hand and its robotic mirror.

---

# 👨‍👦‍👦 Meet The Three Brothers

## 🧠 Brother 1 — The Visionary  
**Laptop + Python + Raspberry Pi (decorative)**  
- Camera input, MediaPipe tracking, finger calculations, filtering, UDP transmission.  
- The Raspberry Pi contributes approximately **95% confidence** and **5% blinking LEDs**.

## 📡 Brother 2 — The Diplomat  
**ESP32‑S3**  
- Wi‑Fi communication, UDP receiver, packet verification, UART bridge, OLED display.  
- Speaks Wi‑Fi, UDP, UART, and I²C.  
- Family role: *Keeps everyone talking.*

## 💪 Brother 3 — The Backbone  
**Arduino Mega**  
- Parse packets, drive servos, smooth motion, safety watchdog.  
- No Linux. No Wi‑Fi. No AI. Just raw **5 V determination**.

---

# ✨ Features

### Computer Vision
- Real‑time MediaPipe hand tracking (21 landmarks)  
- Scale‑invariant finger‑curl ratio  
- Automatic calibration  
- Live OpenCV HUD with gesture labels, FPS, anomaly indicators  

### Embedded Systems
- ESP32‑S3 Wi‑Fi bridge + OLED live diagnostics  
- Arduino Mega motion controller  
- Custom UDP protocol with XOR checksum  
- Custom binary UART protocol  

### Motion Control
- Tendon‑driven fingers with rubber‑band return  
- Smooth servo interpolation (2° per loop)  
- Kalman filter, moving‑average, deadband, velocity/jerk limiter  

### Safety
- Independent communication watchdog  
- Automatic neutral position (90°) after 1 s of silence  
- Packet checksum validation on both links  
- Servo ramping prevents sudden jerks  

### Mechanical Design
- Corrugated cardboard chassis  
- Braided fishing‑line tendons  
- Brass paper‑fastener finger joints  
- Hot‑glue assembly – no 3D‑printed parts  

---

# ⚡ Complete Data Pipeline

---

# 📊 Engineering Snapshot

| Metric | Value |
|--------|-------|
| Total Development Time | 200–250 h |
| Git Commits | 120+ |
| Python Code | ~3,800 lines |
| Embedded Firmware | ~1,000 lines |
| **Total Code** | **~4,800 lines** |
| Controllers | 3 |
| Servos | 5 |
| Degrees of Freedom | 5 |
| Communication | UDP + UART |
| Latency | 80–120 ms |
| FPS | 22–32 |
| Electronics Cost | ~$30 |

---

# 🧰 Hardware

| Component | Model |
|-----------|-------|
| Laptop | Any with webcam |
| ESP32 | ESP32‑S3‑DevKitC‑1 |
| Arduino | Mega 2560 |
| Raspberry Pi | Pi 4 / Zero 2W (decorative) |
| Camera Module | OV7670 (decorative) |
| OLED | SH1106 1.3″ 128×64 I²C |
| Servos | 5× SG90 / MG90S |
| Power | 5 V 3 A external supply |
| Tendons | 20 lb braided fishing line |
| Frame | Corrugated cardboard |

---

# 🔌 Hardware Connections

### ESP32‑S3 ⇄ Arduino Mega (UART + Voltage Divider)

| ESP32‑S3 | Arduino Mega | Notes |
|----------|--------------|-------|
| GPIO17 (TX2) | RX1 (pin 19) | Direct wire |
| GPIO16 (RX2) | TX1 (pin 18) | **1 kΩ series + 2 kΩ to GND** |
| GND | GND | Common ground mandatory |

**Voltage divider** protects the ESP32’s 3.3 V RX pin from the Mega’s 5 V TX.

### Servo Pinout

| Finger | Mega Pin |
|--------|----------|
| Thumb | 2 |
| Index | 3 |
| Middle | 4 |
| Ring | 5 |
| Pinky | 6 |
| Wrist (future) | 7 |

⚠ **Never power servos from the Arduino’s 5 V pin.** Use a dedicated supply and **common ground**.

---

# 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/nexus-bionic-hand.git
   cd nexus-bionic-hand
