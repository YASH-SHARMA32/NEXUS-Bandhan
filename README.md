<div align="center">

# 🤖 NEXUS (बंधन)

### *One Brain. Three Brothers. Infinite Motion.*

### Wireless AI-Powered Tendon-Driven Bionic Hand

<img src="docs/images/banner.png" width="100%" alt="NEXUS Banner">

<p>

![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ESP32-S3](https://img.shields.io/badge/ESP32--S3-Firmware-red?style=for-the-badge)
![Arduino Mega](https://img.shields.io/badge/Arduino-Mega-00979D?style=for-the-badge&logo=arduino)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

</p>

*"Three different controllers from three different companies, somehow sharing one brain."*

<img src="docs/images/demo.gif" width="90%" alt="Demo">

</div>

---

# 🚀 Overview

**NEXUS (बंधन)** is a low-cost, open-source wireless bionic hand capable of mirroring human finger movements in real time using computer vision, embedded systems, and tendon-driven mechanics.

Unlike traditional robotic hands that rely on expensive sensors or custom hardware, NEXUS uses:

- 📷 A standard laptop webcam
- 🧠 MediaPipe AI hand tracking
- 📡 ESP32-S3 wireless communication
- 💪 Arduino Mega servo controller
- 🪝 Fishing-line tendons
- 📦 Cardboard mechanics

The entire system forms a complete cyber-physical pipeline:

```
Human Hand
      │
      ▼
 Laptop Camera
      │
      ▼
MediaPipe Vision
      │
      ▼
Finger Detection
      │
      ▼
 Wi-Fi (UDP)
      │
      ▼
 ESP32-S3
      │
      ▼
 UART
      │
      ▼
Arduino Mega
      │
      ▼
 Servo Motors
      │
      ▼
 Tendons
      │
      ▼
Cardboard Hand
```

The result is a fully wireless robotic hand capable of mirroring finger movements with approximately **80–120 ms total latency**.

---

# 💡 Why "बंधन"?

**बंधन (Bandhan)** means **bond** or **connection** in Hindi.

It represents two different ideas:

- The bond between three completely different controllers.
- The bond between a human hand and its robotic mirror.

The entire project exists because completely unrelated hardware somehow learned to cooperate.

---

# 🎯 Project Goals

NEXUS was built to demonstrate complete robotics engineering from software to mechanics.

The objectives were:

- Learn computer vision
- Learn embedded systems
- Build a complete robotic pipeline
- Create an affordable robotic hand
- Explore prosthetic concepts
- Publish everything as open source

Rather than buying an expensive robotic platform, the goal was:

> **Build everything from scratch. Understand every wire. Every byte. Every servo movement.**

---

# ✨ Features

## Computer Vision

- Real-time MediaPipe hand tracking
- 21 hand landmarks
- Scale invariant finger detection
- Automatic calibration
- Live OpenCV HUD
- Gesture visualization

---

## Embedded Systems

- ESP32-S3 Wi-Fi bridge
- Arduino Mega motion controller
- Custom UART protocol
- Custom UDP protocol
- XOR packet validation
- OLED live diagnostics

---

## Motion Control

- Tendon-driven fingers
- Smooth servo interpolation
- Velocity limiting
- Jerk limiting
- Kalman filtering
- Moving-average smoothing
- Deadband filtering

---

## Safety

- Communication watchdog
- Automatic neutral position
- Packet checksum validation
- Servo ramping
- Stable binary commands

---

## Mechanical Design

- Cardboard chassis
- Fishing-line tendons
- Rubber-band return mechanism
- Brass hinge joints
- Hot-glue assembly
- Modular design

---

# 🏗 System Architecture

```
                    NEXUS

      ┌────────────────────────────┐
      │ Laptop (Python + OpenCV)   │
      │ MediaPipe Vision           │
      └────────────┬───────────────┘
                   │
              UDP over Wi-Fi
                   │
                   ▼
      ┌────────────────────────────┐
      │ ESP32-S3                   │
      │ Wireless Bridge            │
      │ OLED Display               │
      └────────────┬───────────────┘
                   │
               UART 115200
                   │
                   ▼
      ┌────────────────────────────┐
      │ Arduino Mega               │
      │ Servo Controller           │
      └────────────┬───────────────┘
                   │
             Servo PWM Signals
                   │
                   ▼
      ┌────────────────────────────┐
      │ Tendon Driven Hand         │
      └────────────────────────────┘
```

---

# 👨‍👦‍👦 Meet The Three Brothers

## 🧠 Brother 1 — The Visionary

**Laptop + Python + Raspberry Pi**

Responsibilities

- Camera input
- MediaPipe tracking
- Finger calculations
- Filtering
- Gesture analysis
- UDP transmission

The Raspberry Pi contributes approximately **95% confidence** and **5% blinking LEDs**.

---

## 📡 Brother 2 — The Negotiator

**ESP32-S3**

Responsibilities

- Wi-Fi communication
- UDP receiver
- Packet verification
- UART bridge
- OLED display

Languages spoken:

- Wi-Fi
- UDP
- UART
- I²C

Family role:

> Keeps everyone talking.

---

## 💪 Brother 3 — The Backbone

**Arduino Mega**

Responsibilities

- Parse packets
- Drive servos
- Smooth motion
- Safety watchdog

No Linux.

No Wi-Fi.

No AI.

Just raw 5V determination.

---

# ⚡ Complete Data Pipeline

```
Camera
   │
   ▼
MediaPipe
   │
   ▼
21 Hand Landmarks
   │
   ▼
Finger Ratios
   │
   ▼
Angle Calculation
   │
   ▼
Deadband Filter
   │
   ▼
Moving Average
   │
   ▼
Kalman Filter
   │
   ▼
Motion Smoother
   │
   ▼
Binary Decision
   │
   ▼
UDP Packet
   │
   ▼
ESP32-S3
   │
   ▼
UART Frame
   │
   ▼
Arduino Mega
   │
   ▼
Servo Ramping
   │
   ▼
Fishing Line
   │
   ▼
Finger Motion
```

---

# 📊 Engineering Snapshot

| Metric | Value |
|----------|--------|
| Total Development Time | 200–250 Hours |
| Total Git Commits | 120+ |
| Python Code | ~3800 Lines |
| Embedded Firmware | ~1000 Lines |
| Total Codebase | ~4800 Lines |
| Core Files | 6 |
| Controllers | 3 |
| Servos | 5 |
| Degrees of Freedom | 5 |
| Communication Protocols | UDP + UART |
| Latency | 80–120 ms |
| FPS | 22–32 |
| Electronics Cost | ~$30 |

---

# 🧰 Hardware

| Component | Model |
|------------|-------|
| Laptop | HP 15s |
| Camera | HP TrueVision HD |
| ESP32 | ESP32-S3 DevKitC-1 |
| Arduino | Mega 2560 |
| Raspberry Pi | Pi 4 Model B |
| Camera Module | OV7670 |
| OLED | SH1106 |
| Servos | SG90 |
| Power | 5V 3A |
| Tendons | 20 lb Braided Fishing Line |
| Frame | Corrugated Cardboard |

---

# 📦 Repository Structure

```text
nexus-bionic-hand/

├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md

├── python/
│   ├── nexus_hand.py
│   ├── udp_sender.py
│   └── requirements.txt

├── firmware/
│   ├── esp32-s3/
│   └── mega/

├── hardware/
│   ├── wiring_diagram.png
│   ├── pinout.png
│   └── voltage_divider.png

├── docs/
│   ├── images/
│   ├── gifs/
│   └── architecture/

├── config/
│   └── nexus_config.json
```

---

# 🚀 Quick Start

If you just want to see NEXUS moving in under 10 minutes, follow the quick setup below.

*(Continues in Part 2...)*


# 🚀 Quick Start

Get NEXUS up and running in under 10 minutes.

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/nexus-bionic-hand.git

cd nexus-bionic-hand
```

---

## 2. Install Python Dependencies

```bash
pip install -r python/requirements.txt
```

or

```bash
pip install opencv-python mediapipe numpy
```

---

## 3. Upload Firmware

Flash both controllers.

### ESP32-S3

```
firmware/
└── esp32-s3/
      nexus_bridge.ino
```

Board:

```
ESP32S3 Dev Module
```

Settings

- CPU Frequency : 240 MHz
- USB CDC : Enabled
- Upload Speed : 921600

---

### Arduino Mega

```
firmware/
└── mega/
      nexus_muscle.ino
```

Board

```
Arduino Mega 2560
```

Upload normally.

---

## 4. Connect Hardware

```
Laptop
   │
Wi-Fi UDP
   │
ESP32-S3
   │
UART
   │
Arduino Mega
   │
PWM
   │
5 Servos
```

---

## 5. Power

⚠ Never power the servos from the Arduino.

Use

- 5V 3A Adapter
- USB Power Bank

Always connect all grounds together.

---

## 6. Run

```bash
python python/nexus_hand.py --nexus-ip YOUR_ESP32_IP
```

Example

```bash
python python/nexus_hand.py --nexus-ip 192.168.146.24
```

Within seconds

```
PC Link OK
```

appears on the OLED.

Move your hand.

The robotic hand mirrors you.

---

# 🔌 Hardware Connections

## ESP32 ⇄ Arduino Mega

| ESP32 | Arduino Mega |
|---------|--------------|
| GPIO17 TX | RX1 Pin 19 |
| GPIO16 RX | TX1 Pin 18 (Voltage Divider) |
| GND | GND |

Voltage divider

```
Mega TX
   │
 1kΩ
   │──────────── ESP32 RX
   │
 2kΩ
   │
 GND
```

---

## Servo Connections

| Finger | Mega Pin |
|----------|----------|
| Thumb | 2 |
| Index | 3 |
| Middle | 4 |
| Ring | 5 |
| Pinky | 6 |
| Wrist | 7 (Future) |

---

## Power Wiring

```
5V Supply
   │
──────────────
│    │    │
S1   S2   S3...
│
Arduino GND
│
ESP32 GND
```

Common Ground is mandatory.

---

# 🧠 Computer Vision

NEXUS uses **Google MediaPipe Hands**.

The model detects **21 landmarks** every frame.

```
Thumb

2
3
4

Index

5
6
7
8

Middle

9
10
11
12

Ring

13
14
15
16

Pinky

17
18
19
20
```

Instead of comparing angles directly,

NEXUS computes a scale-independent ratio.

\[
r=\frac{Distance(Wrist,Tip)}{Distance(Wrist,Knuckle)}
\]

Advantages

✅ Camera distance independent

✅ Robust

✅ Fast

✅ Easy to calibrate

---

# 🎯 Finger Detection Pipeline

For every frame

```
Camera

↓

MediaPipe

↓

21 Landmarks

↓

Finger Ratio

↓

Angle Mapping

↓

Deadband

↓

Moving Average

↓

Kalman Filter

↓

Motion Smoother

↓

Binary Decision

↓

UDP Packet
```

---

# 🎛 Filtering Pipeline

Several filters remove jitter.

## Deadband

Ignore

```
< 3°
```

movement.

Removes tiny camera noise.

---

## Moving Average

Window Size

```
7 Samples
```

Produces smoother motion.

---

## Kalman Filter

Each finger has an independent

1D Kalman Filter.

Parameters

```
Process Noise

5e-4

Measurement Noise

0.3
```

This removes random landmark vibration.

---

## Motion Smoother

Velocity limit

```
12° / Frame
```

Jerk limit

```
5° / Frame²
```

This prevents unnatural snapping.

---

# 📡 Wireless Communication

Communication happens in two stages.

```
Laptop

↓

UDP

↓

ESP32

↓

UART

↓

Mega
```

---

## UDP Packet

Example

```
$START,
L,
0,
180,
180,
0,
0,
UNKNOWN,
0.95;

R,
0,
180,
180,
0,
0,
UNKNOWN,
0.95

#A3
```

Checksum

```
XOR
```

Validation

```
YES
```

Port

```
8888
```

Average latency

```
<5ms
```

---

## UART Packet

The ESP32 converts the UDP packet into

```
9 Bytes
```

```
[AA]

Thumb

Index

Middle

Ring

Pinky

Wrist

Checksum

[55]
```

Baudrate

```
115200
```

---

# ⚙ Servo Motion

Incoming commands

```
0°

or

180°
```

The Mega never jumps directly.

Instead

```
Current

↓

Target

↓

Current += 2°
```

This creates smooth movement.

No visible jerk.

No sudden snapping.

---

# 🛡 Safety Features

✔ Packet checksum

✔ UART validation

✔ Wi-Fi watchdog

✔ Neutral position fallback

✔ Servo ramping

✔ Motion filtering

✔ Binary protocol

If communication stops

```
1 Second
```

All fingers return to

```
90°
```

---

# 📈 Performance

| Metric | Result |
|---------|---------|
| FPS | 22–32 |
| Total Latency | 80–120 ms |
| UDP Latency | <5 ms |
| Packet Loss | <0.1% |
| CPU Usage | 35–45% |
| RAM Usage | 400–500 MB |
| Communication Range | 10–15 m |

---

# 📊 Code Statistics

| Module | Lines |
|---------|------:|
| Python Vision | ~3800 |
| ESP32 Firmware | ~450 |
| Mega Firmware | ~550 |
| **Total** | **~4800 Lines** |

Development Time

```
200–250 Hours
```

Git Commits

```
120+
```

Built completely from scratch.

No templates.

No robotics framework.

Just code, debugging, and determination.

---

# 🛠 Mechanical Design

Material

- Corrugated cardboard
- Fishing line
- Rubber bands
- Hot glue
- Brass paper fasteners

Each finger has

- 1 Degree of Freedom
- Tendon Pull
- Elastic Return

No gears.

No springs.

No expensive mechanics.

Simple.

Cheap.

Reliable.

---

# 📸 Screenshots

Add these images inside

```
docs/images/
```

```
banner.png

demo.gif

finished_hand.jpg

wiring.jpg

oled.jpg

gui.png

architecture.png

servo_mount.jpg

tendon_system.jpg
```

GitHub will automatically display them beautifully.

---

*(Continues in Part 3: Calibration, Troubleshooting, Roadmap, FAQ, Contributing, License, Acknowledgments, and GitHub polish.)*




































<div align="center">

# 🤖 NEXUS (बंधन)

### Wireless Vision-Based Tendon-Driven Bionic Hand

**One Brain • Three Controllers • Real-Time Motion**

<p>
<img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
<img src="https://img.shields.io/badge/ESP32--S3-Firmware-red.svg">
<img src="https://img.shields.io/badge/Arduino-Mega-teal.svg">
<img src="https://img.shields.io/badge/OpenCV-Computer_Vision-success.svg">
<img src="https://img.shields.io/badge/MediaPipe-Hand_Tracking-orange.svg">
<img src="https://img.shields.io/badge/License-MIT-green.svg">
</p>

*A low-cost wireless bionic hand that mirrors your real hand using computer vision, custom communication protocols, and tendon-driven mechanics.*

</div>

---

# 🎥 Demo

> **Demo Video:** *(Coming Soon)*

Replace with your video:

```
https://youtu.be/YOUR_VIDEO
```

Demo GIF:

```
docs/demo.gif
```

---

# 📖 Overview

NEXUS (बंधन) is an open-source **wireless bionic hand** that tracks a user's hand using a standard laptop webcam and mirrors the movement on a tendon-driven robotic hand in real time.

Unlike many robotic hand projects, NEXUS is designed to be **simple, affordable, and reproducible**, using cardboard, fishing line, hobby servos, and readily available development boards.

The complete pipeline is:

```
Human Hand
      │
      ▼
Laptop Webcam
      │
      ▼
Python + OpenCV + MediaPipe
      │
      ▼
Wi-Fi (UDP)
      │
      ▼
ESP32-S3
      │
      ▼
UART
      │
      ▼
Arduino Mega
      │
      ▼
Servo Motors
      │
      ▼
Fishing-Line Tendons
      │
      ▼
Bionic Hand
```

---

# ✨ Features

- 🖐 Real-time hand tracking
- 🎯 MediaPipe-based finger detection
- 📡 Wireless UDP communication
- ⚡ ESP32-S3 communication bridge
- 🤖 Arduino Mega servo controller
- 🧵 Tendon-driven finger mechanism
- 📺 OLED live status display
- 🔒 Custom checksum protocol
- 🛡 Built-in communication failsafe
- 📂 Fully open source

---

# 📊 Engineering Snapshot

| Metric | Value |
|---------|-------|
| Total Codebase | **6200+ Lines** |
| Python Vision System | **3800+ Lines** |
| Raspberry Pi Firmware | **800+ Lines** |
| ESP32-S3 Firmware | **800+ Lines** |
| Arduino Mega Firmware | **800+ Lines** |
| Controllers | 3 |
| Servo Motors | 5 |
| Computer Vision | MediaPipe |
| Communication | UDP + UART |
| Tracking FPS | 22–28 FPS |
| Latency | <5 ms |
| Cost | ~$30 |

---

# 🏗 System Architecture

| Device | Responsibility |
|---------|----------------|
| **Laptop** | Captures webcam feed and performs MediaPipe hand tracking |
| **Raspberry Pi + OV7670** | Reserved for future onboard edge-AI vision processing |
| **ESP32-S3** | Wireless communication bridge between PC and Arduino |
| **Arduino Mega** | Controls servos and tendon movement |
| **Cardboard Hand** | Physical tendon-driven robotic hand |

---

# ⚙ Hardware

- Arduino Mega 2560
- ESP32-S3
- Raspberry Pi + OV7670 *(optional)*
- 5x Servo Motors
- SH1106 OLED Display
- Corrugated Cardboard
- Fishing Line
- Rubber Bands
- Hot Glue
- External 5V Power Supply

---

# 💻 Software

## Python

- OpenCV
- MediaPipe
- NumPy

Install:

```bash
pip install opencv-python mediapipe numpy
```

---

## ESP32

Required Libraries

- WiFi
- WiFiUDP
- Adafruit GFX
- Adafruit SH110X

---

## Arduino Mega

Uses the built-in Servo library.

---

# 🚀 Quick Start

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/NEXUS.git
```

Install Python dependencies.

```bash
pip install opencv-python mediapipe numpy
```

Upload:

- ESP32 firmware
- Arduino Mega firmware

Run

```bash
python nexus_hand.py
```

Move your hand in front of the webcam and watch the robotic hand mirror your motion.

---

# 📡 Communication Pipeline

```
Laptop
   │
UDP Packets
   │
ESP32-S3
   │
UART
   │
Arduino Mega
   │
Servo Controller
   │
Fishing-Line Tendons
   │
Cardboard Fingers
```

---

# 📁 Repository Structure

```
NEXUS/

├── firmware/
│   ├── esp32/
│   └── mega/
│
├── python/
│   ├── nexus_hand.py
│   ├── filters.py
│   ├── communication.py
│   └── calibration.py
│
├── docs/
│   ├── images/
│   ├── demo.gif
│   ├── wiring.png
│   └── architecture.png
│
├── hardware/
│
├── LICENSE
└── README.md
```

---

# 📈 Current Status

- ✅ Real-time hand tracking
- ✅ Wireless communication
- ✅ Tendon-driven mechanism
- ✅ OLED system monitoring
- ✅ Custom communication protocol
- ✅ Safety watchdog
- ✅ 3 fingers fully calibrated
- 🔄 Remaining fingers under calibration
- 🔄 Wrist rotation in development

---

# 🛣 Roadmap

- [ ] Full five-finger calibration
- [ ] Wrist rotation
- [ ] Force feedback
- [ ] EMG muscle control
- [ ] TensorFlow Lite gesture recognition
- [ ] Mobile app
- [ ] 3D printable chassis
- [ ] Battery-powered version

---

# 📸 Gallery

```
docs/images/hand.jpg

docs/images/electronics.jpg

docs/images/wiring.jpg

docs/images/demo.gif
```

---

# 🤝 Contributing

Contributions are always welcome.

Ideas include:

- Better mechanical design
- 3D printable hand
- Improved filtering
- Faster communication
- Better gesture recognition
- Documentation improvements

Feel free to open an Issue or submit a Pull Request.

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgements

- Google MediaPipe
- OpenCV
- Espressif
- Arduino
- Adafruit
- Open Source Robotics Community

---

<div align="center">

## ⭐ Support the Project

If you enjoyed this project or found it useful, consider giving it a ⭐ on GitHub.

It helps more makers discover NEXUS and motivates future development.

---

**Built with**

☕ Coffee

💻 6200+ Lines of Code

🧵 Fishing Line

📦 Cardboard

🔥 Hot Glue

❤️ Endless Debugging

*"Because every great engineering project begins with an impossible idea."*

**NEXUS (बंधन)**

</div>
