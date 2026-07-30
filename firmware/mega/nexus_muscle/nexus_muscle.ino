// ============================================================================
//  NEXUS MEGA — Arduino Mega 2560 Firmware  v1.0
//  "The Muscle & Safety Layer"
//
//  Role: Receive 9-byte binary frames from ESP32-S3 over UART,
//        drive 6 servos with smooth ramping, and fail safe to neutral
//        if communication is lost.
//
//  Hardware:
//    - Arduino Mega 2560
//    - 6 servos: Thumb(2), Index(3), Middle(4), Ring(5), Pinky(6), Wrist(7)
//    - Servo power: external 5–6 V supply (NOT Mega's 5V pin)
//    - Optional SH1106 128×64 OLED (I²C: SDA=20, SCL=21, addr=0x3C)
//    - ESP32-S3 → Serial1 (RX=19, TX=18) at 115200 baud
//
//  Frame from ESP32-S3 (9 bytes):
//    [0xAA] [Thumb] [Index] [Middle] [Ring] [Pinky] [Wrist] [XOR] [0x55]
//
//  Angles: 0 = OPEN, 180 = CLOSED. Wrist always 90 (neutral).
//
//  Libraries required:
//    - Servo.h        (built-in)
//    - Wire.h         (built-in)
//    - Adafruit SH110X + Adafruit GFX  (if OLED enabled)
// ============================================================================

#include <Servo.h>
#include <Wire.h>

// ── Enable OLED? Set to 1 to use the SH1106 display ─────────────────────────
#define USE_OLED  1

#if USE_OLED
  #include <Adafruit_GFX.h>
  #include <Adafruit_SH110X.h>
#endif

// ============================================================================
//  SECTION 1 — Pin & Frame Definitions
// ============================================================================

#define PIN_THUMB    2
#define PIN_INDEX    3
#define PIN_MIDDLE   4
#define PIN_RING     5
#define PIN_PINKY    6
#define PIN_WRIST    7

#define N_SERVOS     6

#define FRAME_START  0xAA
#define FRAME_END    0x55
#define FRAME_LEN    9       // total bytes in one frame
#define PAYLOAD_LEN  7       // 6 angles + 1 checksum

// ============================================================================
//  SECTION 2 — Timing Constants
// ============================================================================

#define RAMP_STEP         2    // degrees moved per main-loop iteration
#define FAILSAFE_MS    1000    // ms without data → go neutral
#define OLED_UPDATE_MS  250    // ms between OLED refreshes

// ============================================================================
//  SECTION 3 — Servo Objects & State
// ============================================================================

static Servo  servos[N_SERVOS];
static const uint8_t SERVO_PINS[N_SERVOS] = {
    PIN_THUMB, PIN_INDEX, PIN_MIDDLE, PIN_RING, PIN_PINKY, PIN_WRIST
};
static const char* SERVO_NAMES[N_SERVOS] = {
    "THM", "IDX", "MID", "RNG", "PNK", "WRS"
};

static uint8_t target_angle[N_SERVOS];    // commanded by UART
static float   current_angle[N_SERVOS];   // smoothed (float for sub-degree ramp)

// ============================================================================
//  SECTION 4 — UART State Machine
// ============================================================================

typedef enum {
    SM_WAIT_START,   // waiting for 0xAA
    SM_READING,      // receiving payload bytes
    SM_WAIT_END      // expecting 0x55
} uart_state_t;

static uart_state_t  uart_state = SM_WAIT_START;
static uint8_t       uart_buf[PAYLOAD_LEN];   // 6 angles + checksum
static uint8_t       uart_idx   = 0;

// ============================================================================
//  SECTION 5 — Link & Failsafe State
// ============================================================================

static uint32_t last_valid_frame_ms = 0;
static bool     link_ok             = false;
static uint32_t total_frames        = 0;
static uint32_t bad_frames          = 0;

// ============================================================================
//  SECTION 6 — OLED
// ============================================================================

#if USE_OLED
  #define OLED_ADDR  0x3C
  static Adafruit_SH1106G oled(128, 64, &Wire, -1);
  static bool       oled_ok       = false;
  static uint32_t   last_oled_ms  = 0;
#endif

// ============================================================================
//  SECTION 7 — Forward Declarations
// ============================================================================

static void poll_uart();
static void process_frame();
static void check_failsafe();
static void update_servos();
static void go_neutral();
static void update_oled();
static uint8_t xor_check(const uint8_t* buf, uint8_t len);

// ============================================================================
//  SECTION 8 — setup()
// ============================================================================

void setup() {
    // Debug serial (USB)
    Serial.begin(115200);
    Serial.println(F("\n[MEGA] NEXUS Muscle Layer v1.0 — Booting..."));

    // UART from ESP32-S3
    Serial1.begin(115200);
    Serial.println(F("[MEGA] Serial1 ready @ 115200 (RX=19, TX=18)"));

    // Initialise servo angles to neutral
    for (int i = 0; i < N_SERVOS; i++) {
        target_angle[i]  = 90;
        current_angle[i] = 90.0f;
        servos[i].attach(SERVO_PINS[i]);
        servos[i].write(90);
    }
    Serial.println(F("[MEGA] 6 servos attached and set to 90 (neutral)"));

    // Seed the failsafe timer so we don't immediately go lost
    last_valid_frame_ms = millis();

#if USE_OLED
    Wire.begin();
    oled_ok = oled.begin(OLED_ADDR, true);
    if (oled_ok) {
        oled.clearDisplay();
        oled.setTextColor(SH110X_WHITE);
        oled.setTextSize(1);
        oled.setCursor(0, 0);
        oled.print(F("NEXUS MEGA v1.0"));
        oled.setCursor(0, 12);
        oled.print(F("Waiting for ESP32..."));
        oled.display();
        Serial.println(F("[MEGA] OLED OK"));
    } else {
        Serial.println(F("[WARN] OLED not found"));
    }
#endif

    Serial.println(F("[MEGA] Boot complete. Entering main loop."));
}

// ============================================================================
//  SECTION 9 — loop()  (bare-metal super-loop, no delay())
// ============================================================================

void loop() {
    poll_uart();       // 1. Feed incoming bytes into the state machine
    check_failsafe();  // 2. Force neutral if link is lost
    update_servos();   // 3. Ramp each servo toward its target

#if USE_OLED
    update_oled();     // 4. Refresh display (only every OLED_UPDATE_MS)
#endif
}

// ============================================================================
//  SECTION 10 — UART State Machine
// ============================================================================

/**
 * Non-blocking: reads all available bytes from Serial1 and feeds them
 * through the state machine one byte at a time.
 */
static void poll_uart() {
    while (Serial1.available()) {
        uint8_t b = (uint8_t)Serial1.read();

        switch (uart_state) {

            // ── State A: wait for start marker ─────────────────────────
            case SM_WAIT_START:
                if (b == FRAME_START) {
                    uart_idx   = 0;
                    uart_state = SM_READING;
                }
                break;

            // ── State B: collect 7 payload bytes (6 angles + checksum) ─
            case SM_READING:
                uart_buf[uart_idx++] = b;
                if (uart_idx == PAYLOAD_LEN) {
                    uart_state = SM_WAIT_END;
                }
                break;

            // ── State C: wait for end marker, then validate ─────────────
            case SM_WAIT_END:
                if (b == FRAME_END) {
                    process_frame();       // validate & apply
                } else {
                    // Corrupt frame — discard
                    bad_frames++;
                    Serial.print(F("[WARN] Bad frame end byte: 0x"));
                    Serial.println(b, HEX);
                }
                uart_state = SM_WAIT_START;  // always reset
                break;
        }
    }
}

// ============================================================================
//  SECTION 11 — Frame Processor
// ============================================================================

/**
 * Called when a full frame (start + 7 bytes + end) has been received.
 * Validates XOR checksum; if OK, updates target angles and resets failsafe.
 *
 * uart_buf layout:
 *   [0]=Thumb [1]=Index [2]=Middle [3]=Ring [4]=Pinky [5]=Wrist [6]=Checksum
 */
static void process_frame() {
    // Compute XOR of the 6 angle bytes
    uint8_t computed = xor_check(uart_buf, 6);
    uint8_t received = uart_buf[6];

    if (computed != received) {
        bad_frames++;
        char _buf[56];
        snprintf(_buf, sizeof(_buf), "[WARN] Chk mismatch: got=0x%02X exp=0x%02X",
                 received, computed);
        Serial.println(_buf);
        return;
    }

    // Valid frame — update targets
    for (int i = 0; i < N_SERVOS; i++) {
        target_angle[i] = constrain(uart_buf[i], 0, 180);
    }

    last_valid_frame_ms = millis();
    link_ok             = true;
    total_frames++;
}

// ============================================================================
//  SECTION 12 — Failsafe
// ============================================================================

/**
 * If no valid frame has arrived within FAILSAFE_MS, force all servos
 * to neutral (90°). This is independent of the ESP32-S3's own watchdog.
 */
static void check_failsafe() {
    if (millis() - last_valid_frame_ms > FAILSAFE_MS) {
        if (link_ok) {
            // First time entering lost state
            Serial.println(F("[SAFE] PC/S3 link lost — going neutral"));
        }
        link_ok = false;
        for (int i = 0; i < N_SERVOS; i++) {
            target_angle[i] = 90;
        }
    }
}

// ============================================================================
//  SECTION 13 — Smooth Servo Ramping
// ============================================================================

/**
 * Called every main-loop iteration.
 * Moves each servo's current_angle one RAMP_STEP toward its target,
 * then writes the rounded integer to the servo.
 *
 * This runs at ~10–40 kHz depending on OLED load, so RAMP_STEP=2° gives
 * a very smooth 0°→180° transition in about 90 loop cycles (~5–10 ms).
 * Increase RAMP_STEP for faster movement, decrease for slower/smoother.
 */
static void update_servos() {
    for (int i = 0; i < N_SERVOS; i++) {
        float target = (float)target_angle[i];
        float cur    = current_angle[i];

        if (cur < target) {
            cur = min(cur + RAMP_STEP, target);
        } else if (cur > target) {
            cur = max(cur - (float)RAMP_STEP, target);
        }

        current_angle[i] = cur;
        servos[i].write((int)cur);
    }
}

/**
 * Immediately snap all servos to 90° without ramping.
 * Used only during boot (not called during normal operation).
 */
static void go_neutral() {
    for (int i = 0; i < N_SERVOS; i++) {
        target_angle[i]  = 90;
        current_angle[i] = 90.0f;
        servos[i].write(90);
    }
}

// ============================================================================
//  SECTION 14 — OLED HUD
// ============================================================================

#if USE_OLED
/**
 * Updates the OLED display every OLED_UPDATE_MS milliseconds.
 * Completely non-blocking — uses millis() gating.
 *
 * Layout (128×64):
 *   Row 0 (y=0):  "LINK OK" / "LINK LOST" (inverted when lost)
 *   Row 1 (y=12): T:xxx  I:xxx  M:xxx
 *   Row 2 (y=22): R:xxx  P:xxx  W:xxx
 *   Row 3 (y=34): Frames: xxxxxxx
 *   Row 4 (y=46): Bad:xxx
 *   Row 5 (y=56): MEGA v1.0
 */
static void update_oled() {
    if (!oled_ok) return;
    uint32_t now = millis();
    if (now - last_oled_ms < OLED_UPDATE_MS) return;
    last_oled_ms = now;

    oled.clearDisplay();
    oled.setTextSize(1);

    // ── Row 0: Link status ────────────────────────────────────────
    if (link_ok) {
        oled.fillRect(0, 0, 72, 10, SH110X_WHITE);
        oled.setTextColor(SH110X_BLACK);
        oled.setCursor(2, 1);
        oled.print(F("\x07 LINK OK"));
    } else {
        oled.drawRect(0, 0, 72, 10, SH110X_WHITE);
        oled.setTextColor(SH110X_WHITE);
        oled.setCursor(2, 1);
        oled.print(F("! LINK LOST"));
    }

    // ── Row 1: Thumb, Index, Middle ──────────────────────────────
    oled.setTextColor(SH110X_WHITE);
    oled.setCursor(0, 13);
    {
        char _ln[22];
        snprintf(_ln, sizeof(_ln), "T:%3d I:%3d M:%3d",
                 (int)current_angle[0], (int)current_angle[1], (int)current_angle[2]);
        oled.print(_ln);
    }

    // ── Row 2: Ring, Pinky, Wrist ────────────────────────────────
    oled.setCursor(0, 23);
    {
        char _ln[22];
        snprintf(_ln, sizeof(_ln), "R:%3d P:%3d W:%3d",
                 (int)current_angle[3], (int)current_angle[4], (int)current_angle[5]);
        oled.print(_ln);
    }

    // ── Row 3: Angle bars for fingers ────────────────────────────
    // 5 compact bars (not wrist), each 20px wide, 3px high, starting y=34
    for (int f = 0; f < 5; f++) {
        int bx = f * 24 + 2;
        int by = 34;
        oled.drawRect(bx, by, 20, 3, SH110X_WHITE);
        int fill = (int)((current_angle[f] / 180.0f) * 18);
        fill = constrain(fill, 0, 18);
        if (fill > 0) oled.fillRect(bx + 1, by + 1, fill, 1, SH110X_WHITE);
    }

    // ── Row 4: Frame counters ─────────────────────────────────────
    oled.setCursor(0, 40);
    {
        char _ln[24];
        snprintf(_ln, sizeof(_ln), "Frm:%lu Err:%lu", total_frames, bad_frames);
        oled.print(_ln);
    }

    // ── Row 5: Identity ──────────────────────────────────────────
    oled.setCursor(0, 56);
    oled.print(F("NEXUS MEGA v1.0"));

    oled.display();
}
#endif

// ============================================================================
//  SECTION 15 — Utility
// ============================================================================

static uint8_t xor_check(const uint8_t* buf, uint8_t len) {
    uint8_t c = 0;
    for (uint8_t i = 0; i < len; i++) c ^= buf[i];
    return c;
}
