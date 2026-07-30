// ============================================================================
//  NEXUS BRIDGE — ESP32-S3 Firmware  v3.0  ★ HAND ANIMATION EDITION ★
//
//  OLED shows a LIVE PIXEL-ART HAND that opens and closes its fingers
//  in real time — every angle directly maps to a finger curl.
//
//  Hardware:
//    - ESP32-S3 Dev Board
//    - SH1106 1.3" OLED  (I2C: SDA=GPIO1, SCL=GPIO2, addr=0x3C)
//    - Built-in LED GPIO48
//    - Serial2: TX=GPIO17 → 1kΩ → Mega RX1(19) / RX=GPIO16 ← Mega TX1(18)
//
//  Libraries: Adafruit SH110X  +  Adafruit GFX Library
// ============================================================================

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <esp_system.h>
#include <math.h>

// ─── Wi-Fi credentials ────────────────────────────────────────────────────────
static const char*    WIFI_SSID     = "AA";
static const char*    WIFI_PASSWORD = "";

// ─── Network ──────────────────────────────────────────────────────────────────
static const uint16_t UDP_PORT      = 8888;

// ─── UART to Mega ─────────────────────────────────────────────────────────────
static const int  UART2_TX_PIN      = 17;
static const int  UART2_RX_PIN      = 16;
static const long UART2_BAUD        = 115200;

// ─── OLED ─────────────────────────────────────────────────────────────────────
static const int  OLED_SDA          = 1;
static const int  OLED_SCL          = 2;
static const int  OLED_W            = 128;
static const int  OLED_H            = 64;
static const int  OLED_ADDR         = 0x3C;

// ─── LED ──────────────────────────────────────────────────────────────────────
static const int  LED_PIN           = 48;

// ─── Timing ───────────────────────────────────────────────────────────────────
static const uint32_t PC_TIMEOUT_MS    = 150;
static const uint32_t WATCHDOG_SEND_MS = 25;
static const uint32_t OLED_REFRESH_MS  = 50;   // 20 fps

// ─── Frame delimiters ─────────────────────────────────────────────────────────
static const uint8_t FRAME_START = 0xAA;
static const uint8_t FRAME_END   = 0x55;
static const int     N_SERVOS    = 6;

// ============================================================================
//  Shared state
// ============================================================================
static SemaphoreHandle_t g_mutex;
static volatile uint8_t  g_angles[N_SERVOS]  = {90, 90, 90, 90, 90, 90};
static volatile bool     g_new_data          = false;
static volatile uint32_t g_last_packet_ms    = 0;
static volatile uint32_t g_packet_count      = 0;
static volatile bool     g_pc_connected      = false;
static volatile bool     g_wifi_connected    = false;
static          char     g_ip_str[24]        = "...";

// ============================================================================
//  OLED Animation State  (owned by oled_task only — no mutex needed)
// ============================================================================

// Smoothly interpolated angles for fluid motion
static float disp_angles[5] = {90, 90, 90, 90, 90};
static float disp_wrist     = 90.0f;

// Per-finger flash counter (finger state change → 6 flashing frames)
static uint8_t finger_flash[5]     = {0, 0, 0, 0, 0};
static bool    prev_finger_open[5] = {false, false, false, false, false};

// Link status flash (LOST→OK transition)
static uint8_t  link_flash_frames = 0;
static bool     prev_connected    = false;

// Scanline crawler
static uint8_t scan_y = 0;

// Frame counter
static uint32_t anim_frame = 0;

// Packet rate
static uint32_t prev_pkt_count = 0;
static uint32_t prev_rate_ms   = 0;
static uint16_t pkt_rate_hz    = 0;

// ============================================================================
//  Objects
// ============================================================================
static WiFiUDP          udp;
static Adafruit_SH1106G oled(OLED_W, OLED_H, &Wire, -1);

// ============================================================================
//  SECTION 1 — Helpers
// ============================================================================
static uint8_t xor_checksum(const uint8_t* buf, size_t len) {
    uint8_t c = 0;
    for (size_t i = 0; i < len; i++) c ^= buf[i];
    return c;
}
static int parse_hex_byte(const char* p) {
    auto h = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    int hi = h(p[0]), lo = h(p[1]);
    return (hi < 0 || lo < 0) ? -1 : (hi << 4) | lo;
}

// ============================================================================
//  SECTION 2 — UDP Parser
// ============================================================================
static bool parse_packet(const char* buf, size_t len, uint8_t out[N_SERVOS]) {
    const char *dollar = nullptr, *hash = nullptr;
    for (size_t i = 0; i < len; i++) {
        if (buf[i] == '$') dollar = buf + i;
        if (buf[i] == '#') hash   = buf + i;
    }
    if (!dollar || !hash || hash <= dollar) return false;
    uint8_t cmp = 0;
    for (const char* p = dollar; p <= hash; p++) cmp ^= (uint8_t)(*p);
    if ((hash + 2) >= (buf + len)) return false;
    int recv = parse_hex_byte(hash + 1);
    if (recv < 0 || (uint8_t)recv != cmp) return false;
    const char* ls = strstr(dollar, "L,");
    if (!ls) return false;
    ls += 2;
    int v[5]; char* ep; const char* cur = ls;
    for (int i = 0; i < 5; i++) {
        long x = strtol(cur, &ep, 10);
        if (ep == cur) return false;
        if (*ep != ',' && i < 4) return false;
        v[i] = (int)constrain(x, 0, 180);
        cur = ep + 1;
    }
    for (int i = 0; i < 5; i++) out[i] = (uint8_t)v[i];
    out[5] = 90;
    return true;
}

// ============================================================================
//  SECTION 3 — UART Frame
// ============================================================================
static void send_uart_frame(const uint8_t a[N_SERVOS]) {
    uint8_t f[9];
    f[0] = FRAME_START;
    for (int i = 0; i < N_SERVOS; i++) f[1+i] = a[i];
    f[7] = xor_checksum(a, N_SERVOS);
    f[8] = FRAME_END;
    Serial2.write(f, 9);
}

// ============================================================================
//  SECTION 4 — PIXEL HAND DRAWING
//
//  The hand is drawn in a 55×48 pixel canvas starting at (72, 14).
//  Each finger is 3 wide and drawn as a vertical bar whose HEIGHT
//  maps to the angle: 0° (OPEN)  = full height; 180° (CLOSED) = near zero.
//
//  Layout (all coords within the 128×64 OLED):
//
//  ┌──────────────────────────── 128 ────────────────────────────────┐
//  │  STATUS BAR (0–9px)                                              │
//  │  T I M R P  FINGER LEDs (11–14px)                               │
//  │  ┌─────────────────────────┐  HAND (14–62px, x=72..126)         │
//  │  │ 5 angle bars (14–48px)  │                                     │
//  │  │ wrist line  (50–52px)   │                                     │
//  │  └─────────────────────────┘                                     │
//  │  FOOTER (57–63px)                                                │
//  └─────────────────────────────────────────────────────────────────┘
//
//  Left half (0–70px wide): status bar + angle bars + wrist + footer
//  Right half (72–127px): LIVE HAND DRAWING
// ============================================================================

// Finger physical constants for the right half (x=72..127)
//   hx  = horizontal center of each finger in the right panel
//   The panel is 56 px wide (72..127); palm starts at y=46
//   Fingers: T(thumb), I, M, R, P
static const int8_t FNG_HX[5] = {6, 17, 29, 41, 51}; // relative to panel_x=72
static const int8_t FNG_PALM_TOP = 46;                 // y where palm starts
static const int8_t FNG_W       = 5;                   // finger width px
static const int8_t PANEL_X     = 72;

/**
 * Draw the live hand into the right 56×64 section.
 * Each finger's height is: maxH * (1 - angle/180)
 * angle=0 (OPEN)   → finger fully extended (tall)
 * angle=180 (CLOSED) → finger folded (height ≈ 2px)
 *
 * The thumb is shorter and angled.
 * Fingertip caps are drawn as a 3-px arc (top 3 pixels flared).
 */
static void draw_hand(const float angles[5]) {
    // ── Palm ────────────────────────────────────────────────────────
    // Rounded palm base: rect with chamfered corners
    oled.fillRect(PANEL_X + 2, FNG_PALM_TOP, 52, 17, SH110X_WHITE);
    // Chamfer corners
    oled.drawPixel(PANEL_X + 2,      FNG_PALM_TOP,       SH110X_BLACK);
    oled.drawPixel(PANEL_X + 53,     FNG_PALM_TOP,       SH110X_BLACK);
    oled.drawPixel(PANEL_X + 2,      FNG_PALM_TOP + 16,  SH110X_BLACK);
    oled.drawPixel(PANEL_X + 53,     FNG_PALM_TOP + 16,  SH110X_BLACK);

    // ── Fingers ─────────────────────────────────────────────────────
    // Max heights per finger (thumb shorter, middle tallest)
    static const int8_t MAX_H[5] = {20, 28, 32, 26, 20};

    for (int f = 0; f < 5; f++) {
        float ang     = constrain(angles[f], 0.0f, 180.0f);
        float t       = ang / 180.0f;           // 0=open, 1=closed
        int   h       = (int)(MAX_H[f] * (1.0f - t * 0.88f));
        h = max(h, 3);

        int fx = PANEL_X + FNG_HX[f];           // left edge of finger
        int fy = FNG_PALM_TOP - h;              // top of finger

        // Thumb: special shorter + slightly offset
        int fw = (f == 0) ? 4 : FNG_W;

        // Draw finger rectangle
        oled.fillRect(fx, fy, fw, h, SH110X_WHITE);

        // Rounded fingertip: erase two corner pixels
        oled.drawPixel(fx,          fy,      SH110X_BLACK);
        oled.drawPixel(fx + fw - 1, fy,      SH110X_BLACK);

        // Knuckle line (crease at 40% from palm top)
        int knuckle_y = FNG_PALM_TOP - (int)(MAX_H[f] * 0.35f);
        if (knuckle_y > fy + 2) {
            oled.drawFastHLine(fx + 1, knuckle_y, fw - 2, SH110X_BLACK);
        }
    }

    // ── Wrist ──────────────────────────────────────────────────────
    oled.drawFastHLine(PANEL_X + 2, FNG_PALM_TOP + 17, 52, SH110X_WHITE);

    // ── "Glow" outline — 1-px border around the whole hand ─────────
    // Top of finger tips: connect them with a light outline
    // (just the silhouette's topmost pixels are already white)

    // ── Flash effect — invert right panel on finger change ──────────
    // (handled by caller via fingerFlash counters)
}

/**
 * Draw a tiny "cyberpunk" ornament: a corner bracket in the top-right
 * of the right panel to frame the hand.
 */
static void draw_panel_frame(bool connected) {
    uint16_t col = SH110X_WHITE;
    // Top-right corner bracket
    oled.drawFastHLine(PANEL_X,      0, 56,  col);   // top edge
    oled.drawFastVLine(PANEL_X,      0, 64,  col);   // left divider
    // Small corner brackets
    oled.drawFastHLine(PANEL_X + 1,  1, 6,   col);
    oled.drawFastVLine(PANEL_X + 1,  1, 4,   col);
    oled.drawFastHLine(PANEL_X + 49, 1, 6,   col);
    oled.drawFastVLine(PANEL_X + 54, 1, 4,   col);
}

// ============================================================================
//  SECTION 5 — Left panel helpers
// ============================================================================

// Draw 3-bar Wi-Fi icon at (x,y)
static void draw_wifi(int x, int y, bool on) {
    for (int b = 0; b < 3; b++) {
        int bh = 2 + b * 2, bx = x + b * 4, by = y + 6 - bh;
        if (on) oled.fillRect(bx, by, 3, bh, SH110X_WHITE);
        else    oled.drawRect(bx, by, 3, bh, SH110X_WHITE);
    }
}

// Draw horizontal angle bar
static void draw_bar(int x, int y, int w, int h, uint8_t angle, bool flash) {
    oled.drawRect(x, y, w, h, SH110X_WHITE);
    int fill = (int)(((float)angle / 180.0f) * (w - 2));
    fill = constrain(fill, 0, w - 2);
    if (fill > 0) {
        if (flash) {
            for (int i = 0; i < fill; i += 2)
                oled.drawFastVLine(x + 1 + i, y + 1, h - 2, SH110X_WHITE);
        } else {
            oled.fillRect(x + 1, y + 1, fill, h - 2, SH110X_WHITE);
        }
    }
}

// ============================================================================
//  SECTION 6 — BOOT ANIMATION
// ============================================================================
static void boot_animation() {
    static const char* T1 = "NEXUS";
    static const char* T2 = "BRIDGE";
    uint32_t start = millis();
    uint8_t  phase = 0;
    uint32_t phase_ms = millis();

    while (millis() - start < 2600) {
        if (millis() - phase_ms >= 110 && phase < 13) {
            phase++;
            phase_ms = millis();
        }

        oled.clearDisplay();
        oled.setTextColor(SH110X_WHITE);

        // Typewriter title
        oled.setTextSize(2);
        oled.setCursor(2, 0);
        uint8_t c1 = min((int)phase, 5);
        for (uint8_t i = 0; i < c1; i++) oled.print(T1[i]);
        // Blinking cursor
        if (phase < 5 && ((millis() / 130) % 2 == 0))
            oled.fillRect(2 + c1 * 12, 0, 6, 15, SH110X_WHITE);

        if (phase >= 6) {
            oled.setCursor(2, 17);
            uint8_t c2 = min((int)(phase - 6), 6);
            for (uint8_t i = 0; i < c2; i++) oled.print(T2[i]);
            if (phase < 12 && ((millis() / 130) % 2 == 0))
                oled.fillRect(2 + c2 * 12, 17, 6, 15, SH110X_WHITE);
        }

        // Progress bar
        int bf = constrain((phase * 118) / 13, 0, 116);
        oled.drawRect(5, 48, 118, 5, SH110X_WHITE);
        oled.fillRect(6, 49, bf, 3, SH110X_WHITE);

        // Animated dots
        oled.setTextSize(1);
        oled.setCursor(0, 56);
        uint8_t dots = (millis() / 280) % 4;
        for (uint8_t d = 0; d < dots; d++) oled.print('.');

        // Mini hand silhouette on the right — all fingers extended
        float boot_angles[5] = {0,0,0,0,0};
        // Draw panel frame
        oled.drawFastVLine(PANEL_X, 0, 64, SH110X_WHITE);
        draw_hand(boot_angles);

        oled.display();
        vTaskDelay(pdMS_TO_TICKS(35));
    }

    // Show IP for 1 second
    oled.clearDisplay();
    oled.setTextSize(1);
    oled.setTextColor(SH110X_WHITE);
    oled.setCursor(2, 10);
    oled.print("NEXUS BRIDGE");
    oled.setCursor(2, 26);
    oled.print(g_ip_str);
    oled.setCursor(2, 38);
    oled.printf("UDP :%d", UDP_PORT);
    oled.setCursor(2, 52);
    oled.print("ALL SYSTEMS GO");
    draw_hand((const float[]){0,0,0,0,0});
    oled.display();
    vTaskDelay(pdMS_TO_TICKS(1200));
}

// ============================================================================
//  SECTION 7 — MAIN DASHBOARD RENDER
// ============================================================================
static void draw_dashboard(const uint8_t snap[N_SERVOS], bool connected,
                            uint32_t pkt_count, bool glitch) {
    static const char* LABELS = "TIMRP";

    oled.clearDisplay();

    // Random glitch tear line
    if (glitch) {
        oled.drawFastHLine(0, random(12, 56), PANEL_X - 2, SH110X_WHITE);
    }

    // ══════════════════════════════════════════════════════════════
    //  LEFT PANEL  (x = 0..69)
    // ══════════════════════════════════════════════════════════════

    // ── TOP BAR (y=0–9) ──────────────────────────────────────────
    bool bar_flash = (link_flash_frames > 0) && (link_flash_frames % 2 == 0);
    if (connected ^ bar_flash) {
        oled.fillRect(0, 0, 70, 9, SH110X_WHITE);
        oled.setTextColor(SH110X_BLACK);
        oled.setTextSize(1);
        oled.setCursor(2, 1);
        oled.print("\x07 LINK OK");
    } else {
        oled.drawRect(0, 0, 68, 9, SH110X_WHITE);
        oled.setTextColor(SH110X_WHITE);
        oled.setTextSize(1);
        oled.setCursor(2, 1);
        oled.print("! PC LOST");
    }
    if (link_flash_frames > 0) link_flash_frames--;

    // ── FINGER LEDs + LABELS (y=11–18) ───────────────────────────
    const int8_t COL_W = 12, COL_X0 = 2;
    for (int f = 0; f < 5; f++) {
        int cx = COL_X0 + f * COL_W + 4;
        bool open  = (disp_angles[f] <= 90.0f);
        bool flash = (finger_flash[f] > 0) && (finger_flash[f] % 2 == 1);
        if (finger_flash[f] > 0) finger_flash[f]--;

        if (flash) {
            oled.drawCircle(cx, 13, 3, SH110X_WHITE);
            oled.drawPixel(cx, 13, SH110X_WHITE);
        } else if (open) {
            oled.fillCircle(cx, 13, 3, SH110X_WHITE);
        } else {
            oled.drawCircle(cx, 13, 3, SH110X_WHITE);
        }
        oled.setTextColor(open ? SH110X_WHITE : SH110X_WHITE);
        oled.setTextSize(1);
        oled.setCursor(cx - 2, 18);
        oled.print(LABELS[f]);
    }

    // ── ANGLE BARS (y=24–43) ─────────────────────────────────────
    const int8_t BAR_X = 2, BAR_W = 67, BAR_H = 3, BAR_GAP = 4;
    for (int f = 0; f < 5; f++) {
        int by = 25 + f * BAR_GAP;
        bool flash = (finger_flash[f] > 0);
        draw_bar(BAR_X, by, BAR_W, BAR_H, (uint8_t)disp_angles[f], flash);
    }

    // ── WRIST NEEDLE (y=46–51) ────────────────────────────────────
    // Axis line
    oled.drawFastHLine(2, 49, 64, SH110X_WHITE);
    oled.drawFastVLine(2,  47, 4, SH110X_WHITE);
    oled.drawFastVLine(65, 47, 4, SH110X_WHITE);
    oled.drawFastVLine(33, 46, 6, SH110X_WHITE);
    // Needle: map wrist 0–180 to x position 4–63
    int needle_x = (int)(2 + (disp_wrist / 180.0f) * 62);
    needle_x = constrain(needle_x, 4, 63);
    // Triangle needle
    oled.fillTriangle(needle_x, 46, needle_x - 2, 51, needle_x + 2, 51, SH110X_WHITE);

    // ── FOOTER (y=56–63) ─────────────────────────────────────────
    oled.setTextColor(SH110X_WHITE);
    oled.setTextSize(1);
    oled.setCursor(0, 57);
    oled.printf("P:%lu", pkt_count % 99999UL);
    oled.setCursor(38, 57);
    oled.printf("%dHz", pkt_rate_hz);
    draw_wifi(56, 57, connected);

    // Rolling scanline pixel (right edge of left panel)
    oled.drawPixel(69, scan_y, SH110X_WHITE);
    scan_y = (scan_y + 1) % 64;

    // ══════════════════════════════════════════════════════════════
    //  RIGHT PANEL — LIVE HAND  (x = 72..127)
    // ══════════════════════════════════════════════════════════════

    draw_panel_frame(connected);
    draw_hand(disp_angles);

    // When PC lost: draw a glitch X over the hand
    if (!connected) {
        if ((anim_frame / 4) % 2 == 0) {
            // Diagonal X flash
            oled.drawLine(PANEL_X + 2, 2,  PANEL_X + 54, 62, SH110X_WHITE);
            oled.drawLine(PANEL_X + 2, 62, PANEL_X + 54, 2,  SH110X_WHITE);
        }
        oled.setTextColor(SH110X_WHITE);
        oled.setTextSize(1);
        oled.setCursor(PANEL_X + 6, 28);
        oled.print("LOST");
    } else {
        // Show open/closed state label for largest-change finger
        int max_f = 0;
        float max_dev = 0;
        for (int f = 0; f < 5; f++) {
            float dev = fabsf(disp_angles[f] - 90.0f);
            if (dev > max_dev) { max_dev = dev; max_f = f; }
        }
        oled.setTextSize(1);
        oled.setTextColor(SH110X_WHITE);
        oled.setCursor(PANEL_X + 4, 57);
        bool open = disp_angles[max_f] <= 90.0f;
        oled.print(open ? "OPEN" : "CLSD");
    }

    oled.display();
}

// ============================================================================
//  SECTION 8 — FreeRTOS Tasks
// ============================================================================

// ── UDP Receiver ─────────────────────────────────────────────────────────────
static void task_udp_receiver(void* p) {
    static char rx[512];
    for (;;) {
        int n = udp.parsePacket();
        if (n > 0) {
            int r = udp.read(rx, sizeof(rx) - 1);
            if (r > 0) {
                rx[r] = '\0';
                uint8_t parsed[N_SERVOS];
                if (parse_packet(rx, r, parsed)) {
                    if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                        for (int i = 0; i < N_SERVOS; i++) g_angles[i] = parsed[i];
                        g_new_data       = true;
                        g_last_packet_ms = millis();
                        g_packet_count++;
                        g_pc_connected   = true;
                        xSemaphoreGive(g_mutex);
                    }
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

// ── Dispatcher ───────────────────────────────────────────────────────────────
static void task_dispatcher(void* p) {
    uint8_t  la[N_SERVOS];
    uint32_t last_wd = 0;
    for (;;) {
        bool send_now = false, pc_timeout = false;
        if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            pc_timeout = (millis() - g_last_packet_ms) > PC_TIMEOUT_MS;
            if (g_new_data) {
                for (int i = 0; i < N_SERVOS; i++) la[i] = g_angles[i];
                g_new_data = false; send_now = true;
            }
            xSemaphoreGive(g_mutex);
        }
        if (send_now) {
            send_uart_frame(la);
            last_wd = millis();
        } else if (pc_timeout) {
            uint32_t now = millis();
            if (now - last_wd >= WATCHDOG_SEND_MS) {
                if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(2)) == pdTRUE) {
                    for (int i = 0; i < N_SERVOS; i++) la[i] = g_angles[i];
                    xSemaphoreGive(g_mutex);
                }
                send_uart_frame(la);
                last_wd = now;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

// ── Watchdog ──────────────────────────────────────────────────────────────────
static void task_watchdog(void* p) {
    for (;;) {
        if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            g_pc_connected = ((millis() - g_last_packet_ms) <= PC_TIMEOUT_MS);
            xSemaphoreGive(g_mutex);
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

// ── OLED Task ─────────────────────────────────────────────────────────────────
static void task_oled(void* p) {

    // ── Boot animation ────────────────────────────────────────────────────
    boot_animation();

    // ── Main loop ─────────────────────────────────────────────────────────
    uint32_t last_frame = millis();
    prev_rate_ms = millis();

    for (;;) {
        uint32_t now = millis();
        if (now - last_frame < OLED_REFRESH_MS) {
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }
        last_frame = now;
        anim_frame++;

        // ── Snapshot shared state ─────────────────────────────────────────
        uint8_t  snap[N_SERVOS];
        bool     connected;
        uint32_t pkt_count;
        if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
            for (int i = 0; i < N_SERVOS; i++) snap[i] = g_angles[i];
            connected = g_pc_connected;
            pkt_count = g_packet_count;
            xSemaphoreGive(g_mutex);
        } else {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        // ── Link transition flash ─────────────────────────────────────────
        if (connected && !prev_connected) link_flash_frames = 8;
        prev_connected = connected;

        // ── Smooth angle interpolation (lerp 20% per frame) ──────────────
        for (int f = 0; f < 5; f++) {
            float target = (float)snap[f];
            float old    = disp_angles[f];
            disp_angles[f] += (target - old) * 0.20f;
            disp_angles[f]  = constrain(disp_angles[f], 0.0f, 180.0f);

            // Detect open/close transition → trigger finger flash
            bool open_now = (disp_angles[f] <= 90.0f);
            if (open_now != prev_finger_open[f]) finger_flash[f] = 6;
            prev_finger_open[f] = open_now;
        }
        disp_wrist += ((float)snap[5] - disp_wrist) * 0.20f;

        // ── Packet rate (once/sec) ────────────────────────────────────────
        if (now - prev_rate_ms >= 1000) {
            pkt_rate_hz    = (uint16_t)(pkt_count - prev_pkt_count);
            prev_pkt_count = pkt_count;
            prev_rate_ms   = now;
        }

        // ── Glitch logic ──────────────────────────────────────────────────
        bool glitch = false;
        if (!connected && (anim_frame % 7 == 0)) glitch = true;
        if ( connected && (anim_frame % 53 == 0)) glitch = true;

        // ── Render ───────────────────────────────────────────────────────
        draw_dashboard(snap, connected, pkt_count, glitch);
    }
}

// ── LED Task ──────────────────────────────────────────────────────────────────
static void task_led(void* p) {
    bool state = false;
    for (;;) {
        bool conn;
        if (xSemaphoreTake(g_mutex, pdMS_TO_TICKS(2)) == pdTRUE) {
            conn = g_pc_connected;
            xSemaphoreGive(g_mutex);
        } else conn = false;
        uint32_t hp = conn ? 500 : 167;
        state = !state;
        digitalWrite(LED_PIN, state);
        vTaskDelay(pdMS_TO_TICKS(hp));
    }
}

// ============================================================================
//  SECTION 9 — setup()
// ============================================================================
void setup() {
    Serial.begin(115200);
    Serial.println("\n[NEXUS] v3.0 — Hand Animation Edition");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);

    Wire.begin(OLED_SDA, OLED_SCL);
    if (!oled.begin(OLED_ADDR, true)) Serial.println("[WARN] OLED missing");
    oled.clearDisplay();
    oled.setTextColor(SH110X_WHITE);
    oled.setTextSize(1);
    oled.setCursor(0, 0);
    oled.print("Init...");
    oled.display();

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    uint32_t ws = millis();
    while (WiFi.status() != WL_CONNECTED) {
        delay(250);
        digitalWrite(LED_PIN, (millis() / 250) % 2);
        if (millis() - ws > 15000) { ESP.restart(); }
    }
    WiFi.localIP().toString().toCharArray(g_ip_str, sizeof(g_ip_str));
    g_wifi_connected = true;
    g_last_packet_ms = millis();
    Serial.printf("[WIFI] %s\n", g_ip_str);

    udp.begin(UDP_PORT);
    Serial2.begin(UART2_BAUD, SERIAL_8N1, UART2_RX_PIN, UART2_TX_PIN);

    g_mutex = xSemaphoreCreateMutex();
    if (!g_mutex) ESP.restart();

    xTaskCreatePinnedToCore(task_udp_receiver, "udp_rx",     4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(task_dispatcher,   "dispatcher", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(task_watchdog,     "watchdog",   2048, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(task_oled,         "oled",       8192, NULL, 1, NULL, 0);
    xTaskCreatePinnedToCore(task_led,          "led",        1024, NULL, 1, NULL, 0);

    Serial.println("[NEXUS] All tasks running.");
    digitalWrite(LED_PIN, LOW);
}

// ============================================================================
//  SECTION 10 — loop()  (idle)
// ============================================================================
void loop() { delay(1); }
