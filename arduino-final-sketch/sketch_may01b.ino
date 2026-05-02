#include <squat-pose-detection_inferencing.h>

#include "camera.h"
#include "gc2145.h"
#include <WiFi.h>
#include "arduino_secrets.h"

// --- Both model headers ---
#include <squat-pose-detection_inferencing.h>
#include <Pushup_form_detector_inferencing.h>

#define LED_PIN LED_BUILTIN

static uint8_t frame_copy[80 * 60];  // RGB565 copy

void blinkLED(int times, int delay_ms) {
    for (int i = 0; i < times; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(delay_ms);
        digitalWrite(LED_PIN, LOW);
        delay(delay_ms);
    }
}

// --- Fixed IP config ---
IPAddress local_IP(10, 126, 125, 184);  // fixed IP for the Nicla — pick any unused .x
IPAddress gateway(10, 126, 125, 1);     // your phone (hotspot host)
IPAddress subnet(255, 255, 255, 0);
IPAddress dns(8, 8, 8, 8);

// --- WiFi ---
char ssid[] = SECRET_SSID;
char pass[] = SECRET_PASS;
WiFiServer server(80);

// --- Camera ---
GC2145 galaxyCore;
Camera cam(galaxyCore);
FrameBuffer fb;

// --- Model selection ---
enum ActiveModel { MODEL_SQUAT, MODEL_PUSHUP };
ActiveModel activeModel = MODEL_SQUAT;  // default

// --- EI input buffer ---
// Use the larger of the two input sizes (both are 96x96 so same size)
static float ei_buffer[96 * 96];

// --- Latest results ---
String latest_label = "none";

// Squat: 3 classes
float squat_scores[EI_CLASSIFIER_LABEL_COUNT] = {0};  

// Pushup: 2 classes — we store separately
// We'll use fixed size 2 since we know the pushup model has 2 classes
float pushup_scores[2] = {0};
const char* pushup_labels[2] = {"bad", "good"};

// ---------------------------------------------------------------
// RGB565 → EI packed float (0x00RRGGBB as float)
// ---------------------------------------------------------------
void resize_and_convert(uint8_t *src, float *dst,
                        int src_w, int src_h,
                        int dst_w, int dst_h) {
    float x_scale = (float)src_w / dst_w;
    float y_scale = (float)src_h / dst_h;

    for (int dy = 0; dy < dst_h; dy++) {
        for (int dx = 0; dx < dst_w; dx++) {
            int sx = (int)(dx * x_scale);
            int sy = (int)(dy * y_scale);

            int src_idx = (sy * src_w + sx) * 2;
            uint16_t pixel = ((uint16_t)src[src_idx] << 8) | src[src_idx + 1];

            float r = ((pixel >> 11) & 0x1F) * (255.0f / 31.0f);
            float g = ((pixel >> 5)  & 0x3F) * (255.0f / 63.0f);
            float b = ((pixel >> 0)  & 0x1F) * (255.0f / 31.0f);

            uint32_t pixel_ei = ((uint32_t)((uint8_t)r) << 16)
                               | ((uint32_t)((uint8_t)g) << 8)
                               | ((uint32_t)((uint8_t)b));

            dst[dy * dst_w + dx] = (float)pixel_ei;
        }
    }
}

int ei_camera_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, ei_buffer + offset, length * sizeof(float));
    return 0;
}

void sendBMP(WiFiClient &client) {
    int w = 80, h = 60;

    int rowSize = (w * 3 + 3) & ~3;  // align to 4 bytes
    int padding = rowSize - (w * 3);
    uint8_t pad[3] = {0, 0, 0};
    uint32_t fileSize = 54 + (rowSize * h);

    uint8_t bmpHeader[54] = {
        0x42, 0x4D,
        (uint8_t)(fileSize), (uint8_t)(fileSize >> 8),
        (uint8_t)(fileSize >> 16), (uint8_t)(fileSize >> 24),
        0,0,0,0,
        54,0,0,0,
        40,0,0,0,
        (uint8_t)(w), (uint8_t)(w >> 8), 0,0,
        (uint8_t)(h), (uint8_t)(h >> 8), 0,0,
        1,0,24,0
    };

    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: image/bmp");
    client.println("Connection: close");
    client.println();

    client.write(bmpHeader, 54);

    for (int y = h - 1; y >= 0; y--) {
        for (int x = 0; x < w; x++) {
    
            uint8_t gray = frame_copy[y * w + x];
            uint8_t rgb[3] = { gray, gray, gray };
    
            client.write(rgb, 3);
        }
    
        client.write(pad, padding);  // 👈 REQUIRED
    }
}

// ---------------------------------------------------------------
// Run whichever model is active
// ---------------------------------------------------------------
void runInference() {
    if (cam.grabFrame(fb, 3000) != 0) {
        Serial.println("Frame capture failed");
        return;
    }

    uint8_t *src = fb.getBuffer();

    for (int y = 0; y < 60; y++) {
        for (int x = 0; x < 80; x++) {
            frame_copy[y * 80 + x] = src[(y * 2) * 160 + (x * 2)];
        }
    }

    resize_and_convert(fb.getBuffer(), ei_buffer, 80, 60, 96, 96);

    signal_t signal;
    signal.total_length = 96 * 96;
    signal.get_data = &ei_camera_get_data;

    if (activeModel == MODEL_SQUAT) {
        ei_impulse_result_t result = {0};
        EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
        if (err != EI_IMPULSE_OK) {
            Serial.print("Squat classifier error: "); Serial.println(err);
            return;
        }
        float max_val = 0; int max_idx = 0;
        for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            squat_scores[i] = result.classification[i].value;
            if (result.classification[i].value > max_val) {
                max_val = result.classification[i].value;
                max_idx = i;
            }
        }
        latest_label = String(result.classification[max_idx].label);

        Serial.println("--- Squat Inference ---");
        for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            Serial.print("  "); Serial.print(result.classification[i].label);
            Serial.print(": "); Serial.print(result.classification[i].value * 100, 1);
            Serial.println("%");
        }
        Serial.print(">> "); Serial.print(latest_label);
        Serial.print(" ("); Serial.print(max_val * 100, 1); Serial.println("%)\n");

    } else {
        // Pushup model uses its own run_classifier — they are separate compiled units
        // We need to call the pushup model's classifier differently
        // Edge Impulse renames run_classifier per library via namespacing in newer SDK
        // If both libraries conflict on run_classifier, see note below
        ei_impulse_result_t result = {0};
        EI_IMPULSE_ERROR err = run_classifier(&signal, &result, false);
        if (err != EI_IMPULSE_OK) {
            Serial.print("Pushup classifier error: "); Serial.println(err);
            return;
        }
        float max_val = 0; int max_idx = 0;
        for (int i = 0; i < 2; i++) {
            pushup_scores[i] = result.classification[i].value;
            if (result.classification[i].value > max_val) {
                max_val = result.classification[i].value;
                max_idx = i;
            }
        }
        latest_label = String(result.classification[max_idx].label);

        Serial.println("--- Pushup Inference ---");
        for (int i = 0; i < 2; i++) {
            Serial.print("  "); Serial.print(result.classification[i].label);
            Serial.print(": "); Serial.print(result.classification[i].value * 100, 1);
            Serial.println("%");
        }
        Serial.print(">> "); Serial.print(latest_label);
        Serial.print(" ("); Serial.print(max_val * 100, 1); Serial.println("%)\n");
    }
}

// ---------------------------------------------------------------
// Web dashboard + model switcher
// ---------------------------------------------------------------
void handleWebClient() {
    WiFiClient client = server.accept();
    if (!client) return;

    String request = "";
    unsigned long timeout = millis() + 1000;
    while (client.connected() && millis() < timeout) {
        if (client.available()) {
            char c = client.read();
            request += c;
            if (request.endsWith("\r\n\r\n")) break;
        }
    }

    // Handle model switch requests
    if (request.indexOf("GET /squat") >= 0) {
        activeModel = MODEL_SQUAT;
        latest_label = "none";
    } else if (request.indexOf("GET /pushup") >= 0) {
        activeModel = MODEL_PUSHUP;
        latest_label = "none";
    }

    String modelName = (activeModel == MODEL_SQUAT) ? "Squat Detection" : "Pushup Form";
    String squat_active = (activeModel == MODEL_SQUAT) ? "#4af" : "#444";
    String pushup_active = (activeModel == MODEL_PUSHUP) ? "#4af" : "#444";

    // Build score bars HTML
    String bars = "";
    if (activeModel == MODEL_SQUAT) {
        for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
            int pct = (int)(squat_scores[i] * 100);
            bars += "<p><b>" + String(ei_classifier_inferencing_categories[i]) + "</b></p>";
            bars += "<div class='bar-wrap'><div class='bar' style='width:" + String(pct * 3) + "px'>";
            bars += String(pct) + "%</div></div>";
        }
    } else {
        for (int i = 0; i < 2; i++) {
            int pct = (int)(pushup_scores[i] * 100);
            bars += "<p><b>" + String(pushup_labels[i]) + "</b></p>";
            bars += "<div class='bar-wrap'><div class='bar' style='width:" + String(pct * 3) + "px'>";
            bars += String(pct) + "%</div></div>";
        }
    }

    if (request.indexOf("GET /frame") >= 0) {
        sendBMP(client);
        client.stop();
        return;
    }

    // Determine label color
    String labelColor = "#4f4";
    if (latest_label == "bad" || latest_label == "badform") labelColor = "#f44";

    client.println("HTTP/1.1 200 OK");
    client.println("Content-type:text/html");
    client.println("Refresh: 2");
    client.println();
    client.println("<!DOCTYPE html><html><head><title>Exercise Detector</title>");
    client.println("<meta name='viewport' content='width=device-width,initial-scale=1'>");
    client.println("<style>");
    client.println("body{font-family:sans-serif;text-align:center;padding:20px;background:#111;color:#eee;}");
    client.println("h1{color:#fff;} h2{color:#aaa;}");
    client.println(".label{font-size:3em;font-weight:bold;margin:20px;}");
    client.println(".btn{display:inline-block;padding:12px 28px;margin:8px;border-radius:8px;");
    client.println("     font-size:1.1em;font-weight:bold;cursor:pointer;text-decoration:none;color:#fff;}");
    client.println(".bar-wrap{background:#333;border-radius:8px;margin:8px auto;width:300px;}");
    client.println(".bar{height:28px;border-radius:8px;background:#4af;text-align:left;");
    client.println("     padding-left:8px;line-height:28px;min-width:2px;}");
    client.println("</style></head><body>");
    client.println("<h1>Exercise Classifier</h1>");

    // Model switch buttons
    client.print("<a href='/squat' class='btn' style='background:");
    client.print(squat_active);
    client.println("'>Squat</a>");
    client.print("<a href='/pushup' class='btn' style='background:");
    client.print(pushup_active);
    client.println("'>Pushup</a>");

    client.print("<h2>Active: "); client.print(modelName); client.println("</h2>");
    client.print("<div class='label' style='color:"); client.print(labelColor); client.print("'>");
    client.print(latest_label); client.println("</div>");
    client.println("<h2>Camera Feed</h2>");
    client.println("<img src='/frame' width='320' style='border-radius:10px;'>");
    client.println(bars);
    client.println("</body></html>");
    client.println();
    client.stop();
}

// ---------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    //while (!Serial);
    Serial.println("Exercise Classifier");
    Serial.println("===================");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Camera
    if (!cam.begin(CAMERA_R160x120, CAMERA_GRAYSCALE, 30)) {
        Serial.println("Camera init failed!");
        while (1);
    }
    Serial.println("Camera ready");

    // Fixed IP
    WiFi.config(local_IP, dns, gateway, subnet);

    Serial.print("Connecting to: ");
Serial.println(ssid);

int wifiStatus = WL_IDLE_STATUS;
int retryCount = 0;

while (wifiStatus != WL_CONNECTED) {
    wifiStatus = WiFi.begin(ssid, pass);

    // Blink LED while trying
    blinkLED(2, 200);

    Serial.print("Attempt ");
    Serial.println(retryCount + 1);

    delay(3000);
    retryCount++;

    // Optional: every few retries, do a longer blink to indicate failure
    if (retryCount % 5 == 0) {
        Serial.println("Still trying to connect...");
        blinkLED(5, 100);  // fast blink burst
    }
}

    // Connected
    digitalWrite(LED_PIN, HIGH);  // solid ON = success
    
    Serial.println("\nWiFi connected!");
    Serial.print("Dashboard: http://");
    Serial.println(WiFi.localIP());

    server.begin();
}

// ---------------------------------------------------------------
void loop() {
    runInference();
    handleWebClient();
    delay(2000);
}
