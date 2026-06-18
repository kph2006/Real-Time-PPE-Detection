/*
  ============================================================

  COMBINED SAFETY CHECKPOINT — Helmet + Mask + Alcohol (Arduino Uno)
  ============================================================
  This sketch is the single hardware controller for the whole
  system, based on the confirmed-working alcohol_gate_control
  protocol. Python makes the access decision (helmet + mask via
  webcam, alcohol via this board's MQ-3 reading); this sketch
  just drives the outputs and streams the sensor value.

  WIRING:
    Servo Signal  --> Pin 5
    Green LED (+) --> Pin 3  (with 220ohm resistor to GND)
    Red LED (+)   --> Pin 2  (with 220ohm resistor to GND)
    Buzzer (+)    --> Pin 4
    MQ-3 AOUT     --> A1
    5V + GND      --> All components

  SERIAL PROTOCOL (Python <-> Arduino):
    Arduino sends once at boot:  "ARDUINO_READY"
    Arduino sends every 300ms:   "SENSOR:427"
    Python sends:                "ACCESS_GRANTED" or "ACCESS_DENIED"
    Arduino replies:             "STATUS:OPENING_GATE" / "STATUS:GATE_CLOSED"
                                  "STATUS:ACCESS_DENIED" / "STATUS:STANDBY"
  ============================================================
*/

#include <Servo.h>

// ---------- Pin Definitions ----------
const int SERVO_PIN  = 5;
const int GREEN_LED  = 3;
const int RED_LED    = 2;
const int BUZZER_PIN = 4;
const int MQ3_PIN    = A1;   // Analog pin - NO pinMode needed

// ---------- Servo ----------
const int SERVO_CLOSED = 180;
const int SERVO_OPEN   = 90;

// ---------- Timing ----------
const unsigned long SERVO_OPEN_MS  = 2000;
const unsigned long BUZZER_MS      = 2000;
const unsigned long RED_LED_MS     = 5000;
const unsigned long SENSOR_SEND_MS = 300;   // Push sensor value every 300ms

Servo gateServo;
String inputBuffer = "";
unsigned long lastSensorSend = 0;

// ============================================================
void setup() {
  Serial.begin(9600);

  pinMode(GREEN_LED,  OUTPUT);
  pinMode(RED_LED,    OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  // NOTE: Do NOT call pinMode on analog pin A1

  gateServo.attach(SERVO_PIN);
  gateServo.write(SERVO_CLOSED);

  digitalWrite(GREEN_LED,  LOW);
  digitalWrite(RED_LED,    LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // Let MQ-3 settle for 2 seconds before first read
  delay(2000);

  Serial.println("ARDUINO_READY");
}

// ============================================================
void loop() {

  // -- Continuously push sensor reading to Python --
  if (millis() - lastSensorSend >= SENSOR_SEND_MS) {
    lastSensorSend = millis();
    int val = analogRead(MQ3_PIN);
    Serial.print("SENSOR:");
    Serial.println(val);
  }

  // -- Read serial commands from Python --
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        handleCommand(inputBuffer);
      }
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}

// ============================================================
void handleCommand(String cmd) {
  if (cmd == "ACCESS_GRANTED") {
    grantAccess();
  } else if (cmd == "ACCESS_DENIED") {
    denyAccess();
  }
}

// ============================================================
// ACCESS GRANTED: Green LED ON + Servo 0->90->0
// (helmet OK + mask OK + alcohol below threshold)
// ============================================================
void grantAccess() {
  Serial.println("STATUS:OPENING_GATE");

  digitalWrite(RED_LED,   LOW);
  digitalWrite(GREEN_LED, HIGH);

  // Open gate
  gateServo.write(SERVO_OPEN);
  delay(SERVO_OPEN_MS);   // Hold open 2 seconds

  // Close gate
  gateServo.write(SERVO_CLOSED);
  delay(500);

  digitalWrite(GREEN_LED, LOW);

  Serial.println("STATUS:GATE_CLOSED");
}

// ============================================================
// ACCESS DENIED: Red LED (5s) + Buzzer beeps (2s)
// (no helmet OR no mask OR alcohol above threshold)
// ============================================================
void denyAccess() {
  Serial.println("STATUS:ACCESS_DENIED");

  gateServo.write(SERVO_CLOSED);

  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED,   HIGH);

  // Buzzer beeps for BUZZER_MS (2 seconds) using tone/noTone so
  // the CPU is NOT blocked by delay() — keeps serial port responsive.
  unsigned long start = millis();
  bool buzzerOn = false;
  unsigned long lastToggle = start;
  const unsigned long ON_TIME  = 200;
  const unsigned long OFF_TIME = 200;

  while (millis() - start < BUZZER_MS) {
    unsigned long now = millis();
    if (!buzzerOn && (now - lastToggle >= OFF_TIME)) {
      digitalWrite(BUZZER_PIN, HIGH);
      buzzerOn = true;
      lastToggle = now;
    } else if (buzzerOn && (now - lastToggle >= ON_TIME)) {
      digitalWrite(BUZZER_PIN, LOW);
      buzzerOn = false;
      lastToggle = now;
    }
    // Keep reading serial so no bytes are dropped during buzzer
    while (Serial.available() > 0) {
      char c = (char)Serial.read();
      if (c == '\n') {
        inputBuffer.trim();
        if (inputBuffer.length() > 0) handleCommand(inputBuffer);
        inputBuffer = "";
      } else if (c != '\r') {
        inputBuffer += c;
      }
    }
  }
  digitalWrite(BUZZER_PIN, LOW);

  // Red LED stays on for remaining time up to 5s total
  unsigned long elapsed = millis() - start;
  if (elapsed < RED_LED_MS) {
    delay(RED_LED_MS - elapsed);
  }

  digitalWrite(RED_LED, LOW);

  Serial.println("STATUS:STANDBY");
}
