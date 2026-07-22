#include <Servo.h>

Servo s;

// Pins
#define TRIG1 2
#define ECHO1 3
#define TRIG2 4
#define ECHO2 5
#define SERVO_PIN 9
#define BUZZER 8

// Distance limits
#define HAND_DIST 15
#define EMPTY_DIST 5

void setup() {
  pinMode(TRIG1, OUTPUT);
  pinMode(ECHO1, INPUT);
  pinMode(TRIG2, OUTPUT);
  pinMode(ECHO2, INPUT);

  pinMode(BUZZER, OUTPUT);

  s.attach(SERVO_PIN);
  s.write(0); // closed
}

long getDist(int t, int e) {
  digitalWrite(t, LOW); delayMicroseconds(2);
  digitalWrite(t, HIGH); delayMicroseconds(10);
  digitalWrite(t, LOW);
  long d = pulseIn(e, HIGH, 30000);
  return (d == 0) ? 999 : d * 0.0343 / 2;
}

void loop() {

  long d1 = getDist(TRIG1, ECHO1);
  long d2 = getDist(TRIG2, ECHO2);

  // If SENSOR1 detects hand
  if (d1 <= HAND_DIST) {

    // If bin is empty → OPEN
    if (d2 >= EMPTY_DIST) {
      digitalWrite(BUZZER, LOW); // no beep
      s.write(90);
      delay(2000);
    }
    // If bin NOT empty → BUZZ
    else {
      s.write(0); // do not open
      digitalWrite(BUZZER, HIGH); 
      delay(300);
      digitalWrite(BUZZER, LOW);
      delay(300);
    }
  }

  // No hand → close
  else {
    s.write(0);
    digitalWrite(BUZZER, LOW);
  }

  delay(100);
}
