// M1 = low band  -> D3
// M2 = mid band  -> D5
// High band H is read but ignored so motor3 can't interfere.

const int M1_PIN = 3;  // low
const int M2_PIN = 5;  // mid
// const int M3_PIN = 9;  // high (wired but not driven for safety)

void setup() {
  pinMode(M1_PIN, OUTPUT);
  pinMode(M2_PIN, OUTPUT);
  // pinMode(M3_PIN, OUTPUT);

  analogWrite(M1_PIN, 0);
  analogWrite(M2_PIN, 0);
  // analogWrite(M3_PIN, 0);

  Serial.begin(115200);
  while (!Serial) {
    ; // required on Nano 33 BLE
  }
  Serial.setTimeout(5); 
  Serial.println("Haptic translator ready. Expecting: L M H");
}

void loop() {
  if (Serial.available()) {
    // Read three integers sent as: "L M H\n"
    int L = Serial.parseInt();  // low
    int M = Serial.parseInt();  // mid
    int H = Serial.parseInt();  // high

    int c = Serial.peek();
    if (c == '\n' || c == '\r') {
      Serial.read();
    }

    // Clamp to [0,255]
    L = constrain(L, 0, 255);
    M = constrain(M, 0, 255);
    H = constrain(H, 0, 255);

    // Drive motors: only low + mid
    analogWrite(M1_PIN, L);
    analogWrite(M2_PIN, M);
    // High band ignored:
    // analogWrite(M3_PIN, H);

    
    Serial.print("L=");
    Serial.print(L);
    Serial.print(" M=");
    Serial.print(M);
    Serial.print(" H(ignored)=");
    Serial.println(H);
  }
}
