#include <Arduino.h>

#define y_joy A0
#define x_joy A1
 
void setup() {
  Serial.begin(9600);
}

void loop() {
  int x_value=analogRead(x_joy);
  int y_value=analogRead(y_joy);
  int speed = int(map(y_value,0,1023,-100,100));
  int angle = int(map(x_value,0,1023,-45,45));
  //Serial.println("x_value: "+String(x_value)+", y_value: "+String(y_value));
  Serial.println("Speed: "+String(speed)+", Angle: "+String(angle));
  //10*Serial.print("\n");
  delay(200);
}
