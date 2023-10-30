#include <Arduino.h>
// Complete Instructions to Get and Change ESP MAC Address: https://RandomNerdTutorials.com/get-change-esp32-esp8266-mac-address-arduino/

#include "WiFi.h"
 
void setup(){
  Serial.begin(115200);
  //delay(5000);
  WiFi.mode(WIFI_MODE_STA);
  Serial.println("__________________\nMac Address: ");
  Serial.println(WiFi.macAddress());
  Serial.println("__________________");
}
 
void loop(){
   Serial.println("hej");
   delay(1000);
}