#include <Arduino.h>
/*
  Rui Santos
  Complete project details at https://RandomNerdTutorials.com/esp-now-esp32-arduino-ide/
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files.
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
*/
#include <esp_now.h>
#include <WiFi.h>
#include <ESP32Servo.h>


#define servo_pin 23
#define speed_controler_pin 22

// creates two servo objects to control the servo for the direction of the weeles and one for the speed of the wheels 
Servo servo_wheels;
Servo speed_controler;

int y_value=2048;
int x_value=2048;
int angle=map(x_value,0,4095,45,145);
int speed=map(y_value,0,4095,0,180);

typedef struct struct_message {
    char a[32];
    int b=y_value;
    int c=x_value;
} struct_message;

// Create a struct_message called myData
struct_message myData;

// callback function that will be executed when data is received
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  memcpy(&myData, incomingData, sizeof(myData));
  //Serial.println(myData.a); // char message
  int y_value=myData.b;
  int x_value=myData.c;
  int angle=map(x_value,0,4095,45,145);
  int speed=map(y_value,0,4095,0,180);
  String datapakke=(String) "a"+angle+"s"+speed;
  //SerialPort.print(datapakke);
  Serial.println("y_value: "+String(y_value)+"x_value: "+String(x_value));
}

void setup() {
  // Initialize Serial Monitor
  Serial.begin(9600);

  servo_wheels.attach(servo_pin);
  servo_wheels.write(95);
  speed_controler.attach(speed_controler_pin,1000,2000);
  speed_controler.write(90);

  delay(2000);//check if this delay is still needed????

    // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);
  
  // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  
  // Once ESPNow is successfully Init, we will register for recv CB to
  // get recv packer info
  esp_now_register_recv_cb(OnDataRecv);
}
 


void loop() {
  //delay(100);
  y_value=myData.b;
  x_value=myData.c;
  angle=map(x_value,0,4095,45,145);
  speed=map(y_value,0,4095,0,180);
  servo_wheels.write(angle);
  speed_controler.write(speed);
}