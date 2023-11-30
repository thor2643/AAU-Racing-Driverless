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

typedef struct struct_message {
    int angle=95;
    int speed=90;
} struct_message;

// Create a struct_message called myData
struct_message myData;

void setup() {
  // Initialize Serial Monitor
  Serial.begin(115200);

  servo_wheels.attach(servo_pin);
  servo_wheels.write(myData.angle);
  speed_controler.attach(speed_controler_pin,1000,2000);
  speed_controler.write(myData.speed);

  delay(2000);//check if this delay is still needed????

}
 
void decode_data(std::string data_recivd){
    std::string signal =data_recivd;
    //std::string signal = "A150V120";
    std::string temp;
    std::string firstNumberStr, secondNumberStr;

    for (char c : signal) {
        if (std::isdigit(c)) {
            temp += c;
        } else {
            if (!temp.empty()) {
                if (firstNumberStr.empty()) {
                    firstNumberStr = temp;
                } else if (secondNumberStr.empty()) {
                    secondNumberStr = temp;
                }
                temp.clear();
            }
        }
    }
    if (!temp.empty() && secondNumberStr.empty()) {
        secondNumberStr = temp;
    }

    myData.angle = std::stoi(firstNumberStr);
    myData.speed = std::stoi(secondNumberStr);
}

void loop() {
  int angle_1=45;
  int speed_1=90;
  /*
  //make a for loop that loops 100 times and save angle from 45 to 145 and speed from 0 to 180 in a std::string and send it to the decoder
  for (int i = 0; i < 100; i++){
  delay(100);
  servo_wheels.write(myData.angle);
  speed_controler.write(myData.speed);
  Serial.println("angle: "+String(myData.angle)+"speed: "+String(myData.speed));
  String str = "A"+String(angle_1)+"V"+String(speed_1);
  //convert str to std::string
  std::string str_1 = str.c_str();
  decode_data(str_1);
  if (i%10==0){
    speed_1=speed_1+10;
  }
  angle_1=angle_1+1;
  }*/

  servo_wheels.write(myData.angle);
  speed_controler.write(myData.speed);
}