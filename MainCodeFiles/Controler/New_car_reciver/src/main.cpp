#include <Arduino.h>
#include <ESP32Servo.h>
//The cuminication protocol with WiFi and ESP-NOW is based on the example code from https://RandomNerdTutorials.com/esp-now-esp32-arduino-ide/
//Evything else is developed by AAU ROB3 Group 364
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

#define servo_pin 23
#define speed_controler_pin 22
#define green_led 32
#define yellow_led 25
#define red_led 27

// creates two servo objects to control the servo for the direction of the weeles and one for the speed of the wheels 
Servo servo_wheels;
Servo speed_controler;



typedef struct struct_manuel_mode_data {
    bool stop=false;
    bool manuel=false;
    int x=2048;
    int y=2048;
} struct_manuel_mode_data;

typedef struct struct_atomatic_mode_data {
    float angle=95;
    float speed=90;
} struct_atomatic_mode_data;

// Create a struct_message called autoData and manuelData
struct_atomatic_mode_data autoData;
struct_manuel_mode_data manuelData;

float current_angle=autoData.angle;
float current_speed=autoData.speed;
float new_angle=current_angle;
float new_speed=current_speed;

int y_value=manuelData.y;
int x_value=manuelData.x;
int angle_manuel=map(x_value,0,4095,45,145);
int speed_manuel=map(y_value,0,4095,0,180);

// callback function that will be executed when data is received
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  memcpy(&manuelData, incomingData, sizeof(manuelData));
  //Serial.println(myData.a); // char message
  //y_value=manuelData.y;
  //x_value=manuelData.x;
}
void led_on(String colour){
  if (colour=="green"){
    digitalWrite(yellow_led,LOW);
    digitalWrite(red_led,LOW);
    digitalWrite(green_led,HIGH);
  }
  else if (colour=="yellow"){
    digitalWrite(green_led,LOW);
    digitalWrite(red_led,LOW);
    digitalWrite(yellow_led,HIGH);
  }
  else if (colour=="red"){
    digitalWrite(green_led,LOW);
    digitalWrite(yellow_led,LOW);
    digitalWrite(red_led,HIGH);
  }
}
void led_off(){
  digitalWrite(green_led,LOW);
  digitalWrite(yellow_led,LOW);
  digitalWrite(red_led,LOW);
}
void setup() {
  // Initialize Serial Monitor
  Serial.begin(115200);

  servo_wheels.attach(servo_pin);
  servo_wheels.write(autoData.angle);
  speed_controler.attach(speed_controler_pin,1000,2000);
  speed_controler.write(autoData.speed);

  //Led setup
  pinMode(green_led,OUTPUT);
  pinMode(yellow_led,OUTPUT);
  pinMode(red_led,OUTPUT);

  //LED blinks one at a time in 2 seconds
  for (int i=0;i<10;i++){
    led_on("green");
    delay(200);
    led_on("yellow");
    delay(200);
    led_on("red");
    delay(200);
    led_off();
  }
  //delay(2000);//check if this delay is still needed????

  Serial.setTimeout(1);

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
 
void decode_data(std::string data_recivd){
    std::string signal =data_recivd;
    //std::string signal = "A150V120";
    std::string temp;
    std::string firstNumberStr, secondNumberStr;

    //extract the numbers from the string and save them in the variables firstNumberStr and secondNumberStr
    //the first number is the angle and the second is the speed
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
    float angle=std::stoi(firstNumberStr);
    float speed=std::stoi(secondNumberStr);
    //failsafe:
    if (angle>145){
      angle=145;
    }
    if (angle<45){
      angle=45;
    }
    if (speed>180){
      speed=180;
    }
    if (speed<0){
      speed=0;
    }
    //save the data in the struct
    autoData.angle = angle;
    autoData.speed = speed;
}


int dt=0;
void P_controler(){
  //if the chage in time is lagere than 10 ms run the code
  if ((millis()-dt)>10){
    dt=millis();

    //Angle
    float error_angle=autoData.angle-current_angle;
    float Kp_angle=0.5;  
    float diff_angle=Kp_angle*error_angle;
    new_angle=current_angle+diff_angle;
    servo_wheels.write(new_angle);
    current_angle=new_angle;

    //Speed
    float error_speed=autoData.speed-current_speed;
    float Kp_speed=0.02;  
    float diff_speed=Kp_speed*error_speed;
    new_speed=current_speed+diff_speed;
    speed_controler.write(new_speed);
    current_speed=new_speed;
   }
}

bool prev_mode=manuelData.manuel; //true=manuel, true=auto
bool manual_mode_pc=false;
bool stop=false;
void loop() {
  //run the code until the stop command is recived (E-stop), then the cars' speed and angle is set to 0 imidiatly!!
  while (stop==false){
    if (Serial.available() > 0) {
      if (prev_mode!=manuelData.manuel){
        if (prev_mode==false && manuelData.manuel==true){
          prev_mode=true;
          Serial.println("manual");
        }
        else if (prev_mode==true && manuelData.manuel==false){
          prev_mode=false;
          Serial.println("auto");
        }
      }
      // read the incoming byte:
      String str = Serial.readStringUntil('\n');
      //convert str to std::string and save it in str_1
      std::string str_1 = str.c_str();

      if (str_1=="stop" || manuelData.stop==true){
        autoData.angle=95;
        autoData.speed=90;
        stop=true;
        Serial.println("stop");
      }
      else if (str_1=="manual"){
        manual_mode_pc==true;
        led_on("yellow");
      }
      else if (str_1=="auto"){
        manual_mode_pc==false;
        led_on("green");
      }
      else{
        decode_data(str_1);
      }
    }
    if (manuelData.stop==true){
      autoData.angle=95;
      autoData.speed=90;
      manuelData.x=2048;
      manuelData.y=2048;
      if (stop==false){
        Serial.println("stop");
        stop=true;
      }
    }
    if (manuelData.manuel==true){
      led_on("yellow");
      angle_manuel=map(manuelData.x,0,4095,45,145);
      speed_manuel=map(manuelData.y,0,4095,0,180);
      servo_wheels.write(angle_manuel);
      speed_controler.write(speed_manuel);
    }
    else{
      if (manual_mode_pc==false){
        led_on("green");
      }
      P_controler();
    }
  }
  led_on("red");
  servo_wheels.write(autoData.angle);
  speed_controler.write(autoData.speed);
  delay(200);
  led_off();
  delay(200);

}