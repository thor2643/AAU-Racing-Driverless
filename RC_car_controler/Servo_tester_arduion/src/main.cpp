#include <Arduino.h>
#include <Servo.h>
#include <SoftwareSerial.h>

//#define potintiometer A0
#define servo_pin 11
#define speed_controler_pin 10

#define input_pin_A1 0
#define input_pin_A2 1
#define input_pin_A3 2
#define input_pin_A4 3
#define input_pin_A5 4
#define input_pin_A6 5
#define input_pin_A7 6
#define input_pin_A8 7

#define input_pin_B1 12
#define input_pin_B2 13
#define input_pin_B3 A5
#define input_pin_B4 A4
#define input_pin_B5 A3
#define input_pin_B6 A2
#define input_pin_B7 A1
#define input_pin_B8 A0

Servo myservo;  // create servo object to control a servo
Servo motor;    // create motor object as servo becures as they uses the same controle signal

//SoftwareSerial softSerial(10, 11);

String angle="";
String speed="";
String incommping_data="";

int set_angle=95;
//int set_speed=90;
int set_speed=95;
int kill_switch=0;

int prev_angle=set_angle;
int prev_speed=set_speed;

int input_motor_val=90;
int input_angle_val=95;

void setup() {
  Serial.begin(15200);
  //softSerial.begin(9600);
  //pinMode(speed_controler_pin,OUTPUT);
  
  pinMode(input_pin_A1,INPUT);
  pinMode(input_pin_A2,INPUT);
  pinMode(input_pin_A3,INPUT);
  pinMode(input_pin_A4,INPUT);
  pinMode(input_pin_A5,INPUT);
  pinMode(input_pin_A6,INPUT);
  pinMode(input_pin_A7,INPUT);
  pinMode(input_pin_A8,INPUT);

  pinMode(input_pin_B1,INPUT);
  pinMode(input_pin_B2,INPUT);
  pinMode(input_pin_B3,INPUT);
  pinMode(input_pin_B4,INPUT);
  pinMode(input_pin_B5,INPUT);
  pinMode(input_pin_B6,INPUT);
  pinMode(input_pin_B7,INPUT);
  pinMode(input_pin_B8,INPUT);

  myservo.attach(servo_pin);
  motor.attach(speed_controler_pin,1000,2000); //
  myservo.write(input_angle_val);
  motor.write(input_motor_val);


  //motor.writeMicroseconds(set_speed);

  delay(500);
  delay(2000);
}

void loop() {
  delay(1000);
  //int read_1=analogRead(A0);
  //int read_2=analogRead(A1);
  //int read_3=analogRead(A2);

  int read_1A=digitalRead(input_pin_A1);
  int read_2A=digitalRead(input_pin_A2);
  int read_3A=digitalRead(input_pin_A3);
  int read_4A=digitalRead(input_pin_A4);
  int read_5A=digitalRead(input_pin_A5);
  int read_6A=digitalRead(input_pin_A6);
  int read_7A=digitalRead(input_pin_A7);
  int read_8A=digitalRead(input_pin_A8);

  int read_1B=digitalRead(input_pin_B1);
  int read_2B=digitalRead(input_pin_B2);
  int read_3B=digitalRead(input_pin_B3);
  int read_4B=digitalRead(input_pin_B4);
  int read_5B=digitalRead(input_pin_B5);
  int read_6B=digitalRead(input_pin_B6);
  /*int read_1B=0;
  int read_2B=0;
  int read_3B=0;
  int read_4B=0;
  int read_5B=0;
  int read_6B=0;*/
  int read_7B=0;
  int read_8B=0;
  
  /*
  if(analogRead(input_pin_B1)>0){read_1B=1;}else{read_1B=0;}
  if(analogRead(input_pin_B2)>0){read_2B=1;}else{read_2B=0;}
  if(analogRead(input_pin_B3)>0){read_3B=1;}else{read_3B=0;}
  if(analogRead(input_pin_B4)>0){read_4B=1;}else{read_4B=0;}
  if(analogRead(input_pin_B5)>0){read_5B=1;}else{read_5B=0;}
  if(analogRead(input_pin_B6)>0){read_6B=1;}else{read_6B=0;}*/
  read_7B=digitalRead(input_pin_B7);
  read_8B=digitalRead(input_pin_B8);
  //int read_1=map(analogRead(A0),0,657,0,11);
  //int read_2=map(analogRead(A1),0,657,0,11);
  //int read_3=map(analogRead(A2),0,657,0,11);

  input_motor_val=read_1A*1+read_2A*2+read_3A*4+read_4A*8+read_5A*16+read_6A*32+read_7A*64+read_8A*128;
  input_angle_val=read_1B*1+read_2B*2+read_3B*4+read_4B*8+read_5B*16+read_6B*32+read_7B*64+read_8B*128;
  Serial.print(read_1A);
  Serial.print(read_2A);
  Serial.print(read_3A);
  Serial.print(read_4A);
  Serial.print(read_5A);
  Serial.print(read_6A);
  Serial.print(read_7A);
  Serial.print(read_8A);
  Serial.println();
  Serial.print(read_1B);
  Serial.print(read_2B);
  Serial.print(read_3B);
  Serial.print(read_4B);
  Serial.print(read_5B);
  Serial.print(read_6B);
  Serial.print(read_7B);
  Serial.print(read_8B);
  Serial.println("\n____________\n");
  Serial.print("input_motor_val: "+String(input_motor_val)+" input_angle_val:"+String(input_angle_val));
  //Serial.println(input_angle_val);
  Serial.println("\n____________\n");
  motor.write(input_motor_val);
  if (input_angle_val<181){
    myservo.write(input_angle_val);
  }
  /*
  int num = 42;
  int bits[8];

  for (int i = 0; i < 8; i++) {
  bits[i] = bitRead(num, i);
  }
  Serial.println("\n");
  // Print the binary array
  for (int i = 7; i >= 0; i--) {
  Serial.print(bits[i]);
  }
  Serial.println("\n");
  */
  //Serial.println(read_1*100+read_2*10+read_3);
  
  //Serial.println(map(read,0,657,0,95));



  //delay(1000);

  //analogWrite(speed_controler_pin,245);
  //motor.write(130);

  /*
  if (softSerial.available()){
    char data_char=char(softSerial.read());
    if (data_char=='s'){
      Serial.println("angle: "+String(angle));
      set_angle=angle.toInt();
      incommping_data="speed";
      speed="";
;    }
    else if (data_char=='a'){
      Serial.println("speed: "+String(speed));
      set_speed=speed.toInt();
      incommping_data="angle";
      angle="";
    }
    if(incommping_data=="speed" && data_char!='s'){
      speed+=data_char;
    }
    else if(incommping_data=="angle" && data_char!='a'){
      angle+=data_char;
    }
    
    //motor.writeMicroseconds(set_speed);
    //kill_switch=0;
  }

  motor.write(set_speed);
  myservo.write(set_angle);*/
  
  /*if (abs(set_angle-prev_angle)>1){
      myservo.write(set_angle);
      prev_angle=set_angle;
  }

  if (abs(set_speed-prev_speed)>2){
      motor.write(set_speed);
      prev_speed=set_speed;
  }*/

  /*else{
    myservo.write(set_angle);
    //motor.write(set_speed);
    motor.writeMicroseconds(set_speed);
    /*if (kill_switch<500){
    myservo.write(set_angle);
    motor.write(set_speed);
    kill_switch+=1;
    }
    
    else{
      set_angle=100;
      set_speed=90;
      myservo.write(set_angle);
      motor.write(set_speed);
    }*/
  //}
  //Serial.println(set_angle);
  //Serial.println(set_speed);
}
