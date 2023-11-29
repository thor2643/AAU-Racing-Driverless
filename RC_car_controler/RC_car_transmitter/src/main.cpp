#include <Arduino.h>

#define y_joy 34
#define x_joy 35

#include <esp_now.h>
#include <WiFi.h>

//Mac Address: 0C:B8:15:C1:A8:E8
uint8_t Reciver_MAC_Address[] = {0x0C, 0xB8, 0x15, 0xC1, 0xA8, 0xE8};

// Structure to send data multible sets of small data, 
// shall match the structure for the receiver
typedef struct struct_message_package {
  char a[32];
  int b;
  int c;
} struct_message_package;

// Creating a struct message called my_data_package
struct_message_package my_data_package;

esp_now_peer_info_t peerInfo;

// callback when data is sent
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("\r\nLast Packet Send Status:\t");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
}

void setup() {
  // Init Serial Monitor
  Serial.begin(115200);
 
  // Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);

  // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Once ESPNow is successfully Init, we will register for Send CB to
  // get the status of Trasnmitted packet
  esp_now_register_send_cb(OnDataSent);
  
  // Register peer
  memcpy(peerInfo.peer_addr, Reciver_MAC_Address, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  // Add peer        
  if (esp_now_add_peer(&peerInfo) != ESP_OK){
    Serial.println("Failed to add peer");
    return;
  }
}
 
void loop() {
  //Read values from joysticks
  int x_value=analogRead(x_joy);
  int y_value=analogRead(y_joy);

  Serial.println("x_value: "+String(x_value)+", y_value: "+String(y_value)+", button_state: ");
  //¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
  float angle=map(x_value,0,4095,45,145);
  angle=76.65; //96.5 is the middle position of the servo = 0 degrees
  x_value=map(angle,45,145,0,4095);
  Serial.println("angle: "+String(angle)+", new x_value: "+String(x_value));
  //¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
  // Set values to send
  //strcpy(my_data_package.a, "The speed and steering angle for the RC car is:");
  my_data_package.b = y_value;
  my_data_package.c = x_value;
  
  // Send message via ESP-NOW
  esp_err_t result = esp_now_send(Reciver_MAC_Address, (uint8_t *) &my_data_package, sizeof(my_data_package));
   
  if (result == ESP_OK) {
    Serial.println("Sent with success");
  }
  else {
    Serial.println("Error sending the data");
  }
  //¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
  //delay(500);
  //¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤
}

