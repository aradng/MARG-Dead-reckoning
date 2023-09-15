/*
* Brian R Taylor
* brian.taylor@bolderflight.com
* 
* Copyright (c) 2021 Bolder Flight Systems Inc
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the “Software”), to
* deal in the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*/

#include "mpu6500.h"
#include "WiFi.h"
#include "AsyncUDP.h"

const char * ssid = "*******";
const char * password = "*******";

/* Mpu6500 object */
bfs::Mpu6500 imu;
AsyncUDP udp;
IPAddress remote_IP(192, 168, 0, 2);
#define UDP_PORT 1234

void setup() {
  /* Serial to display data */
  Serial.begin(115200);
  while(!Serial) {}
  WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    if (WiFi.waitForConnectResult() != WL_CONNECTED) {
        Serial.println("WiFi Failed");
        while(1) {
            delay(1000);
        }
    }
  /* Start the I2C bus */
  Wire.begin();
  Wire.setClock(400000);
  /* I2C bus,  0x68 address */
  imu.Config(&Wire, bfs::Mpu6500::I2C_ADDR_PRIM);
  /* Initialize and configure IMU */
  if (!imu.Begin()) {
    Serial.println("Error initializing communication with IMU");
    while(1) {}
  }
  /* Set the sample rate divider */
  if (!imu.ConfigSrd(0)) {
    Serial.println("Error configured SRD");
    while(1) {}
  }
}

float accel[3], gyro[3];
const int packet_div = 20;
int counter = 0;
char data[packet_div][100];
char buff[packet_div*100];
unsigned long long int t = millis();

void loop() {
  /* Check if data read */
  if (imu.Read()) {
    imu.new_imu_data();
    accel[0] = imu.accel_x_mps2();
    accel[1] = imu.accel_y_mps2();
    accel[2] = imu.accel_z_mps2();
    gyro[0] = imu.gyro_x_radps();
    gyro[1] = imu.gyro_y_radps();
    gyro[2] = imu.gyro_z_radps();
    sprintf(data[counter], "%f,%f,%f,%f,%f,%f\n", accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]);
    counter++;
//    Serial.print("rd\t");
//    Serial.print(counter);
//    Serial.print("\t");
//    Serial.print(data[counter-1]);
//    Serial.print("\t");
//    Serial.println(millis() - t);
  }
  t = millis();
  if (counter >= packet_div)
  {
//     Serial.print("rc\t");
//     Serial.print(counter);
     counter = 0;
     buff[0] = '\0';
     for (int i = 0; i < packet_div; i++)
      strcat(buff, data[i]);     
     udp.broadcastTo(buff, 1234);
//     Serial.print("send_buffer\t");
//     Serial.println(millis() - t);
  }
//  delay(1);
}
