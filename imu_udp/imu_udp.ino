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

#include "mpu9250.h"
#include "WiFi.h"
#include "AsyncUDP.h"

const char * ssid = "Arad-PC";
const char * password = "pepegaking98";

/* Mpu6500 object */
bfs::Mpu9250 imu;
AsyncUDP udp;
IPAddress remote_IP(192, 168, 1, 3);
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
  imu.Config(&Wire, bfs::Mpu9250::I2C_ADDR_PRIM);
  /* Initialize and configure IMU */
  if (!imu.Begin()) {
    Serial.println("Error initializing communication with IMU");
    while(1) {}
  }
  /* Set the sample rate divider */
  if (!imu.ConfigSrd(4)) {
    Serial.println("Error in configuration of sample rate divider");
    while(1) {}
  }
  if (imu.ConfigAccelRange(bfs::Mpu9250::ACCEL_RANGE_4G))
    Serial.println("accel range set to ACCEL_RANGE_4G");
  if (imu.ConfigGyroRange(bfs::Mpu9250::GYRO_RANGE_500DPS))
    Serial.println("gyro range set to GYRO_RANGE_500DPS");
}

float accel[3], gyro[3], mag[3];
const int packet_div = 6;
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
    mag[0] = imu.mag_x_ut();
    mag[1] = imu.mag_y_ut();
    mag[2] = imu.mag_z_ut();
    sprintf(data[counter], "%f,%f,%f,%f,%f,%f,%f,%f,%f\n", accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2], mag[0],mag[1],mag[2]);
    counter++;
  }
  t = millis();
  if (counter >= packet_div)
  {
     counter = 0;
     buff[0] = '\0';
     for (int i = 0; i < packet_div; i++)
      strcat(buff, data[i]);     
     udp.broadcastTo(buff, 1234);
  }
}
