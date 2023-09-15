# Dead-reckoning with MARG/IMU
## Introduction
 this is a project aimed at using a MARG/IMU sensor to estimate the position of a pedestrian (aka [PDR](https://en.wikipedia.org/wiki/Dead_reckoning#Pedestrian_dead_reckoning_(PDR))).\
 acceleration results are gaited for pedestrian steps to remove integration error build up and only estimate the step length and interpolate the new position from heading in horizental plane and acceleration vector magnitude.\

## Hardware
 The sensor used is the [MPU-6050 6-DoF Accel and Gyro Sensor](https://www.adafruit.com/product/3886) connected to an [ESP32-WROOM-32](https://www.espressif.com/en/products/devkits/esp32-devkitc).\
 ESP32 sends the data to a UDP server running the AHRS and EKF algorithms.

## Software
### ESP32:
runs a simple UDPAsync client sending batch results broadcast through the network.\
 port, batch size and sample rate can be adjusted through `UDP_PORT`, `packet_div` and `imu.ConfigSrd(0)` respectively. default is on port 1234, batch size of 20 and sample rate of 1KHz.
### AHRS - Attitude and Heading Reference System:
runs on a UDP server listening to the ESP32 packets and estimates the orientation of the sensor using the [Madgwick](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/) or [Mahony](https://ahrs.readthedocs.io/en/latest/filters/mahony.html)
 algorithm.\
 the server also runs a simple Kalman filter to estimate the heading of the sensor in the horizental plane. the server can be run using:
 ```shell
 python3 ahrs_udp.py
 ```
     
 this implementaton uses `OpenGL` and `pygame` to display the orientation of the sensor. can adjust the number of samples used to calibrate the sensor in `calibrate()` method adjust the frequency and AHRS algorithm through the `filter` variable.
### EKF - Extended Kalman Filter:
TBD
## Results
<img src="./resources/AHRS_demo.gif" width="640" height="360"/>

##### negligible drift in heading during stationary trasmit and over 5 minutes of walking.