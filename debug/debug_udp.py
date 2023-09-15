import socket
from ahrs.filters import Madgwick
import numpy as np

# Define the UDP server's IP address and port to bind to
server_ip = "0.0.0.0"  # Listen on all available network interfaces
server_port = 1234  # Replace with the port you want to use

# Create a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the server IP and port
udp_socket.bind((server_ip, server_port))

print(f"UDP server is listening on {server_ip}:{server_port}")
madgwick = Madgwick()
Q = np.array([[1., 0., 0., 0.]])

while True:
    try:
        # Receive data from a client
        data, client_address = udp_socket.recvfrom(60*20)  # Adjust the buffer size as needed

        # Print the received data and client address
        # print(f"Received data from {client_address}: {data.decode('utf-8')}")

        # You can add your processing logic here
        data = data.decode('utf-8').splitlines()
        # break

        # accel = np.array([float(data[0]), float(data[1]), float(data[2])])
        # gyro = np.array([float(data[3]), float(data[4]), float(data[5])])

        # use mahoon filter to get the orientation
        #
        # print(f"accel: {accel}")
        # print(f"gyro: {gyro}")
        # np.append(Q, madgwick.updateIMU(Q[-1], gyro, accel))
        # print(f"madgwick: {Q[-1]}")

    except Exception as e:
        print(e)
        print("\nUDP server terminated by user.")
        break


# Close the UDP socket
udp_socket.close()
