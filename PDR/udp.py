import socket
import logging
import numpy as np
import pandas as pd
import itertools

class UDP:
    def __init__(self, port=1234, single=False):
        self.port = port
        self.single = single
        if not single:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind(("0.0.0.0", self.port))
            self.udp_socket.settimeout(0.5)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def read(self):
        if self.single:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind(("0.0.0.0", self.port))
            self.udp_socket.settimeout(0.5)
        marg = []
        data, client_address = self.udp_socket.recvfrom(
            3000
        )  # Adjust the buffer size as needed
        data = data.decode("utf-8").splitlines()
        # print(len(data))
        try:
            for i in data:
                i = i.split(",")
                marg.append(float(i[j]) for j in range(9))
        except Exception as e:
            self.logger.error("Error in reading data")
            self.logger.error(i)
            self.logger.error(data)
            self.logger.error(len(data))
            raise e
        marg = pd.DataFrame(marg, columns=tuple(itertools.product(["accel", "gyro", "mag"], ["x", "y", "z"])))
        return marg
    
    def close(self):
        self.udp_socket.close()
