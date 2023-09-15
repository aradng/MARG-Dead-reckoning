"""
PyTeapot module for drawing rotating cube using OpenGL as per
quaternion or yaw, pitch, roll angles received over serial port.
"""

from pandas.core.arrays.period import dt64arr_to_periodarr
from pandas.tseries import frequencies
import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

import time
import socket
import numpy as np
import pandas as pd
from ahrs.filters import Mahony, Madgwick, AQUA

filter = Mahony(frequency=1000)
# filter = Madgwick(frequency=1000)
# filter = AQUA(frequency=1000)
print(filter.frequency, filter.Dt)

# Define the UDP server's IP address and port to bind to
server_ip = "0.0.0.0"  # Listen on all available network interfaces
server_port = 1234  # Replace with the port you want to use

# Create a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind the socket to the server IP and port
udp_socket.bind((server_ip, server_port))
udp_socket.settimeout(0.5)

print(f"UDP server is listening on {server_ip}:{server_port}")
Q = np.array([1., 0., 0., 0.])
ind =  (('gyro', 'x'),
        ('gyro', 'y'),
        ('gyro', 'z'),
        ('accel', 'x'),
        ('accel', 'y'),
        ('accel', 'z'))
calib = pd.Series(np.zeros(6), index=ind)

def main():
    global Q, calib
    video_flags = OPENGL | DOUBLEBUF
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), video_flags)
    pygame.display.set_caption("PyTeapot IMU orientation visualization")
    resizewin(1280, 720)
    init()
    frames = 0
    ticks = pygame.time.get_ticks()

    while True:
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        if event.type == KEYDOWN and event.key == K_c:
            calib = calibrate()
            print(calib)
            Q = np.array([1., 0., 0., 0.])
        data = read_data() - calib
        # print(f"Time taken to read data: {time.time() - t}")
        # t = time.time()
        for k,v in data.iterrows():
            Q = filter.updateIMU(Q, gyr=v['gyro'].to_numpy(), acc=v['accel'].to_numpy())

        [w, nx, ny, nz] = Q
        draw(w, nx, ny, nz)
        pygame.display.flip()
        frames += 1        
    print("fps: %d" % ((frames*1000)/(pygame.time.get_ticks()-ticks)))

def read_data():
    data, client_address = udp_socket.recvfrom(10000)  # Adjust the buffer size as needed
    gyro = []
    accel = []
    data = data.decode('utf-8').splitlines()
    try:
        for i in data:
            i = i.split(',')
            accel.append([float(i[0]), float(i[1]), float(i[2])])
            gyro.append([float(i[3]), float(i[4]), float(i[5])])
    except Exception as e:
        print("Error in reading data")
        print(i)
        print(data)
        print(len(data))
        raise e
    gyro = pd.DataFrame(gyro, columns=['x', 'y', 'z'])
    accel = pd.DataFrame(accel, columns=['x', 'y', 'z'])
    return pd.concat([gyro, accel], keys=['gyro', 'accel'], axis=1)

def calibrate(n_samples=200):
    df = pd.DataFrame()
    for i in range(n_samples):
        df = pd.concat([df, read_data()], ignore_index=True)
        if i % (n_samples/10) == 0:
            print(f"Calibrating... {i}/{n_samples}")
    df = df.mean()
    df['accel'] = np.array([0, 0, 0])
    return df

def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def draw(w, nx, ny, nz):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    drawText((-2.6, 1.8, 2), "MPU Quetrion Visulization", 18)
    drawText((-2.6, 1.6, 2), f"filter Module {str(filter.__class__).split('.')[2]}", 16)
    drawText((-2.6, -2, 2), "Press Escape to Exit , C to Calibrate", 16)

    [yaw, pitch , roll] = quat_to_ypr([w, nx, ny, nz])
    # drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" %(yaw, pitch, roll), 16)
    drawText((-2.6, -1.8, 2), f"Q: {w:.3f}, x: {nx:.3f}, y: {ny:.3f}, z: {nz:.3f}", 16)
    glRotatef(2 * math.acos(w) * 180.00/math.pi, -1 * nx, nz, ny)
    glBegin(GL_QUADS)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(2.0, 0.2, -1.0)
    glVertex3f(-2.0, 0.2, -1.0)
    glVertex3f(-2.0, 0.2, 1.0)
    glVertex3f(2.0, 0.2, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(2.0, -0.2, 1.0)
    glVertex3f(-2.0, -0.2, 1.0)
    glVertex3f(-2.0, -0.2, -1.0)
    glVertex3f(2.0, -0.2, -1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(2.0, 0.2, 1.0)
    glVertex3f(-2.0, 0.2, 1.0)
    glVertex3f(-2.0, -0.2, 1.0)
    glVertex3f(2.0, -0.2, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(2.0, -0.2, -1.0)
    glVertex3f(-2.0, -0.2, -1.0)
    glVertex3f(-2.0, 0.2, -1.0)
    glVertex3f(2.0, 0.2, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-2.0, 0.2, 1.0)
    glVertex3f(-2.0, 0.2, -1.0)
    glVertex3f(-2.0, -0.2, -1.0)
    glVertex3f(-2.0, -0.2, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(2.0, 0.2, -1.0)
    glVertex3f(2.0, 0.2, 1.0)
    glVertex3f(2.0, -0.2, 1.0)
    glVertex3f(2.0, -0.2, -1.0)
    glEnd()


def drawText(position, textString, size):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def quat_to_ypr(q):
    yaw   = math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
    pitch = -math.asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll  = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
    pitch *= 180.0 / math.pi
    yaw   *= 180.0 / math.pi
    yaw   -= -0.13  # Declination at Chandrapur, Maharashtra is - 0 degress 13 min
    roll  *= 180.0 / math.pi
    return [yaw, pitch, roll]


if __name__ == '__main__':
    main()
    