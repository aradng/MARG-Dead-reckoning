import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
import logging

import time
import socket
import numpy as np
import pandas as pd
from ahrs.filters import Mahony, Madgwick, EKF
from sympy import capture

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
f_handler = logging.FileHandler('ahrs_udp.log')

filter = Mahony(frequency=200, k_P=1, k_I=0.3)
# filter = Madgwick(frequency=200)
# filter = EKF(frequency=200)
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
try:
    calib = pd.read_pickle('calib/calib.pkl')
except:
    ind =  (('gyro', 'x'),
        ('gyro', 'y'),
        ('gyro', 'z'),
        ('accel', 'x'),
        ('accel', 'y'),
        ('accel', 'z'),
        ('mag', 'x'),
        ('mag', 'y'),
        ('mag', 'z'))
    print("No calibration file found")
    calib = pd.Series(np.zeros(9), index=ind)

try:
    A = np.load('calib/hard_soft_iron.npy')
except:
    print("No hard soft iron calibration file found")
    A = np.eye(3)

def main():
    capture = False
    log_data = pd.DataFrame()
    log_accel = pd.DataFrame()
    V = pd.DataFrame(np.zeros(shape=(1,6)), columns = (('accel', 'x'), ('accel', 'y'), ('vel', 'x'), ('vel', 'y'), ('pos', 'x') , ('pos', 'y')))
    global Q, calib, A
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
            calib = calibrate(500)
            print(calib)
            calib.to_pickle('calib/calib.pkl')
            # Q = np.array([1., 0., 0., 0.])
        if event.type == KEYDOWN and event.key == K_d:
            capture = ~capture
        data = read_data() - calib
        for k,v in data.iterrows():
            data.loc[k, 'mag'] = A @ v['mag'].to_numpy()
            if filter.__class__.__name__ == 'EKF':
                Q = filter.update(Q, gyr=v['gyro'].to_numpy(), acc=v['accel'].to_numpy(), mag=v['mag'].to_numpy())
            else:
                Q = filter.updateMARG(Q, gyr=v['gyro'].to_numpy(), acc=v['accel'].to_numpy(), mag=v['mag'].to_numpy())
                # Q = filter.updateIMU(Q, gyr=v['gyro'].to_numpy(), acc=v['accel'].to_numpy()) 
            rotation_matrix = quat_to_rot_mat(Q)
            data.loc[k, 'accel'] = rotation_matrix @ v['accel'].to_numpy()

        if capture:
            log_accel = pd.concat([log_accel, data['accel']])
            log_data = pd.concat([log_data, data])
        [w, nx, ny, nz] = Q
        rotation_matrix = quat_to_rot_mat(Q)
        draw(w, nx, ny, nz, rotation_matrix @ v['accel'].to_numpy())
        pygame.display.flip()
        frames += 1 
    log_data.to_pickle('debug/log_data.pkl')     
    log_accel.to_pickle('debug/log_accel.pkl')
    print("fps: %d" % ((frames*1000)/(pygame.time.get_ticks()-ticks)))

def read_data():
    data, client_address = udp_socket.recvfrom(3000)  # Adjust the buffer size as needed
    gyro = []
    accel = []
    mag = []
    data = data.decode('utf-8').splitlines()
    # print(len(data))
    try:
        for i in data:
            i = i.split(',')
            accel.append([float(i[0]), float(i[1]), float(i[2])])
            gyro.append([float(i[3]), float(i[4]), float(i[5])])
            mag.append([float(i[6]), float(i[7]), float(i[8])])
    except Exception as e:
        print("Error in reading data")
        print(i)
        print(data)
        print(len(data))
        raise e
    gyro = pd.DataFrame(gyro, columns=['x', 'y', 'z'])
    accel = pd.DataFrame(accel, columns=['x', 'y', 'z'])
    mag = pd.DataFrame(mag, columns=['x', 'y', 'z'])
    return pd.concat([gyro, accel, mag], keys=['gyro', 'accel', 'mag'], axis=1)

def calibrate(n_samples=1000):
    df = pd.DataFrame()
    for i in range(n_samples):
        df = pd.concat([df, read_data()], ignore_index=True)
        if i % (n_samples/10) == 0:
            print(f"calibrating gyro... {i}/{n_samples}")
    print(pd.concat([df[:100].mean()['gyro'], df[:200].mean()['gyro'], df[:400].mean()['gyro'], df.mean()['gyro']], axis=1))
    df = df.mean()
    df['accel'] = np.array([0, 0, 0])
    df['mag'] = np.array([26.372621, -24.870084, -80.033550])
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


def draw(w, nx, ny, nz, accel):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    drawText((-2.6, 1.8, 2), "MPU Quetrion Visulization", 18)
    drawText((-2.6, 1.6, 2), f"filter Module {str(filter.__class__).split('.')[2]}", 16)
    drawText((-2.6, -2, 2), "Press Escape to Exit , C to Calibrate, D to capture", 16)

    [yaw, pitch , roll] = quat_to_ypr([w, nx, ny, nz])

    # drawText((-2.6, -1.8, 2), "Yaw: %f, Pitch: %f, Roll: %f" %(yaw, pitch, roll), 16)
    drawText((-2.6, -1.8, 2), f"Q: {w:.3f}, x: {nx:.3f}, y: {ny:.3f}, z: {nz:.3f}", 16)
    drawText((-2.6, -1.9, 2), f"Q accel : x: {accel[0]:.3f}, y: {accel[1]:.3f}, z: {accel[2]:.3f}", 16)
    accel_s = np.array(['0', '0', '0'])
    for i in range(len(accel)):
        if accel[i] > 1:
            accel_s[i] = '+'
        elif accel[i] < -1:
            accel_s[i] = '-'
    drawText((1, -1.9, 2), f"x: {accel_s[0]}, y: {accel_s[1]}, z: {accel_s[2]}", 16)
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

def quat_to_rot_mat(q):
    r00 = 2 * (q[0] * q[0] + q[1] * q[1]) - 1
    r01 = 2 * (q[1] * q[2] - q[0] * q[3])
    r02 = 2 * (q[1] * q[3] + q[0] * q[2])
    r10 = 2 * (q[1] * q[2] + q[0] * q[3])
    r11 = 2 * (q[0] * q[0] + q[2] * q[2]) - 1
    r12 = 2 * (q[2] * q[3] - q[0] * q[1])
    r20 = 2 * (q[1] * q[3] - q[0] * q[2])
    r21 = 2 * (q[2] * q[3] + q[0] * q[1])
    r22 = 2 * (q[0] * q[0] + q[3] * q[3]) - 1

    return np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])

if __name__ == '__main__':
    main()
    