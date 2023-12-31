import numpy as np
import pandas as pd
import pygame
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *
from PDR.pdr import PDR
import time

class Visulize:
    def __init__(self, width, height, pdr):
        self.width = width
        self.height = height

        self.frames = 0
        self.ticks = pygame.time.get_ticks()

        self.capture = False
        self.running = False

        self.pdr = pdr

        self.calib_menu = False

        self.timings = pd.DataFrame()

    def run(self):
        self.init()
        self.running = True
        while self.running:
            t = time.time()
            current = {"frame_start": time.time()}
            self.event = pygame.event.poll()
            self.event_handler()
            t = time.time()
            self.pdr.update()
            current.update({"total": time.time() - t})
            t = time.time()
            self.draw()
            current.update({"draw": time.time() - t})
            pygame.display.flip()
            self.frames += 1
            current.update({"frame_time": time.time() - current["frame_start"]})
            self.timings = pd.concat([self.timings, pd.Series(current)], axis=1, ignore_index=True)

    def init(self):
        pygame.init()
        video_flags = OPENGL | DOUBLEBUF
        self.screen = pygame.display.set_mode((1280, 720), video_flags)
        pygame.display.set_caption("IMU orientation visualization")
        self.resizewin()
        glShadeModel(GL_SMOOTH)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

    def event_handler(self):
        if self.event.type == QUIT or (self.event.type == KEYDOWN and self.event.key == K_ESCAPE):
            self.running = False
            if len(self.pdr.data):
                self.pdr.data.to_csv('data.csv')
                self.pdr.data_f.to_csv('data_f.csv')
                self.pdr.true_accel.to_csv('true_accel.csv')
                self.timings *= 1000
                self.timings.T.to_pickle('timings.pkl')
                self.pdr.plot_path()
        if self.event.type == KEYDOWN and self.event.key == K_f:
            pygame.display.toggle_fullscreen()
        if self.event.type == KEYDOWN and self.event.key == K_c:
            self.calib_menu = ~self.calib_menu
        if self.event.type == KEYDOWN and self.event.key == K_d:
            self.capture = ~self.capture
            self.pdr.capture = self.capture
        if self.event.type == KEYDOWN and self.event.key == K_r:
            self.frames = 0
            self.ticks = pygame.time.get_ticks()
        if self.calib_menu:
            if self.event.type == KEYDOWN and self.event.key == K_1:
                self.pdr.calibrate_mag()
                self.calib_menu = False
            elif self.event.type == KEYDOWN and self.event.key == K_2:
                self.pdr.calibrate_gyro()
                self.calib_menu = False
            elif self.event.type == KEYDOWN and self.event.key == K_3:
                self.pdr.calibrate_accel()
                self.calib_menu = False

    def resizewin(self):
        if self.height == 0:
            self.height = 1
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1.0*self.width/self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()      

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0, 0.0, -7.0)

        self.drawText((-2.6, 1.8, 2), f"filter Module {str(self.pdr.filter.__class__).split('.')[2]}", 18)
        if not self.calib_menu:
            self.drawText((-2.6, -2, 2), "Press Escape to Exit , C to Calibrattion Settings, D to capture", 16)
        else:
            self.drawText((-2.6, -2, 2), "Press 1 for magnetometer calibration, 2 for gyro calibration, 3 for accel calibration", 16)
        
        [yaw, pitch , roll] = self.quat_to_ypr(self.pdr.Q)

        self.drawText((-2.6, -1.8, 2), f"Q: {self.pdr.Q[0]:.3f}, x: {self.pdr.Q[1]:.3f}, y: {self.pdr.Q[2]:.3f}, z: {self.pdr.Q[3]:.3f}", 16)
        self.drawText((2, 1.8, 2) , f"Recording {'X' if self.capture else ' '}", 16, color=(0, 255, 0, 255) if self.capture else (255, 0, 0, 255))
        self.drawText((2,1.9,2), f"fps: {self.frames / ((pygame.time.get_ticks() - self.ticks) / 1000.0):.2f}", 16)
        if self.capture:
            accel = self.pdr.rotation_matrix @ self.pdr.data['accel'].iloc[-1].to_numpy()
            self.drawText((-2.6, -1.9, 2), f"Q accel : x: {accel[0]:.1f}, y: {accel[1]:.1f}, z: {accel[2]:.1f}", 16)
            accel_s = np.array(['0', '0', '0'])
            for i in range(len(accel)):
                if accel[i] > 1:
                    accel_s[i] = '+'
                elif accel[i] < -1:
                    accel_s[i] = '-'
            self.drawText((1, -1.9, 2), f"x: {accel_s[0]}, y: {accel_s[1]}, z: {accel_s[2]}", 16)
        glRotatef(2 * math.acos(self.pdr.Q[0]) * 180.00/math.pi, self.pdr.Q[1], 1 * self.pdr.Q[3], -1 * self.pdr.Q[2])
        self.draw_box()
    
    @staticmethod
    def drawText(position, textString, size, color=(255, 255, 255, 255)):
        font = pygame.font.SysFont("Courier", size, True)
        textSurface = font.render(textString, True, color, (0, 0, 0, 255))
        textData = pygame.image.tostring(textSurface, "RGBA", True)
        glRasterPos3d(*position)
        glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

    @staticmethod
    def quat_to_ypr(q):
        yaw   = math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
        pitch = -math.asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
        roll  = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
        pitch *= 180.0 / math.pi
        yaw   *= 180.0 / math.pi
        # yaw   -= self.pdr.magnetic_declination
        roll  *= 180.0 / math.pi
        return [yaw, pitch, roll]

    @staticmethod
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
    @staticmethod
    def draw_box():
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

from ahrs.filters import Mahony, Madgwick, EKF

if __name__ == "__main__":
    pdr = PDR(lp=True, filter=Mahony, frequency=200)
    vis = Visulize(1280, 720, pdr)
    vis.run()