import enum
import sys

sys.path.append("../")
from PDR.udp import UDP
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PDR.util import BlitManager
import time
from scipy import linalg
from enum import Enum

class sampler(Enum):
    single = 0
    continuous = 1
    linear = 2

class Calibrate(object):
    def __init__(self, norm=None , outlier_percentile=0.05, sampler=sampler.continuous, sensor='mag', N_single=100, lim=200):
        self.outlier_percentile = outlier_percentile
        self.norm = norm
        self.calib_A = np.eye(3)
        self.calib_b = np.zeros(3)
        self.N = N_single
        self.sensor = sensor
        self.sampler = sampler
        self.lim = lim

        self.U = UDP(single=(self.sampler == sampler.single))

        self.fig, ax = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True)
        self.ax = ax.flatten()
        self.data = pd.DataFrame()
        self.t = np.array([])

        self.calibrating = False

    def draw_init(self):
        self.xy = self.ax[0].scatter([], [], c="r", marker=".", animated=True)
        self.ax[0].set_title("XY Data")
        self.ax[0].set_xlabel("X [uT]")
        self.ax[0].set_ylabel("Y [uT]")
        self.ax[0].set_xlim(-self.lim, self.lim)
        self.ax[0].set_ylim(-self.lim, self.lim)
        self.ax[0].axis("equal")
        self.ax[0].grid()

        self.yz = self.ax[1].scatter([], [], c="r", marker=".", animated=True)
        self.ax[1].set_title("YZ Data")
        self.ax[1].set_xlabel("Y [uT]")
        self.ax[1].set_ylabel("Z [uT]")
        self.ax[1].set_xlim(-self.lim, self.lim)
        self.ax[1].set_ylim(-self.lim, self.lim)
        self.ax[1].axis("equal")
        self.ax[1].grid()

        self.xz = self.ax[2].scatter([], [], c="r", marker=".", animated=True)
        self.ax[2].set_title("XZ Data")
        self.ax[2].set_xlabel("X [uT]")
        self.ax[2].set_ylabel("Z [uT]")
        self.ax[2].set_xlim(-self.lim, self.lim)
        self.ax[2].set_ylim(-self.lim, self.lim)
        self.ax[2].axis("equal")
        self.ax[2].grid()

        self.title = self.ax[3].set_title(f"Data {len(self.data)} samples")
        self.bm = BlitManager(self.fig.canvas, [self.xy, self.yz, self.xz, self.title])
        self.calibrating = True

        self.fig.canvas.mpl_connect("key_press_event", self.save)
        self.fig.canvas.mpl_connect("close_event", self.close)

        plt.show(block=False)
        plt.pause(0.1)

    def save(self, event):
        if event.key == "c":
            if self.sampler == sampler.single:
                self.fetch_single()
            self.ax[3].scatter(self.data["x"], self.data["y"], c="r", marker=".")
            self.ax[3].scatter(self.data["y"], self.data["z"], c="b", marker=".")
            self.ax[3].scatter(self.data["x"], self.data["z"], c="g", marker=".")
            self.ax[3].set_xlim(-self.lim, self.lim)
            self.ax[3].set_ylim(-self.lim, self.lim)
            self.ax[3].grid()
            self.ax[3].axis("equal")
            self.fig.canvas.draw()
        if event.key == "r":
            self.data = pd.DataFrame()
            self.ax[3].cla()
            self.ax[3].grid()
            self.fig.canvas.draw()

    def close(self, event):
        self.calibrating = False

    def fetch(self):
        while self.calibrating:
            data = self.U.read()
            self.data = pd.concat([self.data, data[self.sensor]], axis=0).reset_index(
                drop=True
            )
            self.xy.set_offsets(np.c_[self.data["x"], self.data["y"]])
            self.yz.set_offsets(np.c_[self.data["y"], self.data["z"]])
            self.xz.set_offsets(np.c_[self.data["x"], self.data["z"]])
            self.title.set_text(f"Data {len(self.data)} samples")
            self.bm.update()
            self.t = np.append(self.t, time.time())
            if (self.sampler == sampler.linear) and (len(self.data) > self.N):
                self.calibrating = False
                break

        print(f"Average sampling frequency: {1//np.mean(np.diff(self.t))} Hz")

    def fetch_single(self):
        data = pd.DataFrame()
        while len(data) < self.N:
            data = pd.concat([data, self.U.read()[self.sensor]], axis=0).reset_index(
                drop=True
            )
        data = data.mean()
        data = pd.DataFrame([data], columns=data.index)
        self.data = pd.concat([self.data, data], axis=0)
        self.xy.set_offsets(np.c_[self.data["x"], self.data["y"]])
        self.yz.set_offsets(np.c_[self.data["y"], self.data["z"]])
        self.xz.set_offsets(np.c_[self.data["x"], self.data["z"]])
        self.title.set_text(f"Data {len(self.data)} samples")
        self.bm.update()

    def filter_outlier(self):
        for column in self.data.columns:
            lower_bound = self.data[column].quantile(0 + self.outlier_percentile / 2)
            upper_bound = self.data[column].quantile(1 - self.outlier_percentile / 2)
            self.data = self.data[
                (self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)
            ]
        return self.data

    def fit_ellipsoid(self):
        data = self.data.to_numpy()
        # D
        D = np.array(
            [
                data[:, 0] ** 2,
                data[:, 1] ** 2,
                data[:, 2] ** 2,
                2 * data[:, 1] * data[:, 2],
                2 * data[:, 0] * data[:, 2],
                2 * data[:, 0] * data[:, 1],
                2 * data[:, 0],
                2 * data[:, 1],
                2 * data[:, 2],
                np.ones_like(data[:, 0]),
            ]
        )
        # S, S_11, S_12, S_21, S_22
        S = np.dot(D, D.T)
        S_11 = S[:6, :6]
        S_12 = S[:6, 6:]
        S_21 = S[6:, :6]
        S_22 = S[6:, 6:]

        # C
        C = np.array(
            [
                [-1, 1, 1, 0, 0, 0],
                [1, -1, 1, 0, 0, 0],
                [1, 1, -1, 0, 0, 0],
                [0, 0, 0, -4, 0, 0],
                [0, 0, 0, 0, -4, 0],
                [0, 0, 0, 0, 0, -4],
            ]
        )

        # v_1, v_2
        E = np.dot(linalg.inv(C), S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))

        E_w, E_v = np.linalg.eig(E)

        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0:
            v_1 = -v_1
        
        v_2 = np.dot(np.dot(-linalg.inv(S_22), S_21), v_1)  # -S_22^{-1} S_21 v_1

        # quadric-form parameters
        M = np.array(
            [
                [v_1[0], v_1[5], v_1[4]],
                [v_1[5], v_1[1], v_1[3]],
                [v_1[4], v_1[3], v_1[2]],
            ]
        )
        n = np.array([v_2[0], v_2[1], v_2[2]])

        d = v_2[3]

        M_1 = linalg.inv(M)
        self.calib_b = -np.dot(M_1, n)
        self.calib_A = np.real(self.norm / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) * linalg.sqrtm(M))

        print("soft iron matrix: \n", self.calib_A)
        print("hard iron bias: \n", self.calib_b)
    
    def plot(self):
        self.calibrated_data = (self.data.copy() - self.calib_b)
        self.calibrated_data[self.calibrated_data.columns] = self.calibrated_data.apply((lambda x: self.calib_A @ x), axis=1, result_type='expand')

        self.fig, ax = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True)
        self.ax = ax.flatten()

        self.ax[0].scatter(self.data["x"], self.data["y"], c="r", marker=".", label='Raw Meas.')
        self.ax[0].scatter(self.calibrated_data["x"], self.calibrated_data["y"], c="b", marker=".", label='Calibrated Meas.')
        self.ax[0].set_title("XY Data")
        self.ax[0].set_xlabel("X [uT]")
        self.ax[0].set_ylabel("Y [uT]")
        self.ax[0].set_xlim(-self.lim, self.lim)
        self.ax[0].set_ylim(-self.lim, self.lim)
        self.ax[0].axis("equal")
        self.ax[0].grid()
        self.ax[0].legend()

        self.ax[1].scatter(self.data["y"], self.data["z"], c="r", marker=".", label='Raw Meas.')
        self.ax[1].scatter(self.calibrated_data["y"], self.calibrated_data["z"], c="b", marker=".", label='Calibrated Meas.')
        self.ax[1].set_title("YZ Data")
        self.ax[1].set_xlabel("Y [uT]")
        self.ax[1].set_ylabel("Z [uT]")
        self.ax[1].set_xlim(-self.lim, self.lim)
        self.ax[1].set_ylim(-self.lim, self.lim)
        self.ax[1].axis("equal")
        self.ax[1].grid()
        self.ax[1].legend()

        self.ax[2].scatter(self.data["x"], self.data["z"], c="r", marker=".", label='Raw Meas.')
        self.ax[2].scatter(self.calibrated_data["x"], self.calibrated_data["z"], c="b", marker=".", label='Calibrated Meas.')
        self.ax[2].set_title("XZ Data")
        self.ax[2].set_xlabel("X [uT]")
        self.ax[2].set_ylabel("Z [uT]")
        self.ax[2].set_xlim(-self.lim, self.lim)
        self.ax[2].set_ylim(-self.lim, self.lim)
        self.ax[2].axis("equal")
        self.ax[2].grid()
        self.ax[2].legend()

        self.ax[3].scatter(self.calibrated_data["x"], self.calibrated_data["y"], c="r", marker=".", label='Calibrated Meas.')
        self.ax[3].scatter(self.calibrated_data["y"], self.calibrated_data["z"], c="g", marker=".", label='Calibrated Meas.')
        self.ax[3].scatter(self.calibrated_data["x"], self.calibrated_data["z"], c="b", marker=".", label='Calibrated Meas.')
        self.ax[3].set_title("Data")
        self.ax[3].set_xlim(-self.lim, self.lim)
        self.ax[3].set_ylim(-self.lim, self.lim)
        self.ax[3].axis("equal")
        self.ax[3].grid()
        self.ax[3].legend()

        plt.show()

    def run(self):
        self.draw_init()
        # data fetcher 
        if self.sampler == sampler.single:
            plt.show()
        else:
            self.fetch()
        # data filter for continuous sampling
        if self.sampler == sampler.continuous:
            self.data = self.filter_outlier()
        # fit ellipsoid
        if self.sampler != sampler.linear:
            self.fit_ellipsoid()
        else:
            self.calib_b = self.data.mean()
            self.calib_A = np.eye(3)
        self.plot()
        self.U.close()
        
# cali = calibration(norm=54, sampler=sampler.continuous, sensor='mag', lim=200)
# cali = calibration(norm=9.8, sampler=sampler.single, N_single=100, sensor='accel', lim=15)
# cali = calibration(sampler=sampler.linear, N_single=1000, sensor='gyro', lim=0.05)
# cali.calibrate()