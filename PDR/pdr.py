from calendar import c
import numpy as np
import pandas as pd
from ahrs.filters import Mahony, Madgwick, EKF
import ahrs
import logging
import itertools

import matplotlib.pyplot as plt
from pyparsing import col
import PDR.util as util
from PDR.udp import UDP
from calibrate import Calibrate, sampler
from scipy import signal

import time

class PDR:
    def __init__(self, filter=Mahony , frequency=100, lp=False, cutoff=5, order=10 , port=1234, lattitude=None, longitude=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.filter = filter(frequency=frequency)
        self.filter_update = self.filter.update if filter.__name__ == 'EKF' else self.filter.updateMARG
        self.dt = self.filter.Dt
        self.lp = lp
        self.port = port

        # magnetic declination and magnetic norm
        self.magnetic_declination = 0
        self.norm = 55
        if lattitude and longitude:
            wmm = ahrs.utils.WMM()
            wmm.get_field(lattitude, longitude, 0)['I']
            self.norm = wmm.magnetic_elements['I']
            self.magnetic_declination = wmm.magnetic_elements['D']
        self.udp_init()

        self.Q = np.array([1., 0., 0., 0.])

        try:
            self.calib = pd.read_csv('calib.csv', sep='\t', header=[0,1], index_col=[0,1]).T
        except:
            self.logger.warning("No calibration file found")
            columns = tuple(itertools.product(('accel', 'gyro', 'mag'), ('x', 'y', 'z')))
            index = tuple(itertools.product(('A'), ('x', 'y', 'z'))) + (('b','b'),)
            data = np.array([np.eye(3)]*3).reshape(9,3).T
            data = np.append(data, np.zeros(shape=(1,9)), axis=0)
            self.calib = pd.DataFrame(data, columns=columns, index=index)
        
        self.calib_b = self.calib.loc['b', 'b']
        self.calib_A = self.calib.loc['A']
        
        if lp:
            self.lp_init(cutoff=cutoff, order=order, fs=frequency)
        
        self.data = pd.DataFrame()
        self.true_accel = pd.DataFrame()
        self.data_f = pd.DataFrame()
        self.capture = False
        
    def udp_init(self):
        self.udp = UDP()

    def lp_init(self, cutoff=5, order=10, fs=200):
        print(order, cutoff, fs)
        b, a = signal.butter(order, cutoff, fs=fs, btype='lowpass', analog=False)
        self.lfilter = {col :util.LiveLFilter(b, a) for col in self.calib.columns}

    def update(self):
        data = self.udp.read()
        true_accel = np.array([])
        q = []
        data -= self.calib_b
        data_f = []
        data['mag'] = (self.calib_A['mag'] @ data['mag'].T).T
        # data['accel'] = (self.calib_A['accel'] @ data['accel'].T).T
        if self.lp:
            data = data.agg(self.lfilter)
        for idx, v in data.iterrows():            
            self.Q = self.filter_update(self.Q, v['gyro'], v['accel'], v['mag'])
            self.rotation_matrix = self.quat_to_rot_mat()
            true_accel = np.append(true_accel, self.rotation_matrix @ v['accel'])
            q.append(self.Q)
        data_f = pd.DataFrame(data_f, columns=data.columns, index=data.index)
        data['q'] = q
        if self.capture:
            self.data = pd.concat([self.data, data], axis=0)
            self.data_f = pd.concat([self.data_f, data_f], axis=0)
            self.true_accel = pd.concat([self.true_accel, pd.DataFrame(true_accel.reshape(-1, 3), columns=('x', 'y', 'z'))], axis=0)

        return self.Q
    
    def quat_to_rot_mat(self):
        q = self.Q
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
    
    
    def calibrate_gyro(self):
        self.udp.close()
        calib = Calibrate(sampler=sampler.linear, N_single=1000, sensor='gyro', lim=0.05)
        calib.run()
        self.calib_b['gyro'] = calib.calib_b
        self.save_calib()
        self.udp_init()
        return calib.calib_b
        
    def calibrate_accel(self):
        self.udp.close()
        calib = Calibrate(norm=9.8, sampler=sampler.single, N_single=100, sensor='accel', lim=15)
        calib.run()
        self.calib_b['accel'] = calib.calib_b
        self.calib_A['accel'] = calib.calib_A
        self.save_calib()
        self.udp_init()
        return calib.calib_b, calib.calib_A

    def calibrate_mag(self):
        self.udp.close()
        calib = Calibrate(norm=self.norm, sampler=sampler.continuous, sensor='mag', lim=200)
        calib.run()
        self.calib_b['mag'] = calib.calib_b
        self.calib_A['mag'] = calib.calib_A
        self.save_calib()
        self.udp_init()
        return calib.calib_b, calib.calib_A

    def save_calib(self):
        self.calib.T.to_csv('calib.csv', sep='\t')

    def plot_path(self):
        df = self.true_accel.copy()
        # velocity
        df.columns = tuple(itertools.product(['accel'], ['x', 'y', 'z']))
        sf = df.shift(1).apply(lambda x: x * self.dt)
        sf.columns = (('vel', 'x'), ('vel', 'y'), ('vel', 'z'))
        sf = sf.cumsum()
        df = pd.concat([df, sf], axis=1)
        df.fillna(0, inplace=True)
        # position
        sf = pd.DataFrame(0, index=df.index, columns=(('pos', 'x'), ('pos', 'y'), ('pos', 'z')))
        df = pd.concat([df, sf], axis=1)
        df['pos'] = df['vel'].shift(1) * self.dt + df['accel'].shift(1) * (self.dt ** 2) / 2
        df.fillna(0, inplace=True)
        df['pos'] = df['pos'].cumsum()
        # plot
        fig, ax = plt.subplots(1,3, figsize=(15, 10), tight_layout=True)
        ax.flatten ()

        ax[0].plot(df['accel']['x'], df['accel']['y'], label='accel')
        ax[0].set_title('acceleration')
        ax[0].axis('equal')
        ax[0].grid()

        ax[1].plot(df['vel']['x'], df['vel']['y'], label='vel')
        ax[1].set_title('velocity')
        ax[1].axis('equal')
        ax[1].grid()

        ax[2].plot(df['pos']['x'], df['pos']['y'], label='pos')
        ax[2].set_title('position')
        ax[2].axis('equal')
        ax[2].grid()

        plt.show()