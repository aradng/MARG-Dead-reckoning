import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam

from sklearn.utils import shuffle
from time import time

from dataset import *
from model import *
from util import *

def find_latest_checkpoint(checkpoint_format):
    checkpoint_pattern = re.compile(checkpoint_format.replace('{epoch:04d}', '(\d+)'))

    checkpoint_files = [file for file in os.listdir('.') if os.path.isfile(file) and checkpoint_pattern.match(file)]

    if checkpoint_files:
        checkpoint_epochs = [int(checkpoint_pattern.search(file).group(1)) for file in checkpoint_files]
        latest_checkpoint_epoch = max(checkpoint_epochs)
        latest_checkpoint_filename = checkpoint_format.format(epoch=latest_checkpoint_epoch)
        return latest_checkpoint_filename, latest_checkpoint_epoch
    else:
        return None, 0
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Model output name')
    parser.add_argument('--mag', help='enable magnetometer training', action='store_true')
    parser.add_argument('--epochs', type=int, default=500, help='number of total epochs')
    args = parser.parse_args()

    np.random.seed(0)

    window_size = 200
    stride = 10

    x_gyro = []
    x_acc = []
    x_mag = []

    y_delta_p = []
    y_delta_q = []

    imu_data_filenames = []
    gt_data_filenames = []

    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu3.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu1.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu2.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu2.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu4.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu4.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu2.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu7.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu4.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu5.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu3.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu2.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu3.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu1.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu3.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu5.csv')
    imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu4.csv')

    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi3.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi1.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi2.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi2.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi4.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi4.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi2.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi7.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi4.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi5.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi3.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi2.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi3.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi1.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi3.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi5.csv')
    gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi4.csv')
    
    for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):
        cur_gyro_data, cur_acc_data, mag_data, cur_pos_data, cur_ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)

        [cur_x_gyro, cur_x_acc, curr_x_mag], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, mag_data, cur_pos_data, cur_ori_data, window_size, stride)

        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)
        x_mag.append(curr_x_mag)

        y_delta_p.append(cur_y_delta_p)
        y_delta_q.append(cur_y_delta_q)

    x_gyro = np.vstack(x_gyro)
    x_acc = np.vstack(x_acc)
    x_mag = np.vstack(x_mag)

    y_delta_p = np.vstack(y_delta_p)
    y_delta_q = np.vstack(y_delta_q)

    x_gyro, x_acc, x_mag, y_delta_p, y_delta_q = shuffle(x_gyro, x_acc, x_mag, y_delta_p, y_delta_q)

    # Define the checkpoint file name
    checkpoint_format = f'{args.output}_checkpoint_{"{epoch:04d}"}.hdf5'
    last_checkpoint_filename, initial_epoch = find_latest_checkpoint(checkpoint_format)
    
    if initial_epoch >= args.epochs:
        print("model already fully trained")
        print("exiting ....")
        exit()

    # Load the existing model checkpoint if resuming
    try:
        train_model = load_model(last_checkpoint_filename, custom_objects={'CustomMultiLossLayer': CustomMultiLossLayer})

        print("Resuming training from checkpoint.")
    except:
        pred_model = create_pred_model_6d_quat(window_size, mag=args.mag)
        train_model = create_train_model_6d_quat(pred_model, window_size)
        train_model.compile(optimizer=Adam(0.0001), loss=None)
        print("no checkpoints found")
    
    model_checkpoint = ModelCheckpoint(checkpoint_format, monitor='val_loss', save_best_only=True, verbose=1, period=max(args.epochs//10, 1))
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    history = train_model.fit([x_gyro, x_acc, x_mag, y_delta_p, y_delta_q], 
                            initial_epoch=initial_epoch,
                            epochs=args.epochs, 
                            batch_size=32, 
                            verbose=1, 
                            callbacks=[model_checkpoint, tensorboard], 
                            validation_split=0.1)

    pred_model = create_pred_model_6d_quat(window_size, mag=args.mag)
    pred_model.set_weights(train_model.get_weights()[:-2])

    pred_model.save('%s.hdf5' % args.output)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{args.output}_loss.png")

if __name__ == '__main__':
    main()