import numpy as np

def extract_imu_tensor_func_pamap2(in_x):
    # [N, C, Sensor, Time]
    sensor_location = 1

    imu_num = 3
    imu_feature_num = 4 # a16, a8, gyro, mag
    x_imu = []
    for i in range(imu_num):
        _x = in_x[:,:,(0+3*imu_feature_num*i):(3*imu_feature_num+3*imu_feature_num*i),:]
        # a16, a8, gyro, mag
        # to mag, gyro, a16
        _x = _x[:,:,[9,10,11,6,7,8,0,1,2],]
        x_imu.append(_x)

    return x_imu

def extract_imu_tensor_func_ucihar(in_x):
    # [N, C, Sensor, Time]
    # ACC, GYRO, MAG to MAG, GYRO, ACC
    x_imu = [in_x[:,:,[6,7,8,3,4,5,0,1,2],:]]

    return x_imu 

def extract_imu_tensor_func_mighar(in_x):
    # [N, C, Sensor, Time]
    sensor_num = in_x.shape[2]//9
    sid_list = list(range(sensor_num))

    in_x_sid = None
    if in_x.shape[2]%9 == 1: # separation with id
        # ignore it
        #in_x_sid = in_x[:,0:1,-2:-1]
        in_x = in_x[:,:,0:-1,:]
    elif in_x.shape[3]%2 == 1: # combination with id
        # ignore it
        #sid_list = list[np.array(in_x[0,-1,range(0, in_x.shape[-1], 9)])]
        in_x = in_x[:,:,:,0:-1]
    elif in_x.shape[2]%9 != 0 or in_x.shape[3]%2 != 0: # error
        raise ValueError(f'Invalid input shape: {in_x.shape=}')

    x_imu = []
    for i in range(sensor_num):
        six = i*9
        eix = six+9
        cur_imu_x = in_x[:,:,six:eix,:]
        # ACC, GYRO, MAG to MAG, GYRO, ACC
        cur_imu_x = cur_imu_x[:,:,[6,7,8,3,4,5,0,1,2],:]
        x_imu.append(cur_imu_x)

    return x_imu
