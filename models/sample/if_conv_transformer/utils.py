
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
