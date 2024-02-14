import os

import numpy as np
import pandas as pd
from ..core import DataReader


class Motionsense(DataReader):
    def __init__(self, dataset):
        super().__init__(
                dataset = dataset, 
                dataset_origin = 'motionsense',
                win_size = 128,  # 50 hz, 2.56 sec
                data_cols = list(range(12)), # attitude_roll-pitch-yaw + gravity_xyz + gyro_xyz + acc_xyz
                sensor_ids = [0 for _ in range(12)])

    def split_losocv(self, n):
        assert 1 <= n <= 24
        subject_list = [i for i in range(1, 25)]

        subjects = {}
        subjects['test'] = [n]

        if 4 <= n <= 6:
            subjects['validation'] = [3] + [s for s in list(range(4, 7)) if s is not n]
        else:
            subjects['validation'] = [4, 5, 6]

        subjects['train'] = [ s for s in subject_list if s not in subjects['test'] and s not in subjects['validation'] ]
        self.split(subjects)

    def split(self, tr_val_ts_ids = None, label_map = None):
        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': list(range(7, 25)),
                'validation': list(range(4, 7)),
                'test': list(range(1, 4)),
            }
        if label_map is None:
            label_map = [
                (1, 'Stairs Down'), # dws
                (2, 'Stairs Up'),   # ups
                (3, 'Walking'),     # wlk
                (4, 'Jogging'),     # jog
                (5, 'Standing'),    # std
                (6, 'Stting'),      # sit
            ]

        self.split_data(tr_val_ts_ids, label_map)

    def read_data(self):

        act_folders = [
            'dws_1',
            'dws_2',
            'dws_11',
            'jog_9',
            'jog_16',
            'sit_5',
            'sit_13',
            'std_6',
            'std_14',
            'ups_3',
            'ups_4',
            'ups_12',
            'wlk_7',
            'wlk_8',
            'wlk_15',
        ]
        act_name_to_id = {
            'dws' : 1, 
            'ups' : 2, 
            'wlk' : 3, 
            'jog' : 4, 
            'std' : 5, 
            'sit' : 6
        }

        def read_data(filename):
            signals_keys = [
                'attitude.roll',
                'attitude.pitch',
                'attitude.yaw',
                'gravity.x',
                'gravity.y',
                'gravity.z',
                'rotationRate.x',
                'rotationRate.y',
                'rotationRate.z',
                'userAcceleration.x',
                'userAcceleration.y',
                'userAcceleration.z'
            ]

            df = pd.read_csv(os.path.join(self.datapath, filename), sep=" ", header=None)
            signals = df[signals_keys]

            return signals

        user_ids = []
        filepaths = []
        act_labels = []

        for act_folder in act_folders:
            act_name = act_folder.split('_')[0]
            act_id = act_name_to_id[act_name]

            for subject in range(1, 25):
                filename = os.path.join('A_DeviceMotion_data',
                                        act_folder,
                                        f'sub_{subject}.csv')
                
                user_ids.append(subject)
                filepaths.append(filename)
                act_labels.append(act_id)

        self._read_data(
            zip(user_ids, filepaths, act_labels),
            read_data,
            label_col = -1,
        )
