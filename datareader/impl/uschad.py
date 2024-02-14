import os

import numpy as np
import pandas as pd
import scipy.io
from ..core import DataReader


class Uschad(DataReader):

    def __init__(self, dataset):
        super().__init__(
                dataset = dataset, 
                dataset_origin = 'uschad',
                win_size = 256,  # 2.56 sec in 100 hz
                dataset_path = os.path.join('dataset', 'USC-HAD'),
                data_cols = list(range(6)), # acc_xyz + gyro_xyz
                sensor_ids = [0 for _ in range(6)])


    def split(self, tr_val_ts_ids = None, label_map = None):
        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': list(range(5, 15)),
                'validation': list(range(3, 5)),
                'test': list(range(1, 3))
            }

        if label_map is None:
            label_map = [
                (1, 'Walking Forward'),
                (2, 'Walking Left'),
                (3, 'Walking Right'),
                (4, 'Walking Upstairs'),
                (5, 'Walking Downstairs'),
                (6, 'Running Forward'),
                (7, 'Jumping Up'),
                (8, 'Sitting'),
                (9, 'Standing'),
                (10, 'Sleeping'),
                (11, 'Elevator Up'),
                (12, 'Elevator Down')
            ]

        self.split_data(tr_val_ts_ids, label_map)


    def read_data(self):

        user_ids = []
        filepaths = []
        act_labels = []
        for uid in range(1, 15):
            for act_label in range(1, 13):
                for trial in range(1, 6):
                    user_ids.append(uid)
                    filepaths.append(os.path.join(self.datapath,
                                     f'Subject{uid}',
                                     f'a{act_label}t{trial}.mat'))

                    act_labels.append(act_label)
            
        self._read_data(
            zip(user_ids, filepaths, act_labels), 
            lambda filepath: pd.DataFrame(scipy.io.loadmat(filepath)['sensor_readings']),
            x_magnif = 1, # unit is g 
            interpolate_limit = 20, # 20/100 is 0.2 hz
            )
