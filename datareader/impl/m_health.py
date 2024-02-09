import os

import numpy as np
import pandas as pd
from ..core import DataReader


class MHealth(DataReader):
    def __init__(self, dataset):

        self._filelist = [
            'mHealth_subject1.log', 
            'mHealth_subject2.log', 
            'mHealth_subject3.log', 
            'mHealth_subject4.log', 
            'mHealth_subject5.log', 
            'mHealth_subject6.log', 
            'mHealth_subject7.log', 
            'mHealth_subject8.log', 
            'mHealth_subject9.log',
            'mHealth_subject10.log'
        ]

        super().__init__(
                dataset = dataset, 
                dataset_origin = 'm_health', 
                win_size = 128, 
                data_cols = [ i for i in range(24) ],
                # readme seems doubt: acc, ecg1, ecg2, acc, gyro, mag, acc, gyro, mag, label
                # correct probably: acc, ecg1, ecg2, acc, mag, gyro, acc, mag, gyro, label
                dataset_path = os.path.join('dataset', 'MHEALTHDATASET'),
                sensor_ids = [0,0,0, 1,1] + [i for i in range(2,4) for _ in range(3*3)]
                )


    def split_losocv(self, n):
        n -=1
        assert 0 <= n < 10

        subjects = {}
        subjects['test'] = [n]

        if n == 5:
            subjects['validation'] = [4]
        else:
            subjects['validation'] = [5]

        subjects['train'] = [i for i in range(8) if (
            i not in subjects['test'] and 
            i not in subjects['validation']
            )]

        self.split(subjects)


    def split(self, tr_val_ts_ids = None, label_map = None):
        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': [0, 1, 2, 3, 6, 7, 8, 9, 10],
                'validation': [4],
                'test': [5]
            }

        if label_map is None:
            label_map = [
                # (0, 'null'),
                (1, 'Standing_still'),
                (2, 'Sitting and relaxing'),
                (3, 'Lying down'),
                (4, 'Walking'),
                (5, 'Climbing stairs'),
                (6, 'Waist bends forwards'),
                (7, 'Frontal elevetion of arms'),
                (8, 'Knees bending (crouching)'),
                (9, 'Cycling'),
                (10, 'Jogging'),
                (11, 'Running'),
                (12, 'Jump front and back'),
            ]


        self.split_data(tr_val_ts_ids, label_map)

    def read_data(self):
        self._read_data(enumerate(self._filelist),
                       lambda filename: pd.read_csv(os.path.join(self.datapath, filename), sep="\t", header=None),
                       label_col = -1,
                       interpolate_limit = 10 # 10/50 Hz = 0.2 Hz
                       )
