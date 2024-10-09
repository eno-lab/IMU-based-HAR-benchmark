import os

import numpy as np
import pandas as pd
from ..core import DataReader


class Pamap2(DataReader):
    def __init__(self, dataset):

        self._filelist = [
            'subject101.dat', 
            'subject102.dat', 
            'subject103.dat', # 103 does not include labels 5 to 7.
            'subject104.dat', # 104 include only one sample for label:5.
            'subject105.dat', # suitable for validation, enough # for all classes
            'subject106.dat', # suitable for validation, enough # for all classes
            'subject107.dat', 
            'subject108.dat', 
            'subject109.dat' # 109 includes label:24 only
        ]

        self._label_map = [
            # (0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'nordic walking'),
            # (9, 'watching TV'),
            # (10, 'computer work'),
            # (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            # (18, 'folding laundry'),
            # (19, 'house cleaning'),
            # (20, 'playing soccer'),
            # (24, 'rope jumping')
        ]

        super().__init__(
                dataset = dataset, 
                dataset_origin = 'pamap2', 
                win_size = 256, 
                data_cols = [
                    1,  # Label
                    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # IMU Hand
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # IMU Chest
                    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49  # IMU ankle
                ],
                sensor_ids = [i for i in range(3) for _ in range(12)], # 12 cols for three imu
                )


    def init_params_dependig_on_dataest(self):
        if self.dataset.startswith('pamap2-with_rj'):
            self._label_map.append((24, 'rope jumping'))
            self._normal_split_dataset_name.append('pamap2-with_rj')


    def split_losocv(self, n, label_map = None):
        n -=1
        assert 0 <= n <= 8

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

        self.split(subjects, label_map=label_map)


    def split(self, subjects = None, label_map = None):
        if subjects is None:
            subjects = {
                'train': [0, 1, 2, 3, 6, 7, 8],
                'validation': [4],
                'test': [5]
            }

        if label_map is None:
            label_map = self._label_map

        self.split_data(subjects, label_map)


    def read_data(self):
        self._read_data(enumerate(self._filelist),
                       lambda filename: pd.read_csv(os.path.join(self.datapath, 'Protocol', filename), sep=" ", header=None),
                       label_col = -1,
                       interpolate_limit = 20 # 20/100 Hz = 0.2 Hz
                       )
