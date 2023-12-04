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
        self._cols = [ i for i in range(24) ]
        # doubt: acc, ecg1, ecg2, acc, gyro, mag, acc, gyro, mag, label
        # correct probably: acc, ecg1, ecg2, acc, mag, gyro, acc, mag, gyro, label

        self._label_map = [
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
        
        super().__init__(dataset, 'm_health', 128, os.path.join('dataset', 'MHEALTHDATASET'))

        if self.is_ratio():
            self.split_with_ratio()
        elif dataset.startswith('m_health-losocv_'):
            n = int(dataset[len('m_health-losocv_'):])
            self._split_losocv(n)
        elif dataset == 'm_health':
            self._split()
        else:
            raise ValueError(f'invalid dataset name: {dataset}')

    def _split_losocv(self, n,limit=10):
        n -=1
        assert 0 <= n < limit

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

        self._split(subjects, label_map=label_map)


    def _split(self, subjects = None):
        if subjects is None:
            subjects = {
                'train': [0, 1, 2, 3, 6, 7, 8, 9, 10],
                'validation': [4],
                'test': [5]
            }

        self.split_data(subjects, self._label_map)

    def read_data(self):
        self._read_data(enumerate(self._filelist),
                       lambda filename: pd.read_csv(os.path.join(self.datapath, filename), sep="\t", header=None),
                       label_col = -1,
                       interpolate_limit = 10 # 10/50 Hz = 0.2 Hz
                       )
