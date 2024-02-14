import os

import numpy as np
import pandas as pd
from ..core import DataReader


class Mighar(DataReader):

    def __init__(self, dataset: str):

        self._label_map = (
            # (0, 'Other')
            (1, 'walking'),
            (2, 'jogging'),
            (3, 'going_up_stairs'),
            (4, 'going_down_stairs'),
            (5, 'standing'),
            (6, 'sitting'),
            (7, 'typing'),
            (8, 'writing'),
            (9, 'eating_cookie'),
            (10, 'eating_pasta'),
            (11, 'ironing'),
            (12, 'folding'),
            (13, 'tooth_brushing'),
            (14, 'vacuuming'))

        self._file_prefix = 'rawdata'

        self.sensor_num = 396
        self.stypes = ('acc', 'gyr', 'mag')

        super().__init__(
                dataset = dataset,
                dataset_origin = 'mighar',
                win_size = 256, # 2.56 sec in 100 hz
                dataset_path = os.path.join('dataset', 'Meshed_IMU_Garment_HAR_Dataset'),
                data_cols = list(range(3*3*396)),
                sensor_ids = [i for i in range(self.sensor_num) for _ in range(3*len(self.stypes))]
                )


    def init_params_dependig_on_dataest(self):
        if self.dataset.startswith('{self.dataset_origin}-axis_sync'):
            self.dataset_origin = '{self.dataset_origin}-axis_sync'
            self._file_prefix = 'data'


    def split_losocv(self, n, label_map=None):
        assert 1 <= n <= 12

        subjects = {}
        subjects['test'] = [n]

        if n == 9:
            subjects['validation'] = [10]
        else:
            subjects['validation'] = [9]

        subjects['train'] = [i for i in range(1, 13) if (
            i not in subjects['test'] and 
            i not in subjects['validation']
            )]

        self.split(subjects, label_map=label_map)


    def split(self, tr_val_ts_ids = None, label_map = None):
        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'validation': [10],
                'test': [11, 12]
            }

        self.split_data(tr_val_ts_ids, self._label_map if label_map is None else label_map)


    def read_data(self):
        user_ids = []
        filepaths = []
        act_labels = []
        for uid in range(1,13):
            for act_label, act_name in self._label_map:
                act_dir = act_name.replace('_', '')
                for trial in range(1, 3):
                    user_ids.append(uid)
                    filepaths.append(os.path.join(self.datapath,
                                                  'data', 
                                                  f'subject{uid}', 
                                                  act_dir, 
                                                  f'{self._file_prefix}{trial}.csv.gz'))
                    act_labels.append(act_label)

        self._read_data(
                zip(user_ids, filepaths, act_labels), 
                lambda filepath: pd.read_csv(filepath, sep=",", dtype='float32'),
                x_magnif = 1, # unit is g 
                interpolate_limit = 20, # 20/100 is 0.2 hz
                dtype='float32'
                )
