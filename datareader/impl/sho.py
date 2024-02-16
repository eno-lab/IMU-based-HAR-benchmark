import os

import numpy as np
import pandas as pd
import scipy.io
from ..core import DataReader


class Sho(DataReader):

    def __init__(self, dataset):
        self._label_map = np.array([ # (id, tag, tag-dataset)
                (1, 'walking', 'walking'),
                (2, 'sitting', 'sitting'),
                (3, 'standing', 'standing'),
                (4, 'jogging', 'jogging'),
                (5, 'biking', 'biking'),
                (6, 'walking upstairs', 'upstairs'),
                (7, 'walking downstairs', 'downstairs')
            ], dtype=object)
        super().__init__(
                dataset = dataset, 
                dataset_origin = 'sho',
                win_size = 128,  # 2.56 sec in 50 hz
                dataset_path = os.path.join('dataset', 'Activity_Recognition_Dataset'),
                data_cols = [1 + j + 14 * location for location in range(5) for j in range(12)] + [-1],  # acc_xyz + liner_acc_xyz + gyro_xyz + mag_xyz
                sensor_ids = [location for location in range(5) for _ in range(12)])


    def split_losocv(self, n):
        assert 1 <= n <= 10
        subject_list = [i for i in range(1, 11)] 

        subjects = {}
        subjects['test'] = [n]

        if 3 <= n <= 4:
            subjects['validation'] = [2] + [i for i in range(3, 5) if i is not n]
        else:
            subjects['validation'] = list(range(3, 5))

        subjects['train'] = [ s for s in subject_list if s not in subjects['test'] and s not in subjects['validation'] ]

        self.split(subjects)


    def split(self, tr_val_ts_ids = None, label_map = None):
        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': list(range(5, 11)),
                'validation': list(range(3, 5)),
                'test': list(range(1, 3))
            }

        if label_map is None:
            label_map = self._label_map[:, 0:2]

        self.split_data(tr_val_ts_ids, label_map)


    def read_data(self):

        user_ids = []
        filepaths = []
        for uid in range(1, 11):
            user_ids.append(uid)
            filepaths.append(os.path.join(self.datapath,
                                f'Dataset',
                                f'Participant_{uid}.csv'))
        
        act_name_to_id = {l:n for n, _, l in self._label_map}
        act_name_to_id['upsatirs'] = act_name_to_id['upstairs'] # fix tag

        def read_file(filepath):
            df = pd.read_csv(filepath, sep=',', skiprows=1)

            df.iloc[:,-1] = [act_name_to_id[label] for label in df.iloc[:,-1]]

            return df
            
        self._read_data(
            zip(user_ids, filepaths), 
            read_file
            )
