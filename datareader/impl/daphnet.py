import csv
import os

import numpy as np
from ..core import DataReader


class Daphnet(DataReader):
    def __init__(self, dataset):
        self._subjects = [
                ['S01R01.txt', 'S01R02.txt'],
                ['S02R01.txt'],
                ['S02R02.txt'], # it splited to implement ispl benchmark configuration
                ['S03R01.txt', 'S03R02.txt'],
                ['S03R03.txt'], # R03 includes label:1 only. # it splited to implement ispl benchmark configuration
                ['S04R01.txt'], # 1 only
                ['S05R01.txt'], 
                ['S05R02.txt'], # it splited to implement ispl benchmark configuration
                ['S06R01.txt', 'S06R02.txt'], # R02 includes label:1 only.
                ['S07R01.txt', 'S07R02.txt'],
                ['S08R01.txt'],
                ['S09R01.txt'],
                ['S10R01.txt']  # 1 only
        ]

        super().__init__(
                dataset = dataset, 
                dataset_origin = 'daphnet', 
                win_size = 192,  # 3 sec 64Hz
                data_cols = [x for x in range(10)],
                sensor_ids = [ix for ix in range(3) for _ in range(3)] # three sensors for three axes
                )


    def split_losocv(self, n, label_map = None):
        n -=1
        assert 0 <= n < 10

        subject_map = [
                [0],
                [1, 2],
                [3, 4],
                [5],
                [6, 7],
                [8],[9],[10],[11],[12]]

        subjects = {}
        subjects['test'] = []
        subjects['validation'] = []
        subjects['train'] = []
        subjects['test'].extend(subject_map[n])

        if n == 2:
            subjects['validation'].extend(subject_map[4])
        else:
            subjects['validation'].extend(subject_map[2])


        subjects['train'] = [i for i in range(13) if (
            i not in subjects['test'] and 
            i not in subjects['validation']
            )]

        self.split(subjects)


    def split(self, tr_val_ts_ids = None, label_map = None):
        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': [0, 3, 8, 9, 10, 11, 12],
                'validation': [2, 4, 6],
                'test': [1, 5, 7]
            }
        
        if label_map is None:
            label_map = (
                # (0, 'Other')
                (1, 'No freeze'),
                (2, 'Freeze')
            )

        self.split_data(tr_val_ts_ids, label_map)


    def read_data(self):
        ixs = []
        filenames = []
        for i, filelist in enumerate(self._subjects):
            for filename in filelist:
                ixs.append(i)
                filenames.append(filename)

        self._read_data(
                zip(ixs, filenames), 
                lambda filename: pd.read_csv(os.path.join(self.datapath, 'dataset', filename), sep=" ", header=None),
                label_col = -1,
                x_magnif = 0.001,
                interpolate_limit = 13 # 13/64 is almost 0.2 hz
                )
