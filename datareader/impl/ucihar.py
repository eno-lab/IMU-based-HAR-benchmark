import os

import numpy as np
import pandas as pd
from ..core import DataReader


class Ucihar(DataReader):
    def __init__(self, dataset):

        self._ispl = False
        super().__init__(
                dataset = dataset, 
                dataset_origin = 'ucihar', 
                win_size = 128,
                data_cols = None,
                sensor_ids = [0 for _ in range(9)])



    def init_params_dependig_on_dataest(self):
        dataset = self.dataset

        if dataset.startswith('ucihar-ispl'):
            self._ispl = True
            self._normal_split_dataset_name.append('ucihar-ispl')
        elif dataset.startswith('ucihar-orig'):
            self._normal_split_dataset_name.append('ucihar-orig')


    def split_losocv(self, n):
        assert 1 <= n <=30
        subject_list = [i for i in range(1, 31)] 
        n -= 1

        subjects = {}
        subjects['test'] = [subject_list[n]]

        # sub 11, 13, 25 show low accuracy with val 30
        # 14, 17 is show lower accuracy
        if n == 7:
            subjects['validation'] = [subject_list[8], subject_list[22]]
        elif n == 22:
            subjects['validation'] = [subject_list[7], subject_list[24]]
        else:
            subjects['validation'] = [subject_list[7], subject_list[22]]

        subjects['train'] = [ s for s in subject_list if s not in subjects['test'] and s not in subjects['validation'] ]

        self.split(subjects)


    def split(self, tr_val_ts_ids = None, label_map = None):

        ######################
        if self._ispl:
            if tr_val_ts_ids is not  None:
                raise ValueError(f"iSPLInception's split is conflicted with specified split: {tr_val_ts_ids}")
            tr_val_ts_ids = {
                # Original train set = 70% of all subjects
                'train': [
                    1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17,
                    19, 21, 22, 23, 25, 26, 27, 28, 29, 30
                ],
                # 1/3 of test set = 10% of all subjects
                'validation': [
                    4, 12, 20
                ],
                # 2/3 of original test set = 20% of all subjects
                'test': [
                    2, 9, 10, 13, 18, 24
                ]
            }
        ######################
        else:
            """ Original split of ucihar """
            if tr_val_ts_ids is None:
                tr_val_ts_ids = {
                    'train': [
                        1, 3, 5, 6, 8, 11, 14, 15, 16, 17, 19,
                        21, 23, 25, 26, 27, 28, 29, 30
                    ],
                    'validation': [
                        22, 7
                    ],
                    'test': [
                        2, 4, 9, 10, 12, 13, 18, 20, 24,
                    ]
                }

        if label_map is None:
            label_map = [
                (1, 'Walking'),
                (2, 'Walking_Upstairs'),
                (3, 'Walking_Downstairs'),
                (4, 'Sitting'),
                (5, 'Standing'),
                (6, 'Laying')
            ]

        self.split_data(tr_val_ts_ids, label_map)


    def read_data(self):
        signals = [
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
        ]

        print('Loading train')
        x_train = self._load_signals('train', signals)
        y_train = self._load_labels('train')
        print('Loading test')
        x_test = self._load_signals('test', signals)
        y_test = self._load_labels(f'test')
        print("Loading subjects")
        # Pandas dataframes
        subjects_train = self._load_subjects('train')
        subjects_test = self._load_subjects('test')

        self._data = {}
        self._data['X'] = np.concatenate((x_train, x_test), 0)
        _labels = np.concatenate((y_train, y_test), 0)
        self._data['y'] = _labels
        _subjects = np.concatenate((subjects_train, subjects_test), 0)
        self._data['id'] = _subjects

    def _load_signals(self, subset, signals):
        signals_data = []

        for signal in signals:
            filename = os.path.join(
                    self.datapath,
                    subset,
                    'Inertial Signals',
                    f'{signal}_{subset}.txt')
            signals_data.append(
                pd.read_csv(filename, delim_whitespace=True, header=None).values
            )

        # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
        return np.transpose(signals_data, (1, 2, 0))

    def _load_labels(self, target, delimiter=","):
        label_path = os.path.join(self.datapath, target, f'y_{target}.txt')
        with open(label_path, 'rb') as file:
            y = np.loadtxt(label_path, delimiter=delimiter)
        return y

    def _load_subjects(self, target, delimiter=","):
        subject_path = os.path.join(self.datapath, target, f'subject_{target}.txt')
        return np.loadtxt(subject_path, delimiter=delimiter)

