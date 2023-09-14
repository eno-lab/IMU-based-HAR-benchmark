import os

import h5py
import numpy as np
import pandas as pd
import simplejson as json
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from ..core import DataReader


class Ucihar(DataReader):
    def __init__(self, dataset):
        super().__init__(dataset, 'ucihar', 128)

        label_map = [
            (1, 'Walking'),
            (2, 'Walking_Upstairs'),
            (3, 'Walking_Downstairs'),
            (4, 'Sitting'),
            (5, 'Standing'),
            (6, 'Laying')
        ]
        self._id_to_label= [x[1] for x in label_map]

        if self.is_ratio():
            self.split_with_ratio()
        elif dataset.startswith('ucihar_losocv_'):
            n = int(dataset[len('ucihar_losocv_'):])
            self._split_ucihar_losocv(n)
        elif dataset == 'ucihar':
            self._split_ucihar()
        else:
            raise ValueError(f'invalid dataset name: {dataset}')


    # This data is already windowed and segmented
    def _split_ucihar_losocv(self, n):
        assert 1 <= n <=30
        subject_list = [i for i in range(1, 31)] 
        n -= 1

        subjects = {}
        subjects['test'] = [subject_list[n]]

        # sub 11, 13, 25 show low accuracy with val 30
        # 14, 17 is show lower accuracy
        if n == 10:
            subjects['validation'] = [subject_list[12], subject_list[29]]
        elif n == 29:
            subjects['validation'] = [subject_list[10], subject_list[28]]
        else:
            subjects['validation'] = [subject_list[10], subject_list[29]]

        subjects['train'] = [ s for s in subject_list if s not in subjects['test'] and s not in subjects['validation'] ]

        return self._split_ucihar(subjects = subjects)


    # This data is already windowed and segmented
    def _split_ucihar(self, label_map=None, subjects=None):
        if subjects is None:
            subjects = {
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
        print("Data: ", self._data['X'].shape, "Targets: ", self._data['y'].shape, "Subjects: ", self._data['id'].shape)

        def split_uci_data(subjectlist):
            flags = np.in1d(self._data['id'], subjectlist)
            return (self._data['X'][flags], 
                    to_categorical(self._data['y'][flags].astype(int), self.n_classes))

        self._X_train, self._y_train = split_uci_data(subjects['train'])
        self._X_valid, self._y_valid = split_uci_data(subjects['validation'])
        self._X_test, self._y_test= split_uci_data(subjects['test'])

        self._train = tf.data.Dataset.from_tensor_slices((self._X_train, self._y_train))
        self._validation = tf.data.Dataset.from_tensor_slices((self._X_valid, self._y_valid))
        self._test = tf.data.Dataset.from_tensor_slices((self._X_test, self._y_test))


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
        _labels = _labels - 1
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

