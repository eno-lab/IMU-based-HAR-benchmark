import csv
import glob
import sys
import os
import re

import h5py
import numpy as np
import pandas as pd
import simplejson as json


class DataReader:
    def __init__(self, dataset, dataset_origin, win_size):
        self.dataset = dataset
        self.dataset_origin = dataset_origin
        self.datapath = os.path.join('dataset', dataset_origin)
        self._id_to_label = None
        self._X_train = None
        self._y_train = None
        self._X_valid = None
        self._y_valid = None
        self._X_test  = None
        self._y_test  = None
        self._train = None
        self._validation = None
        self._test = None
        self._win_size = win_size
        self._data = {'X': None, 'y': None, 'id': None}

        if not dataset.startswith(self.dataset_origin):
            raise ValueError(f"Invalid dataset name: {self.dataset}")

        if self.is_cached():
            self.load_data()
        else:
            self.read_data()
            self.save_data()


    def read_data(self):
        raise NotImplementedError('Must be implemented')

    def is_cached(self):
        return os.path.exists(os.path.join(self.datapath, f'{self.dataset_origin}.h5'))

    def load_data(self):
        with h5py.File(os.path.join(self.datapath, f'{self.dataset_origin}.h5'), 'r') as f:
            self._data = {}
            self._data['X'] = np.array(f['X'])
            self._data['y'] = np.array(f['y'])
            self._data['id'] = np.array(f['id'])

    def save_data(self):
        with h5py.File(os.path.join(self.datapath, f'{self.dataset_origin}.h5'), mode='w') as f:
            f.create_dataset('X', data=self._data['X'])
            f.create_dataset('y', data=self._data['y'])
            f.create_dataset('id', data=self._data['id'])


    def gen_ispl_style_set(self):
        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.id_to_label, self.n_classes, 

   
    def is_ratio(self):
        ratio_re_match = re.match('{self.dataset_origin}_ratio_(\d+)_(\d+)_(\d+)', self.dataset)
        return ratio_re_match is not None

    def split_with_ratio(self):
        ratio_re_match = re.match('{self.dataset_origin}_ratio_(\d+)_(\d+)_(\d+)', self.dataset)
        r_train = float(ratio_re_match.groups()[0])
        r_valid = float(ratio_re_match.groups()[1])
        r_test  = float(ratio_re_match.groups()[2])

        total_ratio = r_train + r_valid + r_test
        r_train /= total_ratio
        r_valid /= total_ratio
        r_test /= total_ratio

        ix = np.arange(self._data['X'].shape[0])
        ix_train, ix_test = train_test_split(ix, test_size = test_ratio)
        ix_train, ix_valid= train_test_split(ix_train, test_size = valid_ratio/(train_ratio+valid_ratio))

        self._X_train = self._data['X'][ix_train]
        self._X_valid = self.data['X'][ix_valid]
        self._X_test = self.data['X'][ix_test]
        self._y_train = self.data['y'][ix_train]
        self._y_valid = self.data['y'][ix_valid]
        self._y_test = self.data['y'][ix_test]


    @property
    def n_classes(self):
        return len(self._id_to_label) if self._id_to_label is not None else None

    @property
    def id_to_label(self):
        return self._id_to_label

    @property
    def input_shape(self):
        return (None,) + self._data['X'].shape[1:]

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_valid(self):
        return self._X_valid

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_valid(self):
        return self._y_valid

    @property
    def y_test(self):
        return self._y_test

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._validation

    @property
    def test(self):
        return self._test

