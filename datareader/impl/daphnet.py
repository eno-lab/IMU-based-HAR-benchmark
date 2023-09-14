import csv
import os

import h5py
import numpy as np
import pandas as pd
import simplejson as json
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from ..core import DataReader


class Daphnet(DataReader):
    def __init__(self, dataset):
        self._label_map = {
            # (0, 'Other')
            '1': 'No freeze',
            '2': 'Freeze'
        }

        self._subjects = [
                ['S01R01.txt', 'S01R02.txt'],
                ['S02R01.txt'],
                ['S02R02.txt'], # it splited to implement ispl benchmark configuration
                ['S03R01.txt', 'S03R02.txt']
                ['S03R03.txt'], # R03 includes label:1 only. # it splited to implement ispl benchmark configuration
                ['S04R01.txt'], # 1 only
                ['S05R01.txt'], 
                ['S05R02.txt'], # it splited to implement ispl benchmark configuration
                ['S06R01.txt', 'S06R02.txt'], # R02 includes label:1 only.
                ['S07R01.txt', 'S07R02.txt'],
                ['S08R01.txt'],
                ['S09R01.txt']
                ['S10R01.txt']  # 1 only
        ]
        self._cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._cols = [x-1 for x in self._cols]

        super().__init__(dataset, 'daphnet', 192)  # 3 sec 64Hz

        if self.is_ratio():
            self.split_with_ratio()
        elif dataset.startswith('daphnet_losocv_'):
            n = int(dataset[len('daphnet_losocv_'):])
            self._split_daphnet_losocv(n)
        elif dataset == 'daphnet':
            self._split_daphnet()
        else:
            raise ValueError(f'invalid dataset name: {dataset}')


    def _split_daphnet_losocv(self, n, label_map = None):
        n -=1
        assert 0 <= n < 10

        subject_map = [
                [0],
                [1, 2],
                [3, 4],
                [5],
                [6, 7],
                [8],[9],[10],[11],12]]

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

        self._split_daphnet(subjects)


    def _split_daphnet(self, subjects = None):
        if subjects is None:
            subjects = {
                'train': [0, 3, 8, 9, 10, 11, 12],
                'validation': [2, 4, 6],
                'test': [1, 5, 7]
            }

        label_to_id = {x[0]: i for i, x in enumerate(label_map)}
        self._id_to_label = [x[1] for x in label_map]

        _filter = np.in1d(self._data['y'], list(label_to_id.keys()))
        _x = self._data['X'][_filter]
        _id = self._data['id'][_filter]
        _y = [[label_to_id[y]]for y in self._data['y'][_filter]]
        _y = to_categorical(np.asarray(_y, dtype=int), self.n_classes)

        _f_train = np.in1d(_id, subjects['train'])
        _f_valid = np.in1d(_id, subjects['validation'])
        _f_test = np.in1d(_id, subjects['test'])

        self._X_train = _x[_f_train]
        self._y_train = _y[_f_train]
        self._X_valid = _x[_f_valid]
        self._y_valid = _y[_f_valid]
        self._X_test = _x[_f_test]
        self._y_test = _y[_f_test]

        self._train = tf.data.Dataset.from_tensor_slices((self._X_train, self._y_train))
        self._validation = tf.data.Dataset.from_tensor_slices((self._X_valid, self._y_valid))
        self._test = tf.data.Dataset.from_tensor_slices((self._X_test, self._y_test))

    def read_data(self):
        data = []
        seg = []
        subject_ids = []
        labels = []
        label = None

        for i, filelist in enumerate(self._subjects):
            for filename in filelist:
                with open(os.path.join(self.datapath, 'dataset', filename), 'r') as f:
                    reader = csv.reader(f, delimiter=' ')
                    for line in reader:
                        if line[10] == '0':
                            label = None
                            seg = []
                            continue
                        
                        if label is not None:
                            if label != line[10]: # change label
                                seg = []
                                label = line[10]
                        else:
                            label = line[10]

                        elem = [line[ix] for ix in self._cols]
                        if sum([x == 'NaN' for x in elem]) == 0:
                            seg.append([float(x) / 1000 for x in elem[:-1]])
                            if len(seg) == self._win_size:
                                # accepted 
                                data.append(seg)
                                labels.append(int(label))
                                subject_ids.append(i)

                                seg = seg[int(len(seg)//2):] # stride = win_size/2
                        else:
                            label = None # reset
                            seg = [] 

        self._data = {}
        self._data['X'] = np.asarray(data)
        self._data['y'] = np.asarray(labels, dtype=int)
        self._data['id'] = np.asarray(subject_ids)
