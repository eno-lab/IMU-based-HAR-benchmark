import os

import numpy as np
import pandas as pd
from ..core import DataReader
from ..utils import interp_nans, to_categorical

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
        self._cols = [
                1,  # Label
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # IMU Hand
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # IMU Chest
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49  # IMU ankle
            ]
        
        super().__init__(dataset, 'pamap2', 256)

        if self.is_ratio():
            self.split_with_ratio()
        elif dataset.startswith('pamap2_full_losocv_'):
            n = int(dataset[len('pamap2_full_losocv_'):])
            self._split_pamap2_full_losocv(n)
        elif dataset == 'pamap2_full':
            self._split_pamap2_full()
        elif dataset.startswith('pamap2_losocv_'):
            n = int(dataset[len('pamap2_losocv_'):])
            self._split_pamap2_losocv(n)
        elif dataset == 'pamap2':
            self._split_pamap2()
        else:
            raise ValueError(f'invalid dataset name: {dataset}')

    def _split_pamap2_full_losocv(self, n):
        label_map = [
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
            (24, 'rope jumping')
        ]
        self._split_pamap2_losocv(n, label_map=label_map)


    def _split_pamap2_losocv(self, n, label_map = None, limit=8):
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

        self._split_pamap2(subjects, label_map=label_map)


    def _split_pamap2_full(self):
        label_map = [
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
            (24, 'rope jumping')
        ]

        self._split_pamap2(label_map=label_map)


    def _split_pamap2(self, subjects = None, label_map = None):
        if subjects is None:
            subjects = {
                'train': [0, 1, 2, 3, 6, 7, 8],
                'validation': [4],
                'test': [5]
            }

        if label_map is None:
            label_map = [
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

    def read_data(self):
        data = []
        seg = []
        subject_ids = []
        labels = []
        label = None
        for i, filename in enumerate(self._filelist):
            print('Reading file %d of %d' % (i + 1, len(self._filelist)))

            df = pd.read_csv(os.path.join(self.datapath, 'Protocol', filename), sep=" ", header=None)
            df = df.iloc[:,self._cols]
            label_df = df.iloc[:, 0].astype(int)

            df = df.iloc[:, 1:].astype(float)
            df.interpolate(inplace=True, limit=20) # 20/100 = 0.2Hz

            for ix, cur_label in enumerate(label_df):
                #nan_c = 0
                if cur_label == 0:
                    label = None
                    seg = []
                    continue

                if label is not None:
                    if label != cur_label: # change label
                        seg = []
                        label = cur_label
                else:
                    label = cur_label

                seg.append(ix)

                if len(seg) == self._win_size:
                    _sdf = df.iloc[seg,:]
                    if _sdf.isna().any(axis=None):
                        print(f"Warning: skip a segment include NaN. ({min(seg)}:{max(seg)+1}, label={label})")
                        continue

                    # accepted 
                    data.append(_sdf)
                    labels.append(label)
                    subject_ids.append(i)

                    seg = seg[int(len(seg)//2):] # stride = win_size/2

        self._data = {}
        self._data['X'] = np.asarray(data)
        self._data['y'] = np.asarray(labels, dtype=int)
        self._data['id'] = np.asarray(subject_ids)
