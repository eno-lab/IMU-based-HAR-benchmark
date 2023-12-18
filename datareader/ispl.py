# title       :datareader
# description :Script to extract and load the 4 datasets needed for our papers
#              This script is extendable and can accommodate many more datasets
# author      :Ronald Mutegeki
# date        :20210203
# version     :1.0
# usage       :Either execute the file with "dataset_name" and "dataset_path" specified or call it in utils.py.
# notes       :Uses already downloaded datasets to prepare them for our models

# TODO add our information 

import csv
import glob
import sys
import os

import h5py
import numpy as np
import pandas as pd
import simplejson as json


# Structure followed in this file is based on : https://github.com/nhammerla/deepHAR/tree/master/data
class DataReader:
    def __init__(self, dataset, datapath, _type='original'):
        if dataset == 'daphnet':
            self.data, self.idToLabel = self._read_daphnet(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        if dataset.startswith('daphnet_losocv_'):
            n = int(dataset[len('daphnet_losocv_'):])
            self.data, self.idToLabel = self._read_daphnet_losocv(datapath.rstrip("/"), n)
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'opportunity_task_a':
            self.data, self.idToLabel = self._read_opportunity_task_a(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'opportunity_task_b2':
            self.data, self.idToLabel = self._read_opportunity_task_b2(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'opportunity_task_c':
            self.data, self.idToLabel = self._read_opportunity_task_c(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'opportunity_task_a_without_null':
            self.data, self.idToLabel = self._read_opportunity_task_a_without_null(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'opportunity_task_b2_without_null':
            self.data, self.idToLabel = self._read_opportunity_task_b2_without_null(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'opportunity_task_c_without_null':
            self.data, self.idToLabel = self._read_opportunity_task_c_without_null(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset.startswith('opportunity'):
            self.data, self.idToLabel = self._read_opportunity(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset.startswith('pamap2_losocv_'):
            n = int(dataset[len('pamap2_losocv_'):])
            self.data, self.idToLabel = self._read_pamap2_losocv(datapath.rstrip("/"), n)
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'pamap2_full':
            self.data, self.idToLabel = self._read_pamap2_full(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset.startswith('pamap2_full_losocv_'):
            n = int(dataset[len('pamap2_full_losocv_'):])
            self.data, self.idToLabel = self._read_pamap2_full_losocv(datapath.rstrip("/"), n)
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset.startswith('pamap2'):
            self.data, self.idToLabel = self._read_pamap2(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'ucihar':
            self.data, self.idToLabel = self._read_ucihar(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset.startswith('ucihar_losocv_'):
            n = int(dataset[len('ucihar_losocv_'):])
            self.data, self.idToLabel = self._read_ucihar_losocv(datapath.rstrip("/"), n)
            self.save_data(dataset, datapath.rstrip("/") + "/")
        elif dataset == 'ispl':
            self.data, self.idToLabel = self._read_ispl(datapath.rstrip("/"))
            self.save_data(dataset, datapath.rstrip("/") + "/")
        else:
            print('Dataset is not yet supported!')
            sys.exit(0)

    def save_data(self, dataset, path=""):
        f = h5py.File(f'{path}{dataset}.ispl.h5', mode='w')
        for key in self.data:
            f.create_group(key)
            for field in self.data[key]:
                f[key].create_dataset(field, data=self.data[key][field])
        f.close()
        with open(f'{path}{dataset}.ispl.h5.classes.json', 'w') as f:
            f.write(json.dumps(self.idToLabel))
        print('Done.')

    @property
    def train(self):
        return self.data['train']

    @property
    def validation(self):
        return self.data['validation']

    @property
    def test(self):
        return self.data['test']

    @staticmethod
    def _build_losocv(subjects, n, train=[]):
        """ 
            n: 1<= n <= len(subjects) 
            train: Someones must be assigned train. They must be excluded from subjects.
        """

        n -= 1
        train = train.copy()
        valid = []
        test = [subjects[n]]
        if n == 0:
            valid = [subjects[-1]]
            train.extend(subjects[1:-1])
        elif n == 1:
            valid = [subjects[0]]
            train.extend(subjects[2:])
        elif n == 2:
            valid = [subjects[1]]
            train.append(subjects[0])
            train.extend(subjects[3:])
        elif n == len(subjects)-2:
            valid = [subjects[n-1]]
            train.extend(subjects[0:(n-1)])
            train.append(subjects[-1])
        else:
            valid = [subjects[n-1]]
            train.extend(subjects[0:(n-1)])
            train.extend(subjects[(n+1):])

        return {'train': train, 'validation': valid, 'test': test}


    def _read_pamap2_full_losocv(self, datapath, n, label_map = None):
        """ n: 1 <= n <= 8 """
        # The followings must be assinged for train.
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

        dat_files = [
            'subject101.dat', 'subject102.dat', 
            'subject103.dat', # 103 does not include labels 5 to 7.
            'subject104.dat', # 104 include only one sample for label:5.
            'subject105.dat', # suitable for validation, enough # for all classes
            'subject106.dat', # suitable for validation, enough # for all classes
            'subject107.dat', 'subject108.dat', 
            'subject109.dat' # 109 includes label:24 only
        ]
        files = {}
        n -=1
        assert 0 <= n <= 8

        files['test'] = [dat_files[n]]

        if n == 4:
            files['validation'] = [dat_files[5]] # 106
        else:
            files['validation'] = [dat_files[4]] # 105

        files['train'] = [ f for f in dat_files if f not in files['test'] and f not in files['validation'] ]

        return self._read_pamap2(datapath, files = files, label_map = label_map)


    def _read_pamap2_losocv(self, datapath, n, label_map = None):
        """ n: 1 <= n <= 8 """
        # The followings must be assinged for train.
        dat_files = [
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
        files = {}
        n -=1
        assert 0 <= n <= 7

        files['test'] = [dat_files[n]]

        if n == 0:
            files['validation'] = [dat_files[6]] # 107
        else:
            files['validation'] = [dat_files[0]] 
            #files['validation'] = [dat_files[3], dat_files[4]]  # 70.74 ... 
            # 102 だめ、106だめ、105だめ, 108ダメ･･･。う〜ん

#
#        if n == 7:
#            files['validation'] = [dat_files[0]] # 106
#        else:
#            files['validation'] = [dat_files[7]] # 105
#
        files['train'] = [ f for f in dat_files if f not in files['test'] and f not in files['validation'] ]

        return self._read_pamap2(datapath, files = files, label_map = label_map)


    def _read_pamap2_full(self, datapath, n, label_map = None):
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

        return self._read_pamap2(datapath, label_map=label_map)

    def _read_pamap2(self, datapath, files = None, label_map = None, cols = None):
        if files is None:
            files = {
                'train': [
                    'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
                    'subject107.dat', 'subject108.dat', 'subject109.dat'
                ],
                'validation': [
                    'subject105.dat'
                ],
                'test': [
                    'subject106.dat'
                ]
            }
        if label_map is None:
            # not including the activities with few labeled data and "other"
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

        if cols is None:
            # remove the columns we don't need   (Heart rate, temperature, orientation...)
            cols = [
                1,  # Label
                4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # IMU Hand
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # IMU Chest
                38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49  # IMU ankle
            ]

        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        data = {dataset: self._read_pamap2_files(datapath, files[dataset], cols, labelToId)
                for dataset in ('train', 'validation', 'test')}
        return data, idToLabel

    def _read_pamap2_files(self, datapath, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{datapath.rstrip("/")}/Protocol/{filename}', 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    if line[1] not in labelToId.keys():
                        continue

                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) for x in elem[1:]])
                        labels.append(labelToId[elem[0]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    def _read_daphnet_losocv(self, datapath, n):
        """ 
            n : 1<= n <= 10
        """
        subjects = [
                ['S01R01.txt', 'S01R02.txt'],
                ['S02R01.txt', 'S02R02.txt'],
                ['S03R01.txt', 'S03R02.txt', 'S03R03.txt'], # R03 includes label:1 only.
                ['S04R01.txt'], # 1 only
                ['S05R01.txt', 'S05R02.txt'],
                ['S06R01.txt', 'S06R02.txt'], # R02 includes label:1 only.
                ['S07R01.txt', 'S07R02.txt'],
                ['S08R01.txt'],
                ['S09R01.txt'],
                ['S10R01.txt']  # 1 only
        ]

        n -=1
        assert 0 <= n <= 9
        
        train = []
        valid = []
        test = []
        _used = []

        test.extend(subjects[n])
        _used.append(subjects[n])

        if n == 4:
            valid.extend(subjects[6])# 106
            _used.append(subjects[6])
        else:
            valid.extend(subjecits[4]) # 105
            _used.append(subjects[4])

        files['train'] = [ s for s in subjects if s not in _used ]
        files['validation'] = valid
        files['test'] = test

        return self._read_daphnet(datapath, files)


    def _read_daphnet(self, datapath, files=None):
        if files is None:
            files = {
                'train': [
                    'S01R01.txt', 'S01R02.txt',
                    'S03R01.txt', 'S03R02.txt',
                    'S06R01.txt', 'S06R02.txt',
                    'S07R01.txt', 'S07R02.txt',
                    'S08R01.txt', 'S09R01.txt', 'S10R01.txt'
                ],
                'validation': [
                    'S02R02.txt', 'S03R03.txt', 'S05R01.txt'
                ],
                'test': [
                    'S02R01.txt', 'S04R01.txt', 'S05R02.txt'
                ]
            }
        label_map = [
            # (0, 'Other')
            (1, 'No freeze'),
            (2, 'Freeze')
        ]
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]
        cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data = {dataset: self._read_daph_files(datapath, files[dataset], cols, labelToId)
                for dataset in ('train', 'validation', 'test')}
        return data, idToLabel

    def _read_daph_files(self, datapath, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{datapath.rstrip("/")}/dataset/%s' % filename, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for line in reader:
                    elem = []
                    # not including the non related activity
                    if line[10] == "0":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    def _read_opportunity_task_a_without_null(self, datapath):
        files = {
            'train': [
                'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat',
                'S2-ADL2.dat', 'S2-ADL3.dat', 
                'S3-ADL2.dat', 'S3-ADL3.dat',
                'S4-ADL2.dat', 'S4-ADL3.dat'
            ],
            'validation': [
                'S1-ADL1.dat',
                'S2-ADL1.dat',
                'S3-ADL1.dat',
                'S4-ADL1.dat', 
            ],
            'test': [
                'S2-ADL4.dat',
                'S2-ADL5.dat',
                'S3-ADL4.dat',
                'S3-ADL5.dat',
            ]
        }
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            #(0, 'Null'),
            (1, 'Stand'),
            (2, 'Walk'),
            (4, 'Sit'),
            (5, 'Lie'),
        ]
        cols = [
                     2,  3,  4,  5,  6,  7,  8,  9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134,
            244]

        return self._read_opportunity2(datapath, files, label_map, cols)

    def _read_opportunity_task_a(self, datapath):
        files = {
            'train': [
                'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat',
                'S2-ADL2.dat', 'S2-ADL3.dat', 
                'S3-ADL2.dat', 'S3-ADL3.dat',
                'S4-ADL2.dat', 'S4-ADL3.dat'
            ],
            'validation': [
                'S1-ADL1.dat',
                'S2-ADL1.dat',
                'S3-ADL1.dat',
                'S4-ADL1.dat', 
            ],
            'test': [
                'S2-ADL4.dat',
                'S2-ADL5.dat',
                'S3-ADL4.dat',
                'S3-ADL5.dat',
            ]
        }
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            (0, 'Null'),
            (1, 'Stand'),
            (2, 'Walk'),
            (4, 'Sit'),
            (5, 'Lie'),
        ]
        cols = [
                     2,  3,  4,  5,  6,  7,  8,  9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134,
            244]

        return self._read_opportunity2(datapath, files, label_map, cols)

    def _read_opportunity_task_b2_without_null(self, datapath):
        files = {
            'train': [
                'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL2.dat',                                              'S2-Drill.dat',
                'S3-ADL1.dat', 'S3-ADL2.dat',                                              'S3-Drill.dat',
                'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat',                               'S4-Drill.dat'
            ],
            'validation': [
                'S2-ADL3.dat',
                'S3-ADL3.dat',
            ],
            'test': [
                'S2-ADL4.dat',
                'S2-ADL5.dat',
                'S3-ADL4.dat',
                'S3-ADL5.dat',
            ]
        }
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            #(0, 'Null'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        cols = [
            #         2,  3,  4,  5,  6,  7,  8,  9,
            #10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
            #20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            #30, 31, 32, 33, 34, 35, 36, 37,           # there are many NaN in ACCs, expecially 5 and 12
            38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134,
            250]

        return self._read_opportunity2(datapath, files, label_map, cols)

    def _read_opportunity_task_b2(self, datapath):
        files = {
            'train': [
                'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL2.dat',                                              'S2-Drill.dat',
                'S3-ADL1.dat', 'S3-ADL2.dat',                                              'S3-Drill.dat',
                'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat',                               'S4-Drill.dat'
            ],
            'validation': [
                'S2-ADL3.dat',
                'S3-ADL3.dat', 
            ],
            'test': [
                'S2-ADL4.dat',
                'S2-ADL5.dat',
                'S3-ADL4.dat',
                'S3-ADL5.dat',
            ]
        }
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            (0, 'Null'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        cols = [
            #         2,  3,  4,  5,  6,  7,  8,  9,
            #10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
            #20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            #30, 31, 32, 33, 34, 35, 36, 37,
            38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134,
            250]

        return self._read_opportunity2(datapath, files, label_map, cols)

    def _read_opportunity_task_c_without_null(self, datapath):
        files = {
            'train': [
                'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat',                               'S2-Drill.dat',
                'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL2.dat',                               'S3-Drill.dat',
                'S4-ADL1.dat', 'S4-ADL2.dat',                                              'S4-Drill.dat',
            ],
            'validation': [
                'S4-ADL3.dat', 
            ],
            'test': [
                'S4-ADL4.dat',
                'S4-ADL5.dat',
            ]
        }
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            #(0, 'Null'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        cols = [
            38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            250]

        return self._read_opportunity2(datapath, files, label_map, cols)

    def _read_opportunity_task_c(self, datapath):
        files = {
            'train': [
                'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat',                               'S2-Drill.dat',
                'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL2.dat',                               'S3-Drill.dat',
                'S4-ADL1.dat', 'S4-ADL2.dat',                                              'S4-Drill.dat',
            ],
            'validation': [
                'S4-ADL3.dat', 
            ],
            'test': [
                'S4-ADL4.dat',
                'S4-ADL5.dat',
            ]
        }
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            (0, 'Null'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        cols = [
            38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            250]

        return self._read_opportunity2(datapath, files, label_map, cols)

    def _read_opportunity(self, datapath):
        files = {
            'train': [
                               'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                'S2-ADL1.dat',                'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat',
                               'S3-ADL2.dat',                'S3-ADL4.dat', 'S3-ADL5.dat',
                'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat',                               'S4-Drill.dat'
            ],
            'validation': [
                'S1-ADL1.dat',
                'S3-ADL3.dat', 'S3-Drill.dat',
                'S4-ADL4.dat',
            ],
            'test': [
                'S2-ADL2.dat', 'S2-Drill.dat',
                'S3-ADL1.dat',
                'S4-ADL5.dat',
            ]
        }
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        label_map = [
            # (0, 'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]
        cols = [
            38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            103, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
            130, 131, 132, 133, 134,
            250]

        return self._read_opportunity2(detapath, files, label_map, cols)


    def _read_opportunity2(self, datapath, files, label_map, cols):
        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        skip_transient_activity = 0 not in [x[0] for x in label_map]

        cols = [x - 1 for x in cols]  # the last col must be a label

        data = {dataset: self._read_opportunity_files(datapath, files[dataset], cols, labelToId, skip_transient_activity, for_training)
                for dataset, for_training in zip(('train', 'validation', 'test'), (True, True, False))}

        return data, idToLabel

    # this is from https://github.com/nhammerla/deepHAR/tree/master/data and it is an opportunity Challenge reader.
    # It is a python translation for the official one provided by the dataset publishers in Matlab.
    def _read_opportunity_files(self, datapath, filelist, cols, labelToId, skip_transient_activity, for_training):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print(f'Reading file, {filename}: {i+1} of {len(filelist)}')
            with open(f'{datapath.rstrip("/")}/dataset/{filename}', 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                line_count = -1
                for line in reader:
                    line_count += 1
                    elem = []
                    # not including the transient activity if it is not included in the targets
                    if skip_transient_activity and line[cols[-1]] == "0":
                        continue

                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}

    # This data is already windowed and segmented
    def _read_ucihar_losocv(self, datapath, n):
        assert 1 <= n <=30
        subject_list = [i for i in range(1, 31)] 
        n -= 1

        subjects = {}
        subjects['test'] = [subject_list[n]]

        # sub 11, 13, 25 show low accuracy with val 30
        # 10, 14, 16 show lower accuracy
        # 28 x 
        # 16 x
        # 10 x

        subjects['validation'] = [subject_list[14]]
        #if n == 9:
        #    subjects['validation'] = [subject_list[12], subject_list[13], subject_list[29]]
        #elif n == 10:
        #    subjects['validation'] = [subject_list[12], subject_list[9], subject_list[29]]
        #elif n == 29:
        #    subjects['validation'] = [subject_list[10], subject_list[9], subject_list[28]]
        #else:
        #    subjects['validation'] = [subject_list[10], subject_list[9], subject_list[29]]

        subjects['train'] = [ s for s in subject_list if s not in subjects['test'] and s not in subjects['validation'] ]

        return self._read_ucihar(datapath, subjects = subjects)

    # This data is already windowed and segmented
    def _read_ucihar(self, datapath, signals=None, label_map=None, subjects=None):
        if signals is None:
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
        if label_map is None:
            label_map = [
                (1, 'Walking'),
                (2, 'Walking_Upstairs'),
                (3, 'Walking_Downstairs'),
                (4, 'Sitting'),
                (5, 'Standing'),
                (6, 'Laying')
            ]
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

        # labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        print('Loading train')
        x_train = self._load_signals(datapath, 'train', signals)
        y_train = self._load_labels(f'{datapath}/train/y_train.txt')
        print('Loading test')
        x_test = self._load_signals(datapath, 'test', signals)
        y_test = self._load_labels(f'{datapath}/test/y_test.txt')
        print("Loading subjects")
        # Pandas dataframes
        subjects_train = self._load_subjects(f'{datapath}/train/subject_train.txt')
        subjects_test = self._load_subjects(f'{datapath}/test/subject_test.txt')

        _data = np.concatenate((x_train, x_test), 0)
        _labels = np.concatenate((y_train, y_test), 0)
        _labels = _labels - 1
        _subjects = np.concatenate((subjects_train, subjects_test), 0)
        print("Data: ", _data.shape, "Targets: ", _labels.shape, "Subjects: ", _subjects.shape)
        data = {dataset: self.split_uci_data(subjects[dataset], _data, _labels, _subjects)
                for dataset in ('train', 'validation', 'test')}

        return data, idToLabel

    def split_uci_data(self, subjectlist, _data, _labels, _subjects):

        flags = np.in1d(_subjects, subjectlist)
        return {'inputs': _data[flags], 'targets': _labels[flags].astype(int)}


    def _load_signals(self, datapath, subset, signals):
        signals_data = []

        for signal in signals:
            filename = f'{datapath}/{subset}/Inertial Signals/{signal}_{subset}.txt'
            signals_data.append(
                pd.read_csv(filename, delim_whitespace=True, header=None).values
            )

        # Resultant shape is (7352 train/2947 test samples, 128 timesteps, 9 signals)
        return np.transpose(signals_data, (1, 2, 0))

    def _load_labels(self, label_path, delimiter=","):
        with open(label_path, 'rb') as file:
            y_ = np.loadtxt(label_path, delimiter=delimiter)
        return y_

    def _load_subjects(self, subject_path, delimiter=","):
        return np.loadtxt(subject_path, delimiter=delimiter)

    def _read_ispl(self, datapath):
        # create the iSPL dataset from raw data in the dataset folder
        datafiles = glob.glob(f"{datapath}/raw/*sensor*.txt")
        files = {
            'train': [
                datafiles[0],
                datafiles[1],
                datafiles[2]
            ],
            'validation': [
                datafiles[3],
                datafiles[4]
            ],
            'test': [
                datafiles[5]
            ]
        }

        label_map = [
            # (0, 'Idle'),
            (1, 'Walking'),
            (2, 'Standing'),
            (3, 'Sitting'),
            # (4, 'Running')
        ]

        labelToId = {str(x[0]): i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        cols = [
            4, 5, 6,  # Acc x,y,z
            7, 8, 9,  # Gyr x,y,z
            # 10, 11, 12,   # Mag x,y,z
            13, 14, 15,  # lacc x,y,z
            # 16            # Barometer
            0  # ActivityID
        ]

        data = {dataset: self._read_ispl_files(datapath, files[dataset], cols, labelToId)
                for dataset in ('train', 'validation', 'test')}
        return data, idToLabel

    def _read_ispl_files(self, datapath, filelist, cols, labelToId):
        data = []
        labels = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i + 1, len(filelist)))
            with open(f'{filename}', 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for line in reader:
                    elem = []
                    # not including the transient activity
                    if line[0] == "0":
                        continue
                    for ind in cols:
                        elem.append(line[ind])
                    if sum([x == 'NaN' for x in elem]) == 0:
                        data.append([float(x) / 1000 for x in elem[:-1]])
                        labels.append(labelToId[elem[-1]])

        return {'inputs': np.asarray(data), 'targets': np.asarray(labels, dtype=int)}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _dataset = sys.argv[1]
        _datapath = sys.argv[2]
    else:
        _dataset = input('Enter Dataset name e.g. opportunity, daphnet, ucihar, pamap2:')
        _datapath = input('Enter Dataset root folder: ')
    print(f'Reading {_dataset} from {_datapath}')
    dr = DataReader(_dataset, _datapath)
