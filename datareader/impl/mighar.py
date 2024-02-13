import csv
import os

import numpy as np
from ..core import DataReader
import pandas as pd
import h5py


class Mighar(DataReader):

    def __init__(self, dataset: str):

        self._label_map = (
            # (0, 'Other')
            (1, 'walking'),
            (2, 'jogging'),
            (3, 'goingupstairs'),
            (4, 'goingdownstairs'),
            (5, 'standing'),
            (6, 'sitting'),
            (7, 'typing'),
            (8, 'writing'),
            (9, 'eatingcookie'),
            (10, 'eatingpasta'),
            (11, 'ironing'),
            (12, 'folding'),
            (13, 'toothbrushing'))

        self._subjects = [f"subject{i+1}" for i in range(12)]

        self.sensor_num = 396
        self._cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self._cols = [x - 1 for x in self._cols]

        dataset_origin = 'mighar'
        dataset_path = None
        win_size = 256

        self.dataset = dataset
        self.dataset_origin = dataset_origin
        if dataset_path is None:
            self.datapath = os.path.join('dataset', dataset_origin)
        else:
            self.datapath = dataset_path

        self._id_to_label = None
        self._X_train = None
        self._y_train = None
        self._X_valid = None
        self._y_valid = None
        self._X_test = None
        self._y_test = None
        self._train = None
        self._validation = None
        self._test = None
        self._win_size = win_size
        self._data = {'X': None, 'y': None, 'id': None, 'sid': None}

        if not dataset.startswith(self.dataset_origin):
            raise ValueError(f"Invalid dataset name: {self.dataset}")

        self.select_sensors()

        if not self.is_cached():
            self.read_data()
        self.load_data()

        if self.is_ratio():
            # self.split_with_ratio()
            raise NotImplementedError(
                "Split with ratio is not implemented in Mighar")
        elif len(dataset.split("split_")) != 1:
            subjects = {}
            split_info = dataset.split("-split_")[1].split("-")[0].split("_")
            if len(split_info) != 2:
                raise ValueError(f"Invalid dataset options: {self.dataset}")
            for i in [0, 1]:
                use_subjects = []
                subject_ids = split_info[i].split(",")
                for subject_id in subject_ids:
                    subject_num = subject_id.split(":")
                    if len(subject_num) == 1:
                        use_subjects.append(int(subject_num[0]) - 1)
                    else:
                        use_subjects += [
                            i for i in range(
                                int(subject_num[0]) - 1, int(subject_num[1]))
                        ]
                if i == 0:
                    subjects["validation"] = use_subjects
                else:
                    subjects["test"] = use_subjects
            subjects["train"] = [
                i for i in range(12) if i not in subjects["test"]
                and i not in subjects["validation"]
            ]
            self._split(subjects=subjects)
        else:
            self._split()

    def select_sensors(self):
        self.use_sensors = []
        if len(self.dataset.split("-sensors_")) == 1:
            self.use_sensors = [i for i in range(396)]
            self.concat = False
        else:
            self.sensor_info = self.dataset.split("-sensors_")[1].split(
                "-")[0].split("_")
            sensor_ranges = self.sensor_info[0].split(",")
            for sensor_range in sensor_ranges:
                sensor_num = sensor_range.split(":")
                if len(sensor_num) == 1:
                    self.use_sensors.append(int(sensor_num[0]))
                else:
                    self.use_sensors += [
                        i for i in range(int(sensor_num[0]),
                                         int(sensor_num[1]) + 1)
                    ]
            if len(self.sensor_info) == 1:
                self.concat = False
            else:
                self.concat = True

    def _split_losocv(self, n, label_map=None):
        pass

    def _split(self, subjects=None):
        if subjects is None:
            subjects = {
                'train': [0, 3, 8, 9, 10, 11, 12],
                'validation': [2, 4, 6],
                'test': [1, 5, 7]
            }

        self.split_data(subjects, self._label_map)

    def read_data(self):
        filenames = []
        dfs = {}
        for subject_id, subject in enumerate(self._subjects):
            filenames.append((subject_id, subject))
            df = pd.DataFrame()
            for trial in [1, 2]:
                for label_id, label in self._label_map:
                    raw_data = pd.read_csv(
                        os.path.join(self.datapath, "data", subject, label,
                                     f"trial{trial}.csv.gz"))
                    print(
                        f"reading {self.datapath}/data/{subject}/{label}/trial{trial}.csv.gz"
                    )
                    for sensor_id in range(396):
                        sensor_data = raw_data.iloc[:,
                                                    list(
                                                        range((sensor_id * 9 +
                                                               1),
                                                              (sensor_id * 9 +
                                                               10)))].copy()

                        # rename colmuns
                        sensor_data = sensor_data.set_axis([
                            "accx", "accy", "accz", "gyrox", "gyroy", "gyroz",
                            "magx", "magy", "magz"
                        ],
                                                           axis=1)

                        # add label column
                        sensor_data.loc[:, 'sid'] = sensor_id
                        sensor_data.loc[:, 'label'] = label_id
                        df = pd.concat([df, sensor_data])

            dfs[subject] = df

        loop_elements = filenames

        def read_file_func(filename):
            return dfs[filename]

        exist_sensor_col = True
        label_col = -1
        file_sep = " "
        x_magnif = 1
        interpolate_limit = 10
        null_label = 0

        all_data = []
        seg = []
        subject_ids = []
        sensor_ids = []
        labels = []
        label = None

        for i, filename in loop_elements:
            df: pd.DataFrame = read_file_func(filename)
            df = df.iloc[:, self._cols]
            label_df = df.iloc[:, label_col].astype(int)

            if exist_sensor_col:
                if label_col == -1:
                    sensor_df = df.iloc[:, -2].astype(int)
                    df = df.iloc[:, :-2].astype(float)
                elif label_col == 0:
                    sensor_df = df.iloc[:, 1].astype(int)
                    df = df.iloc[:, 2::].astype(float)
            else:
                sensor_df = pd.Series(np.zeros(df.shape[0])).astype(int)
                if label_col == -1:
                    df = df.iloc[:, :-1].astype(float)
                elif label_col == 0:
                    df = df.iloc[:, 1::].astype(float)

            # 13/64 hz = 0.2Hz
            df.interpolate(inplace=True, limit=interpolate_limit)
            if x_magnif != 1:
                df *= x_magnif

            sensor = sensor_df.iat[0]
            for ix, cur_label in enumerate(label_df):
                cur_sensor = sensor_df.iat[ix]
                if cur_label == null_label:
                    label = None
                    seg = []
                    continue

                if label is not None:
                    if label != cur_label or sensor != cur_sensor:  # change label
                        seg = []
                        label = cur_label
                else:
                    label = cur_label

                sensor = sensor_df.iat[ix]
                seg.append(ix)

                if len(seg) == self._win_size:
                    _sdf = df.iloc[seg, :]
                    if _sdf.isna().any(axis=None):
                        print(
                            f"Warning: skip a segment include NaN. ({min(seg)}:{max(seg)+1}, label={label})"
                        )
                        continue

                    # accepted
                    all_data.append(_sdf)
                    labels.append(label)
                    subject_ids.append(i)
                    sensor_ids.append(sensor)

                    seg = seg[int(len(seg) // 2):]  # stride = win_size/2

        self._sensor_data = {}
        os.makedirs(os.path.join(self.datapath, "cache"), exist_ok=True)
        for sensor_id in range(self.sensor_num):
            index = [
                i for i in range(len(all_data)) if sensor_ids[i] == sensor_id
            ]
            with h5py.File(os.path.join(
                    self.datapath, "cache",
                    f'{self.dataset_origin}_{sensor_id}.h5'),
                           mode='w') as f:
                f.create_dataset('X',
                                 data=np.asarray([all_data[i] for i in index]))

                f.create_dataset('y',
                                 data=np.asarray([labels[i] for i in index]))
                f.create_dataset('id',
                                 data=np.asarray(
                                     [subject_ids[i] for i in index]))

    def save_data(self, sensor_id):
        with h5py.File(os.path.join(self.datapath,
                                    f'{self.dataset_origin}_{sensor_id}.h5'),
                       mode='w') as f:
            f.create_dataset('X', data=self._data['X'])
            f.create_dataset('y', data=self._data['y'])
            f.create_dataset('id', data=self._data['id'])

    def is_cached(self):
        return os.path.exists(
            os.path.join(self.datapath, "cache",
                         f'{self.dataset_origin}_0.h5'))

    def load_data(self):
        sensor_data = {}
        sensor_data['X'] = []
        sensor_data['y'] = []
        sensor_data['id'] = []
        for sensor_id in self.use_sensors:
            with h5py.File(os.path.join(
                    self.datapath, "cache",
                    f'{self.dataset_origin}_{sensor_id}.h5'),
                           mode='r') as f:
                sensor_data['X'].append(np.array(f['X']))
                sensor_data['y'].append(np.array(f['y']))
                sensor_data['id'].append(np.array(f['id']))
        if self.concat is True:
            self._data['X'] = np.concatenate(sensor_data['X'], axis=2)
            self._data['y'] = sensor_data['y'][0]
            self._data['id'] = sensor_data['id'][0]
        else:
            self._data['X'] = np.concatenate(sensor_data['X'])
            self._data['y'] = np.concatenate(sensor_data['y'])
            self._data['id'] = np.concatenate(sensor_data['id'])
