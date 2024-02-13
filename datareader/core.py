import os
import re

import h5py
import numpy as np
import pandas as pd
from .utils import interp_nans, to_categorical


class DataReader:
    def __init__(self, dataset, dataset_origin, win_size, dataset_path=None):
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
        ix_train, ix_test = train_test_split(ix, test_size = test_ratio, random_stateint=42)
        ix_train, ix_valid= train_test_split(ix_train, test_size = valid_ratio/(train_ratio+valid_ratio), random_stateint=42)

        self._X_train = self._data['X'][ix_train]
        self._X_valid = self.data['X'][ix_valid]
        self._X_test = self.data['X'][ix_test]
        self._y_train = self.data['y'][ix_train]
        self._y_valid = self.data['y'][ix_valid]
        self._y_test = self.data['y'][ix_test]


    def split_data(self, ids, label_map, x_col_filter=None):
        label_to_id = {x[0]: i for i, x in enumerate(label_map)}
        self._id_to_label = [x[1] for x in label_map]

        _filter = np.in1d(self._data['y'], list(label_to_id.keys()))
        _x = self._data['X'][_filter]
        _id = self._data['id'][_filter]
        _y = [label_to_id[y] for y in self._data['y'][_filter]]
        _y = to_categorical(np.asarray(_y, dtype=int), self.n_classes)

        if x_col_filter is not None:
            _x = _x[:,:,x_col_filter]

        _f_train = np.in1d(_id, ids['train'])
        _f_valid = np.in1d(_id, ids['validation'])
        _f_test = np.in1d(_id, ids['test'])

        self._X_train = _x[_f_train]
        self._y_train = _y[_f_train]
        self._X_valid = _x[_f_valid]
        self._y_valid = _y[_f_valid]
        self._X_test = _x[_f_test]
        self._y_test = _y[_f_test]

    def _read_data(self,
                   loop_elements,
                   read_file_func,
                   label_col=-1,
                   file_sep=" ",
                   x_magnif=1,
                   interpolate_limit=10,
                   null_label=0,
                   exist_sensor_col=False):
        """

        Args:
            loop_elements: used as 'for id, filename in loop_elements'
            read_file_func:
                args: filename
                return: pandas.DataFrame
            label_col: -1 or 0, default -1
            exist_sensor_col: True or False, default False
                If True, sensor_col is -2 when label_col is -1 or 1 when label_col is 0.
        """
        data = []
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

            df.interpolate(inplace=True,
                           limit=interpolate_limit)  # 13/64 hz = 0.2Hz
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
                    _sdf = df.iloc[seg,:]
                    if _sdf.isna().any(axis=None):
                        print(f"Warning: skip a segment include NaN. ({min(seg)}:{max(seg)+1}, label={label})")
                        continue

                    # accepted 
                    data.append(_sdf)
                    labels.append(label)
                    subject_ids.append(i)
                    sensor_ids.append(sensor)

                    seg = seg[int(len(seg)//2):] # stride = win_size/2

        self._data = {}
        self._data['X'] = np.asarray(data)
        self._data['y'] = np.asarray(labels, dtype=int)
        self._data['id'] = np.asarray(subject_ids)

    @property
    def out_loss(self):
        return 'categorical_crossentropy'

    @property
    def out_activ(self):
        return 'softmax'

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
