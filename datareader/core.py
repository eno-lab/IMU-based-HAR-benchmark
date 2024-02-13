import os
import re

import h5py
import numpy as np
import pandas as pd
from .utils import interp_nans, to_categorical


class DataReader:
    def __init__(self, dataset, dataset_origin, win_size, data_cols,
            dataset_path=None, sensor_ids = None):
        self.dataset = dataset
        self.dataset_origin = dataset_origin
        if dataset_path is None:
            self.datapath = os.path.join('dataset', dataset_origin) 
        else:
            self.datapath = dataset_path

        self._sensor_ids = sensor_ids
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
        self._cols = data_cols

        self._separate_target_sensor_ids = None
        self._with_separated_sensor_id = False

        if not dataset.startswith(self.dataset_origin):
            raise ValueError(f"Invalid dataset name: {self.dataset}")

        if self.is_cached():
            self.load_data()
        else:
            self.read_data()
            self.save_data()

        self.init_params_dependig_on_dataest()
        if not self.parse_and_run_data_split_rule_dependig_on_dataest():
            self.parse_and_run_data_split_rule()


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

        self.handling_separate_sensor_settings()

    def handling_separate_sensor_settings(self):
        if self._separate_target_sensor_ids is not None:
            x_l = {'train': [], 'valid':[], 'test': []}
            valid_imu_ids = set(self._sensor_ids)
            invalid_imu_ids = [ix for ix in self._separate_target_sensor_ids if ix not in valid_imu_ids]
            if len(invalid_imu_ids) != 0:
                raise ValueError(f"Invalid sensor id(s): {invalid_imu_ids}")

            shapes = {}
            for mode in ('train', 'valid', 'test'):
                for sensor_id in self._separate_target_sensor_ids:
                    v = eval(f'self._X_{mode}[:, :, np.in1d(self._sensor_ids, sensor_id)]')
                    if mode not in shapes:
                        shapes[mode] = v.shape
                    elif shapes[mode][-1] != v.shape[-1]:  # different sensor axis num
                        raise ValueError(f"number of columns is different on column id: {sensor_id}, type: {mode}")
                    x_l[mode].append(v)

            self._X_train = np.vstack(x_l['train'])
            self._X_test = np.vstack(x_l['test'])
            self._X_valid = np.vstack(x_l['valid'])

            self._y_train = np.concatenate([self._y_train for _ in self._separate_target_sensor_ids], 0)
            self._y_test= np.concatenate([self._y_test for _ in self._separate_target_sensor_ids], 0)
            self._y_valid= np.concatenate([self._y_valid for _ in self._separate_target_sensor_ids], 0)

            if self._with_separated_sensor_id:
                def _tmp(x):
                    _ids = np.repeat(self._separate_target_sensor_ids, x.shape[0]//len(self._separate_target_sensor_ids))[:,np.newaxis, np.newaxis]
                    _ids = np.repeat(_ids, x.shape[1], axis=1)
                    return np.concatenate([x, _ids], -1)

                self._X_train = _tmp(self._X_train)
                self._X_test = _tmp(self._X_test)
                self._X_valid = _tmp(self._X_valid)

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


    def init_params_dependig_on_dataest(self):
        # please implement it if necessary
        return 


    def parse_and_run_data_split_rule_dependig_on_dataest(self):
        """ 
            Please implement it if necessary.
            If do something here (and return True), 
            the parse_and_run_data_split_rule run next will be skipped.
            return: do something (True) or not (False)
        """
        return False


    def parse_and_run_data_split_rule(self):
        dataset = self.dataset
        dataset_origin = self.dataset_origin

        if self.is_ratio():
            self.split_with_ratio()
        elif dataset.startswith(f'{dataset_origin}-losocv_'):
            n = int(dataset[len(f'{dataset_origin}-losocv_'):])
            self.split_losocv(n)
        elif dataset.startswith(f'{dataset_origin}-separate'):
            if self._sensor_ids is None:
                raise NotImplementedError("self._sensor_ids is still None")

            with_sep_ids = False
            if dataset.endswith('_with_sep_ids'):
                with_sep_ids = True
                dataset = dataset[0:-len('_with_sep_ids')]

            sensor_ids = sorted(list(set(self._sensor_ids)))
            if dataset.startswith(f'{dataset_origin}-separate_'):
                sensor_ids = [int(s) for s in dataset[len(f'{dataset_origin}-separate_'):].split("_")]

            self._separate_target_sensor_ids = sensor_ids
            self._with_separated_sensor_id = with_sep_ids

            self.split()
        elif dataset == self.dataset_origin:
            self.split()
        else:
            raise ValueError(f'invalid dataset name: {dataset}')


    def split_losocv(self, n, label_map = None):
        raise NotImplementedError()


    def split(self, tr_val_ts_ids = None, label_map = None):
        """
            tr_val_ts_ids: a dict object like 
                {
                    'train': [i for i in range(1,27)],
                    'validation': [27, 28, 29, 30],
                    'test': [31, 32, 33, 34, 35, 36]
                }
            label_map: a list of label and description like
                [
                    (1, 'Walking'),
                    (2, 'Walking_Upstairs'),
                    (3, 'Walking_Downstairs'),
                    (4, 'Sitting'),
                    (5, 'Standing'),
                    (6, 'Laying')
                ]
        """ 
        raise NotImplementedError()


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
