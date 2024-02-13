import os

import numpy as np
import scipy.io
from ..core import DataReader


class Uschad(DataReader):
    def __init__(self, dataset):
        self._win_stride = 100  # 100 hz, 1 sec
        super().__init__(
                dataset = dataset, 
                dataset_origin = 'uschad',
                win_size = 200,  # 100 hz, 2 sec
                data_cols = None,
                sensor_ids = [0 for _ in range(6)])

    def split(self, tr_val_ts_ids = None, label_map = None):
        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': [
                    *range(5, 15)
                ],
                'validation': [
                    *range(3, 5)
                ],
                'test': [
                    *range(1, 2)
                ]
            }

        if label_map is None:
            label_map = [
                (1, 'Walking Forward'),
                (2, 'Walking Left'),
                (3, 'Walking Right'),
                (4, 'Walking Upstairs'),
                (5, 'Walking Downstairs'),
                (6, 'Running Forward'),
                (7, 'Jumping Up'),
                (8, 'Sitting'),
                (9, 'Standing'),
                (10, 'Sleeping'),
                (11, 'Elevator Up'),
                (12, 'Elevator Down')
            ]

        self.split_data(tr_val_ts_ids, label_map)

    def read_data(self):
        x = []      # signal data   (N, L, C)
        y = []      # action number (N)
        id = []     # subject id    (N)
        for subject in range(1, 15):
            for action in range(1, 13):
                for trial in range(1, 6):
                    filename = os.path.join(self.datapath,
                                            f'Subject{subject}',
                                            f'a{action}t{trial}.mat')
                    
                    data_dict = scipy.io.loadmat(filename)
                    full_signals = data_dict['sensor_readings']

                    # Apply a window function to obtain fixed length data.
                    for window_begin in range(0, len(full_signals) - self._win_size, self._win_stride):
                        window_end = window_begin + self._win_size
                        fixed_length_signals = full_signals[window_begin:window_end]

                        # save fixed length signals and tags.
                        x.append(fixed_length_signals)
                        # y.append(data_dict['activity_number'][0])
                        y.append(action)
                        id.append(data_dict['subject'][0])

        self._data = {}
        self._data['X'] = np.array(x)
        self._data['y'] = np.array(y, dtype=int)
        self._data['id'] = np.array(id, dtype=int)
