import os

import numpy as np
import pandas as pd
from ..core import DataReader


class Motionsense(DataReader):
    def __init__(self, dataset):
        self._win_stride = 64  # 50 hz, 1.28 sec
        super().__init__(
                dataset = dataset, 
                dataset_origin = 'motionsense',
                win_size = 128,  # 50 hz, 2.56 sec
                data_cols = None,
                sensor_ids = [0 for _ in range(12)])

    def split(self, tr_val_ts_ids = None, label_map = None):
        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': [
                    *range(7, 25)
                ],
                'validation': [
                    *range(4, 7)
                ],
                'test': [
                    *range(1, 4)
                ]
            }
            "dws","ups", "wlk", "jog", "std", "sit"
        if label_map is None:
            label_map = [
                (1, 'Stairs Down'), # dws
                (2, 'Stairs Up'),   # ups
                (3, 'Walking'),     # wlk
                (4, 'Jogging'),     # jog
                (5, 'Standing'),    # std
                (6, 'Stting'),      # sit
            ]

        self.split_data(tr_val_ts_ids, label_map)

    def read_data(self):
        x = []      # signal data   (N, L, C)
        y = []      # activity number (N)
        subject_id = []     # subject id    (N)

        activity_folders = [
            'dws_1',
            'dws_2',
            'dws_11',
            'jog_9',
            'jog_16',
            'sit_5',
            'sit_13',
            'std_6',
            'std_14',
            'ups_3',
            'ups_4',
            'ups_12',
            'wlk_7',
            'wlk_8',
            'wlk_15',
        ]
        activity_name_to_id = {
            'dws' : 1, 
            'ups' : 2, 
            'wlk' : 3, 
            'jog' : 4, 
            'std' : 5, 
            'sit' : 6
        }

        signals_keys = [
            'attitude.roll',
            'attitude.pitch',
            'attitude.yaw',
            'gravity.x',
            'gravity.y',
            'gravity.z',
            'rotationRate.x',
            'rotationRate.y',
            'rotationRate.z',
            'userAcceleration.x',
            'userAcceleration.y',
            'userAcceleration.z'
        ]


        for activity_folder in activity_folders:
            activity_name = activity_folder.split('_')[0]
            activity_id = activity_name_to_id[activity_name]

            for subject in range(1, 25):
                filename = os.path.join(self.datapath,
                                        'A_DeviceMotion_data',
                                        activity_folder,
                                        f'sub_{subject}.csv')
                
                df = pd.read_csv(filename)
                full_signals = df[signals_keys]

                # Apply a window function to obtain fixed length data.
                for window_begin in range(0, len(full_signals), self._win_stride):
                    window_end = window_begin + self._win_size
                    if(window_end > len(full_signals)):
                        break
                    fixed_length_signals = full_signals[window_begin:window_end]

                    if fixed_length_signals.isna().any(axis=None):
                        print(f"Warning: skip a segment include NaN. ({window_begin}:{window_end}, activity={activity_name}, subject={subject})")
                        continue

                    # save fixed length signals and tags.
                    x.append(fixed_length_signals)
                    y.append(activity_id)
                    subject_id.append(subject)

        self._data = {}
        self._data['X'] = np.array(x)
        self._data['y'] = np.array(y, dtype=int)
        self._data['id'] = np.array(subject_id, dtype=int)
