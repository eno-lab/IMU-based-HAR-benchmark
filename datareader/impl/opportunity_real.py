import os

import numpy as np
import pandas as pd
from ..core import DataReader


class OpportunityReal(DataReader):
    def __init__(self, dataset):
        # names are from label_legend.txt of Opportunity dataset
        # except 0-ie Other, which is an additional label
        self._label_map = {
            #(0, 'Null'),
            '406516', 'Open Door 1',
            '406517', 'Open Door 2',
            '404516', 'Close Door 1',
            '404517', 'Close Door 2',
            '406520', 'Open Fridge',
            '404520', 'Close Fridge',
            '406505', 'Open Dishwasher',
            '404505', 'Close Dishwasher',
            '406519', 'Open Drawer 1',
            '404519', 'Close Drawer 1',
            '406511', 'Open Drawer 2',
            '404511', 'Close Drawer 2',
            '406508', 'Open Drawer 3',
            '404508', 'Close Drawer 3',
            '408512', 'Clean Table',
            '407521', 'Drink from Cup',
            '405506', 'Toggle Switch'
        }

        self._filelist = [
            'S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
            'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat',
            'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat',
            'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat' 
            ]
        self._cols = [
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

        self._cols = [x-1 for x in self._cols]

        super().__init__(dataset, 'opportunity_real', 32, os.path.join('dataset', 'opportunity')) # 30Hz, almost 1 sec, cut off 2% of samples 

        if self.is_ratio():
            self.split_with_ratio()
        elif dataset == 'opportunity_real-task_b2':
            self._split_opportunity_task_b2()
        elif dataset == 'opportunity_real-task_b2_no_null':
            self._split_opportunity_task_b2_no_null()
        elif dataset == 'opportunity_real-task_c':
            self._split_opportunity_task_c()
        elif dataset == 'opportunity_real-task_c_no_null':
            self._split_opportunity_task_c_no_null()
        elif dataset == 'opportunity_real':
            self._split_opportunity()
        else:
            raise ValueError(f'invalid dataset name: {dataset}')


    def _split_opportunity_task_b2_no_null(self):
        label_map = [
            #(0, 'Other'),
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

        self._split_opportunity(files, label_map)


    def _split_opportunity_task_b2(self, label_map = None):
        files = {
            'train': [    1 , 2,  3,  4,  5,
                       6,      8,         11,
                      12, 13,             17,
                      18, 19, 20,         23],
            'validation': [0, 7, 14],
            'test':  [9, 10, 15, 16]
        }

        self._split_opportunity(files)


    def _split_opportunity_task_c_no_null(self):
        label_map = [
            #(0, 'Other'),
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

        self._split_opportunity_task_c(label_map)


    def _split_opportunity_task_c(self, label_map = None):
        files = {
            'train': [ 0,  1,  2,  3,  4, 
                       6,      8,         11,
                      12, 13,             17,
                      18, 19,             23],
            'validation': [5, 7, 14, 20],
            'test':  [21, 22]
        }
        cols = [
            38, 39,
            40, 41, 42, 43, 44, 45, 46,
            51, 52, 53, 54, 55, 56, 57, 58, 59,
            64, 65, 66, 67, 68, 69,
            70, 71, 72, 77, 78, 79,
            80, 81, 82, 83, 84, 85,
            90, 91, 92, 93, 94, 95, 96, 97, 98,
            250]
        cols = [x-1 for x in cols[:-1]] # remove label
        _cols = self._cols[:-1] # remove label

        col_filter =  np.in1d(_cols, cols)

        self._split_opportunity(files, label_map = label_map, x_col_filter = col_filter)


    def _split_opportunity(self, files = None, label_map = None, x_col_filter=None):
        if files is None:
            files = {
                'train': [     1 , 2,  3,  4,  5, 
                           6,      8,  9, 10,  
                              13,     15, 16,    
                          18, 19, 20,         23],
                'validation': [0, 14, 17, 21],
                'test':  [7, 11, 12, 22]
            }

        if label_map is None:
            label_map = [
                (0, 'Other'),
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


        self.split_data(files, label_map, x_col_filter)


    def read_data(self):
        data = []
        seg = []
        file_ids = []
        labels = []
        for i, filename in enumerate(self._filelist):
            print(f'Reading file, {filename}: {i+1} of {len(self._filelist)}')

            df = pd.read_csv(os.path.join(self.datapath, 'dataset', filename), sep=" ", header=None)
            
            df = df.iloc[:,self._cols]
            label_df = df.iloc[:, -1].astype(int)

            df = df.iloc[:, :-1].astype(float)
            df.interpolate(inplace=True, limit=6) # 6/30 = 0.2 Hz
            df /= 1000

            for low in range(0, len(label_df), int(self._win_size//2)):
                high = low + self._win_size
                if high > df.shape[0]:
                    break

                seg_labels = label_df[low:high]
                #seg_labels = seg_labels[seg_labels != 0]
                #if len(seg_labels) == 0:
                #    continue

                #if len(seg_labels) < self._win_sizewin_size//2:
                #    #print(f"Warning: skip a segment whose overhalf of labels are 0. ({low}:{high}, label={label})")
                #    continue

                label = seg_labels.mode().iloc[0]
                #print(f"{len(seg_labels)=}:{label}")
                seg = df.iloc[low:high,:]
                if seg.isna().any(axis=None):
                    print(f"Warning: skip a segment include NaN. ({low}:{high}, label={label})")
                    continue

                data.append(seg)
                labels.append(label)
                file_ids.append(i)
                
        self._data = {}
        self._data['X'] = np.asarray(data)
        self._data['y'] = np.asarray(labels, dtype=int)
        self._data['id'] = np.asarray(file_ids)
