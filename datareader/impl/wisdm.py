import os
import io

import numpy as np
import pandas as pd
from ..core import DataReader
from ..utils import interp_nans, to_categorical


class Wisdm(DataReader):
    def __init__(self, dataset):

        super().__init__(
                dataset = dataset, 
                dataset_origin = 'wisdm', 
                win_size = 64,  # 20 hz, almost 3 sec
                data_cols = [
                    0,  # subjects
                    1,  # labels
                    # 2, # timestamp
                    3, 4, 5, # acc x,y,z, 10 = 1g
                ],
                dataset_path = os.path.join('dataset', 'WISDM_ar_v1.1'),
                sensor_ids = [0, 0, 0]
                )


    def init_params_dependig_on_dataest(self):
        self._id_to_label = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
        self._label_map = [(i, label) for i, label in enumerate(self._id_to_label)]


    def split_losocv(self, n):
        assert 1 <= n <= 37 

        subjects = {}
        subjects['test'] = [n]

        if n == 34:
            subjects['validation'] = [33, 36]
        elif n == 36:
            subjects['validation'] = [33, 34]
        else:
            subjects['validation'] = [34, 36]

        subjects['train'] = [i for i in range(1, 37) if (
            i not in subjects['test'] and 
            i not in subjects['validation']
            )]

        self.split(subjects)


    def split(self, tr_val_ts_ids = None, label_map = None):
        ##################
        # data summary
        ##################
        #         1      2    3      4    5    6    7    8      9     10  
        #  0  426.0  389.0  429  201.0  407  412  364  567  428.0  431.0  
        #  1  365.0  390.0  364   28.0  212  392  303  341    NaN  400.0  
        #  2    NaN    NaN   52   40.0   54   54   83   88    NaN    NaN  
        #  3    NaN    NaN   93    NaN   49   22   77  107    NaN   54.0  
        #  4   97.0    NaN  108   43.0  108   43  113  142    NaN  137.0  
        #  5   92.0    NaN  104   56.0  105   38   69  107    NaN  121.0  
        #
        #       11   12   13     14     15     16     17   18   19   20  
        #  0 403.0  357  432  459.0  383.0  416.0  320.0  415  584  435  
        #  1 414.0  409  408  439.0  425.0    NaN   95.0  396  536  429  
        #  2   NaN   75   38    NaN    NaN   98.0    NaN   47   83  520  
        #  3   NaN   54   54    NaN    NaN   64.0    NaN   64   70  178  
        #  4 142.0   83  150  269.0   50.0   46.0  184.0   78  137  156  
        #  5  85.0   90  137   93.0   42.0   51.0  121.0   77   81  150  
        #
        #     21     22     23   24     25     26   27     28   29     30
        #  0 414  233.0  218.0  206  231.0  438.0  413  471.0  411  418.0
        #  1 316  206.0  409.0  407  215.0  394.0  398    NaN  423    NaN
        #  2  52    NaN    NaN   22    NaN    NaN   68    NaN   74   50.0
        #  3  94    NaN    NaN   17    NaN    NaN   53   42.0   52  102.0
        #  4 155  177.0  156.0   96    NaN  115.0  102   93.0  153  136.0
        #  5 128  117.0   62.0   92    NaN  122.0  110   97.0  139  124.0
        #
        #     31   32   33   34     35   36
        #  0 559  410  492  443  236.0  205
        #  1 466  405   93  426  415.0  400
        #  2  70  100  107   51   52.0   82
        #  3  86   54   52   43   34.0   63
        #  4 151  123   68  124    NaN  175
        #  5 125   74  144   89    NaN  133

        if tr_val_ts_ids is None:
            tr_val_ts_ids = {
                'train': [i for i in range(1,27)],
                'validation': [27, 28, 29, 30],
                'test': [31, 32, 33, 34, 35, 36]
            }

        self.split_data(tr_val_ts_ids, self._label_map)


    def read_data(self):
        lines = []
        with open(os.path.join(self.datapath, 'WISDM_ar_v1.1_raw.txt'), 'r') as f:
            # clean up the data
            for l in f.readlines():
                l = l[:-1] # remove \n
                if not l:
                    continue 
                if l.endswith(";"):
                    _sp = l.split(',')
                    if len(_sp) == 7: # subject 21, like "21,Walking,117003963904000,0.65,9.51,5.13,;"
                        lines.append(f'{",".join(_sp[0:6])};')
                    else:
                        lines.append(l)
                elif l == "11,Walking,1867172313000,4.4,4.4,":
                    lines.append(f"{l}NaN;")

#                else:
#                    raise ValueError(f'Invalid value {l}')

        lines = ''.join(lines).split(';')
        txt = '\n'.join(lines)
        
        df = pd.read_csv(io.StringIO(txt), sep=",", header=None)

        _subject_ids = df.iloc[:,0].astype(int)
        _labels = df.iloc[:,1]
        df = df.iloc[:,3:].astype(float)
        df.interpolate(inplace=True, limit=4) # 20 hz = 0.2Hz
        df *= 0.1

        self._id_to_label = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
        label_to_id = {label: i for i, label in enumerate(self._id_to_label)}
        _labels = [label_to_id[l] for l in _labels]

        data = []
        seg = []
        subject_ids = []
        labels = []
        label = None
        subject_id = None

        for ix, z in enumerate(zip(_subject_ids, _labels)):
            cur_subject_id, cur_label = z

            if subject_id is not None:
                if subject_id != cur_subject_id: # change subject_id
                    seg = []
                    subject_id = cur_subject_id
                    label = None
                    print(f'start subject id: {subject_id}')
            else:
                subject_id = cur_subject_id
                print(f'start subject id: {subject_id}')

            if label is not None:
                if label != cur_label: # change label
                    seg = []
                    label = cur_label
                    print(f'start label: {label}')
            else:
                label = cur_label
                print(f'start label: {label}')

            seg.append(ix)

            if len(seg) == self._win_size:
                _sdf = df.iloc[seg,:]
                if _sdf.isna().any(axis=None):
                    print(f"Warning: skip a segment include NaN. ({min(seg)}:{max(seg)+1}, label={label})")
                    continue

                # accepted 
                data.append(_sdf)
                labels.append(label)
                subject_ids.append(subject_id)

                seg = seg[int(len(seg)//2):] # stride = win_size/2

        self._data = {}
        self._data['X'] = np.asarray(data)
        self._data['y'] = np.asarray(labels, dtype=int)
        self._data['id'] = np.asarray(subject_ids)

        print(self._data['X'].shape)
        print(self._data['y'].shape)
        print(self._data['id'].shape)
