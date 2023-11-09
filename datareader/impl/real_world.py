import os

import numpy as np
import pandas as pd
import zipfile
from ..core import DataReader


class RealWorld(DataReader):
    def __init__(self, dataset):

        """
            no data (not used):
              proband2 > climbingup > forearm
              proband6 > jumping > thigh 

            zip in zip:
              proband4  > climbingup & climbingdown
              proband7  > climbingup & climbingdown
              proband14 > climbingup & climbingdown

            anomaly file name:
              proband4  > walking 
              proband6  > sitting 
              proband7  > sitting 
              proband8  > standing 
              proband13 > walking
            
                
        """

        self.users = [ f'proband{i}' for i in range(1,16) ]
        self.loc = ("chest", "forearm", "head", "shin", "thigh", "upperarm", "waist")
        self.activities = ('climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking')
        self.stypes = ('acc', 'gyr', 'mag')
        self.stype_fn_map = {'acc': 'acc', 'gyr': 'Gyroscope', 'mag': 'MagneticField'}

        self.label_map = [(i, ac) for i, ac in enumerate(self.activities)]
        self._cols = [ i for i in range(len(self.loc)*3*len(self.stypes) +1) ]

        
        super().__init__(dataset, 'real_world', 128, os.path.join('dataset', 'realworld2016_dataset'))

        if self.is_ratio():
            self.split_with_ratio()
        elif dataset.startswith('real_world-losocv_'): 
            n = int(dataset[len('real_world-losocv_'):])
            self._split_real_world_losocv(n)
        elif dataset == 'real_world':
            self._split_real_world()
        else:
            raise ValueError(f'invalid dataset name: {dataset}')


    def _split_real_world_losocv(self, n, limit=15):
        n -=1
        assert 0 <= n < limit

        subjects = {}
        subjects['test'] = [n]

        if n == 10:
            subjects['validation'] = [9, 12]
        elif n == 11:
            subjects['validation'] = [9, 11]
        else:
            subjects['validation'] = [10, 11]

        subjects['train'] = [i for i in range(len(self.users)) if (
            i not in subjects['test'] and 
            i not in subjects['validation']
            )]

        self._split_real_world(subjects)


    def _split_real_world(self, subjects = None):
        if subjects is None:
            subjects = {
                'train': [i for i in range(11)],
                'validation': [10, 11],
                'test': [12, 13]
            }

        self.split_data(subjects, self.label_map)


    def read_data(self):
        def read_func(user):
            total_data = []

            def cleanup_and_append_act_data(act_data, act_id):
                act_data.sort_index(inplace=True)

                pd_ix = (~(act_data.isna()).to_numpy().T*np.arange(1,act_data.shape[0]+1)).T

                max_index = pd_ix.max(axis=0).min()
                pd_ix[pd_ix==0] = pd_ix.shape[0]
                min_index = pd_ix.min(axis=0)
                min_index = min_index[min_index < pd_ix.shape[0]].max() # remove the colums filled w/ nan
                min_index = act_data.index[min_index]
                max_index = act_data.index[max_index]
                if min_index % 20 != 0:
                    min_index = (min_index // 20 + 1) * 20
                if max_index % 20 != 0:
                    max_index = (max_index // 20) * 20

                seq_ix = np.arange(min_index, max_index+1, 20) # where resampled at 
                lacked_ix = seq_ix[~np.in1d(seq_ix, act_data.index.to_numpy())] # extract not included ones.
                act_data = pd.concat((act_data, pd.DataFrame(index=lacked_ix)), axis=1)
                act_data.sort_index(inplace=True) # not necessary. just in a case.
                act_data.interpolate(inplace=True)
                act_data = act_data.loc[seq_ix,:] # resampling
                act_data = act_data.reset_index() # the removed index will be placed on the first column
                act_data.iloc[:,0] = act_id # overwrite the first column with the activity id
                total_data.append(act_data)


            for act_id, act in enumerate(self.activities):
                if user == 'proband2' and act == 'climbingup':
                    continue
                if user == 'proband6' and act == 'jumping':
                    continue

                f_infix = act
                if ((user == 'proband4' and act == 'walking') or
                    (user == 'proband6' and act == 'sitting') or
                    (user == 'proband7' and act == 'sitting') or
                    (user == 'proband8' and act == 'standing') or
                    (user == 'proband13' and act == 'walking')):
                    f_infix = f'{act}_2'

                if act.startswith('climbing') and user in ['proband4', 'proband7', 'proband14']:
                    for f_ix in range(1,4):
                        act_data = None
                        for loc in self.loc:
                            for st in self.stypes:
                                pz = zipfile.ZipFile(os.path.join(self.datapath, user, 'data', f"{st}_{act}_csv.zip"))
                                print(pz)

                                if f_ix == 1:
                                    if st == 'acc':
                                        zf_infix = f"{act}_{f_ix}"
                                    else:
                                        zf_infix = f"{act}"
                                    f_infix = f"{act}"
                                else:
                                    zf_infix = f"{act}_{f_ix}"
                                    f_infix = f"{act}_{f_ix}"


                                with pz.open(f'{st}_{zf_infix}_csv.zip') as pzf:
                                    z = zipfile.ZipFile(pzf)
                                    print(z)
                                    with z.open(f'{self.stype_fn_map[st]}_{f_infix}_{loc}.csv') as f:
                                        p = pd.read_csv(f, index_col=0)

                                    p = pd.DataFrame(p.to_numpy()).set_index(0)
                                    if act_data is None:
                                        act_data = p
                                    else:
                                        act_data = pd.concat((act_data, p), axis=1)
                        cleanup_and_append_act_data(act_data, act_id)
                else:
                    act_data = None
                    for loc in self.loc:
                        for st in self.stypes:
                            z = zipfile.ZipFile(os.path.join(self.datapath, user, 'data', f"{st}_{act}_csv.zip"))
                            print(z)
                            with z.open(f'{self.stype_fn_map[st]}_{f_infix}_{loc}.csv') as f:
                                p = pd.read_csv(f, index_col=0)

                            p = pd.DataFrame(p.to_numpy()).set_index(0)
                            if act_data is None:
                                act_data = p
                            else:
                                act_data = pd.concat((act_data, p), axis=1)
                    cleanup_and_append_act_data(act_data, act_id)

            total_data = pd.concat(total_data, ignore_index=True)

            return total_data

        self._read_data(enumerate(self.users),
                       read_func,
                       label_col = 0,
                       interpolate_limit = 10 # 10/50 Hz = 0.2 Hz
                       )
