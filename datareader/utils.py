import importlib 
import numpy as np


def gen_datareader(dataset):
    """
    dataset: 'daphnet', 
             'daphnet-losocv_i', where 1 <= i <= 10
             'wisdm', 
             'wisdm-losocv_i', where 1 <= i <= 36
             'pamap2',       # exclude 24: rope jumping 
             'pamap2-full',  # include 24: rope jumping
             'pamap2-losocv_i', where 1 <= i <= 8
             'pamap2-full-losocv_i', where 1 <= i <= 9
             'opportunity, 
             'opportunity-real, # include Null and split ignoring label boundary 
             'ucihar', 
             'ucihar-losocv_i', where 1 <= i <=  30
             'ucihar-orig', 
             # 'ispl' # not implemented 
    """
    dataset_origin = dataset.split('-')[0].lower()
    parts = []

    for p in dataset_origin.split('_'):
        if len(p) == 0:
            continue
        elif len(p) == 1:
            parts.append(p.upper())
        else: 
            parts.append(p[0].upper() + p[1:])
    
    cls_name = ''.join(parts)
    mod = importlib.import_module(f'.impl.{dataset_origin}', "datareader")
    return eval(f'mod.{cls_name}(dataset)')


def interp_nans(y):
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    if nans.any():
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def to_categorical(y, class_num=None):
    if class_num is None:
        class_num = max(y)+1
    return np.eye(class_num, dtype=np.float32)[y]
