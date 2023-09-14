import importlib 
import numpy as np

def gen_datareader(dataset):
    """
    dataset: 'daphnet', 
             'daphnet_losocv_i', where 1 <= i <= 10
             'pamap2', 
             'pamap2_full', 
             'pamap2_losocv_i', where 1 <= i <= 8
             'pamap2_full_losocv_i', where 1 <= i <= 9
             'opportunity, 
             'ucihar', 
             'ucihar_losocv_i', where 1 <= i <=  30
             'ispl'
    """
    dataset_origin = dataset.split('_')[0].lower()
    cls_name = dataset_origin[0].upper() + dataset_origin[1:]
    mod = importlib.import_module(f'.impl.{dataset_origin}', "datareader")
    return eval(f'mod.{cls_name}(dataset)')


def interp_nans(y):
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    if nans.any():
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y

