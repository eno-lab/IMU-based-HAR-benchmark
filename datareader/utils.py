import importlib 
import numpy as np
import re


def gen_datareader(dataset):
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


def parse_suffix_options(dataset, dataset_origin):
    options = dataset.split('-')

    flags = {}
    remaining = []

    def check_duplicated(opt):
        if f'is_{opt}' in flags and flags[f'is_{opt}']:
            raise ValueError('duplicated {opt} suffix: {dataset}')
        flags[f'is_{opt}'] = True

    for ix, option in enumerate(options):
        if option == dataset_origin:
            remaining.append(ix)
            continue 

        ratio_re_match = re.match('^ratio_(\d+)_(\d+)_(\d+)$', option)
        if ratio_re_match is not None:
            check_duplicated('ratio')
            flags['train_ratio'] = float(ratio_re_match.groups()[0])
            flags['valid_ratio'] = float(ratio_re_match.groups()[1])
            flags['test_ratio']  = float(ratio_re_match.groups()[2])
        elif option.startswith(f'losocv_'):
            check_duplicated('losocv')
            flags['losocv_n'] = int(option[len('losocv_'):])
        elif option.startswith('separation'):
            check_duplicated('separation')
            with_sid = False
            if option.endswith('_with_sid'):
                with_sid = True
                option = option[0:-len('_with_sid')]

            if option.startswith(f'separation_'):
                sensor_ids = [int(s) for s in option[len(f'separation_'):].split("_")]

            flags['sep_sids'] = sensor_ids
            flags['sep_with_sid'] = with_sid 

        elif option.startswith('combination'):
            check_duplicated('combination')
            with_sid = False
            if option.endswith('_with_sid'):
                with_sid = True
                option = option[0:-len('_with_sid')]

            if option.startswith(f'combination_'):
                sensor_ids = [int(s) for s in option[len(f'combination_'):].split("_")]

            flags['cmb_sids'] = sensor_ids
            flags['cmb_with_sid'] = with_sid
        else:
            remaining.append(ix)

    return '-'.join([options[ix] for ix in remaining]), flags
