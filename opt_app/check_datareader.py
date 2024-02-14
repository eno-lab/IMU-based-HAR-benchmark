from datareader import gen_datareader

import numpy as np
import pandas as pd

#dataset = 'pamap2'
#dataset = 'opportunity'
#dataset = 'daphnet'
#dataset = 'ucihar'
#dataset = 'real_world'
#dataset = 'opportunity_real'
#dataset = 'mighar'
#dataset = 'mighar-combination_90_70_101_244_with_sid'
#dataset = 'uschad'
dataset = 'wisdm'

dr = gen_datareader(dataset)
X_train, y_train, X_val, y_val, X_test, y_test, labels, n_classes = dr.gen_ispl_style_set()


print("Data Distribution: \n\n")
print(f"Train:  X -> {X_train.shape} Class count -> {list(np.bincount(y_train.argmax(1)))} \n\n"
             f"{pd.DataFrame(y_train.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
print(f"Validation:  X -> {X_val.shape} Class count -> {list(np.bincount(y_val.argmax(1)))} \n\n"
             f"{pd.DataFrame(y_val.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
print(f"Test:  X -> {X_test.shape} Class count -> {list(np.bincount(y_test.argmax(1)))} \n\n"
             f"{pd.DataFrame(y_test.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
