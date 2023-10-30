#
# convert cm to table 
#
# Example:
# 
# [[10 2]
#  [6  4]]  and labels 'No freeze' and 'freeze' 
#
# to 
#
# \toprule
#  & A & B \\  
# \midrule
# A: No freeze & 10 & 2 \\  
# B: freeze    &  6 & 4 \\  
# \bottomrule
#




import numpy as np
import pandas as pd
import io
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import argparse  

parser = argparse.ArgumentParser()
parser.add_argument('--cm_text_file', required=True)
# [[10 2]
#  [6  4]]
parser.add_argument('--label_file', required=True)
#
# label1
# label2
# label3
# ...
#
args = parser.parse_args()

KEY='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

lines = []
labels = []

with open(args.label_file) as f:
    for l in f.readlines():
        if l.endswith('\n'):
            l = l[:-1]
        l.strip()
        if l : # empty check
            labels.append(l)

label_ix = 0
with open(args.cm_text_file) as f:
    for l in f.readlines():
        if l.endswith('\n'):
            l = l[:-1]
        l = l.replace('[', '')
        l = l.replace(']', '')
        
        for _ in range(5):
            l = l.replace('  ', ' ')
        l = l.strip()

        if l : # empty check
            l = l.replace(' ', ' & ')
            l = f'{KEY[label_ix]}: {labels[label_ix]} & {l} \\\\'
            label_ix += 1
            lines.append(l)

header = f" & {' & '.join(KEY[0:len(lines)])}\\\\"
lines = ["\\toprule", header,"\\midrule",] + lines + ["\\bottomrule",]
print('\n'.join(lines))
