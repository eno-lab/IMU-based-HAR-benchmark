import numpy as np
import pandas as pd
import io
import os

import argparse  
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument('--losocv_file_prefix', required=True)
parser.add_argument('--best_selection_metrics', default='mf1')

args = parser.parse_args()

base_dir = os.path.dirname(args.losocv_file_prefix)
file_prefix = os.path.basename(args.losocv_file_prefix)
score_list = []

print(base_dir)

for file in os.listdir(base_dir):
    if not file.startswith(file_prefix):
        continue

    print(file)

    lines = []
    best_score = None
    with open(os.path.join(base_dir, file)) as f:
        for l in f.readlines():
            if l.endswith('\n'):
                l = l[:-1]  # remove \n 

            if not ((l.startswith('[[') or l.startswith(' [')) 
                and (l.endswith(']]') or l.endswith(']'))):
                continue

            end_flag = l.endswith(']]')

            l = l.replace('[', '')
            l = l.replace(']', '')

            for _ in range(5):
                l = l.replace('  ', ' ')

            lines.append(l if l[0] != ' ' else l[1:])

            if not end_flag:
                continue

            stio = io.StringIO('\n'.join(lines))
            cm = pd.read_csv(stio, sep=' ', header=None)

            n_classes = len(cm)
            y_true = np.repeat(np.arange(n_classes), cm.sum(axis=1))
            y_pred = []

            for i in range(n_classes):
                y_pred.extend(np.repeat(np.arange(n_classes), cm.iloc[i,:]))

            y_pred = np.array(y_pred)

            cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

            clr = classification_report(y_true, y_pred,
                                        labels=np.arange(n_classes),
                                        output_dict=True, 
                                        zero_division=0.0)

            score = {'mf1': clr['macro avg']['f1-score'], 
                    'wf1': clr['weighted avg']['f1-score'],
                    'acc': clr['accuracy'],
                    #'loss': scores[0],
                    'cm': cm}


            if best_score is None or best_score[args.best_selection_metrics] < score[args.best_selection_metrics]:
                best_score = score
            lines = []
    score_list.append(best_score)

cm = None
for s in score_list:
    cm = s['cm'] if cm is None else cm + s['cm']

print(cm)
n_classes = len(cm)
y_true = np.repeat(np.arange(n_classes), cm.sum(axis=1))
y_pred = []

for i in range(n_classes):
    y_pred.extend(np.repeat(np.arange(n_classes), cm[i,:]))

y_pred = np.array(y_pred)

clr = classification_report(y_true, y_pred,
                            labels=np.arange(n_classes),
                            digits=4, 
                            zero_division=0.0)

print(clr)
