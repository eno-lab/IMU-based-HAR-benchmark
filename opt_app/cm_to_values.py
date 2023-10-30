#
# recalculate accuracy and so on from cm text like
# [[10 6]
#  [ 5 4]]
#
# Such cm text can be found in reports.
#
import numpy as np
import pandas as pd
import io
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import argparse  

parser = argparse.ArgumentParser()
parser.add_argument('filepath')
args = parser.parse_args()

lines = []
with open(args.filepath) as f:
    for l in f.readlines():
        l = l.replace('[', '')
        l = l.replace(']', '')
        for _ in range(5):
            l = l.replace('  ', ' ')
        lines.append(l if l[0] != ' ' else l[1:])

stio = io.StringIO(''.join(lines))
cm = pd.read_csv(stio, sep=' ', header=None)

n_classes = len(cm)
y_true = np.repeat(np.arange(n_classes), cm.sum(axis=1))
y_pred = []
for i in range(n_classes):
    y_pred.extend(np.repeat(np.arange(n_classes), cm.iloc[i,:]))

y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
print(cm)

classificationReport = classification_report(y_true, y_pred,
                                             labels=np.arange(n_classes),
                                             digits=4,
                                             )

print(classificationReport)

