# 
# reconstruct the output of optim_summary.sh
# 
# exmaple:
#   optim_summary.sh tsf opportunity 20231030 | head -n 10 > top_ten_result.txt
#   python3 optim_result_formatter.py top_ten_result.txt
# 

import argparse  
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("filename")

summary = None
args = parser.parse_args()
with open(args.filename, 'r') as f:
    for rank, line in enumerate(f.readlines()):
        if line[-1] == '\n':
            line = line[:-1]
        scores, vals, labels = line.split('|')

        if summary is None:
            index = ['mf1', 'loss', 'acc']
            index.extend(labels.split(','))
            summary = pd.DataFrame(index=index)

        loss, m_f1, acc = scores.split(',')
        values = []
        values.append(loss)
        values.append(m_f1)
        values.append(acc)
        values.extend(vals.split(','))
        summary.insert(summary.shape[-1], rank, values)

    if summary is not None:
        summary.to_csv(f'{args.filename}_summary.csv')
