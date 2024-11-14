#!/bin/sh
# Usage  : optim_summary.sh MODEL DATASET SUFFIX
# Examples
#    optim_summary.sh tsf opportunity 20231030
#    optim_summary.sh tsf opportunity 20231030 | head -n 10
#    optim_summary.sh tsf opportunity 20231030 | sort -t, -k2 | head -n 10
#    optim_summary.sh tsf opportunity 20231030 | sort -t, -k3 -r | head -n 10
#
echo "select DISTINCT (select group_concat(substr(round(value,4) || '000000', 1, 6)) from trial_values where trial_values.trial_id = tv.trial_id) as val, (select group_concat(substr('    ' || param_value, -4, 4)) from trial_params where trial_params.trial_id = tv.trial_id), (select group_concat(param_name) from trial_params where trial_params.trial_id = tv.trial_id) from trial_values as tv order by val desc" | sqlite3 ./$1_$2_$3.db 
