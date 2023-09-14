# IMU-based HAR Benchmark 

This is a benchmark utility to benchmark IMU-based HAR models. 

Features
- benchmark setup with iSPLInception's proposal
- Leave one subject out CV for ucihar, daphnet, and pamap2 (new)
- Ratio based split (new)
- Clean segmentation for opportunity (new)

# How to run
```
usage: main.py [-h] --datasets DATASETS --model_name MODEL_NAME [--ispl_datareader] [--class_weight] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--patience PATIENCE]

optional arguments:
  -h, --help            show this help message and exit
  --datasets DATASETS
  --model_name MODEL_NAME
  --ispl_datareader
  --class_weight
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --patience PATIENCE
```

Exmaple dataset settings
```
['ucihar']
['daphnet']
['pamap2']
['opportunity']
[f'ucihar_losocv_{i}' for i in range(1, 31)]
[f'daphnet_losocv_{i}' for i in range(1, 11)]
[f'pamap2_losocv_{i}' for i in range(1, 9)]
['ucihar_ratio_70_20_10']
['daphnet_ratio_70_20_10']
['pamap2_ratio_70_20_10']
['opportunity_ratio_70_20_10']
```

Exmaple model names (please see models/)
```
ispl.cnn
ispl.cnn_lstm
ispl.ispl_inception
...
```

# How to adding your models
Please manage your models with a separate repository with add-on style (copy and combined to this benchmark).
It allow you to select license of your model's source code.

If you build your model, please let us know to list below.

This bemchmark call the following to get preconfiged models.
```
models.model\_name.gen_pretrained_model(dataset_name)
```

An example of separate model repository is 
- README.md
- models/your\_model/core.py

The rTsfNet is an exmple:
- see: https://hgoehoge/tsf
- TODO: update

# add-ons
- rTsfNet: https://hgoehoge/tsf

Please let us know to list here if you build your model.

# Note 

The original is the benchmark system publiced on iSPLInception ( https://github.com/rmutegeki/iSPLInception/ ).

