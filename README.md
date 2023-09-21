# IMU-based HAR Benchmark 

This is a benchmark utility to benchmark IMU-based HAR models. 

# How to run
```
usage: [CUDA_VISIBLE_DEVICES=N] main.py [-h] --datasets DATASETS --model_name MODEL_NAME [--ispl_datareader] [--class_weight] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--patience PATIENCE]

optional arguments:
  -h, --help                # show this help message and exit
  --datasets DATASETS	    # evaluate dataset or datasets for cv
  --model_name MODEL_NAME   # module will be evaluated
  --ispl_datareader 	    # use ispl implementation 
  --class_weight	    # use class weight
  --epochs EPOCHS 	    # default 300
  --batch_size BATCH_SIZE   # default: 64
  --patience PATIENCE	    # defualt: 50

environment variable
  CUDA_VISIBLE_DEVICES=N    # GPU selection. -1 disable GPU.
```

# Available Dataset
DATASETS is handled via 'eval'.

Available Datasets specifications are
- ['daphnet'], 
- [f'daphnet-losocv_{i}' for i in range(1,11)]
- ['wisdm']
- ['wisdm-losocv_{i}' for i in range(1, 37)]
- ['pamap2']                                  # exclude 24: rope jumping 
- ['pamap2-full']                             # include 24: rope jumping
- ['pamap2-losocv_{i}' for i in range(1, 9)]  # subject 9 include 24 only, so ignored
- ['pamap2-full-losocv_{i}' for i in range(1, 10)] 
- ['opportunity']			      # ispl based split
- ['opportunity-real']                        # ispl based split, include Null and split ignoring label boundary 
- ['ucihar']
- ['ucihar-losocv_{i}' for i range(1, 31)]
- ['ucihar-ispl']			      # ispl based split

# Model implementation
Model instance will be get through the following call.
```
models.MODEL_NAME.get_preconfiged_model(DATASET_NAME)

# DATASET_NAME is elements of DATASETS.
```

# Related repositories 
- rTsfNet: https://github.com/eno-lab/rTsfNet

Please let us know to add your repository to this list.

# How to adding your models
Please manage your models with a separate repository with add-on style (copy and combined to this benchmark).
It allow you to select license of your model's source code.

This benchmark system get the model instance with the following sort.
```
models.MODEL_NAME.get_preconfiged_model(DATASET_NAME)
# DATASET_NAME is elements of DATASETS.
```

An exmaple is rTsfNet: https://github.com/eno-lab/rTsfNet .

# Acknowledgement
This bemchmark system is forked from iSPLInception ( https://github.com/rmutegeki/iSPLInception/ ).
