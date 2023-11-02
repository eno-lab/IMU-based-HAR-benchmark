# IMU-based HAR Benchmark 

This is a benchmark utility for IMU-based HAR models. 

# LICENSE
Use of this benchmark in publications must be acknowledged by referencing the following publication [1]. 
We recommend to refer to this benchmark as the "IMU-based HAR Benchmark" in publications.

- Yu Enokibori. 2023. rTsfNet: a DNN model with Multi-head 3D Rotation and Time Series Feature Extraction for IMU-based Human Activity Recognition. arXiv:2310.19283 [cs.HC]
- https://doi.org/10.48550/arXiv.2310.19283 (or https://arxiv.org/abs/2310.19283 )

## NOTE
The DOI and citation text will be updated for formal one. So, please check here before you make a submission.

# How to run
```
usage: [CUDA_VISIBLE_DEVICES=N] python3 -m main [-h] --datasets DATASETS --model_name MODEL_NAME [--ispl_datareader] [--class_weight] [--epochs EPOCHS] 
               [--boot_strap_epochs BOOT_STRAP_EPOCHS] [--batch_size BATCH_SIZE] [--patience PATIENCE]
               [--shuffle_on_train] [--lr_magnif LR_MAGNIF] [--lr_magnif_on_plateau LR_MAGNIF_ON_PLATEAU] [--lr_auto_adjust_based_bs] [--mixed_precision MIXED_PRECISION]
               [--pretrained_model PRETRAINED_MODEL] [--two_pass] [--skip_train] [--best_selection_metric BEST_SELECTION_METRIC]
               [--optuna] [--optuna_study_suffix OPTUNA_STUDY_SUFFIX] [--optuna_num_of_trial OPTUNA_NUM_OF_TRIAL]

optional arguments:
  -h, --help                                                # show this help message and exit
  --datasets DATASETS	                                    # evaluate dataset or datasets for cv
  --model_name MODEL_NAME                                   # module will be evaluated
  --ispl_datareader                                         # use iSPL implementation datareaders for ucihar, daphnet, pamap2 and opportunity
  --class_weight                                            # use class weight                                                    
  --epochs EPOCHS                                           # default: 350
  --boot_strap_epochs BOOT_STRAP_EPOCHS                     # default: 0, protection for early stop
  --batch_size BATCH_SIZE                                   # default: 64
  --patience PATIENCE                                       # defualt: 50, for early stop
  --shuffle_on_train                                        # enable shuffle. False in ISPL's condition
  --lr_magnif LR_MAGNIF                                     # default: 1
  --lr_magnif_on_plateau LR_MAGNIF_ON_PLATEAU               # default: 0.8
  --lr_auto_adjust_based_bs                                 # disable auto lr adjustment based on batch size, NOT RECOMMENDED for most cases
  --mixed_precision MIXED_PRECISION                         # default: None, global policy for, e.g., tf.keras.mixed_precision.set_global_policy
  --pretrained_model PRETRAINED_MODEL                       # default: None, the path for a pretrained model file
  --two_pass                                                # two path training. EXPERIMENTAL, 1st: no shuffle/class_weight, 2nd shuf/clw(if enabled) 
  --skip_train                                              # evaluation only mode
  --best_selection_metric BEST_SELECTION_METRIC             # metric to select best one from the best-low-val-loss or the final-epoch models
  --optuna                                                  # run optuna based parameter optimization 
  --optuna_study_suffix OPTUNA_STUDY_SUFFIX                 # study name suffix on optuna
  --optuna_num_of_trial OPTUNA_NUM_OF_TRIAL                 # num of trial of optuna

environment variable
  CUDA_VISIBLE_DEVICES=N                                    # GPU selection. -1 disable GPU.
```

# Available Dataset @ 2023/10/13
DATASETS is handled via 'eval'.

Available Datasets specifications are
```
- ['daphnet'], 
- [f'daphnet-losocv_{i}' for i in range(1,11)]
- ['wisdm']
- [f'wisdm-losocv_{i}' for i in range(1, 37)]
- ['pamap2']                                        
- ['pamap2-with_rj']                                # include the label 24: rope jumping that is optional activity
- [f'pamap2-losocv_{i}' for i in range(1, 9)]       # subject 9 include the label 24 only, so ignored
- [f'pamap2-full-losocv_{i}' for i in range(1, 10)] 
- ['opportunity']			            # iSPL based train/test set split
- ['opportunity\_real']                             # iSPL based train/test set split, include Null, split ignoring label boundary 
- ['opportunity\_real-task\_b2']                    # task b2 of opportunity challenge, include Null, split ignoring label boundary, exclude 12 ACCs.
- ['opportunity\_real-task\_b2\_no\_null']          # task b2 of opportunity challenge, exclude Null, split ignoring label boundary, exclude 12 ACCs.
- ['opportunity\_real-task\_c']                     # task c of opportunity challenge, include Null, split ignoring label boundary 
- ['opportunity\_real-task\_c\_no\_null']           # task c of opportunity challenge, exclude Null, split ignoring label boundary 
- ['opportunity\_real\_last\_label']                # select the last label of segments instead of 'mode' value. 
- ['opportunity\_real\_last\_label\_b2']            # select the last label of segments instead of 'mode' value. 
- ['opportunity\_real\_last\_label\_b2\_no\_null']  # select the last label of segments instead of 'mode' value. 
- ['opportunity\_real\_last\_label\_c']             # select the last label of segments instead of 'mode' value. 
- ['opportunity\_real\_last\_label\_c\_no\_null']   # select the last label of segments instead of 'mode' value. 
- ['ucihar']                                        # the same as ['ucihar-orig']
- [f'ucihar-losocv_{i}' for i in range(1, 31)]
- ['ucihar-ispl']                                   # iSPL based split
```

## Links for the datasets
- [Daphnet](https://doi.org/10.24432/C56K78)
- [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)
- [PAMAP2](https://doi.org/10.24432/C5NW2H)
- [OPPORTUNITY](https://doi.org/10.24432/C5M027)
- [UCIHAR](https://doi.org/10.24432/C54S4K)

## How to locate the downloaded files

Please see [dataset\_file\_list.txt](dataset_file_list.txt).


# Directories
```
- logs             # logs for TensorBoard are stored.
- reports          # summary reports are stored.
- images           # images of loss/acc graphs, confusion matrix, etc. are stored.
- trained_models   # trained models/weights are stored.
- dataset          # please put the dataset as-is downloaded from their website. plz see dataset_file_list.txt.
- opt_app          # optional utilities. Please see the header of the files.
```

# Related repositories @ 2023/10/13
- rTsfNet: https://github.com/eno-lab/rTsfNet

Please let us know to add your repository to this list!!

# How to add your models
Please manage your models with a separate repository with add-on style (copy and combined to this benchmark).
It allow you to select license of your model's source code.

This benchmark system get the model instance with the following sort.
```
models.MODEL_NAME.get_preconfiged_model(input_shape, n_classes, out_loss, out_activ, DATASET_NAME, metrics, lr_magnif=lr_magnif)

or 

config = models.MODEL_NAME.get_config(DATASET_NAME, lr_magnif)
models.MODEL_NAME.gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config) 

or 

config = models.MODEL_NAME.get_optim_config(DATASET_NAME, trial, lr_magnif)
models.MODEL_NAME.gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config) 

# DATASET_NAME is elements of DATASETS.
```

An exmaple is rTsfNet. Please looking into https://github.com/eno-lab/rTsfNet .

# Performance comparison
Pleaase see [REGISTERED PERFORMANCES](https://github.com/eno-lab/IMU-based-HAR-benchmark/wiki)

We are welcome to register the performance of new algorithms. Please let us know.

# Dataset settings 
Please see [wiki](https://github.com/eno-lab/IMU-based-HAR-benchmark/wiki)

# Acknowledgement
This bemchmark system is forked from iSPLInception ( https://github.com/rmutegeki/iSPLInception/ ).
