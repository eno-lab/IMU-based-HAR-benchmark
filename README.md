# IMU-based HAR Benchmark 

This is a benchmark utility for IMU-based HAR models. 

# NEWS
- Adding Keras 3 support
  - Keras 2 support has also been kept. (Some sample models only work with Keras 2. See models/sample/keras2.)
- Adding pyTorch support 


# LICENSE
Use of this benchmark in publications must be acknowledged by referencing the following publication [1]. 
We recommend to refer to this benchmark as the "IMU-based HAR Benchmark" in publications.

- Yu Enokibori. 2024. rTsfNet: a DNN Model with Multi-head 3D Rotation and Time Series Feature Extraction for IMU-based Human Activity Recognition. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 8, 4, Article 202 (December 2024), 26 pages, https://doi.org/10.1145/3699733
- ~Yu Enokibori. 2023. rTsfNet: a DNN model with Multi-head 3D Rotation and Time Series Feature Extraction for IMU-based Human Activity Recognition. arXiv:2310.19283 [cs.HC],  https://doi.org/10.48550/arXiv.2310.19283 (or https://arxiv.org/abs/2310.19283 )~

~## NOTE The DOI and citation text will be updated for a formal one. So, please check here before you make a submission.~

# How to run
```
usage: [CUDA_VISIBLE_DEVICES=N] python3 -m main [-h] --datasets DATASETS --model_name MODEL_NAME 
               [--ispl_datareader] [--class_weight] [--epochs EPOCHS] [--boot_strap_epochs BOOT_STRAP_EPOCHS]
               [--batch_size BATCH_SIZE] [--early_stopping_patience EARLY_STOPPING_PATIENCE] [--shuffle_on_train] [--lr_magnif LR_MAGNIF]
               [--lr_magnif_on_plateau LR_MAGNIF_ON_PLATEAU] [--reduce_lr_on_plateau_patience REDUCE_LR_ON_PLATEAU_PATIENCE] [--lr_auto_adjust_based_bs]
               [--mixed_precision MIXED_PRECISION] [--pretrained_model PRETRAINED_MODEL] [--label_smoothing LABEL_SMOOTHING] [--skip_train]
               [--best_selection_metrics BEST_SELECTION_METRICS] [--optuna] [--optuna_study_suffix OPTUNA_STUDY_SUFFIX]
               [--optuna_num_of_trial OPTUNA_NUM_OF_TRIAL] [--optuna_nsga_ii_population_size OPTUNA_NSGA_II_POPULATION_SIZE]
               [--optuna_predefine_params_N OPTUNA_PREDEFINE_PARAMS_N] [--optuna_predefined_trials_dir OPTUNA_PREDEFINED_TRIALS_DIR]
               [--optuna_invalid_predefined_trials_dir OPTUNA_INVALID_PREDEFINED_TRIALS_DIR]
               [--optuna_running_predefined_trials_dir OPTUNA_RUNNING_PREDEFINED_TRIALS_DIR]
               [--optuna_finished_predefined_trials_dir OPTUNA_FINISHED_PREDEFINED_TRIALS_DIR] [--optuna_committed_results_dir OPTUNA_COMMITTED_RESULTS_DIR]
               [--optuna_tell_predefined_trial_results] [--optuna_run_a_predefined_trial OPTUNA_RUN_A_PREDEFINED_TRIAL]
               [--downsampling_ignore_rate DOWNSAMPLING_IGNORE_RATE] [--tensorboard] [--use_data_normalization] [--show_epoch_time_detail] [--keras_no_jit]
               [--tf_debug_v2] [--tf_debug_v2_log_dir TF_DEBUG_V2_LOG_DIR]

optional arguments:
  -h, --help                                                # show this help message and exit
  --datasets DATASETS                                       # evaluate dataset or datasets for cv
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
  --skip_train                                              # evaluation only mode
  --best_selection_metric BEST_SELECTION_METRIC             # metric to select best one from the best-low-val-loss or the final-epoch models
  --optuna                                                  # run optuna based parameter optimization 
  --optuna_study_suffix OPTUNA_STUDY_SUFFIX                 # study name suffix on optuna
  --optuna_num_of_trial OPTUNA_NUM_OF_TRIAL                 # num of trial of optuna
  --optuna_nsga_ii_population_size                          # population_size of NSGA-II algorithm of optuna
  --optuna_predefine_params_N                               # generate predefined optuna params. 
                                                            # 0 disables the feature, and -1 uses the value of optuna_nsga_ii_population_size.
  --optuna_predefined_trials_dir                            # The output directory of predefined optuna params. 
                                                            # default: 'optuna_predefined_trials'
  --optuna_invalid_predefined_trials_dir                    # The directory that invalid params detected on run are stored. 
                                                            # default: 'optuna_invalid_predefined_trials'
  --optuna_running_predefined_trials_dir                    # The directory that running predfined params are placed.
                                                            # default: 'optuna_running_predefined_trials'
  --optuna_finished_predefined_trials_dir                   # The directory that results of predfined params will be stored.
                                                            # default: 'optuna_finished_predefined_trials'
  --optuna_committed_results_dir                            # The directory that committed results will be stored.
                                                            # default: 'optuna_commited_results'
  --optuna_tell_predefined_trial_results                    # Tell the results stored on the directory pointed by --optuna_finished_predefined_trials_dir.
  --optuna_run_a_predefined_trial                           # Run a trial with predfined params
  --downsampling_ignore_rate RATE                           # Speed up option, using a part of dataset. rate: 0<= rate < 1. default 0. 
                                                            # E.g., if 0.3 is set for 100 samples, 70 samples are selected uniformly.
  --tensorboard                                             # enable TensorBoard outputs to ./logs directory.
  --use_data_normalization                                  # apply z transform for each axes of the data
  --show_epoch_time_detail                                  # show consumed times (median) of batchs.
  --keras_no_jit                                            # set model.jit_comple=False on Keras
  --tf_debug_v2	                                            # If using TF, enable tf's debug_v2.
  --tf_debug_v2_log_dir                                     # the directory that logs of tf's debug_v2 will be stored. default: 'tfdbg2_logdir'

environment variable
  CUDA_VISIBLE_DEVICES=N                                    # GPU selection. -1 disable GPU.
```

## Examples
```
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['ucihar']" --model_name 'ispl.ispl_inception' --shuffle_on_train
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['ucihar']" --model_name 'sample.mch_cnn_gru' --patience 50 --epochs 600 --shuffle_on_train
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['m_health']" --model_name 'sample.deep_conv_lstm' --boot_strap_epochs 150 --epochs 350 --shuffle_on_train
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['m_health']" --model_name 'sample.keras2.mobile_hart_xs' --boot_strap_epochs 150 --epochs 350 --shuffle_on_train
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['opportunity']" --model_name 'tsf' --boot_strap_epochs 150 --patience 50 --epochs 350 --shuffle_on_train
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['ucihar']" --model_name 'tsf' --boot_strap_epochs 150 --patience 50 --epochs 350 --optuna --optuna_study_suffix 20231101 --optuna_num_of_trial 600

# pyTorch sample
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['ucihar']" --model_name 'sample.torch_deep_conv_lstm' --boot_strap_epochs 150 --epochs 350 --shuffle_on_train
CUDA_VISIBLE_DEVICES=0 python3 -m main --datasets "['ucihar']" --model_name 'sample.if_conv_transformer.If_ConvTransformer_W' --boot_strap_epochs 150 --epochs 350 --shuffle_on_train
```
The last two examples with model_name 'tsf' require the rTsfNet model. Please see the [Related repositories] section.

## Note
Tensorflow 2.15 has [a bug](https://github.com/tensorflow/tensorflow/issues/62607) on LayerNormalization.
If you evaluate tensorflow models including LayerNormalization,  
please use the other versions, such as 2.14, or set 1e-7 for the epsilon attribute of LayerNormalization.

# Available Dataset
DATASETS is handled via 'eval'.

Available Datasets specifications are
```
- ['daphnet'], 
- ['wisdm']
- ['pamap2']                                        
- ['pamap2-with_rj']                                # include the label 24: rope jumping that is optional activity
- ['opportunity']                                   # iSPL based train/test set split
- ['opportunity_real']                              # iSPL based train/test set split, include Null, split ignoring label boundary 
- ['opportunity_real-task_b2']                      # task b2 of opportunity challenge, include Null, split ignoring label boundary, exclude 12 ACCs.
- ['opportunity_real-task_b2_no_null']              # task b2 of opportunity challenge, exclude Null, split ignoring label boundary, exclude 12 ACCs.
- ['opportunity_real-task_c']                       # task c of opportunity challenge, include Null, split ignoring label boundary 
- ['opportunity_real-task_c_no_null']               # task c of opportunity challenge, exclude Null, split ignoring label boundary 
- ['opportunity_real_last_label']                   # select the last label of segments instead of 'mode' value. 
- ['opportunity_real_last_label_b2']                # select the last label of segments instead of 'mode' value. 
- ['opportunity_real_last_label_b2_no_null']        # select the last label of segments instead of 'mode' value. 
- ['opportunity_real_last_label_c']                 # select the last label of segments instead of 'mode' value. 
- ['opportunity_real_last_label_c_no_null']         # select the last label of segments instead of 'mode' value. 
- ['ucihar']                                        # the same as ['ucihar-orig']
- ['ucihar-ispl']                                   # iSPL based split
- ['real_world'] 
- ['m_health'] 
- ['mighar']                                        # memory should be >= 64 GiB. 128 GiB is recommended. 
- ['mighar-offset_calib']                           # Using Offset calibrated data instead of raw data.
- ['uschad']                                        
- ['motion_sense']                                  
- ['sho']                                           
```

## Suffix options 

- Separation and combination suffix options can not be used simultaneously.
- LOSOCV and ratio suffix options can not be used simultaneously.

Examples of acceptable combinations are
```
xxxx-combination_0_1_2-losocv_1
xxxx-combination_0_1_2-ratio_70_10_20
xxxx-separation_0_1_2-losocv_1
xxxx-separation_0_1_2-ratio_70_10_20
```
### Note
The combination and separation sensor options can be used to select a single sensor from many. E.g., \['pamap2-separation\_0'\] and \['pamap2-combination\_0'\].

### Separation sensor option
If add the following suffix for a dataset specification, the sensors included in the dataset are handled indivisually.

```-separation[_0_1_2_3...][_with_sid]```

For example, if \['pamap2-separation'\] is specified, since the pamap2 including three sensors, three samples are generated on time _t_.
In contrast to that, if \['pamap2'\] is specified, one sample including data of the three sensors is generated on time _t_.
If \['pamap2-separation\_0\_2'\] is specified, two samples from sensors 0 and 2 are generated on time _t_.
If \['pamap2-separation\_with\_sid'\] is specified, three samples are generated on time _t_; however, each sample has a sensor ID value, such as 0, 1, and 2, on an additional channel placed on the last. 
The last channel is filled by an identical value.
This extra values should be removed at input mapping phases of DNN.

### Combination sensor option
If add the following suffix for a dataset specification, samples generated for the specified sensors only.

```-combination_0_1_2_3...[_with_sid]```

For example, if \['pamap2-combination\_0\_2'\] is specified, since the pamap2 including three sensors, generated sampels are consists of the values from these two specified sensors.
If \['pamap2--combination\_0\_2\_with\_sid'\] is specified, each sample has sensor ID values, such as 0 and 2, for each axis.
The sensor IDs are placed on a extra entry of window. For example, if the data has a shape [100, 256, 6], meaning 100 samples with 256 window length with 6 channels, the shape will be changed for [100, 25*7*, 6]. 
The extra entries placed on \[:, 256, :\] have sensor ID values. For example, \[0, 0, 0, 2, 2, 2\]. 
This extra values should be removed at input mapping phases of DNN.

### Leave one subject out cross-validation (LOSOCV) option

```-losocv_n```

For example, if \['pamap2-losocv_1'\] is specified, the data of subject 1 is used for test data.
The following sort is suitable to run LOSOCV, \[f'pamap2-losocv_{i}' for i in range(1, 10)\].

Several DataReaders do not support this LOSOCV option, such as the DataReader for the Opportunity. 

### Ratio option

```-ratio_Tr_V_Ts```

**DO NOT USE IT**

**DO NOT USE IT**

**DO NOT USE IT**

**This option makes meaningless results. It has remained to keep compatibility with several studies.**

For example, if \['pamap2-ratio_70_10_20'\] is specified, 70 % of the data is selected randomly for training, 10 % for validation, and 20 % for testing.

## Links for the datasets
- [Daphnet](https://doi.org/10.24432/C56K78)
- [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php)
- [PAMAP2](https://doi.org/10.24432/C5NW2H)
- [OPPORTUNITY](https://doi.org/10.24432/C5M027)
- [UCIHAR](https://doi.org/10.24432/C54S4K)
- [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld)
- [mHealth](https://doi.org/10.24432/C5TW22)
- [MIG-HAR](https://github.com/eno-lab/Meshed_IMU_Garment_HAR_Dataset)
- [USC-HAD](https://sipi.usc.edu/had/)
- [MotionSense](https://github.com/mmalekzadeh/motion-sense)
- [SHO](https://www.researchgate.net/publication/266384007_Sensors_Activity_Recognition_DataSet)

## How to locate the downloaded dataset files

Please put downloaded or decompressed datasets into the dataset directory as is, such as via drag and drop.
The expected file paths are listed in [dataset\_file\_list.txt](dataset_file_list.txt).
Please see this to check how to locate downloaded dataset files.

# Directories
```
- logs             # logs for TensorBoard are stored.
- reports          # summary reports are stored.
- images           # images of loss/acc graphs, confusion matrix, etc. are stored.
- trained_models   # trained models/weights are stored.
- dataset          # please put the dataset as-is downloaded from their website. plz see dataset_file_list.txt.
- opt_app          # optional utilities. Please see the header of the files.
- models           # Models should be stored here. In this repository, there are several sample models and the models used in iSPLInception.
```

# Related repositories @ 2023/10/13
- rTsfNet: https://github.com/eno-lab/rTsfNet

Please let us know so we can add your repository to this list!!

If so, please make [a new issue](https://github.com/eno-lab/IMU-based-HAR-benchmark/issues) with the "Model registration request" label with the URL and so on.

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

If you would like to register your benchmark results, please make [a new issue](https://github.com/eno-lab/IMU-based-HAR-benchmark/issues) with the "Benchmark result registration request" label with evidence, such as trained models and reports.

We are welcome to register the performance of new algorithms. Please let us know.

# Dataset settings 
Please see [wiki](https://github.com/eno-lab/IMU-based-HAR-benchmark/wiki)

# Acknowledgement
This bemchmark system is forked from iSPLInception ( https://github.com/rmutegeki/iSPLInception/ ).
