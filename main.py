import os
import gc
import sys
import pandas as pd
import numpy as np
import importlib
from datetime import datetime
from time import time
from utils import evaluate_model

from sklearn.metrics import confusion_matrix, classification_report

from datareader import gen_datareader

import argparse  

def zero_one_float_type(arg):
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("Argument must be between 0 and 1")
    return f

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--ispl_datareader', action='store_true')
parser.add_argument('--class_weight', action='store_true')
parser.add_argument('--epochs', type=int, default=350)
parser.add_argument('--boot_strap_epochs', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--early_stopping_patience', type=int, default=50)
parser.add_argument('--shuffle_on_train', action='store_true')
parser.add_argument('--lr_magnif', type=float, default=1)
parser.add_argument('--lr_magnif_on_plateau', type=float, default=0.8)
parser.add_argument('--reduce_lr_on_plateau_patience', type=int, default=10)
parser.add_argument('--lr_auto_adjust_based_bs', action='store_true')
parser.add_argument('--mixed_precision', default=None)
parser.add_argument('--pretrained_model', default=None)
parser.add_argument('--label_smoothing', type=zero_one_float_type, default=0.0, help='0 is disable, min=0, max=1')
parser.add_argument('--skip_train', action='store_true')
parser.add_argument('--best_selection_metrics', default='mf1')
parser.add_argument('--optuna', action='store_true')
parser.add_argument('--optuna_study_suffix', default='')
parser.add_argument('--optuna_num_of_trial', type=int, default=10)
parser.add_argument('--downsampling_ignore_rate', type=float, default=0)
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--use_data_normalization', action='store_true')
parser.add_argument('--show_epoch_time_detail', action='store_true')
parser.add_argument('--keras_no_jit', action='store_true')
parser.add_argument('--tf_debug_v2', action='store_true')
parser.add_argument('--tf_debug_v2_log_dir', default='tfdbg2_logdir')

args = parser.parse_args()

import optuna
import logging
import sys
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

total_prediction = []
total_true = []

epochs = args.epochs
batch_size = args.batch_size
early_stopping_patience = args.early_stopping_patience
reduce_lr_on_plateau_patience = args.reduce_lr_on_plateau_patience
shuffle_on_train = args.shuffle_on_train
lr_magnif = args.lr_magnif
downsampling_ignore_rate = args.downsampling_ignore_rate
if not (0<= downsampling_ignore_rate < 1):
    sys.exit(f"Invalid downsampling_ignore_rate, {downsampling_ignore_rate}. Valid range is 0<= rate < 1.")
    
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 200)

# it is not suitable for many case.
# not recommended
if args.lr_auto_adjust_based_bs and batch_size != 64:
    # Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach to Neural Network Training
    # Diego Granziol, Stefan Zohren, Stephen Roberts
    # for Adam
    lr_magnif *= np.sqrt(batch_size)/np.sqrt(64) # it seems to big

datasets = eval(args.datasets) 
model_name = args.model_name
training_id = f'{time()*1000:.0f}'

print("===============")
print(f"{datasets=}")
print(f"{model_name=}")
print("===============")


IS_TF = False
# ...................................................................................#
# init framework
# ...................................................................................#
mod = importlib.import_module(f'models.{model_name}')
framework_name  = mod.get_dnn_framework_name()
if framework_name in ('keras', 'tensorflow'):
    from keras_utils import IS_KERAS_VERSION_GE_3
    if IS_KERAS_VERSION_GE_3:
        import keras
        if keras.config.backend() == 'tensorflow':
            IS_TF = True
    else:
        from tensorflow import keras

    if framework_name == 'tensorflow':
        IS_TF = True

    if  IS_TF:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # ...................................................................................#
        # for debug
        if args.tf_debug_v2:
            debug_log_dir = args.tf_debug_v2_log_dir
            os.makedirs(debug_log_dir, exist_ok=True)
            tf.config.set_soft_device_placement(True) # necessary for TPU
            tf.debugging.experimental.enable_dump_debug_info(
                    debug_log_dir,
                    tensor_debug_mode="FULL_HEALTH",
                    circular_buffer_size=-1)
        # ...................................................................................#

    if args.mixed_precision is not None:
        keras.mixed_precision.set_global_policy(args.mixed_precision)

elif framework_name in ('pytorch', 'torch'):
    import torch
else:
    raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")
# ...................................................................................#

for dataset in datasets:

    dataset_origin = dataset.split('-')[0]
    file_prefix = f"{training_id}_{model_name}_{dataset}"

    if args.optuna:
        study_name = f'{model_name}_{dataset}_{args.optuna_study_suffix}'  # Unique identifier of the study.
        storage_name = f"sqlite:///{study_name}.db"
        study = optuna.create_study(directions=["maximize", "minimize", "maximize", "minimize"],
                                    study_name=study_name, 
                                    storage=storage_name, 
                                    #pruner=optuna.pruners.HyperbandPruner(), 
                                    load_if_exists=True)

    if args.ispl_datareader:
        from ispl_utils import load_dataset, get_loss_and_activation
        X_train, y_train, X_val, y_val, X_test, y_test, labels, n_classes = load_dataset(dataset)
        recommended_out_loss, recommended_out_activ = get_loss_and_activation(dataset)
        file_prefix = f"{file_prefix}_ispl-datareader"
    else:
        dr = gen_datareader(dataset)
        X_train, y_train, X_val, y_val, X_test, y_test, labels, n_classes = dr.gen_ispl_style_set()
        recommended_out_loss, recommended_out_activ = dr.recommended_out_loss, dr.recommended_out_activ

    if args.label_smoothing > 0:
        if IS_TF:
            recommended_out_loss = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        else:
            raise NotImplementedError(f'label_smoothing is supported on tf only')


    if downsampling_ignore_rate > 0:
        def downsampling(X, y):
            assert X.shape[0] == y.shape[0]
            _len = X.shape[0]
            _ix = np.round(np.linspace(0, _len-1, round(_len*(1-downsampling_ignore_rate)))).astype(int)
            return X[_ix], y[_ix]
        X_train, y_train = downsampling(X_train, y_train)
        X_val, y_val = downsampling(X_val, y_val)
        X_test, y_test = downsampling(X_test, y_test)

    normalization_mean = None
    normalization_std = None
    if args.use_data_normalization:

        _sp = X_train.shape
        _mean = np.mean(X_train.reshape((_sp[0]*_sp[1],_sp[2])), axis=0, keepdims=True)
        _std  = np.std(X_train.reshape((_sp[0]*_sp[1],_sp[2])), axis=0, keepdims=True)
        X_train = (X_train-_mean)/_std
        X_val   = (X_val-_mean)/_std
        X_test  = (X_test-_mean)/_std

        normalization_mean = _mean
        normalization_std = _std

    _file_prefix = file_prefix

    # Models to run
    try:
        # Loss and acc
        METRICS = [
            'accuracy'
        ]

        def objective(trial):

            if trial is not None:
                file_prefix = f'{_file_prefix}_trial-{trial.number}'
            else:
                file_prefix = f'{_file_prefix}'

            # Report Generation
            report_name = os.path.join("reports", dataset_origin, f"{file_prefix}_report.txt")
            os.makedirs(os.path.dirname(report_name), exist_ok=True)

            # Path to images
            img_path = os.path.join("images", dataset_origin, file_prefix)
            os.makedirs(img_path, exist_ok=True)

            with open(report_name, "w") as report:
                print("################# Information #################")
                print(f"Dataset: {dataset}")
                print(f"Training_id: {training_id}")
                print(f"Epochs: {epochs}")
                print(f"Train:{X_train.shape} | {y_train.shape}")
                print(f"Validation:{X_val.shape} | {y_val.shape}")
                print(f"Test:{X_test.shape} | {y_test.shape}")

                print(f"Models being trained : {model_name}")

                report.write(f"This is the report for the {dataset} Dataset\n\n")

                report.write('################### Args ###################\n')
                _vars = vars(args)
                for _k in _vars:
                    report.write(f'{_k}: {_vars[_k]}\n')
                report.write('############################################\n\n')

                report.write("Data Distribution: \n\n")
                report.write(f"Train:  X -> {X_train.shape} Class count -> {list(np.bincount(y_train.argmax(1)))} \n\n"
                             f"{pd.DataFrame(y_train.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
                report.write(f"Validation:  X -> {X_val.shape} Class count -> {list(np.bincount(y_val.argmax(1)))} \n\n"
                             f"{pd.DataFrame(y_val.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
                report.write(f"Test:  X -> {X_test.shape} Class count -> {list(np.bincount(y_test.argmax(1)))} \n\n"
                             f"{pd.DataFrame(y_test.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")

                start = time()

                if normalization_mean is not None:
                    report.write('############################################\n\n')
                    report.write("Data Normalization: \n")
                    report.write(f"Means: {normalization_mean} \n")
                    report.write(f"STDs: {normalization_std} \n\n")

                    print('############################################\n')
                    print("Data Normalization")
                    print(f"Means: {normalization_mean}")
                    print(f"STDs: {normalization_std}\n")
                    print('############################################\n')


                print('###############################################################################')
                print(f"Training {model_name} : {datetime.now()}")
                print('###############################################################################')
                log_dir = None
                if args.tensorboard:
                    log_dir = os.path.abspath(os.path.join('logs', 'fit', dataset, file_prefix))
                    os.makedirs(log_dir, exist_ok=True)

                if framework_name in ('tensorflow', 'keras'):
                    # .keras model is faster than .h5 on keras3. (on Keras2, .h5 is better)
                    save_name = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}.{'keras' if IS_KERAS_VERSION_GE_3 else 'h5'}")) 
                elif framework_name in ('pytorch', 'torch'):
                    save_name = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}.pkl"))
                else:
                    raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")

                input_shape = X_train.shape

                model = None
                clw = args.class_weight
                for _ in range(1):

                    try:
                        if args.optuna:
                            hyperparameters  = mod.get_optim_config(dataset, trial, lr_magnif=lr_magnif)
                        else:
                            hyperparameters  = mod.get_config(dataset, lr_magnif=lr_magnif)

                        model = mod.gen_model(input_shape, n_classes, 
                                              recommended_out_loss, recommended_out_activ, 
                                              METRICS, hyperparameters)
                        if args.pretrained_model is not None:
                            if framework_name in ('tensorflow', 'keras'):
                                model.load_weights(args.pretrained_model) 
                            elif framework_name in ('pytorch', 'torch'):
                                model.load_state_dict(torch.load(args.pretrained_model, weights_only=True))
                            else:
                                raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")

                        for key, item in hyperparameters.items():
                            print(f"{key.replace('_', ' ').capitalize()}: {item}")
                    except Exception as e:
                        raise NotImplementedError(f'The model {model_name} is not implemented enough yet: {e}')

                    if framework_name in ('tensorflow', 'keras') and IS_KERAS_VERSION_GE_3 and args.keras_no_jit:
                        model.jit_compile=False

                    history = None
                    if args.skip_train:
                        best_model_path = None
                        history = None
                    else:
                        #--------------------------------------------------------------#
                        # Training
                        #--------------------------------------------------------------#
                        try:
                            # Train and evaluate the current model on the dataset. Save the trained models and histories
                            model, history, best_model_path = evaluate_model(model, X_train, y_train, X_val, y_val,
                                                            early_stopping_patience=early_stopping_patience, 
                                                            boot_strap_epochs = args.boot_strap_epochs,
                                                            _epochs=epochs, _save_name=save_name, _log_dir=log_dir,
                                                            shuffle_on_train = shuffle_on_train,
                                                            batch_size = batch_size, no_weight = not clw,
                                                            lr_magnif_on_plateau = args.lr_magnif_on_plateau,
                                                            reduce_lr_on_plateau_patience=reduce_lr_on_plateau_patience,
                                                            show_epoch_time_detail = args.show_epoch_time_detail,
                                                            framework_name=framework_name)


                            print('###############################################################################')
                        except Exception as e:
                            raise e

                        #--------------------------------------------------------------#

                    # **************** Time to summarize the model's performance ****************
                    report.write(f"{model_name} Model : {datetime.now()}\n")
                    # Training History
                    if history is not None:
                        report.write(f"Model History \n{pd.DataFrame(history.history)}\n\n")

                    model_summary = ''
                    if framework_name in ('tensorflow', 'keras'):
                        model_str = []
                        model.summary(print_fn=lambda x: model_str.append(x))
                        model_summary = "\n".join(model_str)
                    elif framework_name in ('pytorch', 'torch'):
                        from torchinfo import summary
                        # torchinfo.summary with device=None or device=torch.device('cuda') consumes 
                        # twice cuda memory with the model.
                        model_summary = str(summary(model, input_shape, verbose=0, device=torch.device('cpu')))
                        model.cuda()

                    report.write(model_summary)
                    report.write('\n\n')
                    report.write("+++Hyperparameters+++\n")
                    report.write(f"Number of Epochs: {epochs}\n")
                    report.write(f"Batch Size: {batch_size}\n")

                    for key, item in hyperparameters.items():
                        report.write(f"{key.replace('_', ' ').capitalize()}: {item}\n")

                    best_model_weight_path = None
                    best_score = None
                    best_predictions = None
                    for model_type in ["last_model", "best_model"]:
                        if model_type == 'best_model':
                            if best_model_path is None:
                                continue

                            if framework_name in ('tensorflow', 'keras'):
                                model.load_weights(best_model_path) 
                            elif framework_name in ('pytorch', 'torch'):
                                model.load_state_dict(torch.load(best_model_path, weights_only=True))
                            else:
                                raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")

                        print('###############################################################################')
                        print(model_type)
                        print('###############################################################################')

                        if framework_name in ('tensorflow', 'keras'):
                            # .keras model is faster than .h5 on keras3. (on Keras2, .h5 is better)
                            _save_model_path = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}_{model_type}.{'keras' if IS_KERAS_VERSION_GE_3 else 'h5'}")) # h5 is fast in keras2
                        elif framework_name in ('pytorch', 'torch'):
                            _save_model_path = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}_{model_type}.pkl"))
                        else:
                            raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")

                        #--------------------------------------------------------------#
                        # save trained model and get scores
                        #--------------------------------------------------------------#
                        if framework_name in ('tensorflow', 'keras'):
                            if not args.skip_train:
                                model.save(_save_model_path) 

                            # Results for each model
                            val_scores = model.evaluate(X_val, y_val, verbose=1)
                            val_predictions = model.predict(X_val).argmax(1)
                            scores = model.evaluate(X_test, y_test, verbose=1)
                            predictions = model.predict(X_test).argmax(1)
                        elif framework_name in ('pytorch', 'torch'):
                            if not args.skip_train:
                                torch.save(model.state_dict(), _save_model_path) 

                            from torch_utils import calc_loss_acc_output
                            model.eval()
                            loss_function = model.get_loss_function()
                            loss_val, acc_val, val_predictions = calc_loss_acc_output(model, loss_function, X_val, y_val)
                            loss_test, acc_test, predictions = calc_loss_acc_output(model, loss_function, X_test, y_test)
                            model.train()
                            val_scores = [loss_val, acc_val]
                            scores = [loss_test, acc_test]

                        else:
                            raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")
                        #--------------------------------------------------------------#

                        report.write('###############################################################################\n')
                        report.write(model_type)
                        report.write('###############################################################################\n')

                        report.write('\n\n')
                        report.write(f"Test Accuracy: {scores[1] * 100}\n")
                        report.write(f"Test Loss: {scores[0]}\n")
                        report.write('\n\n\n')

                        classificationReport = classification_report(y_test.argmax(1), predictions,
                                                                     labels=np.arange(n_classes), 
                                                                     target_names=labels,
                                                                     digits=4,
                                                                     )
                    
                        print(classificationReport)
                        report.write(f"Classification Report\n{classificationReport}\n\n\n")

                        print("Confusion Matrix:")
                        cm = confusion_matrix(y_test.argmax(axis=1), predictions, labels=np.arange(n_classes))
                        print(cm)
                        report.write(f"Confusion Matrix\n{cm}\n\n\n")

                        print("Normalised Confusion Matrix: True")
                        normalised_confusion_matrix = np.around(confusion_matrix(y_test.argmax(axis=1),
                                                                                 predictions,
                                                                                 labels=np.arange(n_classes),
                                                                                 normalize='true') * 100, decimals=2)
                        print(normalised_confusion_matrix)
                        normalised_confusion_matrix_df = pd.DataFrame(normalised_confusion_matrix, index=labels, columns=labels)
                        report.write(f"Normalised Confusion Matrix: True\n{normalised_confusion_matrix_df}\n\n\n")

                        val_clr = classification_report(y_val.argmax(1), val_predictions,
                                                    labels=np.unique(y_val.argmax(1)), output_dict=True, zero_division=0.0)

                        clr = classification_report(y_test.argmax(1), predictions,
                                                    labels=np.unique(y_test.argmax(1)), output_dict=True, zero_division=0.0)

                        pass_score = {
                                'mf1': clr['macro avg']['f1-score'], 
                                'wf1': clr['weighted avg']['f1-score'],
                                'acc': scores[1],
                                'loss': scores[0],
                                'val_mf1': val_clr['macro avg']['f1-score'], 
                                'val_wf1': val_clr['weighted avg']['f1-score'],
                                'val_acc': val_scores[1],
                                'val_loss': val_scores[0]}

                        if best_model_weight_path is None or best_score[args.best_selection_metrics] < pass_score[args.best_selection_metrics]:
                            best_score = pass_score
                            best_model_weight_path = _save_model_path
                            best_predictions = predictions

                    text = f"Finished working on: {model_name} at: {datetime.now()} -> {time() - start}"
                    print(text)
                    report.write(f"{text}\n\n")

                    if not args.optuna:
                        total_prediction.append(best_predictions)
                        total_true.append(y_test.argmax(axis=1))

                    if model is not None:
                        del model

                    if framework_name in ('tensorflow', 'keras'):
                        keras.backend.clear_session()
                    elif framework_name in ('pytorch', 'torch'):
                        torch.cuda.empty_cache()
                    else:
                        raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")
                    gc.collect()

                    return best_score['mf1'], best_score['loss'], best_score['val_mf1'], best_score['val_loss']

        if args.optuna:
            study.optimize(objective, n_trials=args.optuna_num_of_trial)
            # TODO add best scors for total_prediction and total_true to support optuna optim for losocv, etc.
            # but is it necessary?
        else:
            objective(None)

    except Exception as ex:
        print("Caught an exception: ", ex)
        import traceback
        pyver = sys.version_info
        if pyver.major >=3 and pyver.minor >= 10: 
            print("".join(traceback.format_exception(ex))) # for 3.10
        else:
            print("".join(traceback.format_exception(None, ex, ex.__traceback__)))
        continue

    print('**Finished**')

if not args.optuna and len(datasets) > 1:
    total_report_name = os.path.join('reports', dataset_origin, f"{training_id}_{model_name}_total_report.txt")
    os.makedirs(os.path.dirname(total_report_name), exist_ok=True)

    total_prediction = np.concatenate(total_prediction)
    total_true = np.concatenate(total_true)
    with open(total_report_name, "w") as report:

        classificationReport = classification_report(total_true, total_prediction,
                                                     labels=np.arange(n_classes),
                                                     target_names=labels,
                                                     digits=4,
                                                     )

        print(classificationReport)
        report.write(f"Classification Report\n{classificationReport}\n\n\n")

        print("Confusion Matrix:")
        cm = confusion_matrix(total_true, total_prediction, labels=np.arange(n_classes))
        print(cm)
        report.write(f"Confusion Matrix\n{cm}\n\n\n")

        print("Normalised Confusion Matrix: True")
        normalised_confusion_matrix = np.around(confusion_matrix(total_true,
                                                                 total_prediction,
                                                                 labels=np.arange(n_classes),
                                                                 normalize='true') * 100, decimals=2)
        print(normalised_confusion_matrix)
        normalised_confusion_matrix_df = pd.DataFrame(normalised_confusion_matrix, index=labels, columns=labels)
        report.write(f"Normalised Confusion Matrix: True\n{normalised_confusion_matrix_df}\n\n\n")


