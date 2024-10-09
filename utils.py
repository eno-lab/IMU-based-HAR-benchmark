# title       :utils
# description :Script that contains utilities that are required by or shared with other scripts
# author      :Ronald Mutegeki
# date        :20210203
# version     :1.0
# usage       :Call it in main.py.
# notes       :Majority imports are done in here. Processing of the dataset, Model training
#              and evaluation are also done within this script.

# TODO add our information

import ast
import os
import re

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import stats
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
from datareader.ispl import DataReader

from sklearn.model_selection import train_test_split


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}. Let's avoid too many logs

# **************** Default Setting ******************
# Setting default parameters for plots and figures
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Dejavu Sans'
matplotlib.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# **************** Dataset Configurations ******************
def get_data_configuration(dataset):
    if dataset.startswith('daphnet'):
        n_signals = 9  # Ankle acc - x,y,z; Upper Leg acc - x,y,z; Trunk acc - x,y,z;
        win_size = 192  # 1 sec -> 64 | 2.56 (3) sec -> 192 | 5 sec -> 320 # sampling rate = 64hz
        n_classes = 2  # 1 - no freeze (walk, stand, turn); 2 - freeze
        n_steps = 3  # Since we are using 3 seconds and 192 is divisible by 3
        length = 64  # Split each window of 192 time steps into sub sequences for the cnn

    elif dataset == 'ispl':
        n_signals = 9  # acc - x,y,z; gyro - x,y,z; lacc - x,y,z;
        win_size = 128  # 2.56 sec -> 128 | 5 sec -> 256
        n_classes = 3  # 1 - walking; 2 - standing; 3 - sitting; 4 - running
        n_steps = 4  # 128 is divisible by 4
        length = 32  # Split each window of 128 time steps into sub sequences for the cnn

    elif dataset.startswith('pamap2'):
        n_signals = 36  # IMU hand, chest, ankle
        win_size = 256  # 2.56 sec -> 256 | 5.12 sec -> 512 # sampling rate is 100hz
        n_classes = 11  # Removed 7 classes due to lack of sufficient training data
        n_steps = 4  # 256 is divisible by 4
        length = 64  # Split each window of 256 time steps into sub sequences for the cnn

    elif dataset == 'opportunity_task_a':
        n_signals = 77  # ...
        win_size = 90  # ~2.56 (3) sec -> 90 | 5 sec -> 150  # Sampling rate is 30Hz
        n_classes = 5 
        n_steps = 3  # Since we are using 3 seconds and 90 is divisible by 3
        length = 30  # Split each window of 90 time steps into sub sequences for the cnn

    elif dataset == 'opportunity_task_a_without_null':
        n_signals = 77  # ...
        win_size = 90  # ~2.56 (3) sec -> 90 | 5 sec -> 150  # Sampling rate is 30Hz
        n_classes = 4 
        n_steps = 3  # Since we are using 3 seconds and 90 is divisible by 3
        length = 30  # Split each window of 90 time steps into sub sequences for the cnn

    elif dataset == 'opportunity_task_b2':
        n_signals = 77  # ...
        win_size = 90  # ~2.56 (3) sec -> 90 | 5 sec -> 150  # Sampling rate is 30Hz
        n_classes = 18
        n_steps = 3  # Since we are using 3 seconds and 90 is divisible by 3
        length = 30  # Split each window of 90 time steps into sub sequences for the cnn

    elif dataset == 'opportunity_task_b2_without_null':
        n_signals = 77  # ...
        win_size = 90  # ~2.56 (3) sec -> 90 | 5 sec -> 150  # Sampling rate is 30Hz
        n_classes = 17
        n_steps = 3  # Since we are using 3 seconds and 90 is divisible by 3
        length = 30  # Split each window of 90 time steps into sub sequences for the cnn

    elif dataset == 'opportunity_task_c':
        n_signals = 45  # ...
        win_size = 90  # ~2.56 (3) sec -> 90 | 5 sec -> 150  # Sampling rate is 30Hz
        n_classes = 18
        n_steps = 3  # Since we are using 3 seconds and 90 is divisible by 3
        length = 30  # Split each window of 90 time steps into sub sequences for the cnn

    elif dataset == 'opportunity_task_c_without_null':
        n_signals = 45  # ...
        win_size = 90  # ~2.56 (3) sec -> 90 | 5 sec -> 150  # Sampling rate is 30Hz
        n_classes = 17
        n_steps = 3  # Since we are using 3 seconds and 90 is divisible by 3
        length = 30  # Split each window of 90 time steps into sub sequences for the cnn

    elif dataset.startswith('opportunity'):
        n_signals = 77  # ...
        win_size = 90  # ~2.56 (3) sec -> 90 | 5 sec -> 150  # Sampling rate is 30Hz
        n_classes = 17
        n_steps = 3  # Since we are using 3 seconds and 90 is divisible by 3
        length = 30  # Split each window of 90 time steps into sub sequences for the cnn

    elif dataset.startswith('ucihar'):
        n_signals = 9  # acc - x,y,z; gyro - x,y,z; bacc - x,y,z;
        win_size = 128  # 2.56 seconds
        n_classes = 6  # 1 - Walking; 2 - Walking_Upstairs; 3 - Walking_Downstairs; 4 - Sitting; 5 - Standing; 6 - Laying
        n_steps = 4  # 128 is divisible by 4
        length = 32  # Split each window of 128 time steps into sub sequences for the cnn

    return n_signals, win_size, n_classes, n_steps, length


def get_loss_and_activation(dataset):
    if dataset == 'daphnet':
        return 'binary_crossentropy', 'sigmoid'
    else:
        return CategoricalCrossentropy(reduction=Reduction.AUTO, name='output_loss'), 'softmax'


def windows(data, size):
    start = 0
    while start < len(data):
        yield int(start), int(start + size)
        start += (size / 2)


def segment(x, y, window_size, dataset_signals=9):
    print(f"X: {x.shape}\nY: {y.shape}\nWin_size: {window_size}\nSignals: {dataset_signals}")
    segments = np.zeros(((len(x) // (window_size // 2)) - 1, window_size, dataset_signals))
    labels = np.zeros(((len(y) // (window_size // 2)) - 1))
    i_segment = 0
    i_label = 0
    for (start, end) in windows(x, window_size):
        if len(x[start:end]) == window_size:
            m = stats.mode(y[start:end])
            segments[i_segment] = x[start:end]
            labels[i_label] = m[0]
            i_label += 1
            i_segment += 1
    return segments, labels


# Function that loads a specified dataset
# dataset _type is either 'original' or train_test_split
def _load_dataset(dataset='ucihar', datapath='dataset/ucihar', _type='original'):
    datapath = datapath.rstrip("/")
    # Check if our desired dataset has not yet been generated
    if not os.path.exists(f'{datapath}/{dataset}.ispl.h5'):
        DataReader(dataset, datapath)

    # class labels
    with open(f'{datapath}/{dataset}.ispl.h5.classes.json', 'r') as f:
        labels = ast.literal_eval(f.read())

    with h5py.File(f'{datapath}/{dataset}.ispl.h5', 'r') as f:
        X_train, y_train = np.array(f['train']['inputs']), np.array(f['train']['targets'])
        X_val, y_val = np.array(f['validation']['inputs']), np.array(f['validation']['targets'])
        X_test, y_test = np.array(f['test']['inputs']), np.array(f['test']['targets'])

    return X_train, y_train, X_val, y_val, X_test, y_test, labels


def transform_y(y, nr_classes, dataset):
    # Transforms y, a list with one sequence of A timesteps
    # and B unique classes into a binary Numpy matrix of
    # shape (A, B)
    if dataset.startswith("ucihar") or dataset.startswith("ispl"):
        y = np.array([a - 1 for a in y])
    ybinary = to_categorical(y, nr_classes)
    return ybinary


# Model and dataset evaluation
def evaluate_model(_model, _X_train, _y_train, _X_test, _y_test, 
                   _epochs=20, early_stopping_patience=10, boot_strap_epochs=0,
                   batch_size=64, _save_name='trained_models/please_provide_a_name.h5',
                   _log_dir='logs/fit', no_weight=True, shuffle_on_train=False,
                   lr_magnif_on_plateau=0.8,
                   reduce_lr_on_plateau_patience=10,
                   framework_name='tensorflow'):
    """
    Returns the best trained model and history objects of the currently provided train & test set
    """

    class_weight = {}
    def calc_cl_w():
        print(_y_train.shape)
        cls_freq = _y_train.mean(axis=0) 
        print(cls_freq)
        _w = np.max(cls_freq)/cls_freq
        print(_w)
        return _w/np.sum(_w)*_y_train.shape[1]
    
    for i, w in zip(range(_y_train.shape[1]), calc_cl_w()):
        class_weight[i] = w if not no_weight else 1

    print(len(class_weight))
    for key in class_weight:
        print(f'{key=}, {class_weight[key]=}')
    print(class_weight)

    checkpoint_path = _save_name
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if framework_name == 'tensorflow':
        callbacks = []
        callbacks.append(EarlyStopping(patience=early_stopping_patience, 
                                       start_from_epoch=boot_strap_epochs))

        # Create checkpoint callback
        callbacks.append(ModelCheckpoint(checkpoint_path,
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=False,
                                         verbose=0))
        # Tensorboard Callback
        if _log_dir is not None:
            callbacks.append(TensorBoard(log_dir=_log_dir, histogram_freq=1))

        # Reduce Learning rate after plateau
        callbacks.append(ReduceLROnPlateau(monitor='loss', 
                                           factor=lr_magnif_on_plateau,
                                           patience=reduce_lr_on_plateau_patience,
                                           min_lr=0.00000001, verbose=1))

        # Training the model
        history = _model.fit(_X_train,
                             _y_train,
                             batch_size=batch_size,
                             validation_data=(_X_test, _y_test),
                             epochs=_epochs,
                             verbose=1,
                             shuffle=shuffle_on_train,
                             use_multiprocessing=True,
                             class_weight = class_weight,
                             callbacks=callbacks)

    elif framework_name == 'pytorch':
        raise NotImplementedError("Please someone implements it and send a pull request!! {framework_name=}")
        import torch.utils.data as Data
        from torch_utils import calc_loss_acc_output

        torch_dataset = Data.TensorDataset(torch.FloatTensor(_X_train), torch.tensor(_y_train).long())
        train_loader = Data.DataLoader(dataset = torch_dataset,
                                       batch_size = batch_size,
                                       shuffle = shuffle_on_train,
                                       drop_last = _X_train.shape[0] % batch_size == 1
                                       )


        parameters = ContiguousParams(_model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               'min', 
                                                               factor=lr_magnif_on_plateau, 
                                                               patience=reduce_lr_on_plateau_patience,
                                                               min_lr=0.00000001, 
                                                               verbose=True)
        optimizer = _model.get_optimizer()
        loss_function = _model.get_loss_function()
        
        start_time = time.time()
        log_training_duration = []
        best_val_loss = 1000000000

        true_sum_data = torch.tensor(0)
        loss_sum_data = torch.tensor(0)
        for epoch in range (epochs):

            true_sum_data *= 0
            loss_sum_data *= 0
            _model.prepare_before_epoch(epoch)
            for step, (x,y) in enumerate(train_loader):
                batch_x = x.cuda()
                batch_y = y.cuda()
                output_bc = _model(batch_x)[0]
                
                # cal the sum of pre loss per batch 
                loss = loss_function(output_bc, batch_y)
                loss_sum_data = loss_sum_data + loss_bc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()
                true_num_bc = torch.sum(pred_bc == y).data
                true_sum_data = true_sum_data + true_num_bc

            loss_train = loss_sum_data.data.item()/y_data.shape[0]
            acc_train = true_sum_data.data.item()/y_data.shape[0]

            # validation
            _model.eval()
            loss_valid, acc_valid, _ = calc_loss_acc_output(_model, loss_function, _X_test, _y_test)
            _model.train()

            # update lr
            scheduler.step(loss_train)
            lr = optimizer.param_groups[0]['lr']

            # log lr&train&validation loss&acc per epoch
            lr_results.append(lr)
            loss_train_results.append(loss_train)    
            accuracy_train_results.append(acc_train)
            loss_validation_results.append(loss_valid)    
            accuracy_validation_results.append(acc_valid)

            # print training process
            print('Epoch:', (epoch+1), '|lr:', lr,
                  '| train_loss:', loss_train, 
                  '| train_acc:', accuracy_train, 
                  '| validation_loss:', loss_validation, 
                  '| validation_acc:', accuracy_validation)

            # log training time 
            per_training_duration = time.time() - start_time
            log_training_duration.append(per_training_duration)

            # save best model
            if best_val_loss > loss_valid:
                best_val_loss = loss_valid
                torch.save(_model.state_dict(), checkpoint_path)

            history = pd.DataFrame(data = np.zeros((EPOCH,5),dtype=np.float), 
                                   columns=['train_acc','train_loss','val_acc','val_loss','lr'])
            history['accuracy'] = accuracy_train_results
            history['loss'] = loss_train_results
            history['val_accuracy'] = accuracy_validation_results
            history['val_loss'] = loss_validation_results
            history['lr'] = lr_results

            class HistoryWrapper:
                history = history
                epoch = list(range(1, epochs+1)

            history = HistoryWrapper()

    else:
        raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")

    return _model, history, checkpoint_path


def load_dataset(dataset, plot_details=False):
    """
    dataset: 'daphnet', 
             'daphnet_losocv_i', where 1 <= i <= 10
             'pamap2', 
             'pamap2_full', 
             'pamap2_losocv_i', where 1 <= i <= 8
             'pamap2_full_losocv_i', where 1 <= i <= 9
             'opportunity, 
             'ucihar', 
             'ucihar_losocv_i', where 1 <= i <=  30
             'ispl'
    """

    ratio_re_match = re.match('(.*)_ratio_(\d+)_(\d+)_(\d+)', dataset)
    if ratio_re_match:
        dataset = ratio_re_match.groups()[0]

    if dataset.startswith('opportunity_'):
        datapath = f'dataset/opportunity'
    elif dataset.startswith('pamap2_'):
        datapath = f'dataset/pamap2'
    elif dataset.startswith('daphnet_'):
        datapath = f'dataset/daphnet'
    elif dataset.startswith('ucihar_'):
        datapath = f'dataset/ucihar'
    else:
        datapath = f'dataset/{dataset}'

    n_signals, win_size, n_classes, n_steps, length =  get_data_configuration(dataset)

    # ................................................................................................ #
    # **************** 1. Load Dataset ******************
    # These have preconfigured train/test splits
    X_train, y_train_int, X_val, y_val_int, X_test, y_test_int, labels = _load_dataset(dataset, datapath)

    if not dataset.startswith('ucihar'):
        X_train, y_train_int = segment(X_train, y_train_int, win_size, dataset_signals=n_signals)
        X_val, y_val_int = segment(X_val, y_val_int, win_size, dataset_signals=n_signals)
        X_test, y_test_int = segment(X_test, y_test_int, win_size, dataset_signals=n_signals)

    y_train = transform_y(y_train_int, n_classes, dataset)
    y_val = transform_y(y_val_int, n_classes, dataset)
    y_test = transform_y(y_test_int, n_classes, dataset)

    if ratio_re_match is not None:
        ratio_sum = 0
        for i in [1, 2, 3]:
            ratio_sum += int(ratio_re_match.groups()[i])

        train_ratio  = int(ratio_re_match.groups()[1])/ratio_sum
        valid_ratio  = int(ratio_re_match.groups()[2])/ratio_sum
        test_ratio  =  int(ratio_re_match.groups()[3])/ratio_sum

        X = np.concatenate([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])
        ix = np.arange(X.shape[0])
        ix_train, ix_test = train_test_split(ix, test_size = test_ratio)
        ix_train, ix_valid= train_test_split(ix_train, test_size = valid_ratio/(train_ratio+valid_ratio))

        X_train = X[ix_train]
        X_valid = X[ix_valid]
        X_test = X[ix_test]
        y_train = y[ix_train]
        y_valid= y[ix_valid]
        y_test = y[ix_test]

    # Using original train/val/test split from datareader.py

    if plot_details:
        frequencies = y_train_int.mean(axis=0) * 100
        frequencies_df = pd.DataFrame(frequencies, index=labels, columns=['frequency'])
        frequencies_df.plot(kind='bar', legend=False)
        plt.show()

    return X_train, y_train, X_val, y_val, X_test, y_test, labels, n_classes


# ................................................................................................ #
# Plotting useful results from training
def plot_classification_report(cr, title='Classification Report ', with_avg_total=False,
                               cmap=plt.get_cmap('plasma'), path="images/cr.png"):
    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2: (len(lines) - 4)]:
        #         print(line)
        t = line.split()
        if len(t) <= 0:
            continue
        #         print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        #         print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[-2].split()[2:-1]
        #         print(aveTotal)
        classes.append('avg/total')
        vAveTotal = [float(x) for x in aveTotal]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.savefig(path)
    plt.show()


def plot_metrics(history, model, dataset, image_path):
    metrics = ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
            #plt.ylim([0, np,min([plt.ylim()[1], 5])])
        elif metric == 'accuracy':
            plt.ylim([0.3, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()
        plt.title(f"{name} of {model} on the {dataset} dataset")

    plt.savefig(image_path, bbox_inches='tight')
    plt.show()

