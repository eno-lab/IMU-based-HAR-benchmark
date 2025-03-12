import os
import re

import numpy as np
import pandas as pd
from keras_utils import IS_KERAS_VERSION_GE_3, StopWithNan

def evaluate_model(_model, _X_train, _y_train, _X_test, _y_test, 
                   _epochs=20, early_stopping_patience=10, boot_strap_epochs=0,
                   batch_size=64, _save_name='trained_models/please_provide_a_name.h5',
                   _log_dir='logs/fit', no_weight=True, shuffle_on_train=False,
                   lr_magnif_on_plateau=0.8,
                   reduce_lr_on_plateau_patience=10,
                   show_epoch_time_detail=False,
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


    if framework_name in ('tensorflow', 'keras'):
        if not IS_KERAS_VERSION_GE_3:
            from tensorflow import keras
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
        else:
            import keras
            from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

    time_callback = None
    if show_epoch_time_detail:
        import time
        import array
        if framework_name in ('tensorflow', 'keras'):
            class TimeHistory(keras.callbacks.Callback):
                def on_train_begin(self, logs={}):
                    self.train_batch_times = np.zeros((2,100000000), dtype=int)
                    self.train_count = 0

                def on_test_begin(self, logs={}):
                    self.valid_batch_times = np.zeros((2,100000000), dtype=int)
                    self.valid_count = 0

                def on_train_batch_begin(self, batch, logs=None):
                    self.train_batch_times[0, self.train_count] = time.perf_counter_ns()

                def on_train_batch_end(self, batch, logs=None):
                    self.train_batch_times[1, self.train_count] = time.perf_counter_ns()
                    self.train_count += 1

                def on_test_batch_begin(self, batch, logs=None):
                    self.valid_batch_times[0, self.valid_count] = time.perf_counter_ns()

                def on_test_batch_end(self, batch, logs=None):
                    self.valid_batch_times[1, self.valid_count] = time.perf_counter_ns()
                    self.valid_count += 1
        if framework_name in ('pytorch', 'torch'):
            class TimeHistory():
                def on_train_begin(self, logs={}):
                    self.train_batch_times = np.zeros((2,100000000), dtype=int)
                    self.train_count = 0

                def on_test_begin(self, logs={}):
                    self.valid_batch_times = np.zeros((2,100000000), dtype=int)
                    self.valid_count = 0

                def on_train_batch_begin(self, batch, logs=None):
                    self.train_batch_times[0, self.train_count] = time.perf_counter_ns()

                def on_train_batch_end(self, batch, logs=None):
                    self.train_batch_times[1, self.train_count] = time.perf_counter_ns()
                    self.train_count += 1

                def on_test_batch_begin(self, batch, logs=None):
                    self.valid_batch_times[0, self.valid_count] = time.perf_counter_ns()

                def on_test_batch_end(self, batch, logs=None):
                    self.valid_batch_times[1, self.valid_count] = time.perf_counter_ns()
                    self.valid_count += 1

        time_callback = TimeHistory()

    checkpoint_path = _save_name
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if framework_name in ('tensorflow', 'keras'):
        callbacks = []
        if time_callback is not None:
            callbacks.append(time_callback)
        callbacks.append(StopWithNan())
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
        if lr_magnif_on_plateau > 0 and reduce_lr_on_plateau_patience > 0:
            callbacks.append(ReduceLROnPlateau(monitor='loss', 
                                               factor=lr_magnif_on_plateau,
                                               patience=reduce_lr_on_plateau_patience,
                                               min_lr=0.00000001, verbose=1))
        elif lr_magnif_on_plateau < 0 or reduce_lr_on_plateau_patience < 0:
            callbacks.append(ReduceLROnPlateau(monitor='loss', 
                                               factor=0.1,
                                               patience=1000000000000,
                                               min_lr=0.00000001, verbose=1))


        _fit_config = {'batch_size': batch_size,
                       'validation_data': (_X_test, _y_test),
                       'epochs': _epochs,
                       'verbose': 1,
                       'shuffle': shuffle_on_train,
                       'class_weight':  class_weight,
                       'callbacks': callbacks
                      }
        
        if not IS_KERAS_VERSION_GE_3:
            _fit_config['use_multiprocessing'] = True
        # Training the model
        history = _model.fit(_X_train,
                             _y_train,
                             **_fit_config)

    elif framework_name in ('pytorch', 'keras'):
        import torch.utils.data as Data
        from torch_utils import calc_loss_acc_output
        import torch
        import time

        _model.cuda()

        torch_dataset = Data.TensorDataset(torch.FloatTensor(_X_train), torch.FloatTensor(_y_train))

        train_loader = Data.DataLoader(dataset = torch_dataset,
                                       batch_size = batch_size,
                                       shuffle = shuffle_on_train,
                                       drop_last = _X_train.shape[0] % batch_size != 0
                                       )



        optimizer = _model.get_optimizer()
        loss_function = _model.get_loss_function()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               'min', 
                                                               factor=lr_magnif_on_plateau, 
                                                               patience=reduce_lr_on_plateau_patience,
                                                               min_lr=0.00000001)
        
        train_start_time = time.time()
        start_time = train_start_time
        log_training_duration = []
        best_val_loss = 1000000000

        true_sum_data = torch.tensor(0)
        loss_sum_data = torch.tensor(0)
        if time_callback is not None:
            time_callback.on_train_begin()


        # init lr&train&test loss&acc log
        lr_results = [] 
        loss_train_results = []
        accuracy_train_results = []
        loss_validation_results = []
        accuracy_validation_results = []

        for epoch in range (_epochs):
            true_sum_data *= 0
            loss_sum_data *= 0

            _tmp_total=0
            _model.train()
            _model.prepare_before_epoch(epoch)
            for step, (x,y) in enumerate(train_loader):
                batch_x = x.cuda()
                batch_y = y.cuda()

                if time_callback is not None:
                    time_callback.on_train_batch_begin(step)

                output_bc = _model(batch_x)
                if time_callback is not None:
                    time_callback.on_train_batch_end(step)
                
                # cal the sum of pre loss per batch 
                loss = loss_function(output_bc, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum_data = loss_sum_data + loss
                #pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()
                pred_bc = torch.argmax(output_bc, dim=1)
                _tmp_total += pred_bc.shape[0]
                #print(torch.sum(pred_bc == torch.argmax(batch_y, dim=1)))
                true_num_bc = torch.sum(pred_bc == torch.argmax(batch_y, dim=1)).data
                true_sum_data = true_sum_data + true_num_bc


            loss_train = loss_sum_data.data.item()/_y_train.shape[0]
            acc_train = true_sum_data.data.item()/_y_train.shape[0]

            # validation
            _model.eval()
            loss_valid, acc_valid, _ = calc_loss_acc_output(_model, loss_function, 
                                                            _X_test, _y_test,
                                                            time_callback)
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

            # log training time 
            _end_time = time.time()
            per_training_duration = _end_time - start_time
            start_time = _end_time
            log_training_duration.append(per_training_duration)

            # print training process
            print(f'Epoch: {epoch+1:4d}' +
                  f'| time: {per_training_duration:.6f}' +
                  f'| lr: {lr:.6f}' +
                  f'| train_loss: {loss_train:.6f}' + 
                  f'| train_acc: {acc_train:.6f}' +
                  f'| val_loss: {loss_valid:.6f}' +
                  f'| val_acc: {acc_valid:.6f}')


            # save best model
            if best_val_loss > loss_valid:
                best_val_loss = loss_valid
                torch.save(_model.state_dict(), checkpoint_path)

        history = pd.DataFrame(data = np.zeros((len(accuracy_train_results),5),dtype=float), 
                               columns=['accuracy','loss','val_accuracy','val_loss','lr'])
        history['accuracy'] = accuracy_train_results
        history['loss'] = loss_train_results
        history['val_accuracy'] = accuracy_validation_results
        history['val_loss'] = loss_validation_results
        history['lr'] = lr_results

        class HistoryWrapper:

            def __init__(self):
                self.history = history
                self.epoch = list(range(1, _epochs+1))

        history = HistoryWrapper()

    else:
        raise NotImplementedError(f"Invalid DNN framework is specified. {framework_name=}")

    if time_callback is not None:
        train_batch_time = np.median(np.diff(time_callback.train_batch_times[:,:time_callback.train_count], axis=0))
        valid_batch_time = np.median(np.diff(time_callback.valid_batch_times[:,:time_callback.valid_count], axis=0))
        for _ in range(5):
            print('\n')
        _model.summary()
        print(f'{train_batch_time=} ns')
        print(f'{valid_batch_time=} ns')

        for _ in range(5):
            print('\n')

    return _model, history, checkpoint_path

