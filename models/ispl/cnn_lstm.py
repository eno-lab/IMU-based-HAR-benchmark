from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPool1D, Concatenate, Activation, Add, GlobalAveragePooling1D, \
    Dense, LSTM, TimeDistributed, Reshape, BatchNormalization, Bidirectional, Flatten, MaxPooling1D, Dropout, \
    SeparableConv1D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy, Reduction
import tensorflow as tf

# CNN-LSTM Model
def cnn_lstm(x_shape,
             n_classes,
             out_loss, 
             out_activ, 
             n_hidden=128,
             learning_rate=0.01,
             n_steps=4,
             regularization_rate=0.01,
             cnn_depth=3,
             lstm_depth=2,
             metrics=['accuracy']):
    """ CNN1D_LSTM version 1: Divide 1 window into several smaller frames, then apply CNN to each frame
    - Input data format: [None, n_frames, n_timesteps, n_signals]"""
    if n_steps is None:
        n_steps = -1
        considered_steps = [4, 3, 5, 2]
        for n in considered_steps:
            if input_shape[1] % n == 0:
                n_steps = n
                break

        if n_steps < 0:
            raise ValueError(f"Failed to auto n_steps selection. The length of input cannot be divided by any of {considered_steps}")

    elif x_shape[1] % n_steps != 0:
        raise ValueError(f"The length of input cannot be divided by {n_steps}")

    length = int(x_shape[1]//n_steps)

    _input_shape = x_shape[1:]
    m = Sequential()

    m.add(Reshape((n_steps, length, x_shape[-1]), input_shape=_input_shape))
    m.add(BatchNormalization())
    m.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')))
    m.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu')))
    m.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    m.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, activation='relu')))
    m.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))
    m.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    m.add(TimeDistributed(Flatten()))
    for _ in range(lstm_depth-1):
        m.add(LSTM(n_hidden, return_sequences=True,
                   kernel_regularizer=l2(regularization_rate)))
    m.add(LSTM(n_hidden))
    m.add(Dropout(0.5))
    m.add(Dense(100, activation='relu',
                kernel_regularizer=l2(regularization_rate)))
    m.add(Dense(n_classes, activation=out_activ))

    m.compile(loss=out_loss,
              optimizer=Adam(learning_rate=learning_rate, amsgrad=True),
              weighted_metrics=metrics)
    return m


def get_config(dataset, lr_magnif=1):
    # Give model specific configurations
    return {'n_hidden': 512, 
            'n_steps': None, # Auto
            'learning_rate': 0.0005 * lr_magnif, 'cnn_depth': 3, 'lstm_depth': 2,
            'regularization_rate': 0.000093}


def gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config):
    return cnn_lstm(input_shape, n_classes, out_loss, out_activ, metrics=metrics, **config)


def gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=['accuracy'], lr_magnif=1):
    config = get_config(dataset, lr_magnif)
    return gen_model(input_shape, n_classes, out_loss, out_activ, metrics, config), config


def get_optim_config(dataset, trial, lr_magnif=1):
    raise NotImplementedError("No config for optimization")


def get_dnn_framework_name():
    return 'tensorflow'
